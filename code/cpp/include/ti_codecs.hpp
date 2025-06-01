#pragma once

// #include <string>
// #include <limits>
#include <iostream>
#include <unordered_map>
#include <cstring>
#include <span>
#include <functional>

#include "interface/InterfaceTypes.hpp"
#include "runtime/RuntimeTypes.hpp"

#include <zstd.h> // ZSTD general purpose compression

#include "roaring.hh"

#ifdef ENABLE_FASTPFOR
#include "fastpfor/codecfactory.h"
#endif

#ifdef ENABLE_MORE_COMPRESSIONS
#include "fastbit/bitvector.h"
#include "dtl/bitmap/util/bitmap_tree.hpp"
#include "dtl/bitmap/teb_builder.hpp"
#include "dtl/bitmap/teb_flat.hpp"
#include "dtl/bitmap/teb_wrapper.hpp"
#include "dtl/bitmap/teb_legacy.hpp"

#endif

namespace TeamIndex {

    // number of bytes a list may be larger (after "compression") than the input, without falling back to CodecID::COPY
    static float SIZE_OVERHEAD_TOLERANCE = 1.0;
    static unsigned TINY_SAVINGS_THRESHOLD = 256; // in byte

    static float BITMAP_SPARSITY_LIMIT = 0.01; // in %, when to skip bitmap based compression techniques

    struct Codec {
        using EncodeFunction = std::function<ListSizeCompressed(const std::vector<IDType>&, std::span<BufferType>, std::size_t)>;// last parameter is the domain cardinality of the IDs
        using DecodeFunction = std::function<std::span<IDType>(BufferType*, ListSizeCompressed, IDType*, ListCardinality)>;

        EncodeFunction encode; 
        DecodeFunction decode; 
        unsigned int minimum_list_size = 0; // in byte. Used to allocate enough for small lists, when metadata is large
    private:
    };


    template <bool aredistinct=false>
    static void delta(const std::vector<IDType>& in, std::span<IDType>& out) {
        if (in.empty()) return;
        out[0] = in[0];
        for (size_t k = 0; k < in.size()-1; ++k) {
            if constexpr (aredistinct) out[k] = in[k + 1] - in[k] - 1;
            else out[k+1] = in[k + 1] - in[k];
        }
    }

    template <bool aredistinct=false, typename container = std::span<IDType>>
    static void undelta_inplace(container& vec) {
        for (size_t k = 1; k < vec.size(); ++k) {
            if constexpr (aredistinct) vec[k] = vec[k-1] + vec[k] + 1;
            else vec[k] = vec[k-1] + vec[k];
        }
    }

    //// Compression and Decompression functions:
    // storing without compression is a simple copy:

    static Codec::EncodeFunction
    encode_copy = [](const std::vector<IDType>& input, std::span<BufferType> output, std::size_t) -> ListSizeCompressed
    {
        // just copy data
        memcpy(output.data(), input.data(), input.size()*sizeof(IDType));

        return input.size()*sizeof(IDType); // in byte
    };

    static Codec::DecodeFunction
    decode_copy = [](BufferType *input,
                     ListSizeCompressed size,
                     IDType* output,
                     ListCardinality expected_card)-> std::span<IDType>
    {
        assert(size == sizeof(IDType)*expected_card); // holds true for no compression
        // memcpy(output, input, size);
        return {reinterpret_cast<IDType*>(input), expected_card}; // return where we stored the result
    };


    template<int compression_level>
    static Codec::EncodeFunction
    encode_zstd = [](const std::vector<IDType>& input, std::span<BufferType> output, std::size_t) -> ListSizeCompressed
    {
        // apply delta encoding
        auto aligned_input_ptr = reinterpret_cast<IDType*>(std::aligned_alloc(64U, input.size()*sizeof(IDType)));
        std::span<IDType> deltas{aligned_input_ptr, input.size()};
        TeamIndex::delta<false>(input, deltas);

        // apply ZSTD compression on deltas:
        auto size_bytes = ZSTD_compress(output.data(), output.size(), deltas.data(), deltas.size()* sizeof(IDType), compression_level);

        free(aligned_input_ptr);
        return size_bytes;
    };

    static Codec::DecodeFunction
    decode_zstd = [](BufferType *input,
                     ListSizeCompressed size,
                     IDType* output,
                     ListCardinality expected_card)-> std::span<IDType>
    {
        auto decompressed_size =  sizeof(IDType)*expected_card;

        auto size_bytes = ZSTD_decompress(output, decompressed_size, input, size); // revert ZStd compression
        assert(decompressed_size == size_bytes);

        std::span<IDType> result{output, expected_card};

        undelta_inplace<false>(result); // revert delta encoding

        return result;
    };


#ifdef ENABLE_FASTPFOR

    template<CodecID codec_id, bool use_delta>
    static Codec::EncodeFunction
    encode_fastpyfor = [](const std::vector<IDType>& input, std::span<BufferType> output, std::size_t) -> ListSizeCompressed
    {
        if (input.empty()) return 0; // we do nothing, if there is nothing to compress.
        assert((output.size() % sizeof(uint32_t)) == 0);
        std::size_t out_size = output.size() / sizeof(uint32_t); // size as a number of 32-bit values, not bytes

        auto factory = FastPForLib::CODECFactory(); // TODO make global


        auto codec = factory.getFromName(TeamIndex::to_string(codec_id)); 
    //    assert(iter != FastPForLib::CODECFactory::scodecmap.end());
//        auto codec = iter->second;

        // we try to compress tentatively and might overwrite the result with a simple copy of the list
        std::size_t out_buff_size = input.size()*1.5;
        auto aligned_input_ptr = reinterpret_cast<IDType*>(std::aligned_alloc(64U, input.size()*sizeof(IDType)));
        auto aligned_output_ptr = reinterpret_cast<uint32_t*>(std::aligned_alloc(64U, out_buff_size*sizeof(IDType)));
        memset(aligned_output_ptr, 0, out_buff_size*sizeof(IDType));

        std::span<IDType> deltas{aligned_input_ptr,input.size()};

        uint32_t* out = reinterpret_cast<uint32_t*>(output.data());
        if constexpr (use_delta) {
            TeamIndex::delta<false>(input, deltas);
            codec->encodeArray(deltas.data(), deltas.size(), aligned_output_ptr, out_size);
        }
        else {
            codec->encodeArray(input.data(), input.size(), aligned_output_ptr, out_size);
        }

        auto size_byte = out_size*sizeof(uint32_t);

        if (size_byte > output.size()) {
            std::cerr << "\tBuffer overflow during compression, allocated enough memory? "
                    << out_size << "/"  << output.size()/sizeof(uint32_t) << std::endl;
        }

        memcpy(out, aligned_output_ptr, size_byte);

        free(aligned_input_ptr);
        free(aligned_output_ptr);

        assert(out_size > 0);

        return size_byte; // return size of bytes occupied by the compressed list
    };

    template<CodecID codec_id, bool use_delta>
    static Codec::DecodeFunction
    decode_fastpyfor = [](BufferType *input,
                          ListSizeCompressed size, // in byte
                          IDType* output,
                          ListCardinality expected_card) -> std::span<IDType> {

        auto comp_size = size/sizeof(uint32_t); // "size" is in bytes to number of 32-bit integers
        std::size_t cardinality = expected_card; // local copy, can be changed by the decodeArray function below

        auto factory = FastPForLib::CODECFactory();

        auto codec = factory.getFromName(TeamIndex::to_string(codec_id));

        auto aligned_input_ptr = reinterpret_cast<uint32_t*>(std::aligned_alloc(64U, size));
        auto aligned_output_ptr = reinterpret_cast<uint32_t*>(std::aligned_alloc(64U, cardinality*sizeof(uint32_t)));
        memmove(aligned_input_ptr, input, size); // make sure the data is on aligned memory

        codec->decodeArray(aligned_input_ptr, comp_size, aligned_output_ptr, cardinality);
        assert(cardinality == expected_card); // we know how large it actually is, so this should match in any case

        memmove(output, aligned_output_ptr, cardinality*sizeof(uint32_t));
        std::span<IDType> result{output, cardinality};

        free(aligned_input_ptr);
        free(aligned_output_ptr);

        if constexpr (use_delta) {
            undelta_inplace<false>(result); // undo delta encoding
        }

        return result;
    };

    template<int compression_level>
    static Codec::EncodeFunction
    encode_varintzstd = [](const std::vector<IDType>& input, std::span<BufferType> output, std::size_t) -> ListSizeCompressed
    {
        std::span<BufferType> iterm_buffer_span = {reinterpret_cast<BufferType*>(std::malloc(output.size())), output.size()};

        // apply Delta encoding + Varint encoding
        auto size_bytes_first = encode_fastpyfor<CodecID::VARINT, true>(input, iterm_buffer_span, 0);

        // apply ZStd on top
        auto size_bytes = ZSTD_compress(output.data(), output.size(), iterm_buffer_span.data(), size_bytes_first, compression_level);

        // free intermediate buffer
        free(iterm_buffer_span.data());
        return size_bytes; // in byte
    };

    static Codec::DecodeFunction
    decode_varintzstd = [](BufferType *input,
                     ListSizeCompressed size,
                     IDType* output,
                     ListCardinality expected_card)-> std::span<IDType>
    {
        auto decompressed_size =  sizeof(IDType)*expected_card;
        auto iterm_buffer = reinterpret_cast<BufferType*>(std::malloc(decompressed_size));

        // first revert ZStd compression:
        auto size_bytes_interm = ZSTD_decompress(iterm_buffer, decompressed_size, input, size);

        // then revert varint- and delta encoding:
        auto result = decode_fastpyfor<CodecID::VARINT, true>(iterm_buffer, size_bytes_interm, output, expected_card);

        free(iterm_buffer);

        return result; // return where we stored the result
    };

#endif

    class CodecFactory {

        public:
        CodecFactory() {
            init_factory();
        }

        Codec operator[](CodecID codec_id) const {
            assert((static_cast<unsigned>(codec_id) < CODEC_COUNT) && (codec_id != CodecID::UNKNOWN));
            return _codecs[static_cast<std::size_t>(codec_id)];
        }

        private:
        void init_factory() {
        
        _codecs[static_cast<std::size_t>(CodecID::COPY)] = Codec{encode_copy, decode_copy, 0};
        _codecs[static_cast<std::size_t>(CodecID::ZSTD)] = Codec{encode_zstd<0>, decode_zstd, 0};
        _codecs[static_cast<std::size_t>(CodecID::ZSTDMORE)] = Codec{encode_zstd<ZSTD_MORE_LEVEL>, decode_zstd, 0}; // apply more aggressive compression, value taken from cmake definition
        _codecs[static_cast<std::size_t>(CodecID::ZSTDFASTER)] = Codec{encode_zstd<-1>, decode_zstd, 0}; // faster compression

        _codecs[static_cast<std::size_t>(CodecID::ROARING)] = Codec{[](const std::vector<IDType>& input,
                                            std::span<BufferType> output, std::size_t) -> ListSizeCompressed
                                         {
                                            if (input.empty())
                                                return 0; // we do nothing, if there is nothing to compress.

                                            // Then, we write actual data. In this case, we simply copy the bytes:
                                            roaring::Roaring bm(input.size(), input.data());

                                            assert(bm.cardinality() == input.size());
                                            auto comp_size = bm.getFrozenSizeInBytes();
                                            if (comp_size > output.size()) {
                                                throw std::runtime_error("Output buffer too small: "+std::to_string(comp_size)+">"+std::to_string(output.size()));
                                            }

                                            bm.writeFrozen(reinterpret_cast<char*>(output.data()));
                                            return comp_size;
                                         },
                                         [] (BufferType *input,
                                            ListSizeCompressed size,
                                            IDType* output,
                                            ListCardinality expected_card) -> std::span<IDType>
                                         {
                                            
                                            static_assert(sizeof(uint32_t) == sizeof(IDType));
                                            
                                            // THIS IS NOT THE WAY!!!
                                            const roaring::Roaring bm = roaring::Roaring::frozenView(input, size);
                                            assert(bm.cardinality() == expected_card);
                                            bm.toUint32Array(output);
                                            return {output, bm.cardinality()}; // return where we stored the result
                                         }};


#ifdef ENABLE_FASTPFOR
        _codecs[static_cast<std::size_t>(CodecID::VARINT)] = Codec{encode_fastpyfor<CodecID::VARINT,true>, decode_fastpyfor<CodecID::VARINT,true>, 0};
        _codecs[static_cast<std::size_t>(CodecID::VARINTZSTD)] = Codec{encode_varintzstd<0>, decode_varintzstd, 0};
        _codecs[static_cast<std::size_t>(CodecID::SIMDOPTPFOR)] = Codec{encode_fastpyfor<CodecID::SIMDOPTPFOR,true>, decode_fastpyfor<CodecID::SIMDOPTPFOR,true>, 0};
        _codecs[static_cast<std::size_t>(CodecID::SIMDSIMPLEPFOR)] = Codec{encode_fastpyfor<CodecID::SIMDSIMPLEPFOR,true>, decode_fastpyfor<CodecID::SIMDSIMPLEPFOR,true>, 0};
        _codecs[static_cast<std::size_t>(CodecID::SIMDFASTPFOR256)] = Codec{encode_fastpyfor<CodecID::SIMDFASTPFOR256,true>, decode_fastpyfor<CodecID::SIMDFASTPFOR256,true>, 0};
#endif

#ifdef ENABLE_MORE_COMPRESSIONS
        fact_codecs[static_cast<std::size_t>(CodecID::WAH)] = Codec{[](const std::vector<IDType>& input,
                                         std::span<BufferType> output, std::size_t domain_size) -> ListSizeCompressed
                                      {
                                          // check for sparsity in an uncompressed bitvector representation:
                                          if (input.size()*100.0/domain_size < BITMAP_SPARSITY_LIMIT) {
                                              return std::numeric_limits<ListSizeCompressed>::max();
                                          }
                                          // involves a conversion (with copy) to ibis:array_t, but we don't care here:
                                          wah::bitvector bm;
                                          for (auto idx: input) {
                                              bm.setBit(idx, 1); // set bit to 1 at location idx
                                          }
                                          if (bm.compressible()) { // not required..?
                                              bm.compress();
                                          }
                                          assert(bm.cnt() == input.size());
//                                          bm.print(std::cout);

                                          wah::array_t<wah::bitvector::word_t> wah_words;
                                          bm.write(wah_words);
                                          auto size_bytes = wah_words.size()*sizeof(wah::bitvector::word_t);
                                          assert(size_bytes == bm.getSerialSize());
                                          if (wah_words.size()*sizeof(wah::bitvector::word_t) > output.size()) {
                                              // not enough space allocated, copy probably results in segfault!
                                              // do nothing and return size that triggers fallback_mechanism:
//                                              return input.size()*sizeof(IDType) + SIZE_OVERHEAD_TOLERANCE + 1;
                                              return std::numeric_limits<ListSizeCompressed>::max();
                                          }

                                          memcpy(output.data(), wah_words.data(), sizeof(wah::bitvector::word_t)*wah_words.size());

                                          return size_bytes;
                                      },
                                      [](BufferType *input,
                                         ListSizeCompressed size_bytes,
                                         IDType* output,
                                         ListCardinality expected_card)-> std::span<IDType>
                                      {

                                          auto word_count = size_bytes/sizeof(wah::bitvector::word_t);
                                          assert(size_bytes % sizeof(wah::bitvector::word_t) == 0);
                                          auto input_ptr = reinterpret_cast<wah::bitvector::word_t*>(input);

                                          wah::bitvector bm(input_ptr, word_count);// makes a copy
                                          assert(bm.cnt() == expected_card);

                                          // convert WAH bitvector to (uncompressed) list of ids:
                                          wah::bitvector::pit pit(bm); // iterator over active bits in a WAH bitvector
                                          auto i = 0u;
                                          while (*pit < bm.size()) {
                                              output[i++] = *pit;
                                              pit.next();
                                          }
                                          assert(expected_card == i);
                                          assert(i == bm.cnt());
                                          return {output, bm.cnt()};
                                      }, 0 // no minimum allocation size, with copy-fallback
        };
        _codecs[static_cast<std::size_t>(CodecID::TEB)] = Codec{[](const std::vector<IDType>& input,
                                         std::span<BufferType> output, std::size_t domain_size) -> ListSizeCompressed
                                      {
                                          // check for sparsity in an uncompressed bitvector representation:
                                          if (input.size()*100.0/domain_size < BITMAP_SPARSITY_LIMIT) {
                                              return std::numeric_limits<ListSizeCompressed>::max();
                                          }
                                          dtl::bitmap_tree<> tree(domain_size);
                                          tree.init_tree(input);
                                          dtl::teb_builder builder(std::move(tree));

                                          auto size_bytes = builder.serialized_size_in_words()*sizeof(dtl::teb_word_type);
                                          if (size_bytes > output.size()) {
                                              // not enough space allocated, copy probably results in segfault!
                                              // do nothing and return size that triggers fallback_mechanism:
//                                              return input.size()*sizeof(IDType) + SIZE_OVERHEAD_TOLERANCE + 1;
                                              return std::numeric_limits<ListSizeCompressed>::max();
                                          }

                                          builder.serialize(reinterpret_cast<dtl::teb_word_type*>(output.data()));

                                          return size_bytes;
                                      },
                                      [](BufferType *input,
                                         ListSizeCompressed size_bytes,
                                         IDType* output,
                                         ListCardinality expected_card)-> std::span<IDType>
                                      {
                                          dtl::teb_flat teb_flat(reinterpret_cast<dtl::teb_word_type*>(input));
                                          // simply decompress the TreeEncodedBitmap to a list of ids:
                                          auto i = 0u;
                                          for (auto iter = dtl::teb_iter(teb_flat); !iter.end(); iter.next_off()) {
                                              for (auto j = 0u; j < iter.length(); j++) {
                                                  output[i++] = iter.pos()+j;
                                              }
                                          }
                                          assert(expected_card == i);
                                          return {output, i};
                                      }, 0 // no minimum allocation size, no copy-fallback
        };
#endif
        }
        std::array<Codec, CODEC_COUNT> _codecs;
    };

    const static CodecFactory CODEC_FACTORY; 



    static std::tuple<ListSizeCompressed, CodecID> encode(CodecID codec_id, const std::vector<IDType>& input, std::span<BufferType> output, std::size_t domain_size) {
        assert((static_cast<unsigned>(codec_id) < CODEC_COUNT) and (codec_id != CodecID::UNKNOWN));

        Codec codec = TeamIndex::CODEC_FACTORY[codec_id];
//        std::cout << "\tNumber of ids: " << input.size()
//                << " / Relative list size: " << std::setprecision(4) << std::setw(7) << input.size()*100.0/domain_size << std::setprecision(0) << std::setw(0)
//                << "% (" << TeamIndex::to_string(codec_id) << ")" << std::endl;

        // try compression
        std::size_t size_byte = codec.encode(input, output, domain_size);
        
        //---------------------------------------------------------------------
        // Page-level heuristic
        //
        // Let  ⌈x/P⌉  be the number of whole 4 KiB pages needed for x bytes.
        // ❶  If encoded size starts a *new* page while COPY would stay in the
        //    current page, prefer COPY.
        // ❷  For non‑Roaring, non‑COPY codecs require at least one *full page*
        //    of savings before keeping the compression.
        //---------------------------------------------------------------------
        auto pages = [](std::size_t bytes) {
            return (bytes + TeamIndex::PAGESIZE - 1) / TeamIndex::PAGESIZE;
        };

        std::uintptr_t page_off =
            reinterpret_cast<std::uintptr_t>(output.data()) & (TeamIndex::PAGESIZE - 1);

        std::size_t raw_bytes  = input.size() * sizeof(IDType);
        std::size_t comp_pages = pages(page_off + size_byte);
        std::size_t raw_pages  = pages(page_off + raw_bytes);

        bool adds_page      = (comp_pages > raw_pages);
        bool saves_a_page   = (raw_pages - comp_pages) >= 1;
        
        if (codec_id != CodecID::COPY) {
            if (adds_page) {               // rule ❶
                size_byte = encode_copy(input, output, domain_size);
                return {size_byte, CodecID::COPY};
            }
            if ((codec_id != CodecID::ROARING) && !saves_a_page) {   // rule ❷
                size_byte = encode_copy(input, output, domain_size);
                return {size_byte, CodecID::COPY};
            }
        }

        return {size_byte, codec_id};
    }

    static std::span<IDType> decode(CodecID codec_id, BufferType* input, ListSizeCompressed compressed_size, IDType* output, ListCardinality expected_card) {
        assert(input != nullptr);
        assert(compressed_size > 0);
        assert(expected_card > 0);
        assert((static_cast<unsigned>(codec_id) < CODEC_COUNT) and (codec_id != CodecID::UNKNOWN));

        auto codec = TeamIndex::CODEC_FACTORY[codec_id];

        return codec.decode(input, compressed_size, output, expected_card);
    } 
};