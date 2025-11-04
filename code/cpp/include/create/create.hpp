#pragma once

#include <iostream>
#include <algorithm>
#include <math.h>
#include <span>
#include <vector>
#include <unordered_map>
// #include <memory>
#include <numeric>
#include <fstream>
#include <ostream>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "ti_codecs.hpp"

namespace py = pybind11;


namespace TeamIndex {
    using CodecCounts = std::unordered_map<CodecID, std::size_t>;

    class BatchConverter {
    public:
        using data_type = DataType;

        using InvertedList = std::vector<IDType>;
        using InvertedListBuffer = std::vector<InvertedList *>;
        using Offsets = std::vector<BlockCount>; // the underlying type is the same as for Idsoffs.
        using BinCards = std::vector<IDType>; // the underlying type is the same as for Ids
        using Codecs = std::vector<uint8_t>; // the underlying type is the same as for Ids


        BatchConverter(const DataType *bin_edge_data, size_t bin_cnt, size_t cols): _number_of_tuples(0) {

            /// now, we derive the number of bins per dimension and the total
            /// number of bins for this Team:
            _number_of_bins = 1;
            _shape.resize(cols, 2); // initialize with 1, as we have at least one bin per dimension

            _bin_edges.resize(cols);

            // copy bin data to internal storage, but only take ascending values - some columns may have fewer bins than "bin_cnt"
            for (auto col = 0u; col < cols; col++) {
                DataType last_value = bin_edge_data[col * bin_cnt + 0];
                _bin_edges[col].emplace_back(last_value);
                for (size_t bin_id = 1; bin_id < bin_cnt; bin_id++) {
                    auto current_value = bin_edge_data[col * bin_cnt + bin_id];
                    /// count only edge values that are larger then the
                    /// previous. This allows for variable number of bins per
                    /// attribute:
                    if (current_value < last_value) {
                        break;
                    }
                    _bin_edges[col].emplace_back(current_value);
                    _shape[col]++;
                    last_value = current_value;
                }

                _number_of_bins *= _shape[col];
            }

            // std::cout << "Number of bins: " << _number_of_bins << std::endl;
            // for (auto col = 0u; col < cols; col++) {
            //     std::cout << "Bin edges for column " << col << ": [";
            //     for (auto bin_id = 0u; bin_id < _shape[col]; bin_id++) {
            //         std::cout << _bin_edges[col][bin_id] << ", ";
            //     }
            //     std::cout << "] " << _shape[col] << std::endl;
            // }
        }

        ~BatchConverter() {
            // free up all inverted lists that were not fetched!
            for (auto &[il, rows]: _part_inv_lst) {
                for (auto flat_bin_id = 0u; flat_bin_id < il.size(); flat_bin_id++) {
                    delete il[flat_bin_id];
                }
            }
        }

        void process_batch(const py::buffer &batch, IDType first_id) {

            py::buffer_info info = batch.request(false);
            /* Some sanity checks ... */
            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            if (info.shape[0] <= 0 or info.shape[1] != static_cast<ssize_t>(_shape.size()))
                throw std::runtime_error("Incompatible shape!");

            if (info.format == py::format_descriptor<DataType>::format()) {
                process_batch_impl<DataType>((DataType *) info.ptr,
                static_cast<size_t>(info.shape[1]),
                static_cast<size_t>(info.shape[0]),
                first_id);
            }
            
            else if (info.format == py::format_descriptor<DataTypeSP>::format()) {
                process_batch_impl<DataTypeSP>((DataTypeSP *) info.ptr,
                    static_cast<size_t>(info.shape[1]),
                    static_cast<size_t>(info.shape[0]),
                    first_id);
            }
            else {
                throw std::runtime_error(
                    "Incompatible format! Expecting " + py::format_descriptor<DataType>::format() + " but got " +
                    info.format + " instead!");
            }

        }

        template<typename VALUE_TYPE>
        void process_batch_impl(VALUE_TYPE *data, size_t cols, size_t rows, IDType first_id) {

            ////////////////////////////////////////////////////////////////////////////
            /// NOTE: data in "data" is assumed to be in fortran/column-first order! ///
            ////////////////////////////////////////////////////////////////////////////

            auto current_bc = BinCards(_number_of_bins);
            double average_batch_bin_cardinality = ((double) rows) / current_bc.size();
            auto current_il = InvertedListBuffer(_number_of_bins);

            /// "data" arrives in columnar fashion; we'll therefore process it that way.
            /// Multi-dimensional coordinates are converted to 1d
            /// D*[((i_0 * d_1 + i_1 ) d_2 + i_2 ) d_3 ….. + i_(n-1)-1 )S_{n-1} + i_{n-1}]
            /// and are computed in place:
            ///

//            size_t* bin_ids = (size_t*) std::calloc(rows, sizeof(size_t));
            std::vector<size_t> bin_ids;
            bin_ids.resize(rows, 0);

            if (_shape.size() > 1) { // multi-dimensional Team
                /// first two column/innermost term:
                for (auto row = 0u; row < rows; row++) {
                    size_t &flat_bin_id = bin_ids[row];
                    flat_bin_id = get_quantile<VALUE_TYPE>(data[row], 0) * _shape[1];
                    flat_bin_id += get_quantile<VALUE_TYPE>(data[1 * rows + row], 1);

                    if (cols == 2) [[unlikely]] { // TODO: move this check to outside of the loop and copy the loop
                        // in case we have only two columns, we won't visit the loop below
                        current_bc[flat_bin_id]++;

                        /// append id to temporary lists:
                        if (current_il[flat_bin_id] == nullptr) {
                            current_il[flat_bin_id] = new InvertedList();
                            // current_il[flat_bin_id]->reserve(std::ceil(average_batch_bin_cardinality));
                        }
                        current_il[flat_bin_id]->push_back(first_id + row);
                    }
                }

                /// other columns:
                for (auto col = 2u; col < cols; col++) {

                    for (size_t row = 0; row < rows; row++) {
                        size_t &flat_bin_id = bin_ids[row];
                        flat_bin_id *= _shape[col];
                        flat_bin_id += get_quantile<VALUE_TYPE>(data[col * rows + row], col);

                        /// use resulting id in last run for an increment operation:
                        current_bc[flat_bin_id] += (col + 1 == cols);

                        if (col + 1 == cols) [[unlikely]] { // is this the last dimension?

                            /// append id to temporary lists:
                            if (current_il[flat_bin_id] == nullptr) {
                                current_il[flat_bin_id] = new InvertedList();
                                // current_il[flat_bin_id]->reserve(std::ceil(average_batch_bin_cardinality));
                            }
                            current_il[flat_bin_id]->push_back(first_id + row);
                        }
                    }
                }

            } else { // single dimensional Team

                for (auto bin_id = 0u; bin_id < current_il.size(); bin_id++) {
                    current_il[bin_id] = new InvertedList();
                    // current_il[bin_id]->reserve(std::ceil(average_batch_bin_cardinality));
                }

                // simply iterate over all tuples of this batch and add it to the list of the respective bin
                for (auto row = 0u; row < rows; row++) {
                    size_t bin_id = get_quantile<VALUE_TYPE>(data[row], 0);
                    current_bc[bin_id]++; // statistics
                    current_il[bin_id]->push_back(first_id + row); // fill inverted list
                }
            }
            _number_of_tuples += rows;
            _part_inv_lst.emplace_back(std::make_pair(std::move(current_il), rows));
            _bcs.emplace_back(std::move(current_bc));
        }

        void merge_results() {
            /// first, merge bin-counts:

            assert(_part_inv_lst.size() == _bcs.size());

            if (_bcs.size() < 2) {
                return;
//                throw std::runtime_error("Nothing to merge!");
            }
            {
//                teamio::Timer bcs_timer = teamio::Timer( "Merge BCS");
                /// Note:   first bincount buffer will contain the merged result
                /// Note2:  While floating-point addition is non-commutative,
                ///         integer addition is. This allows the usage of std::reduce,
                ///         but reduce does not allow to std::move the lhs of binary ops
                _bcs[0] = std::accumulate(_bcs.begin(), _bcs.end(),
                                          BinCards(_number_of_bins),
                                          this->combine_bcs);

            }
            /// second, concatenate all lists
            ///
            /// Note:   order within the vector matters
            ///         or ids wont be in sorted order!
            ///         Subsequently, while we we have associativity,
            ///         we have NO commutativity!
            ///         Subsequently, we require std::accumulate instead of
            ///         std::reduce.
            {
//                teamio::Timer il_timer = teamio::Timer( "Merge InvLists");
                _part_inv_lst[0] = std::accumulate(_part_inv_lst.begin(),
                                                   _part_inv_lst.end(),
                                                   std::pair<InvertedListBuffer, size_t>{
                                                           /* neutral element of concatenation: */
                                                           InvertedListBuffer(_number_of_bins),
                                                           /* neutral element of addition: */
                                                           0},
                                                   this->combine_invlists);
            }
            /// first entries in _bcs and _partial_inverted_lists now contain the merged results.
            /// erase other, intermediate results:
            _bcs.erase(_bcs.begin() + 1, _bcs.end());
            _part_inv_lst.erase(_part_inv_lst.begin() + 1, _part_inv_lst.end());

        }

        [[nodiscard]]
        std::tuple<std::span<BufferType>, BinCards, Offsets, std::vector<size_t>, Codecs> compress_result(CodecID codec_id, unsigned batch_id= 0) const
        {
            auto codec = TeamIndex::CODEC_FACTORY[codec_id];

            if (batch_id >= _part_inv_lst.size())
                throw std::runtime_error("Invalid batch: " + std::to_string(batch_id));

            const auto &[lists, cardinality] = _part_inv_lst[batch_id];

            Offsets offs;
            offs.resize(_bcs[batch_id].size()+1, 0ul); // track where the list begins (a starting page id!)

            std::vector<size_t> compressed_sizes;
            compressed_sizes.resize(_bcs[batch_id].size(), 0ul); // track actual size of the list

            Codecs codecs;
            codecs.resize(_bcs[batch_id].size(), static_cast<uint8_t>(CodecID::UNKNOWN)); // track the used codec

            std::size_t running_offs = 0ul; // in terms of bytes/sizeof(BufferType)
            unsigned i = 0u;

            // contains all compressed inv. lists in a continuous stream of bytes and their headers.
            // size is just an upper bound, final size should be equal or smaller (when compressed)
            long uncompressed_size = cardinality*sizeof(IDType);

            // estimate buffer size (as a number of bytes):
            // allocate an additional 50 % and 1 MiB more, just in case compression turns out to be inefficient.
            // further, allocate one page per list extra to account for padding.
            // note that we will not actually compress, in that case.
            std::size_t estimated_buffer_size = _bcs[batch_id].size()*(codec.minimum_list_size+PAGESIZE) + uncompressed_size * 1.2;
            estimated_buffer_size = ((estimated_buffer_size * sizeof(BufferType) + PAGESIZE - 1u) / PAGESIZE) * PAGESIZE;
            std::cout << "Allocating " << estimated_buffer_size << " byte as global buffer." << std::endl;

            BufferType* buffer = nullptr;// reinterpret_cast<BufferType*>(calloc(estimated_buffer_size, sizeof(BufferType)));
            auto ret = posix_memalign(reinterpret_cast<void**>(&buffer), PAGESIZE, estimated_buffer_size);
            
            if (ret == ENOMEM) {
                throw std::runtime_error("Not enough memory for buffer!");
            }
            else if (ret == EINVAL) {
                throw std::runtime_error("Invalid alignment for buffer!");
            }
            else if (ret != 0) {
                throw std::runtime_error("Unknown error while allocating buffer!");
            }

            BufferType* buffer_ptr = reinterpret_cast<BufferType*>(buffer); // this pointer will be advanced during compression

            std::cout << "Trying to compress " << lists.size() << " lists with " << TeamIndex::to_string(codec_id) << " ..." << std::endl;
            for (std::vector<IDType>* inv_list_ptr :  lists) { // iterate over all inverted lists, some might be empty
                offs[i] = (running_offs + PAGESIZE - 1) / PAGESIZE; // Note: empty lists "begin" with offsets of the previous lists
                if (inv_list_ptr == nullptr) {
                    // skip empty bins otherwise
                    compressed_sizes[i] = 0;
                    codecs[i] = static_cast<uint8_t>(CodecID::UNKNOWN);
                    i++;
                    continue;
                }

                assert(not inv_list_ptr->empty()); // we did create a list, so it should not be empty! Issue with duplicate quantiles?

                // we use double to be sure the buffer for this list is not too small, causing write to unallocated memory
                // note, that we abort the compression, if it is inefficient, returning the list un-compressed.
                // So a list is always at most as large as before
                std::size_t expected_list_size = inv_list_ptr->size()*sizeof(IDType)*2+codec.minimum_list_size+PAGESIZE;
                
                if (estimated_buffer_size-running_offs < expected_list_size) {
                    throw std::runtime_error("Estimated sized exceeds buffer size!");
                }

                // encode list and write to buffer_ptr.
                std::span<BufferType> buffer_span = {buffer_ptr, expected_list_size};
                // Signature: static std::tuple<ListSizeCompressed, CodecID> encode(CodecID codec_id, const std::vector<IDType>& input, std::span<BufferType> output, std::size_t domain_size)
                auto [compr_size, selected_codec] = TeamIndex::encode(codec_id, *inv_list_ptr, buffer_span, _number_of_tuples);
                assert(compr_size > 0);


                // note: We store lists at a PAGESIZE-aligned address in the buffer (and later in storage). This requires padding
                // auto padding = (PAGESIZE - (compr_size % PAGESIZE)) % PAGESIZE;
                auto size_in_buffer = (compr_size + PAGESIZE - 1) & ~(PAGESIZE - 1);
                assert((size_in_buffer % PAGESIZE) == 0);

                if (size_in_buffer > estimated_buffer_size-offs[i] * PAGESIZE) {
                    throw std::runtime_error("Not enought space in buffer!");
                }


                running_offs += size_in_buffer; // advance offset for later access into the byte stream
                buffer_ptr = reinterpret_cast<BufferType*>(reinterpret_cast<uint8_t*>(buffer_ptr) + size_in_buffer); // advance pointer by size of this list
                
                // store page offset, list size and codec
                compressed_sizes[i] = compr_size;
                codecs[i] = static_cast<uint8_t>(selected_codec);
                i++;
            }
            offs[i] = (running_offs + PAGESIZE - 1) / PAGESIZE; // Final offset
            
            std::cout << "\tFilled buffer size: " << running_offs << " byte" << std::endl;
            std::cout << "\tUnused buffer space: " << static_cast<signed long>(estimated_buffer_size)-static_cast<signed long>(running_offs) << " byte" << std::endl;
            /// erase what we return via this function from the member vectors:
            auto bcs_buffer = _bcs[batch_id];

            return {{buffer, running_offs}, bcs_buffer, offs, compressed_sizes, codecs};
        }

        Codecs dump_index(CodecID codec_id,
                     const std::string& inv_file_name,
                     const std::string& offs_file_name,
                     const std::string& compr_sizes_file_name,
                     const std::string& codecs_file_name,
                     const std::string& bcs_file_name) {
            merge_results();

            auto [inv_lists, cards, offsets, compr_sizes, codecs] = compress_result(codec_id, 0);

            dump_streams(inv_lists,    cards,         offsets,        compr_sizes,           codecs, 
                        inv_file_name, bcs_file_name, offs_file_name, compr_sizes_file_name, codecs_file_name);

            free(inv_lists.data());
            return codecs;
        }


        /**
         * Method that allows returning the index data to python. Simply calls compress_result() for actual work.
         */
        [[nodiscard]]
        std::tuple<py::array_t<BufferType>, BinCards, Offsets, std::vector<size_t>, Codecs>
        get_result(CodecID codec_id, unsigned batch_id = 0) {
            auto [invl, bcs, offs, compr_sizes, codecs] = compress_result(codec_id, batch_id);

            // Create a Python object that will free the allocated
            // memory when destroyed:
            py::capsule free_when_done(invl.data(), [](void *f) {
                auto *data_ptr = reinterpret_cast<BufferType *>(f);
                free(data_ptr);
            });


            return {py::array_t<BufferType>(
                    {invl.size()}, // shape is one-dimensional
                    {sizeof(BufferType)}, // C-style contiguous strides for double
                    (invl.data()), // the data pointer
                    free_when_done),
                    bcs,
                    offs,
                    compr_sizes,
                    codecs};
        }

        [[nodiscard]]
        std::size_t intermediate_result_count() const {
            return _bcs.size();
        }

    private:
        void dump_streams(std::span<BufferType> inverted_lists, BinCards& cardinalities, Offsets& offsets, std::vector<size_t>& compr_sizes, Codecs& codecs,
                        const std::string& inv_file_name,
                        const std::string& bcs_file_name,
                        const std::string& offs_file_name,
                        const std::string& compr_sizes_file_name,
                        const std::string& codec_ids_file_name) const {


            std::ofstream inv_stream(inv_file_name, std::ios::binary);
            std::ofstream bcs_stream(bcs_file_name, std::ios::binary);
            std::ofstream offs_stream(offs_file_name, std::ios::binary);
            std::ofstream compr_sizes_stream(compr_sizes_file_name, std::ios::binary);
            std::ofstream codecs_stream(codec_ids_file_name, std::ios::binary);

            inv_stream.write(reinterpret_cast<char*>(inverted_lists.data()), inverted_lists.size()*sizeof(BufferType));
            bcs_stream.write(reinterpret_cast<char*>(cardinalities.data()), cardinalities.size()*sizeof(BinCards::value_type));
            offs_stream.write(reinterpret_cast<char*>(offsets.data()), offsets.size()*sizeof(Offsets::value_type));
            compr_sizes_stream.write(reinterpret_cast<char*>(compr_sizes.data()), compr_sizes.size()*sizeof(std::vector<size_t>::value_type));
            codecs_stream.write(reinterpret_cast<char*>(codecs.data()), codecs.size()*sizeof(Codecs::value_type));
        }

        static BinCards combine_bcs(BinCards&& lhs, BinCards& rhs) {
            std::transform(lhs.begin(), lhs.end(),
                           rhs.begin(),
                           lhs.begin(), std::plus<>());
            return lhs;
        };

        static InvertedList* append(InvertedList*& lhs, InvertedList *&rhs) {
            if (rhs != nullptr) {
                if (lhs == nullptr) {
//                    lhs = new InvertedList();
//                    lhs->reserve(rhs->size());
                    lhs = rhs;
                    rhs = nullptr;
                } else {
                    lhs->insert(std::end(*lhs), std::begin(*rhs), std::end(*rhs));
                    delete rhs; // free memory, we will erase() rhs anyway
                }
            }
            return lhs;
        };

        static std::pair<InvertedListBuffer, size_t> combine_invlists(std::pair<InvertedListBuffer, size_t> &&lhs,
                         std::pair<InvertedListBuffer, size_t> &rhs) {
            assert(lhs.first.size() == rhs.first.size());
            std::transform(lhs.first.begin(), lhs.first.end(),
                           rhs.first.begin(),
                           lhs.first.begin(), append);
            lhs.second += rhs.second;
            return lhs;
        };

        /**
         * @brief get_quantile Helper function, that determines the bin of a
         * value for the specified attribute/column.
         *
         * Requires data in _quantile_data to be in sorted, columnar storage
         * order.
         * An ascending range of values may be ended by 0 values, however.
         *
         * Due to the low number of quantiles per attribute,
         * this is just a simple scan.
         */
        template<typename VALUETYPE> unsigned
        get_quantile(const VALUETYPE value, const unsigned column) const {

            auto &bin_edges = _bin_edges[column];
            for (unsigned bin_id = 0; bin_id < _shape[column]; bin_id++) {
                if (value < bin_edges[bin_id]) {
                    return bin_id;
                }
            }
            return _shape[column] - 1;
        }

        size_t _number_of_bins;
        size_t _number_of_tuples;
        std::vector<unsigned> _shape;

        // const DataType *_quantile_data;
        std::vector<std::vector<DataType>> _bin_edges;

        std::vector<BinCards> _bcs;
        std::vector<std::pair<InvertedListBuffer, size_t>> _part_inv_lst;
    };
}
