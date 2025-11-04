#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <tuple>


namespace TeamIndex {

    constexpr std::size_t PAGESIZE = 4096; // used for alignment

    using IDType = std::uint32_t; // type used to represent a single ID in an uncompressed inverted list
    using DataType = double; // type of the data to be indexed.
    using DataTypeSP = float; // type of the data to be indexed (half precision)
    using BufferType = char; // representation of a single byte in the file-stream that stores a (compressed) TeamIndex

    using TeamName = std::string; // internal name for a team; might be different to the one shown to the user
    using TeamID = unsigned;
    constexpr TeamID SMALLEST_TEAM = 0u;
    constexpr TeamID PRIMARY_EXPANSION = 0u;
    using Path = std::string;
    using FileDescriptor = int; // filesystem file descriptor; files are opened on creation of the TeamIndex class

    using RequestID = unsigned; // ID of the request, a number from 0 to N-1, where N is the number of requests
    using BlockCount = unsigned; // number of blocks of size BLOCKSIZE to be read from file, start at RequestOffset
    using StartBlock = unsigned; // offset of a specific inverted list in the byte stream read from file
    using ListCardinality = unsigned; // number of IDs inverted list (after decompression)
    using ListSizeCompressed = std::size_t; // size in bytes of a specific inverted list in the byte stream
    using GroupID = unsigned; // used to union together elements of the same group before further processing
    
    enum struct CodecID: unsigned char { // should have no values larger than 8 bit, see Header struct below
        UNKNOWN=0,
        COPY=1,
        ZSTD=2,
        ZSTDMORE=3,
        ZSTDFASTER=4,

        VARINT=5,
        VARINTZSTD=6, // combination of VARINT and ZSTD
        SIMDOPTPFOR=7,
        SIMDSIMPLEPFOR=8,
        SIMDFASTPFOR256=9,

        ROARING=10,
        WAH=11, 
        TEB=12    
    };
    constexpr unsigned CODEC_COUNT = 13;

    using CompressedListInfo = std::tuple<StartBlock, ListCardinality, CodecID, ListSizeCompressed, GroupID>; // position and size (compressed and uncompressed) of a list within a request
    using WithinRequestOffsets = std::vector<CompressedListInfo>; // not required for I/O, but for later decompression/extraction of inverted lists


    struct alignas(PAGESIZE) Page {
        char padding[PAGESIZE]; // AFAIK not necessary for GCC/Clang
    };
    static_assert(sizeof(Page) == PAGESIZE, "Unexpected Page struct size!");


    /**
     * Query- and Team specific information to resolve index access.
     */
    struct TeamMetaInfo {
        explicit TeamMetaInfo(const TeamIndex::TeamName& _name,
                                 std::size_t _total_size_comp,
                                 ListCardinality _total_cardinality,
                                 std::size_t _request_cnt,
                                 std::size_t _list_cnt,
                                 Path _team_file_path,
                                 bool _is_included,
                                 bool _expand,
                                 unsigned _group_count,
                                 unsigned _min_group_size,
                                 unsigned _max_group_size) :
                             team_name(_name),
                             total_size_comp(_total_size_comp),
                             total_cardinality(_total_cardinality),
                             request_cnt(_request_cnt),
                             list_cnt(_list_cnt),
                             team_file_path(_team_file_path),
                             is_included(_is_included),
                             expand(_expand),
                             group_count(_group_count),
                             min_group_size(_min_group_size),
                             max_group_size(_max_group_size)
                         {
                            // assert(group_count > 0);
                         }

        TeamName team_name;
        std::size_t total_size_comp; // total number of requested bytes (compressed) for all requested lists
        ListCardinality total_cardinality; // total cardinality/number of all lists 
        std::size_t request_cnt;
        std::size_t list_cnt;
        Path team_file_path;
        bool is_included;
        bool expand; // whether to expand the lists of this team, such that each list combination creates a new "pipeline"
        unsigned group_count; // number of leaf groups/unions in this team
        unsigned min_group_size; // minimum count of leafs in all leaf unions
        unsigned max_group_size; // maximum count of leafs in all leaf unions
    };
}


