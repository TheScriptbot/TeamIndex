#pragma once

#include "common_types.hpp"

#include <optional>
#include <unordered_map>
#include <stdexcept>

namespace TeamIndex {

    /// utility stuff:
    inline static std::string to_string(CodecID id) {
        switch (id) {
            case CodecID::UNKNOWN:
                return "unknown";
            case CodecID::COPY:
                return "copy";
            case CodecID::ZSTD:
                return "zstd";
            case CodecID::ZSTDMORE:
                return "zstdmore";
            case CodecID::ZSTDFASTER:
                return "zstdfaster";
            case CodecID::ROARING:
                return "roaring";
#ifdef ENABLE_FASTPFOR
            case CodecID::VARINT:
                return "varint"; // must be the same string as used by the codefactory.h in the FastPfor lib
            case CodecID::VARINTZSTD:
                return "varint+zstd";
            case CodecID::SIMDOPTPFOR:
                return "simdoptpfor";
            case CodecID::SIMDSIMPLEPFOR:
                return "simdsimplepfor";
            case CodecID::SIMDFASTPFOR256:
                return "simdfastpfor256";
#endif
#ifdef ENABLE_MORE_COMPRESSIONS
            case CodecID::WAH:
                return "wah";
            case CodecID::TEB:
                return "teb";
#endif
            default:
                return "ERROR: Codec does not exist!";
        }
    }

    inline static std::unordered_map<std::string, CodecID> init_codec_id_map() {
        std::unordered_map<std::string, CodecID> codec_id_map;
        codec_id_map[to_string(CodecID::COPY)] = CodecID::COPY;
        codec_id_map[to_string(CodecID::ZSTD)] = CodecID::ZSTD;
        codec_id_map[to_string(CodecID::ZSTDMORE)] = CodecID::ZSTDMORE;
        codec_id_map[to_string(CodecID::ZSTDFASTER)] = CodecID::ZSTDFASTER;
        codec_id_map[to_string(CodecID::ROARING)] = CodecID::ROARING;

#ifdef ENABLE_FASTPFOR
        codec_id_map[to_string(CodecID::VARINT)] = CodecID::VARINT;
        codec_id_map[to_string(CodecID::VARINTZSTD)] = CodecID::VARINTZSTD;
        codec_id_map[to_string(CodecID::SIMDOPTPFOR)] = CodecID::SIMDOPTPFOR;
        codec_id_map[to_string(CodecID::SIMDSIMPLEPFOR)] = CodecID::SIMDSIMPLEPFOR;
        codec_id_map[to_string(CodecID::SIMDFASTPFOR256)] = CodecID::SIMDFASTPFOR256;
#endif
#ifdef ENABLE_MORE_COMPRESSIONS
        codec_id_map[to_string(CodecID::WAH)] = CodecID::WAH;
        codec_id_map[to_string(CodecID::TEB)] = CodecID::TEB;
#endif
        return codec_id_map;
    }

    static CodecID string_to_codec(const std::string& name) {
        static auto codec_id_map = init_codec_id_map();

        auto iter = codec_id_map.find(name);
        if (iter == codec_id_map.end()) {
            throw std::runtime_error("Codec \'" + name + "\' not found! Option disabled during build?");
        }
        return iter->second;
    }

    enum struct StorageBackendID: unsigned char {
        LIBURING=1,
        DEFAULT=1,
        DRAM=2,
        SPDK
    };

    inline static std::string to_string(StorageBackendID id) {
        switch (id) {
        case StorageBackendID::LIBURING:
            return "liburing";
        case StorageBackendID::DRAM:
            return "dram";
        // case StorageBackendID::SPDK:
        //     return "spdk";
        default:
                return "ERROR: StorageBackendID does not exist!";
        };
    }

    inline static std::unordered_map<std::string, StorageBackendID> init_backend_id_map() {
        std::unordered_map<std::string, StorageBackendID> backend_id_map;
        backend_id_map[to_string(StorageBackendID::LIBURING)] = StorageBackendID::LIBURING;
        backend_id_map[to_string(StorageBackendID::DEFAULT)] = StorageBackendID::DEFAULT;
        backend_id_map[to_string(StorageBackendID::DRAM)] = StorageBackendID::DRAM;
        return backend_id_map;
    }
    
    static StorageBackendID string_to_backend(const std::string& name) {
        static auto id_map = init_backend_id_map();

        auto iter = id_map.find(name);
        if (iter == id_map.end()) {
            throw std::runtime_error("Backend \'" + name + "\' not found!");
        }
        return iter->second;
    }

    /**
     * Specification of a single I/O request of compressed inverted list data, which results in one or more
     * inverted lists. Information for decompression is contained in the decomp_info field/vector.
     */
    struct RequestInfo {

        explicit RequestInfo(RequestID _rid,
                    TeamName _team_name,
                    StartBlock _start_block,
                    BlockCount _total_block_cnt,
                    WithinRequestOffsets _decomp_info):
            rid(_rid),
            team_name(_team_name),
            start_block(_start_block),
            total_block_cnt(_total_block_cnt),
            decomp_info(_decomp_info)
        {}

        RequestID rid;
        TeamName team_name; // from which Team index do we need to read?
        StartBlock start_block; // where does the I/O request start in the file?
        BlockCount total_block_cnt; // how much data do we have to read from the file?

        // which bytes in the request correspond to relevant inv. lists and how large will they be after decompression?
        WithinRequestOffsets decomp_info; // irrelevant for I/O but relevant for decompression
    };

    /**
     * Object that primarily contains time measurements
     */
    struct ExecutionStatistics {
        std::size_t input_cardinality = 0; // number of ids read from the input
        std::optional<std::size_t> result_cardinality = std::nullopt; // null, if no result is computed
        std::optional<std::size_t> plan_construction_runtime = std::nullopt;
        std::optional<std::size_t> executor_runtime = std::nullopt;
        std::optional<std::string> task_stats_path = std::nullopt;
    };
} // namespace TeamIndex