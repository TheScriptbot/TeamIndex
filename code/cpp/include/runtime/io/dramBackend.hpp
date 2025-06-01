#pragma once

#include "runtime/RuntimeTypes.hpp"
#include "utils.hpp"

#include <vector>
#include <string>
#include <cassert>

#include <fstream>
#include <iostream>
#include <filesystem>

namespace TeamIndex::Storage {

    class DRAMAccessor {

        public:
        
        DRAMAccessor() = default;
        
        ~DRAMAccessor() {
            for (auto span : _team_buffers) {
                free(span.data());
            }
        }

        /**
         * To be called in place of any "I/O" and buffer allocation by the runtime.
         * 
         * start_block is the "global" offset (as a number of blocks/pages) in the entire buffer of this Team.
         * 
         * size is in bytes, i.e., without padding
         */
        [[nodiscard]]
        BufferType* get_list_ptr(TeamID tid, StartBlock start_block) const {
            assert(tid < _team_buffers.size());
            assert(_team_buffers[tid].size() > start_block);
            return reinterpret_cast<BufferType*>(_team_buffers[tid].data()+start_block);
        }

        

        void register_Teams(const std::vector<Path>& import_paths) {
            for (const auto& path : import_paths) {
                // check if file exists and open it
                if (!std::filesystem::exists(path)) {
                    throw std::runtime_error("File does not exist: \'" + path + "\'");
                }

                std::ifstream file(path, std::ios::binary | std::ios::ate);
                if (!file) {
                    throw std::runtime_error("Failed to open file: \'" + path + "\'");
                }

                // determine size of file as a multiple of PAGESIZE (it should be by default)
                std::streamsize file_size = file.tellg();
                file.seekg(0, std::ios::beg);
                BlockCount block_cnt = (file_size + PAGESIZE - 1) / PAGESIZE;

                // allocate an aligned buffer..
                auto buffer = reinterpret_cast<Page*>(get_new_io_buffer(block_cnt));
                if (!buffer) {
                    throw std::runtime_error("Unable to allocate "+ std::to_string(block_cnt*PAGESIZE) + " byte!");
                }

                // and fill it with the content of the file
                file.read(reinterpret_cast<char*>(buffer), file_size);
                if (!file) {
                    throw std::runtime_error( "Error reading file: \'" + path + "\'!");
                }

                // store filled buffer
                _team_buffers.push_back({buffer, block_cnt});
                // std::cout << "Loaded " << block_cnt << " blocks from " << path << std::endl; 
                file.close();
            }
        }
        private:

        // members
        std::vector<std::span<Page>> _team_buffers;
        
    };
}