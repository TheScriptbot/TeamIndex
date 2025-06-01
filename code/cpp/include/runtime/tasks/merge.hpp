#pragma once

#include "common_types.hpp"
#include "utils.hpp"

#include <iterator>
#include <cassert>
#include <span>


namespace TeamIndex {

    static inline void copy_if_set(IDType*& destination, IDType* source, unsigned int& survivors, unsigned char mask) {
        for (unsigned short bit_pos = 0; bit_pos < 8; ++bit_pos) {
            if (mask & (1u << bit_pos)) {
                *destination++ = source[bit_pos];  // Copy and advance destination pointer
                survivors++;
            }
        }
    }

    class MergeFunction {
        public:

        MergeFunction(std::span<std::span<unsigned char>> survivors, std::span<IDType>& smallest_team): 
            _survivors(survivors), _smallest_team(smallest_team) 
            {}

        void operator() () {
            // auto x = 0u;
            // for (auto& bm : _survivors) {
            //     std::cout << x++ << "-th bitmap: [";
            //     for (auto i = 0; i < 10; i++)
            //         std::cout << std::to_string(bm[i]) << " ";
            //     std::cout << "...] ("; 

            //     int count = 0;
            //     for (unsigned char byte : bm) {
            //         count += std::popcount(byte);  // C++20 function to count bits
            //     }
            //     std::cout << count << " matches)" << std::endl;
            // }


            // reduce all _survivor_bms.size()-many bitmaps to a single bitmap:
            if (_survivors.empty()) {
                return;
            }
            if (_survivors.size() > 1) {
                // we merge the second bitmap into the first bitmap
                for (TeamID tid = 1; tid < _survivors.size(); tid++)
                    combine_bms(_survivors[0].data(), _survivors[tid].data()); // result stored in first bitmap
            }

            // std::cout << "After bitmap merge: [";
            // for (auto i = 0; i < 10; i++)
            //     std::cout << std::to_string(_survivors[0][i]) << " ";
            // std::cout << "...] ("; 
            // int count = 0;
            // for (unsigned char byte : _survivors[0]) {
            //     count += std::popcount(byte);  // C++20 function to count bits
            // }
            // std::cout << count << " matches)" << std::endl;

            // reduce _smallest_team to the survivors of the intersection/logical-and above. This gives the final result!
            reduce_ids();

        }
    private:
        void combine_bms(unsigned char* lhs, unsigned char* rhs) {
            auto n = _smallest_team.size(); // number of bits, might not be a multiple of 8

// #ifdef NO_SIMD // i.e., we will work 64-word wise
//            auto byte_cnt = n/8;
//            for (auto i = 0u; i < byte_cnt; i++) {
//                rhs[i] &= lhs[i];
//            }
//            // deal with the rest:
//            if (n % 8) rhs[byte_cnt-1] &= lhs[byte_cnt-1];

            auto lhs_ = reinterpret_cast<std::uint64_t*>(lhs);
            auto rhs_ = reinterpret_cast<std::uint64_t*>(rhs);
            auto dword_cnt = n/64;

            for (auto i = 0u; i < dword_cnt; i++)
                lhs_[i] &= rhs_[i]; // log. AND for 64 bit at once

            // deal with the rest:
            if (n % 64)
                lhs_[dword_cnt-1] &= rhs_[dword_cnt-1];
// #endif
        }

        void reduce_ids() {
// #ifdef NO_SIMD
            // TODO: check and test Jan's branch-less code:
//            @Jan:
//            Mhh. Ich denke mal laut.
//                    Du könntest einen neuen Vector machen, mit der Größe (anzahl 1 in mask) + 1 .
//                    Das erste element (+1) ist ein "Müll Element, das immer von nicht überlebenden überschrieben wird

//            for i = 0; i < vec.length(); ++i:
//              new_vec[mask[i] * survivors] = vec[i]
//              survivors += mask[i]

//            Dann landen alle nicht überlebenden in new_vec[0]

            // TODO: use masked store/SIMD instead?
            auto survivor_position = 0u; // also number of elements after reduction, this will be incremented along the way
            IDType* destination = _smallest_team.data();
            IDType* current = _smallest_team.data(); // accessed via j
            unsigned char* masks = _survivors[0].data();
            for (auto byte_j = 0u; byte_j < _survivors[0].size(); byte_j++, current += 8 /* advance by 8 values */) {
                // test i-th bit. If set, copy corresponding value and advance survivors pointer
                copy_if_set(destination, current, survivor_position, masks[byte_j]);
            }

            // std::cout << "Printing some materialized results: [";
            // for (auto id : std::span<IDType>{_smallest_team.data(), 100}) {
            //     std::cout << id << ", ";
            // }
            // std::cout << "..] " << survivor_position << " in total "<< std::endl;
            _smallest_team = {_smallest_team.data(), survivor_position}; // resize to relevant values, i.e., those that were copied
// #endif
        }
        std::span<std::span<unsigned char>> _survivors;
        std::span<IDType>& _smallest_team;

    };

}