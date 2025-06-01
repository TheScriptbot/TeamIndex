#pragma once
#include <span>


namespace TeamIndex {

    static inline int set_bit_atomic(unsigned char* ptr, unsigned int pos) {
        return __atomic_fetch_or(ptr + pos / 8, true << pos % 8, __ATOMIC_SEQ_CST);
    }
    static inline int unset_bit_atomic(unsigned char* ptr, unsigned int pos) {
        return __atomic_fetch_and(ptr + pos / 8, ~(true << pos % 8), __ATOMIC_SEQ_CST);
    }

    template<bool is_complement>
    class IntersectionFunction {
        public:
        IntersectionFunction(const std::span<IDType> smallest_team,
                            const std::span<IDType> inverted_list,
                            std::span<unsigned char>& survivors):
                _smallest_team(smallest_team), _inverted_list(inverted_list), _survivors(survivors)
        {
            assert(_survivors.size() == _smallest_team.size() / 8 + (_smallest_team.size() % 8 != 0));
        }
        void operator() () {
            auto left_idx = 0u;
            auto match_cnt = 0u;
            // scan smallest-team-ids entirely and find matches
            // std::cout << "Comparing " << _inverted_list.size() << " vs. " <<  _smallest_team.size() << " values" << std::endl;

            // std::cout << "Bitmap before: [";
            // for (auto i = 0; i < 20; i++)
            //     std::cout << std::to_string(_survivors[i]) << " ";
            // std::cout << "...] " << std::endl; 

            for (auto right_id : _inverted_list) { // the other Team's inverted list is "right hand side"
                // perform set intersection of an inverted list with the smallest team by marking common ids in the survivor bit-vector

                // perform intersection/search for matches with the other list:
                while (_smallest_team[left_idx] < right_id) {
                    // skip irrelevant ids of lhs by incrementing counter. TODO: implement galloping
                    if (++left_idx > _smallest_team.size()) break;
                }
                if (_smallest_team[left_idx] == right_id) { // element found -> set corresponding bit
                    if constexpr(is_complement) {
                        unset_bit_atomic(_survivors.data(), left_idx);
                    }
                    else {
                        set_bit_atomic(_survivors.data(), left_idx);
                    }
                    match_cnt++;
                }
            }
            // std::cout << "Bitmap after: [";
            // for (auto i = 0; i < 20; i++)
            //     std::cout << std::to_string(_survivors[i]) << " ";
            // std::cout << "...] " << std::endl; 
            // std::cout << "match_cnt: " << match_cnt << std::endl;
        }
        private:
            const std::span<IDType> _smallest_team; // is aligned buffer
            const std::span<IDType> _inverted_list; // is page-aligned buffer
            std::span<unsigned char>& _survivors;
    };


}