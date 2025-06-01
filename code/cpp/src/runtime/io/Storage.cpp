#include "runtime/io/Storage.hpp"

#include <fcntl.h> // file access
#include <memory> // For std::unique_ptr
#include <cassert>

#include <iostream>

namespace TeamIndex {

    namespace Storage {
        template <typename IOBackend>
        StorageAccessor<IOBackend>::StorageAccessor(const StorageConfig& cfg):
            _cfg(cfg), _remaining_submissions_counter(cfg.queue_pair_count), _remaining_awaits_counter(cfg.queue_pair_count) {
            
            if (cfg.queue_pair_count > 0) {
                _requests.resize(cfg.queue_pair_count);

                // initialize liburing backend:
                _backend = std::make_unique<IOBackend>(cfg.queue_pair_count, cfg.liburing_cfg); // 128 is the queue depth

                for (unsigned i = 0; i < cfg.queue_pair_count; i++) {
                    _remaining_submissions_counter[i].value = 0;
                }
            }
        }

        template <typename IOBackend>
        StorageAccessor<IOBackend>::~StorageAccessor() {}

        template <typename IOBackend>
        inline void StorageAccessor<IOBackend>::register_Teams(const std::vector<TeamMetaInfo>& team_infos)
        {   
            assert(team_infos.size() > 0);

            if constexpr (std::is_same<IOBackend, liburingBackend>::value) {
                // create liburing backend: instantiate rings, etc
                liburingBackend* backend = static_cast<liburingBackend*>(_backend.get());

                std::vector<Path> files;
                for (auto &team_info: team_infos) {
                    files.emplace_back(team_info.team_file_path);
                }
                // actually open files
                backend->register_files(files); // open file descriptors and close previously opened ones
            }
            else {
                // currently, error..
            }

            // create numeric ids for team names for convenient access
            _team_ids.clear();
            TeamID id = 0;
            for (auto& team_info : team_infos) {
                _team_ids[team_info.team_name] = id++;
            }

        }

        template<typename IOBackend>
        void StorageAccessor<IOBackend>::register_request(unsigned queue_pair_id, IORequest&& request) {
            _requests[queue_pair_id].emplace_back(std::move(request));
            _remaining_submissions_counter[queue_pair_id].value++;
        }

        template <typename IOBackend>
        size_t StorageAccessor<IOBackend>::submit_request_async(unsigned queue_pair_id) {
            
            // Synchronization is done by the caller
            auto remaining = _remaining_submissions_counter[queue_pair_id].value;
            if (remaining == 0) {
                return 0;
            }

            auto pos = _requests[queue_pair_id].size()-remaining;

            // 
            auto& req = _requests[queue_pair_id][pos];
            if (req.buff == nullptr) {
                throw std::runtime_error("Buffer for request " + std::to_string(req.req_id) + " is nullptr!");
            }
            _backend->trigger_read_async(queue_pair_id, req); // async
            _remaining_submissions_counter[queue_pair_id].value--;
            _remaining_awaits_counter[queue_pair_id].value++;


            return _remaining_submissions_counter[queue_pair_id].value; // how many requests are left
        }

        template <typename IOBackend>
        IORequest& StorageAccessor<IOBackend>::get_finished_request(unsigned queue_pair_id) {
            auto& req = _backend->expect_finished_request(queue_pair_id);
            _remaining_awaits_counter[queue_pair_id].value--;
            return req;
        }
        
        template <typename IOBackend>
        size_t StorageAccessor<IOBackend>::submit_batch(unsigned queue_pair_id) {
            unsigned remaining = _remaining_submissions_counter[queue_pair_id].value;
            if (remaining == 0) {
                return 0;
            }
            auto batch_size = std::min(remaining, _cfg.submit_batch_size);
            auto offset = _requests[queue_pair_id].size()-remaining;

            std::span<IORequest> requests{_requests[queue_pair_id].data()+offset, batch_size};
            // try to submit requests, may fail so "actual_cnt" may be 0
            auto actual_cnt = _backend->submit_batch(queue_pair_id, requests);

            _remaining_submissions_counter[queue_pair_id].value -= actual_cnt;
            _remaining_awaits_counter[queue_pair_id].value += actual_cnt;

            return actual_cnt;
        }
        template <typename IOBackend>
        size_t StorageAccessor<IOBackend>::await_batch(unsigned queue_pair_id, std::vector<IORequest*>& arrivals) {
            arrivals.reserve(arrivals.size()+_cfg.await_batch_size);
            
            auto cnt = _backend->await_batch(queue_pair_id, arrivals, _cfg.await_batch_size, 0); // note: arrivals may be empty after this call, we do not wait!
            
            _remaining_awaits_counter[queue_pair_id].value -= cnt;
            return cnt;
        }

        template <typename IOBackend>
        size_t StorageAccessor<IOBackend>::remaining_submissions(unsigned queue_pair_id) const {
            return _remaining_submissions_counter[queue_pair_id].value;
        }
                template <typename IOBackend>
        size_t StorageAccessor<IOBackend>::remaining_arrivals(unsigned queue_pair_id) const {
            return _remaining_awaits_counter[queue_pair_id].value;
        }

        template class StorageAccessor<liburingBackend>;
    } // namespace Storage
} // namespace TeamIndex

