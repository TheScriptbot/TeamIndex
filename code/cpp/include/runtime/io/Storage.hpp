#pragma once
#include "runtime/RuntimeTypes.hpp"
#include "IStorageBackend.hpp"
#include "liburingBackend.hpp"

#include <memory> // For std::unique_ptr


namespace TeamIndex::Storage {

    using DEFAULT_BACKEND = liburingBackend;

    struct StorageConfig {
        unsigned submit_batch_size = 16;
        unsigned await_batch_size = 16;
        unsigned queue_pair_count = 1;
        liburingBackend::Config liburing_cfg;
    };

    /**
     * This class creates I/O requests and passes them to the storage backend.
     * 
     * To be used concurrenctly, where each worker/worker_id has its own sets of resources (both backend and requests).
     * 
     * 
     */
    template<typename IOBackend=DEFAULT_BACKEND>
    class StorageAccessor {
        struct alignas(64) AlignedCounter {
            size_t value{0};
        };

        public:
        StorageAccessor(const StorageConfig& cfg);
        ~StorageAccessor();
        /**
         * Register a team by opening the file containing the inverted lists to be referenced by IORequests.
         * 
         * Initialization function.
         */
        void register_Teams(const std::vector<TeamMetaInfo>& team_infos);
                
        /**
         * Register a request by adding an IORequest object to the internal list (one per worker_id).
         * This function also looks up the associated file descriptor, assuming the Team's file was registered before.
         * 
         * Initialization function.
         */
        void register_request(unsigned queue_pair_id, IORequest&& request);


        /**
         * Asynchronously submits the next request for the given worker.
         * 
         * To be called from that worker only.
         * 
         * Runtime function.
         */
        size_t submit_request_async(unsigned ring_id);

        /**
         * Asynchronously submit a batch of requests (see config) for the given worker.
         * 
         * To be called from that worker only.
         * 
         * Runtime function.
         */
        // size_t submit_request_async(unsigned worker_id);

        /**
         * Syncronously waits for the next request to finish.
         * 
         * To be called from that worker only.
         * 
         * Runtime function.
         */
        [[nodiscard]] IORequest& get_finished_request(unsigned ring_id);

        size_t submit_batch(unsigned ring_id);

        size_t await_batch(unsigned ring_id, std::vector<IORequest*>& arrivals);

        [[nodiscard]] size_t remaining_submissions(unsigned ring_id) const;

        [[nodiscard]] size_t remaining_arrivals(unsigned ring_id) const;

        private:

        const StorageConfig& _cfg;

        std::vector<std::vector<IORequest>> _requests; // globally created before running the workload
        std::vector<AlignedCounter> _remaining_submissions_counter; // how many requests have been submitted by each worker
        std::vector<AlignedCounter> _remaining_awaits_counter; // how many requests have been submitted by each worker
        std::unique_ptr<IStorageBackend> _backend;

        // liburing specific (which is based on a file system/file descriptor):
        std::unordered_map<TeamName, TeamID> _team_ids; // for convenient access by name in vectors etc
    };

} // namespace TeamIndex