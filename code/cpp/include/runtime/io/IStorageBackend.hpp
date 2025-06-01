#pragma once

#include "runtime/RuntimeTypes.hpp"

#include <new> // For std::align
#include <atomic>
#include <span>

namespace TeamIndex::Storage {

    constexpr size_t CACHE_LINE_SIZE = 64;

    class IStorageBackend {
        /**
         * Used to count in-flight I/Os during processing. Is updated frequently
         * and core-locally, but may be read by other cores?
         */
        struct IOCounter {
            alignas(CACHE_LINE_SIZE) std::atomic<unsigned> value{0};
        };
    public:
        virtual ~IStorageBackend() = default;
        
        virtual void trigger_read_async(unsigned thread_id, IORequest& request) = 0;
        virtual unsigned submit_batch(unsigned thread_id, std::span<IORequest> request) = 0;
        
        [[nodiscard]] virtual IORequest& expect_finished_request(unsigned thread_id) = 0;
        virtual unsigned await_batch(unsigned thread_id, std::vector<IORequest*>& arrivals, unsigned batch_size, unsigned timeout) = 0;
        
        [[nodiscard]] inline bool no_in_flights(size_t thread_id) const {
            return !_in_flight_counters[thread_id].value.load(std::memory_order_relaxed); // relaxed is enough, because we only need to know if it is zero or not
        }

        [[nodiscard]] inline std::size_t queue_count() const {
            return _in_flight_counters.size();
        }
        
    protected:
        IStorageBackend(unsigned ring_count): _in_flight_counters(ring_count) {};
        
        inline std::atomic<unsigned>& operator[](size_t thread_id) {
            return _in_flight_counters[thread_id].value;
        }

        std::vector<IOCounter> _in_flight_counters; // supposed to have a length according to the number of queue pairs

    };
} // namespace