#pragma once

#include "IStorageBackend.hpp"

#include <liburing.h>

#include <stdexcept>
#include <vector>
#include <memory>

namespace TeamIndex::Storage {

    /**
     * Actual implementation, which simply exposes basic async read and write 
     * functionality to be called from the runtime.
     * 
     */
    class liburingBackend : public IStorageBackend {
        public:
        struct Config {
            unsigned queue_depth = 128;
            bool o_direct = true;
            bool sq_poll = true; // avoid syscalls, let the kernel poll for submissions instead
            bool io_poll = false;
            unsigned sq_thread_idle = 4000; // only used with sq_poll = true
        };

        private:
        /**
         * Capsule class for one io_uring, which is a core-local ressource.
         * 
         * Also has some statistics variables.
         */
        class IOUring {
        public:
            explicit IOUring(const liburingBackend::Config& ucfg);
            ~IOUring() noexcept;
            [[nodiscard]] inline io_uring* get() {
                return &ring;
            }
            void register_file_descriptors(std::vector<int>& file_descriptors);
            // members:
        private:
            struct io_uring_params _ring_params;
            struct io_uring ring;
        };

    public:

        liburingBackend(unsigned ring_count, Config ucfg);
        ~liburingBackend();

        void trigger_read_async(unsigned ring_id, IORequest& request) override;
        unsigned submit_batch(unsigned thread_id, std::span<IORequest> request) override;

        [[nodiscard]] IORequest& expect_finished_request(unsigned ring_id) override;
        unsigned await_batch(unsigned thread_id, std::vector<IORequest*>& arrivals, unsigned batch_size, unsigned timeout) override;
        
        void register_files(const std::vector<Path>& files);
    
    private:
        inline bool prepare_submission(struct io_uring* ring, IORequest& request) {

            // try to accuire a sq entry, may fail due to the queue being full
            struct io_uring_sqe *sqe = io_uring_get_sqe(ring);

            // ringbuffer full, we need to submit other requests first
            if (sqe == nullptr) {
                // Notifies kernel and moves requests into submission queue (ring buffer) with kernel
                // and begin of (asynchronous) I/O processing
                io_uring_submit(ring) | panic_on_err("Error when submitting request before new SQ entry!", false);

                sqe = io_uring_get_sqe(ring);
                if (!sqe) {
                    // liburing: Failed repeatedly, we may try again later
                    return false;
                }
            }

            //// prepare next request:
            int pos;
            // we either use a file descriptor or just a relative offset, depending on if we registered the file descriptors beforehand
            if (_ucfg.sq_poll) {
                pos = (int) request.tid; // tid is in range [0,team_count)
                if (pos >= _fds.size()) {
                    throw std::runtime_error("TeamID does not indicate a valid file!");
                }
            }
            else {
                pos = _fds[request.tid];
            }

            // fill in the struct
            io_uring_prep_read(sqe, pos, request.buff, request.size(), request.offset());
            
            if (_ucfg.sq_poll) {
                // tell liburing we are using fixed-files (i.e., the TeamID as a position) instead of file descriptors
                sqe->flags |= IOSQE_FIXED_FILE;
            }
            // register request data for convenient access upon completion
            io_uring_sqe_set_data(sqe, (void*) &request);
            return true;
        }

        /// Member variables:
        Config _ucfg;
        __kernel_timespec _await_timeout;
        std::vector<FileDescriptor> _fds; 
        std::vector<std::unique_ptr<IOUring>> _rings;
    };

} // namespace