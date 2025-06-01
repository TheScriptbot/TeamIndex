#include "runtime/io/liburingBackend.hpp"
#include "utils.hpp"

#include <cstring>
#include <iostream>


namespace TeamIndex {
    namespace Storage {
        liburingBackend::IOUring::IOUring(const liburingBackend::Config& ucfg) {

            memset(&_ring_params, 0, sizeof(_ring_params));
            if (ucfg.sq_poll) {
                _ring_params.flags |= IORING_SETUP_SQPOLL; // kernel poll mode; user+kernel polling, but no syscalls
                _ring_params.sq_thread_idle = ucfg.sq_thread_idle; // how long the kernel waits for new I/Os (in milliseconds)
            }

            io_uring_queue_init_params(ucfg.queue_depth, &ring, &_ring_params) | panic_on_err("Failed to initialize io_uring", false);

            if (!(_ring_params.features & IORING_FEAT_EXT_ARG)) {
                throw std::runtime_error("Failed when initialize liburing: IORING_FEAT_EXT_ARG is required but not supported!");
            }
            
        }

        liburingBackend::IOUring::~IOUring() noexcept {
            // if (_ucfg.sq_poll) {
            //     io_uring_unregister_files(&ring) | panic_on_err("Unable to unregister files", false);
            // }
            io_uring_queue_exit(&ring);
        }

        void liburingBackend::IOUring::register_file_descriptors(std::vector<int>& file_descriptors) {
            if (file_descriptors.empty()) {
                return;
            }
            io_uring_register_files(&ring, file_descriptors.data(), file_descriptors.size()) | panic_on_err("Unable to register files!", false);
        }
        
        liburingBackend::liburingBackend(unsigned ring_count, liburingBackend::Config ucfg): IStorageBackend(ring_count), _ucfg(std::move(ucfg)), _await_timeout{.tv_sec = 0, .tv_nsec = 0} {
            for (unsigned i = 0; i < this->queue_count(); i++) {
                _rings.emplace_back(std::make_unique<IOUring>(_ucfg));
            }
        }
        liburingBackend::~liburingBackend() {
            for (auto fd: _fds) {
                close(fd);
            }
        }

        void liburingBackend::register_files(const std::vector<Path>& files) {
            // clean-up previously opened files
            if (!_fds.empty()) {
                for (auto fd : _fds) {
                    close(fd);
                }
            }
            _fds.resize(files.size());

            auto i = 0u;
            for (const auto& file: files) {
                int flags = O_RDONLY;
                if (_ucfg.o_direct) {
                    flags |= O_DIRECT;
                }
                 // open files and store file descriptor
                _fds[i] = open(file.c_str(), flags) | panic_on_err("liburing: Error opening \'" + file + "\'", true);
                i++;
            }

            // register open descriptors in the kernel to be used by liburing (necessary for SQ polling + more efficient)            
            if (_ucfg.sq_poll) {
                // register files in the kernel for each ring
                for (auto& ring_ptr : _rings) {
                    ring_ptr->register_file_descriptors(_fds);
                }
            }
        }

        void liburingBackend::trigger_read_async(unsigned thread_id, IORequest& request)
        {
            // get a reference to a queue entry to be filled in the following
            auto* ring = _rings[thread_id]->get();

            prepare_submission(ring, request);

            //// actually submit the prepared request to the system
            io_uring_submit(ring) | panic_on_err("liburing: Failed to submit new SQE!", false); 

            // track in-flight request count
            _in_flight_counters[thread_id].value.fetch_add(1, std::memory_order_relaxed);
        }

        unsigned liburingBackend::submit_batch(unsigned thread_id, std::span<IORequest> requests)
        {
            auto* ring = _rings[thread_id]->get();

            auto submission_cnt = 0u;
            for (auto& request : requests) {
                if (prepare_submission(ring, request)) {
                    submission_cnt++;
                }
                else {
                    break; // need to stop, otherwise it's hard to track what got actually
                }
            }
            if (submission_cnt) {
                //// submit prepared requests to the system
                io_uring_submit(ring) | panic_on_err("liburing: Failed to submit batch of new SQEs!", false); 

                _in_flight_counters[thread_id].value.fetch_add(submission_cnt, std::memory_order_relaxed);
            }
            return submission_cnt; // return how many submission were actually made
        }

        IORequest &liburingBackend::expect_finished_request(unsigned thread_id)
        {
            auto* ring = _rings[thread_id]->get();

            struct io_uring_cqe *cqe;
            
            // wait until completion!
            io_uring_wait_cqe(ring, &cqe) | panic_on_err("liburing: Failed to patiently wait for completion queue entry, unknown error", false);

            // io_uring_

            auto res = cqe->res | panic_on_err("Result of read was unsuccessful", false);

            // Got something, retrieve request from completion queue entry
            IORequest* request = (IORequest*) io_uring_cqe_get_data(cqe);
            io_uring_cqe_seen(ring, cqe);

            if (res != request->size()) {
                throw std::runtime_error("Read has unexpected size: " + std::to_string(res) + " (expected: " + std::to_string(request->size()) + ")");
            }

            _in_flight_counters[0].value.fetch_sub(1, std::memory_order_relaxed);


            return *request;
        }

        unsigned liburingBackend::await_batch(unsigned thread_id, std::vector<IORequest*>& arrivals, unsigned batch_size, unsigned timeout) {
            /**
             * TODO: Actively clean up missing in-flight operations. We never await them, so if something goes wrong or they take too long, 
             * may be stuck in the kernel queues, I think. 
             * 
             */
            auto* ring = _rings[thread_id]->get();
            // io_uring_cqe *cqes[batch_size];
            // auto ret = io_uring_wait_cqes(ring, cqes, batch_size, &_await_timeout, 0) | panic_on_err("wait_cqes was unsuccessful", false);

            struct io_uring_cqe *cqe;
            unsigned head;

            io_uring_for_each_cqe(ring, head, cqe) {

                // process arrival:
                auto res = cqe->res | panic_on_err("Result of read was unsuccessful", false);
                arrivals.emplace_back(reinterpret_cast<IORequest*>(io_uring_cqe_get_data(cqe)));

                // stop, if batch is full
                if (arrivals.size() >= batch_size) {
                    break;
                }
            }

            if (arrivals.size()) {
			    io_uring_cq_advance(ring, arrivals.size());
            }
            
            return arrivals.size();
        }

    } // namespace Storage

} // namespace TeamIndex
