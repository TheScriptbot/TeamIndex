#pragma once
#include "common_types.hpp"

#include <functional>

#include <execinfo.h> // backtrace_symbols_fd
#include <thread>

namespace TeamIndex::Storage {

    /**
     * All information needed to aquire inverted list data, may span multiple lists in one request.
     * Does not contain information on actual list size in bytes, works in terms of PAGESIZE-sized blocks.
     */
    struct IORequest {
        explicit IORequest(RequestID _req_id,
                            BufferType* _buff,
                            std::function<void()> _callback,
                            StartBlock _start_block,
                            BlockCount _block_count,
                            TeamID _tid):
                req_id(_req_id),
                buff(_buff),
                callback(_callback),
                start_block(_start_block),
                block_count(_block_count),
                tid(_tid)
        {}

        RequestID req_id;
        BufferType* buff; // will be managed elsewhere
        std::function<void()> callback; // callback to be called upon completion    

        // liburing specific fields:
        // 0 for the first block of the Team. Offset in byte is: start_block*BLOCKSIZE:
        StartBlock start_block; 
        BlockCount block_count;
        TeamID tid;


        [[nodiscard]] std::size_t offset() const {
            return start_block*PAGESIZE;
        }
        [[nodiscard]] std::size_t size() const {
            return block_count*PAGESIZE;
        }

        // // SPDK-specific fields:
    // #ifdef ENABLE_SPDK
    //     struct spdk_nvme_ns* spdk_namespace; // SPDK namespace
    //     struct spdk_nvme_qpair* spdk_qpair;  // SPDK queue pair
    //     uint64_t spdk_lba_start;         // Logical block address for SPDK I/O
    //     uint32_t spdk_lba_count;   // Number of blocks for SPDK I/O
    // #endif
    };


    static bool set_processor_affinity(std::thread& thread, unsigned int core_id) {
        // from https://taskflow.github.io/taskflow/ExecuteTaskflow.html
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_t native_handle = thread.native_handle();
        return pthread_setaffinity_np(native_handle, sizeof(cpu_set_t), &cpuset) == 0;
    }


    /** Convert errno to exception
     * @throw std::runtime_error / std::system_error
     * @return never
     */
    [[noreturn]]
    static void panic(std::string_view sv, int err) {
#ifndef NDEBUG
        // copied function from https://github.com/CarterLi/liburing4cpp

        // https://stackoverflow.com/questions/77005/how-to-automatically-generate-a-stacktrace-when-my-program-crashes
        void *array[32];

        // get void*'s for all entries on the stack
        auto size = backtrace(array, 32);

        // print out all the frames to stderr
        fprintf(stderr, "Error: errno %d:\n", err);
        backtrace_symbols_fd(array, size, STDERR_FILENO);

        // __asm__("int $3");
#endif

        throw std::system_error(err, std::generic_category(), sv.data());
    }

    struct panic_on_err {
        // copied function from https://github.com/CarterLi/liburing4cpp
        panic_on_err(std::string_view _command, bool _use_errno)
                : command(_command), use_errno(_use_errno) {}

        std::string_view command;
        bool use_errno;
    };

    inline int operator|(int ret, panic_on_err &&poe) {
        // copied function from https://github.com/CarterLi/liburing4cpp
        if (ret < 0) {
            if (poe.use_errno) {
                panic(poe.command, errno);
            } else {
                if (ret != -ETIME) panic(poe.command, -ret);
            }
        }
        return ret;
    }


} // namespace TeamIndex