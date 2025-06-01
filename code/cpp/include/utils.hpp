#pragma once


#include <chrono>
#include <optional>
#include <iostream>
#include <immintrin.h>


namespace TeamIndex {

    // template <class T> static bool needPaddingTo64bytes(const T *inbyte) {
    //     return (reinterpret_cast<uintptr_t>(inbyte) & 63) != 0;
    // }
   static size_t round_up_to_alignment(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
    /**
     * We only use aligned buffers. 
     * This needs to be freed!
     */
    template<unsigned alignment=PAGESIZE>
    static BufferType* get_new_io_buffer(BlockCount block_count) {
        return static_cast<BufferType*>(std::aligned_alloc(alignment, round_up_to_alignment(block_count*PAGESIZE*sizeof(BufferType), alignment)));
    }
    
    template<std::size_t alignment=alignof(__m256)>
    static IDType* get_new_decompression_buffer(ListCardinality cardinality) {
        return static_cast<IDType*>(std::aligned_alloc(alignment, round_up_to_alignment(cardinality*sizeof(IDType), alignment)));
    }

    class Timer {
    public:
        using clock_type = std::chrono::high_resolution_clock;

        explicit Timer(std::string&& name, std::size_t volume = 0, bool start_stopped= false):
                _timer_name(std::move(name)), _volume(volume), _stopped(start_stopped)
        {
            if (not _stopped) _start = clock_type::now();
        }

        ~Timer() {
            if (_verbose_on_death) {
                if (not _stopped) {
                    _stop = clock_type::now();
                }
                auto duration = _stop - _start;
                auto sec = std::chrono::duration_cast<std::chrono::seconds>(duration);
                auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(duration - sec);
                auto us  = std::chrono::duration_cast<std::chrono::microseconds>(duration - sec - ms);
                auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(duration - sec - ms - us);

                // ANSI escape sequences for color (yellow bold in this case)
                const char* colorStart = "\033[1;32m";
                const char* colorEnd = "\033[0m";

                // Build the output strings: only highlight the largest non-zero unit.
                std::string secStr = (sec.count() > 0) 
                    ? std::string(colorStart) + std::to_string(sec.count()) + "s" + colorEnd 
                    : std::to_string(sec.count()) + " s";
                std::string msStr;
                if (sec.count() == 0 && ms.count() > 0)
                    msStr = std::string(colorStart) + std::to_string(ms.count()) + "ms" + colorEnd;
                else
                    msStr = std::to_string(ms.count()) + "ms";
                std::string usStr;
                if (sec.count() == 0 && ms.count() == 0 && us.count() > 0)
                    usStr = std::string(colorStart) + std::to_string(us.count()) + "µs" + colorEnd;
                else
                    usStr = std::to_string(us.count()) + " µs";
                std::string nsStr;
                if (sec.count() == 0 && ms.count() == 0 && us.count() == 0 && ns.count() > 0)
                    nsStr = std::string(colorStart) + std::to_string(ns.count()) + "ns" + colorEnd;
                else
                    nsStr = std::to_string(ns.count()) + "ns";

                // Print all in one line: timer name, the four parts and additional volume info if available.
                std::cout << _timer_name << ":\n\t" 
                        << secStr << " " << msStr << " " << usStr << " " << nsStr;
                        
                if (_volume > 0) {
                    double lifetime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
                    // Bandwidth: MB/s (assuming 1 MB = 1e6 bytes). Multiply by 1000 to adjust scale as in original.
                    double bandwidth = (lifetime_ns > 0) ? (_volume * 1000.0 / lifetime_ns) : 0.0;
                    std::cout << "\n\tVolume: " << _volume/4096 << " Pages / "
                             << std::fixed << std::setprecision(2)
                            << (float)_volume/1000/1000 << " MB" 
                            << "\n\tBandwidth: " << bandwidth << " MB/s";
                }
                std::cout << std::endl;
            }
        }


        auto& start() {
            _stopped = false;
            _start = clock_type::now();
            return _start;
        };

        /**
         * @brief stop Stops timer, returns lifetime in nanoseconds
         * @return
         */
        auto& stop() {
            if (not _stopped) {
                _stop = clock_type::now();
                _stopped = true;
            }
            return _stop;
        }

        long duration() {
            if (not _stopped) {
                _stop = clock_type::now();
                _stopped = true;
            }
            return std::chrono::duration_cast<std::chrono::nanoseconds>( _stop-_start).count();
        }

        bool is_stopped() const {
            return _stopped;
        };

        void set_volume(std::size_t volume) {
            _volume = volume;
        }
        
        void set_verbose_on_death() {
            _verbose_on_death = true;
        }

        [[nodiscard]] const std::string& get_name() const {
            return _timer_name;
        }
        
    private:
        clock_type::time_point _start;
        clock_type::time_point _stop;
        std::string _timer_name;
        std::size_t _volume; // associate the time with a data volume to calculate throughput
        bool _stopped;
        bool _verbose_on_death = false;
    };


    static std::string getCurrentDateTime() {
        auto now = std::time(nullptr);
        std::tm tm{};
        localtime_r(&now, &tm);
    
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y_%m_%d-%H_%M_%S");
        return oss.str();
    }

}