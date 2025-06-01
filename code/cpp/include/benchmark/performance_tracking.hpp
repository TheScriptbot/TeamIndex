#include <fstream>
#include <nlohmann/json.hpp>

#include <chrono>

#include <vector>
#include <map>

#include <iomanip>

namespace TeamIndex {

    enum class TaskTypes : unsigned {
        Leaf,
        LeafUnionPrepare,
        LeafUnion,
        ExpandedInit,
        ExpandedIntersection,
        ExpandedSubtraction,
        DistributedIntersection,
        DistributedSubtraction,
        BigInnerUnion,
        BigOuterUnion,
        BigInnerIntersection,
        BigOuterIntersection,
        TeamUnion,
        TeamSubtraction,
        Materialize
    };
    static std::string to_string(TaskTypes type) {
        const std::string task_pfx = "Task::";
        switch (type)
        {
            case TaskTypes::Leaf:
                return task_pfx+"Leaf"; 
            case TaskTypes::LeafUnionPrepare:
                return task_pfx+"LeafUnionPrepare"; 
            case TaskTypes::LeafUnion:
                return task_pfx+"LeafUnion"; 
            case TaskTypes::ExpandedInit:
                return task_pfx+"ExpandedInit"; 
            case TaskTypes::ExpandedIntersection:
                return task_pfx+"ExpandedIntersection"; 
            case TaskTypes::ExpandedSubtraction:
                return task_pfx+"ExpandedSubtraction"; 
            case TaskTypes::DistributedIntersection:
                return task_pfx+"DistributedIntersection"; 
            case TaskTypes::DistributedSubtraction:
                return task_pfx+"DistributedSubtraction"; 
            case TaskTypes::BigInnerUnion:
                return task_pfx+"BigInnerUnion"; 
            case TaskTypes::BigOuterUnion:
                return task_pfx+"BigOuterUnion"; 
            case TaskTypes::BigInnerIntersection:
                return task_pfx+"BigInnerIntersection"; 
            case TaskTypes::BigOuterIntersection:
                return task_pfx+"BigOuterIntersection";
            case TaskTypes::TeamUnion:
                return task_pfx+"TeamUnion";
            case TaskTypes::TeamSubtraction:
                return task_pfx+"TeamSubtraction";
            case TaskTypes::Materialize:
                return task_pfx+"Materialize";

        default:
            return task_pfx+std::to_string(static_cast<unsigned>(type));
        }
    }

    class TaskTracker {

        public:
        using clock_type = std::chrono::high_resolution_clock;
        // using Tag = TaskTypes;
        using StatPosition = std::size_t;

        struct TaskStats {
            public:
            TaskTypes ttype;
            clock_type::time_point start_time;
            clock_type::time_point stop_time;
            
            std::optional<std::size_t> input_volume;
            std::optional<std::size_t> output_volume;
            std::optional<unsigned> worker_id;

            TaskStats(TaskTypes ttype) : ttype(ttype) {}

            void start() {
                start_time = clock_type::now();
            }
            void stop() {
                stop_time = clock_type::now();
            }
        };
        TaskTracker() = default;

        // returns a reference to the stats object, so tasks can update their stats on their own
        void* register_task(TaskTypes ttype) {           
            _stats.emplace_back(ttype);
            _type_count[ttype]++;
            StatPosition pos = _stats.size()-1;
            
            auto ret = reinterpret_cast<void*>(pos);
            
            return ret;
        }

        TaskStats& get_stats(StatPosition pos) {
            return _stats.at(pos);
        }

        public:

        std::string dump_stats(const clock_type::time_point& program_start_time,
                        const clock_type::time_point& program_stop_time,
                        const std::string& file_path = "./stats.json") const {
            
            std::stringstream ss_json_start;
            auto time_t_start = std::chrono::system_clock::to_time_t(program_start_time);
            std::tm tm_start = *std::localtime(&time_t_start);
            ss_json_start << std::put_time(&tm_start, "%Y-%m-%dT%H:%M:%SZ");
            std::string timestamp_start_json = ss_json_start.str();
                        
            std::stringstream ss_json_stop;
            ss_json_stop.clear();
            auto time_t_stop = std::chrono::system_clock::to_time_t(program_stop_time);
            std::tm tm_stop = *std::localtime(&time_t_stop);
            ss_json_stop << std::put_time(&tm_stop, "%Y-%m-%dT%H:%M:%SZ");
            std::string timestamp_stop_json = ss_json_stop.str();

            std::stringstream ss_filename;
            ss_filename << std::put_time(&tm_start, "%Y-%m-%dT%H:%M:%SZ"/* "%Y%m%d_%H%M%S"*/);
            std::string timestamp_filename = ss_filename.str();

            // Store task type counts
            std::ofstream outfile(file_path);

            if (outfile.is_open()) {
                nlohmann::json output_json;
                
                nlohmann::json metadata;
                metadata["execution_start"] = timestamp_start_json;
                metadata["execution_stop"] = timestamp_stop_json;
                // Calculate execution time in nanoseconds
                auto execution_duration = program_stop_time - program_start_time;
                auto execution_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(execution_duration).count();
                metadata["execution_time_ns"] = execution_time_ns;
                
                output_json["metadata"] = metadata;

                nlohmann::json task_counts;
                for (const auto& [type, count] : _type_count) {
                    task_counts[to_string(type)] = count;
                }
                output_json["task_counts"] = task_counts;
        
                // Store individual task statistics
                nlohmann::json task_statistics = nlohmann::json::array();
                for (const auto& stat : _stats) {
                    nlohmann::json stat_json;
                    stat_json["type"] = to_string(stat.ttype);
                    
                    if (stat.start_time != clock_type::time_point{}) {
                        auto relative_start = stat.start_time - program_start_time;
                        auto relative_stop = stat.stop_time - program_start_time;
                        
                        auto start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(relative_start).count();
                        assert(start_ns > 0);
                        stat_json["start_ns"] = start_ns;
                        stat_json["stop_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(relative_stop).count();
                    } else {
                        stat_json["start_ns"] = nlohmann::json::value_t::null;
                        stat_json["stop_ns"] = nlohmann::json::value_t::null;
                    }                  
                    
                    // Handle optional input_volume
                    if (stat.input_volume.has_value()) {
                        stat_json["input_volume"] = stat.input_volume.value();
                    } else {
                        stat_json["input_volume"] = nlohmann::json::value_t::null;
                    }
                    // Handle optional output_volume
                    if (stat.output_volume.has_value()) {
                        stat_json["output_volume"] = stat.output_volume.value();
                    } else {
                        stat_json["output_volume"] = nlohmann::json::value_t::null;
                    }
                    // Handle optional worker_id
                    if (stat.worker_id.has_value()) {
                        stat_json["worker_id"] = stat.worker_id.value();
                    } else {
                        stat_json["worker_id"] = nlohmann::json::value_t::null;
                    }
                    task_statistics.push_back(stat_json);
                }
                output_json["task_statistics"] = task_statistics;
        
                // Write the JSON object to the specified file

                outfile << output_json.dump(4); // Use dump(4) for pretty printing (indentation of 4 spaces)
                outfile.close();
                // std::cout << "Performance statistics dumped to: " << full_filename << std::endl;
            } else {
                std::cerr << "\n\nERROR opening file for writing: " << file_path << "\n" << std::endl;
            }
            return file_path;
        }

        private:

        std::vector<TaskStats> _stats;
        std::map<TaskTypes, std::size_t> _type_count;
    };


}