#pragma once

#include "interface/InterfaceTypes.hpp"
#include "io/Storage.hpp"
#include "utils.hpp"

#include <taskflow/taskflow.hpp>

#include <optional>
#include <utility>
#include <span>


namespace TeamIndex {

    struct PlanConfig {
        unsigned ise_count;
        unsigned outer_union_term_count;
        unsigned outer_union_group_count;
        unsigned outer_intersection_term_count;
        unsigned outer_intersection_group_count;
        unsigned leaf_union_list_parallel_threshold; // to determine which implementation to use when#
        unsigned distributed_intersection_parallel_threshold;
        unsigned long table_cardinality;
    };

    struct ExecutorConfig {
        unsigned worker_count = 1;
        StorageBackendID backend = StorageBackendID::DEFAULT;
        std::optional<std::string> print_execution_plan = std::nullopt;
        std::optional<std::string> print_task_stats = std::nullopt;
        std::optional<std::string> print_result_stats = std::nullopt;
        std::optional<std::string> experiment_name = std::nullopt;
        bool verbose = false;
        bool return_result = true;
    };
    
    class TeamIndexExecutor {
    public:
        TeamIndexExecutor() = default;
        [[nodiscard]]
        static std::tuple<std::span<IDType>, ExecutionStatistics> run(std::vector<std::vector<RequestInfo>> &request_infos,
                                                                                std::vector<TeamMetaInfo> &team_workload_infos,
                                                                                const PlanConfig& pcfg,
                                                                                const ExecutorConfig& cfg,
                                                                                const Storage::StorageConfig& storage_cfg);

        /** 
         * "Dry-run" the workload by only executing the I/O part (using the same runtime elements as the actual "run_workload").
         * 
         * Used as a baseline.
         */
        [[nodiscard]]
        static std::tuple<std::span<IDType>, ExecutionStatistics> run_read_all(std::vector<std::vector<RequestInfo>> &request_infos,
                                                                                std::vector<TeamMetaInfo> &team_workload_infos,
                                                                                const ExecutorConfig& cfg,
                                                                                const Storage::StorageConfig& storage_cfg);
    };

    /**
     * Taskflow worker interface that pins workers to specific CPU cores.
     */
    class PinnedWorker : public tf::WorkerInterface {
        public:
        // to call before the worker enters the scheduling loop
        void scheduler_prologue(tf::Worker& w) override {

            // now affine the worker to a particular CPU core equal to its id
            if(!Storage::set_processor_affinity(w.thread(), w.id())) {
                printf("failed to pin worker %lu to CPU core %lu\n", w.id(), w.id());
            }
        }

        // to call after the worker leaves the scheduling loop
        void scheduler_epilogue(tf::Worker& w, std::exception_ptr) override {

        }
    };

} // namespace TeamIndex
