#include "common_types.hpp"
#include "interface/InterfaceTypes.hpp"
#include "runtime/runtime.hpp"

#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>

using json = nlohmann::json;
namespace TeamIndex {

    static void deserialize_from_json(const std::string& input_path,
                                std::string& query,
                                std::vector<std::vector<RequestInfo>>& request_infos,
                                std::vector<TeamMetaInfo>& team_workload_infos,
                                PlanConfig& pcfg,
                                ExecutorConfig& cfg,
                                Storage::StorageConfig& storage_cfg) {
        std::ifstream file(input_path);
        if (!file) {
            throw std::runtime_error("Failed to open JSON file: " + input_path);
        }

        json data;
        file >> data;

        query = data["query"];
        
        for (const auto& req_group : data["request_infos"]) {
            std::vector<RequestInfo> group;
            for (const auto& req : req_group) {
                WithinRequestOffsets decomp_info;
                for (const auto& offset : req["decomp_info"]) {
                    decomp_info.emplace_back(offset[0], offset[1], string_to_codec(offset[2]), offset[3], offset[4]);
                }
                group.emplace_back(req["rid"], req["team_name"], req["start_block"], req["total_block_cnt"], decomp_info);
            }
            request_infos.push_back(std::move(group));
        }
        
        for (const auto& team : data["team_workload_infos"]) {
            team_workload_infos.emplace_back(team["team_name"],
                team["total_size_comp"],
                team["total_cardinality"],
                team["request_cnt"],
                team["list_cnt"],
                team["team_file_path"],
                team["is_included"],
                team["expand"],
                team["group_count"],
                team["min_group_size"],
                team["max_group_size"]);
        }

        pcfg.ise_count = data["plan_config"]["ise_count"];
        pcfg.outer_union_term_count = data["plan_config"]["outer_union_term_count"];
        pcfg.outer_union_group_count = data["plan_config"]["outer_union_group_count"];
        pcfg.outer_intersection_term_count = data["plan_config"]["outer_intersection_term_count"];
        pcfg.outer_intersection_group_count = data["plan_config"]["outer_intersection_group_count"];
        pcfg.leaf_union_list_parallel_threshold = data["plan_config"]["leaf_union_list_parallel_threshold"];
        pcfg.distributed_intersection_parallel_threshold = data["plan_config"]["distributed_intersection_parallel_threshold"];        
        
        cfg.worker_count = data["executor_config"]["worker_count"];
        cfg.backend = string_to_backend(data["executor_config"]["backend"]);
        cfg.print_execution_plan = data["executor_config"].contains("print_execution_plan") ? std::optional<std::string>{data["executor_config"]["print_execution_plan"]} : std::nullopt;
        cfg.print_task_stats = data["executor_config"].contains("print_task_stats") ? std::optional<std::string>{data["executor_config"]["print_task_stats"]} : std::nullopt;
        cfg.print_result_stats = data["executor_config"].contains("print_result_stats") ? std::optional<std::string>{data["executor_config"]["print_result_stats"]} : std::nullopt;
        cfg.experiment_name = data["executor_config"].contains("experiment_name") ? std::optional<std::string>{data["executor_config"]["experiment_name"]} : std::nullopt;
        cfg.verbose = data["executor_config"]["verbose"];
        cfg.return_result = data["executor_config"]["return_result"];

        storage_cfg.submit_batch_size = data["storage_config"]["submit_batch_size"];
        storage_cfg.await_batch_size = data["storage_config"]["await_batch_size"];
        storage_cfg.queue_pair_count = data["storage_config"]["queue_pair_count"];
        storage_cfg.liburing_cfg.queue_depth = data["storage_config"]["liburing_cfg"]["queue_depth"];
        storage_cfg.liburing_cfg.o_direct = data["storage_config"]["liburing_cfg"]["o_direct"];
        storage_cfg.liburing_cfg.sq_poll = data["storage_config"]["liburing_cfg"]["sq_poll"];
        storage_cfg.liburing_cfg.io_poll = data["storage_config"]["liburing_cfg"]["io_poll"];
        storage_cfg.liburing_cfg.sq_thread_idle = data["storage_config"]["liburing_cfg"]["sq_thread_idle"];
    }
}
