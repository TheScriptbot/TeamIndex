#include "json_import.hpp"

#include "runtime/runtime.hpp"

using namespace TeamIndex;

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 2) {
        std::cerr << "Invalid number of arguments!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <json_file_path.json>" << std::endl;
        return 1;
    }

    std::vector<std::vector<RequestInfo>> request_infos;
    std::vector<TeamMetaInfo> team_workload_infos;
    PlanConfig pfg;
    ExecutorConfig cfg;
    Storage::StorageConfig storage_cfg;

    std::string query;
    std::cout << "Deserializing query specification from " << argv[1] << std::endl;
    try {
        deserialize_from_json(argv[1], query, request_infos, team_workload_infos, pfg, cfg, storage_cfg);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    auto cnt = 0u;
    auto vol_cnt  = 0u;
    for (auto& req_group : request_infos) {
        for (auto& req_info: req_group) {
            vol_cnt += req_info.total_block_cnt;
            cnt++;
        }
    }

    std::cout << "Deserialization successful, got " << cnt << " requests ("<< vol_cnt << " 4KiB pages) for " 
        << team_workload_infos.size() << " Teams!" << std::endl;

    std::cout << "Query: " << query << std::endl;

    std::cout << "Executing...\n" << std::endl;

    auto [ids, stats] = TeamIndexExecutor::run(request_infos, team_workload_infos, pfg, cfg, storage_cfg);
    
    if (cfg.return_result and ids.data() == nullptr and ids.size() != 0) {
        throw std::runtime_error("Result is null, but return_result is set to true!");
    }
    else if (not cfg.return_result and ids.data() != nullptr) {
        throw std::runtime_error("Result is not null, but return_result is set to false!");
    }
    if (cfg.return_result)
        std::cout << "Materialized result size: " << ids.size() << std::endl;
    else {
        if (stats.result_cardinality.has_value())
            std::cout << "Result size: " << stats.result_cardinality.value() << std::endl;
        else
            std::cout << "Result size: NULL!" << std::endl;
    }
    
    // it's our resposibility to clean up the result:
    free(ids.data());

    return 0;
}