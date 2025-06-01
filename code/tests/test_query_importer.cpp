#include "json_import.hpp"

using namespace TeamIndex;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <json_file_path.json>" << std::endl;
        return 1;
    }

    std::vector<std::vector<RequestInfo>> request_infos;
    std::vector<TeamMetaInfo> team_workload_infos;
    ExecutorConfig cfg;
    Storage::StorageConfig storage_cfg;
    std::string query;
    try {
        deserialize_from_json(argv[1], query, request_infos, team_workload_infos, cfg, storage_cfg);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    auto cnt = 0u;
    for (auto& req_group : request_infos)
        for (auto& req_info: req_group) {
            cnt++;
        }

    std::cout << "Deserialization successful." << std::endl;
    std::cout << "Got " << cnt << " requests for " << team_workload_infos.size() << " Teams!" << std::endl;
    return 0;
}
