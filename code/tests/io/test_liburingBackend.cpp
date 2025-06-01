#include "runtime/io/Storage.hpp"
#include "interface/InterfaceTypes.hpp"
#include "utils.hpp"

#include <iostream>
#include <span>

/**
 * This test accesses a test index file and reads the first few bytes, printing the first values of the first list to console.
 * 
 */
void test1() {
    std::cout << "Test1" << std::endl;
    unsigned ring_id = 0;
    TeamIndex::BlockCount request_size = 4; // read just the first page
    TeamIndex::StartBlock start_block = 4;
    TeamIndex::ListCardinality cardinality = 3189;
    TeamIndex::Path test_team_path = "../../tests/test_index/A-E-C.copy.lists";
    TeamIndex::TeamName team_name = "A-E-C";



    // setup I/O access class
    TeamIndex::Storage::liburingBackend::Config ucfg;
    TeamIndex::Storage::StorageConfig cfg;
    TeamIndex::Storage::StorageAccessor<> accessor(cfg);

    // define some work
    std::vector<TeamIndex::TeamMetaInfo> tmis = {TeamIndex::TeamMetaInfo{team_name, 
        cardinality*sizeof(TeamIndex::IDType), cardinality, 1, 1, test_team_path, false}};

    accessor.register_Teams(tmis);

    auto buff = TeamIndex::get_new_io_buffer(request_size);

    auto callback = [buff, cardinality](){
        std::cout << "List length: " << cardinality << std::endl;
        auto values = std::span<TeamIndex::IDType>(reinterpret_cast<TeamIndex::IDType*>(buff), cardinality);
        std::cout << "Values: [" << std::endl;
        for (auto value : values) {
            std::cout << value << ", ";
        }
        std::cout << "..." << std::endl;
    };
    TeamIndex::Storage::IORequest req{0, buff, callback, start_block, request_size, 0};
    accessor.register_request(ring_id, std::move(req));
    

    // actual work:
    auto remaining_req = accessor.submit_request_async(ring_id);

    if (remaining_req != 0) {
        throw std::runtime_error("Expected 0 remaining submissions, got " + std::to_string(remaining_req));
    }

    auto& finished_req = accessor.get_finished_request(ring_id);
    
    // req.callback();
    std::cout << "Done." << std::endl;
}


/**
 * This test accesses a test index file and reads the first few bytes, printing the first values of the first list to console.
 * 
 */
void test2() {
    std::cout << "Test2" << std::endl;
    unsigned ring_id = 0;
    TeamIndex::BlockCount request_size = 4; // read just the first page
    TeamIndex::StartBlock start_block = 4;
    TeamIndex::ListCardinality cardinality = 3189;
    TeamIndex::Path test_team_path = "../../tests/test_index/A-E-C.copy.lists";
    TeamIndex::TeamName team_name = "A-E-C";



    // setup I/O access class
    TeamIndex::Storage::liburingBackend::Config ucfg;
    TeamIndex::Storage::StorageConfig cfg;
    TeamIndex::Storage::StorageAccessor<> accessor(cfg);

    // define some work
    std::vector<TeamIndex::TeamMetaInfo> tmis = {TeamIndex::TeamMetaInfo{team_name,
        cardinality*sizeof(TeamIndex::IDType), cardinality, 1, 1, test_team_path, false}};

    accessor.register_Teams(tmis);

    auto buff = TeamIndex::get_new_io_buffer(request_size);

    auto callback = [buff, cardinality](){
        std::cout << "List length: " << cardinality << std::endl;
        auto values = std::span<TeamIndex::IDType>(reinterpret_cast<TeamIndex::IDType*>(buff), cardinality);
        std::cout << "Values: [" << std::endl;
        for (auto value : values) {
            std::cout << value << ", ";
        }
        std::cout << "..." << std::endl;
    };

    TeamIndex::Storage::IORequest req{0, buff, callback, start_block, request_size, 0};
    accessor.register_request(ring_id, std::move(req));
    

    // actual work:
    auto remaining_req = accessor.submit_batch(ring_id);

    // auto& req = accessor.get_finished_request(ring_id);
    std::vector<TeamIndex::Storage::IORequest*> arrivals;
    do {
        accessor.await_batch(ring_id, arrivals);
    } while (arrivals.size() < 1);

    // for (auto req: arrivals)
    //     req->callback();

    std::cout << "Done." << std::endl;
}

int main() {
    test1();
    test2();
}