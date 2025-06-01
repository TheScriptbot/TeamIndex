#include "runtime/io/dramBackend.hpp"
#include "interface/InterfaceTypes.hpp"



int main() {
    using namespace TeamIndex;

    TeamID tid = 0; // has no meaning here
    StartBlock offset = 16; // where to read from the file (as a number of 4KiB blocks)
    BlockCount list_size = 4; // I/O size
    ListCardinality cardinality = 100; // printed value count, should usually be the number of values in the list
    
    std::vector<Path> files = {"../../tests/test_index/F-H.copy.lists", "../../tests/test_index/B-I.copy.lists"};

    auto callback = [cardinality](TeamIndex::Storage::IORequest& req){
        std::cout << "Req ID: " << req.req_id << std::endl;
        std::cout << "I/O Size: " << req.block_count << " blocks" << std::endl;
        std::cout << "Printed value count: " << cardinality << std::endl;
        std::uint64_t max_cnt = (req.size())/sizeof(TeamIndex::IDType);
        auto values = std::span<TeamIndex::IDType>(reinterpret_cast<TeamIndex::IDType*>(req.buff), cardinality);
        std::cout << "Values: [" << std::endl;
        for (auto value : values) {
            std::cout << value << ", ";
        }
        std::cout << "..." << std::endl;
    };



    auto backend = Storage::dramBackend(files);

    Storage::IORequest req(0, nullptr, callback, offset, list_size, tid);

    req.buff = backend.get_list(tid, offset);

    req.callback(req);
}