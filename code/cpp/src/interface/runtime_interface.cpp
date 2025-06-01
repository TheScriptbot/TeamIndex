#include "runtime/runtime.hpp"
#include "interface/InterfaceTypes.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std::string_literals;

void define_runtime_interface(py::module &m) {
    
    py::class_<TeamIndex::RequestInfo>(m, "RequestInfo")
        .def(py::init<TeamIndex::RequestID,
                TeamIndex::TeamName,
                TeamIndex::StartBlock,
                TeamIndex::BlockCount,
                TeamIndex::WithinRequestOffsets>())
        .def_readwrite("rid", &TeamIndex::RequestInfo::rid)        
        .def_readwrite("team_name", &TeamIndex::RequestInfo::team_name)
        .def_readwrite("start_block", &TeamIndex::RequestInfo::start_block)
        .def_readwrite("total_block_cnt", &TeamIndex::RequestInfo::total_block_cnt)
        .def_readwrite("decomp_info", &TeamIndex::RequestInfo::decomp_info)
        .def("__repr__",
                [](const TeamIndex::RequestInfo &rinfo) {
                    return "<TeamIndex::RequestInfo - ID: " + std::to_string(rinfo.rid)
                    + " block_count: " + std::to_string(rinfo.total_block_cnt)
                    + ", team: " + rinfo.team_name
                    + ", inverted list count: " + std::to_string(rinfo.decomp_info.size())
                    + ">";
                }
        );
    py::class_<TeamIndex::PlanConfig>(m, "PlanConfig")
            .def(py::init<>())
            .def(py::init<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned long>())
            .def_readwrite("ise_count", &TeamIndex::PlanConfig::ise_count)
            .def_readwrite("outer_union_term_count", &TeamIndex::PlanConfig::outer_union_term_count)
            .def_readwrite("outer_union_group_count", &TeamIndex::PlanConfig::outer_union_group_count)
            .def_readwrite("outer_intersection_term_count", &TeamIndex::PlanConfig::outer_intersection_term_count)
            .def_readwrite("outer_intersection_group_count", &TeamIndex::PlanConfig::outer_intersection_group_count)
            .def_readwrite("leaf_union_list_parallel_threshold", &TeamIndex::PlanConfig::leaf_union_list_parallel_threshold)
            .def_readwrite("distributed_intersection_parallel_threshold", &TeamIndex::PlanConfig::distributed_intersection_parallel_threshold)
            .def_readwrite("table_cardinality", &TeamIndex::PlanConfig::table_cardinality)
            
            .def("__repr__",
                    [](const TeamIndex::PlanConfig &pcfg) {
                        return "<TeamIndex::PlanConfig - "
                          "ise_count: " + std::to_string(pcfg.ise_count)
                        + ", outer_union_term_count: " + std::to_string(pcfg.outer_union_term_count)
                        + ", outer_union_group_count: " + std::to_string(pcfg.outer_union_group_count)
                        + ", outer_intersection_term_count: " + std::to_string(pcfg.outer_intersection_term_count)
                        + ", outer_intersection_group_count: " + std::to_string(pcfg.outer_intersection_group_count)
                        + ", leaf_union_list_parallel_threshold: " + std::to_string(pcfg.leaf_union_list_parallel_threshold)
                        + ", distributed_intersection_parallel_threshold: " + std::to_string(pcfg.distributed_intersection_parallel_threshold)
                        + ", table_cardinality: " + std::to_string(pcfg.table_cardinality)
                        + ">";
                    }
            );
        

    py::class_<TeamIndex::TeamMetaInfo>(m, "TeamMetaInfo")
        .def(py::init<const TeamIndex::TeamName&, std::size_t, TeamIndex::ListCardinality,
            std::size_t, std::size_t, TeamIndex::Path,
            bool, bool, unsigned, unsigned, unsigned>())
        .def_readonly("team_name", &TeamIndex::TeamMetaInfo::team_name)
        .def_readonly("total_size_comp", &TeamIndex::TeamMetaInfo::total_size_comp)
        .def_readonly("total_cardinality", &TeamIndex::TeamMetaInfo::total_cardinality)
        .def_readonly("request_cnt", &TeamIndex::TeamMetaInfo::request_cnt)
        .def_readonly("list_cnt", &TeamIndex::TeamMetaInfo::list_cnt)
        .def_readonly("team_file_path", &TeamIndex::TeamMetaInfo::team_file_path)
        .def_readonly("is_included", &TeamIndex::TeamMetaInfo::is_included)
        .def_readonly("expand", &TeamIndex::TeamMetaInfo::expand)
        .def_readonly("group_count", &TeamIndex::TeamMetaInfo::group_count)
        .def_readonly("min_group_size", &TeamIndex::TeamMetaInfo::min_group_size)
        .def_readonly("max_group_size", &TeamIndex::TeamMetaInfo::max_group_size)
        .def("__repr__",
                [](const TeamIndex::TeamMetaInfo &tinfo) {
                    return "<TeamIndex::TeamMetaInfo - name: " + tinfo.team_name
                            + ",\n\ttotal compressed size: " + std::to_string(tinfo.total_size_comp)
                            + " byte,\n\ttotal cardinality: " + std::to_string(tinfo.total_cardinality)
                            + ",\n\tis_included = " + std::to_string(tinfo.is_included)
                            + ",\n\texpand = " + std::to_string(tinfo.expand)
                            + ",\n\tgroup count = " + std::to_string(tinfo.group_count)
                            + ",\n\tmin_group_size = " + std::to_string(tinfo.min_group_size)
                            + ",\n\tmax_group_size = " + std::to_string(tinfo.max_group_size)
                            + ">";
                }
        );

    py::enum_<TeamIndex::StorageBackendID> backend_enum(m,"StorageBackendID");
    backend_enum.value("default", TeamIndex::StorageBackendID::DEFAULT, "Default storage backend. Probably liburing.");
    backend_enum.value("liburing", TeamIndex::StorageBackendID::LIBURING, "liburing storage backend.");
    backend_enum.value("dram", TeamIndex::StorageBackendID::DRAM, "DRAM backend for benchmarking.");
    // backend_enum.value("",TeamIndex::StorageBackendID::SPDK, "SPDK storage backend.");

    m.def("backend_id_to_string", py::overload_cast<TeamIndex::StorageBackendID>(&TeamIndex::to_string));
    m.def("string_to_backend_id", &TeamIndex::string_to_backend);

    auto ucfg_to_string = [](const TeamIndex::Storage::liburingBackend::Config &cfg) {
        return "<TeamIndex::liburingBackendConfig - queue_depth:" + std::to_string(cfg.queue_depth)
                + ", o_direct: " + std::to_string(cfg.o_direct)
                + ", sq_poll: " + std::to_string(cfg.sq_poll)
                + ", io_poll: " + std::to_string(cfg.io_poll)
                + ", sq_thread_idle: " + std::to_string(cfg.sq_thread_idle)
                + ">";
    };

    py::class_<TeamIndex::Storage::liburingBackend::Config>(m, "liburingBackendConfig")
        .def(py::init<>())
        .def(py::init<unsigned, bool, bool, bool, unsigned>())
        .def_readwrite("queue_depth", &TeamIndex::Storage::liburingBackend::Config::queue_depth)
        .def_readwrite("o_direct", &TeamIndex::Storage::liburingBackend::Config::o_direct)
        .def_readwrite("sq_poll", &TeamIndex::Storage::liburingBackend::Config::sq_poll)
        .def_readwrite("io_poll", &TeamIndex::Storage::liburingBackend::Config::io_poll)
        .def_readwrite("sq_thread_idle", &TeamIndex::Storage::liburingBackend::Config::sq_thread_idle)
        .def("__repr__", ucfg_to_string);

    py::class_<TeamIndex::Storage::StorageConfig>(m, "StorageConfig")
        .def(py::init<>())
        .def(py::init<unsigned, unsigned, unsigned, TeamIndex::Storage::liburingBackend::Config>())
        .def_readwrite("submit_batch_size", &TeamIndex::Storage::StorageConfig::submit_batch_size)
        .def_readwrite("await_batch_size", &TeamIndex::Storage::StorageConfig::await_batch_size)
        .def_readwrite("queue_pair_count", &TeamIndex::Storage::StorageConfig::queue_pair_count)
        .def_readwrite("liburing_cfg", &TeamIndex::Storage::StorageConfig::liburing_cfg)
        .def("__repr__",
                [ucfg_to_string](const TeamIndex::Storage::StorageConfig &cfg) {
                    return "<TeamIndex::StorageConfig - queue_pair_count: " + std::to_string(cfg.queue_pair_count)
                            + ", submit_batch_size: " + std::to_string(cfg.submit_batch_size)
                            + ", await_batch_size: " + std::to_string(cfg.await_batch_size)
                            + ", " + ucfg_to_string(cfg.liburing_cfg)
                            + ">";
                }
        );
    py::class_<TeamIndex::ExecutorConfig>(m, "ExecutorConfig")
        .def(py::init<>())
        .def(py::init<unsigned,
            TeamIndex::StorageBackendID,
            std::optional<std::string>,
            std::optional<std::string>,
            std::optional<std::string>,
            std::optional<std::string>,
            bool,
            bool>())
        .def_readwrite("worker_count", &TeamIndex::ExecutorConfig::worker_count)
        .def_readwrite("backend", &TeamIndex::ExecutorConfig::backend)
        .def_readwrite("print_execution_plan", &TeamIndex::ExecutorConfig::print_execution_plan)
        .def_readwrite("print_task_stats", &TeamIndex::ExecutorConfig::print_task_stats)
        .def_readwrite("print_result_stats", &TeamIndex::ExecutorConfig::print_result_stats)
        .def_readwrite("experiment_name", &TeamIndex::ExecutorConfig::experiment_name)
        .def_readwrite("verbose", &TeamIndex::ExecutorConfig::verbose)
        .def_readwrite("return_result", &TeamIndex::ExecutorConfig::return_result)
        .def("__repr__",
                [](const TeamIndex::ExecutorConfig &cfg) {
                    return "<TeamIndex::ExecutorConfig - worker_count:" + std::to_string(cfg.worker_count)
                            + "backend:" + TeamIndex::to_string(cfg.backend)
                            + (cfg.print_execution_plan ? ", plan print path: " + cfg.print_execution_plan.value() : std::string(""))
                            + (cfg.print_task_stats ? ", task stats print path: " + cfg.print_task_stats.value() : std::string(""))
                            + (cfg.print_result_stats ? ", result stats print path: " + cfg.print_result_stats.value() : std::string(""))
                            + (cfg.experiment_name ? ", experiment name: " + cfg.experiment_name.value() : std::string(""))
                            + ", verbose: " + (cfg.verbose ? "true" : "false")
                            + ", return_result: " + (cfg.return_result ? "true" : "false")
                            + ">";
                }
        );
    py::class_<TeamIndex::ExecutionStatistics>(m, "ExecutionStatistics")
//            .def(py::init<std::optional<std::size_t>, std::optional<std::size_t>, std::optional<std::size_t>, std::optional<std::size_t>, std::optional<std::size_t>, std::optional<std::size_t>>())
            .def_readonly("input_cardinality", &TeamIndex::ExecutionStatistics::input_cardinality)
            .def_readonly("result_cardinality", &TeamIndex::ExecutionStatistics::result_cardinality)
            .def_readonly("plan_construction_runtime", &TeamIndex::ExecutionStatistics::plan_construction_runtime)
            .def_readonly("executor_runtime", &TeamIndex::ExecutionStatistics::executor_runtime)
            .def_readonly("task_stats_path", &TeamIndex::ExecutionStatistics::task_stats_path)
            .def("__repr__",
                 [](const TeamIndex::ExecutionStatistics &stats) {
                     return "<TeamIndex::ExecutionStatistics: "s
                            + "\n\tinput_cardinality: " + std::to_string(stats.input_cardinality)
                            + (stats.result_cardinality ? "\n\tresult_cardinality: " + std::to_string(stats.result_cardinality.value())+" ids" : ""s)
                            + (stats.plan_construction_runtime ? "\n\tplan_construction_runtime: " + std::to_string(stats.plan_construction_runtime.value()/1000u)+"µs" : ""s)
                            + (stats.executor_runtime ? "\n\texecutor_runtime: " + std::to_string(stats.executor_runtime.value()/1000u)+"µs" : ""s)
                            + (stats.task_stats_path ? "\n\ttask_stats_path: " + stats.task_stats_path.value() : ""s)
                            + ">";
                 }
            );

    m.def("run", [] (std::vector<std::vector<TeamIndex::RequestInfo>> &request_infos,
                std::vector<TeamIndex::TeamMetaInfo> &team_workload_infos,
                const TeamIndex::PlanConfig& pcfg,
                const TeamIndex::ExecutorConfig& cfg,
                const TeamIndex::Storage::StorageConfig& storage_cfg) {

            auto [ids, stats] = TeamIndex::TeamIndexExecutor::run(request_infos, team_workload_infos, pcfg, cfg, storage_cfg);
            
            
            if (not cfg.return_result) {
                assert(ids.data() == nullptr);
                // we do not expect a materialized result, so we won't return one to python either!
                // The actual result size can be obtained from stats
                return std::make_pair(py::array_t<TeamIndex::IDType>({0},{sizeof(TeamIndex::IDType)},nullptr), stats);
            }
            if (ids.data() == nullptr or ids.size() == 0) {
                free(ids.data());
                // no results! Return empty array
                return std::make_pair(py::array_t<TeamIndex::IDType>({0},{sizeof(TeamIndex::IDType)},nullptr), stats);
            }
            // Create a Python object that will free the allocated
            // memory when destroyed:
            py::capsule free_when_done(ids.data(), [](void *result) {
                free(result);
            });

            return std::make_pair(py::array_t<TeamIndex::IDType>(
                        {ids.size()}, // shape
                        {sizeof(TeamIndex::IDType)}, // C-style contiguous strides for double
                        ids.data(), // the data pointer
                        free_when_done),
                    stats);
        }, py::return_value_policy::move);
    
    m.def("run_read_all", [] (std::vector<std::vector<TeamIndex::RequestInfo>> &request_infos,
                std::vector<TeamIndex::TeamMetaInfo> &team_workload_infos,
                const TeamIndex::ExecutorConfig& cfg,
                const TeamIndex::Storage::StorageConfig& storage_cfg) {

            auto [ids, stats] = TeamIndex::TeamIndexExecutor::run_read_all(request_infos, team_workload_infos, cfg, storage_cfg);

            if (ids.data() == nullptr) {
                // no results! Return empty array
                return std::make_pair(py::array_t<TeamIndex::IDType>({0},{sizeof(TeamIndex::IDType)},nullptr), stats);
            }

            // Create a Python object that will free the allocated
            // memory when destroyed:
            py::capsule free_when_done(ids.data(), [](void *f) {
                auto *data_ptr = reinterpret_cast< TeamIndex::IDType *>(f);
                free(data_ptr);
            });

            return std::make_pair(py::array_t<TeamIndex::IDType>(
                        {ids.size()}, // shape
                        {sizeof(TeamIndex::IDType)}, // C-style contiguous strides for double
                        ids.data(), // the data pointer
                        free_when_done),
                    stats);
        }, py::return_value_policy::move);
}
