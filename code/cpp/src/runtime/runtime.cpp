#include "runtime/runtime.hpp"
#include "runtime/io/dramBackend.hpp"
#include "ti_codecs.hpp"

#include "runtime/task_observer.hpp"
#include "runtime/tasks/intersection.hpp"
#include "runtime/tasks/merge.hpp"

#include "roaring.hh"

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/sort.hpp>
#include <taskflow/algorithm/reduce.hpp>

namespace TeamIndex {
   
    /**
     * @brief Evaluate the index with increased parallelism by making use of an algebraic expansion of the index-expression.
     * 
     * 
     * Example with 4 Teams, where $A_Q$, $B_Q$, $C_Q$ and $D_Q$ denote the relevant set of lists/bins for each Team:
     * 
     * $$R =  (\binunion_{A_i \in A_Q} A_i) \cup (\binunion_{B_j \in B_Q} B_j) \cup (\binunion_{C_k \in C_Q} C_k) \cup (\binunion_{D_l \in D_Q} D_l)$$
     * 
     * For the sake of illustration, we access the complement sets of $C$ and $D$, such that we have to subtract their values (instead of intersect with them):
     * 
     * $$R =  (\binunion_{A_i \in A_Q} A_i) \cup (\binunion_{B_j \in B_Q} B_j) \setminus (\binunion_{C_k \in C_Q^C} C_k) \setminus (\binunion_{D_l \in D_Q^C} D_l)$$
     * 
     * One possible expansion of this term is with 4 Teams:
     * 
     * $$R = \bigcup_{i,j,k}^n A_i \cap B_j \setminus C_k \setminus (\bigcup_l D_l)$$.
     * 
     * We expand the terms/lists of certain Teams, e.g. $A_i$, $B_j$ and $C_k$ into their individual lists,to form all possible combinations of these lists.
     * Lists from other Teams (only $D_Q^C$ in this case) are exhaustively combined in each of these expressions.
     * The goal is to form independent tasks that can be executed in parallel. 
     * Choosing the Teams to expand allows to control the number of these parallel tasks.
     * In this example, we would have $|A_Q| \cdot |B_Q| \cdot |C_Q^C|$ parallel tasks.
     * $D_Q^C$ is not expanded, such that all of it's lists are common dependencies of all tasks.
     * 
     * Not choosing to expand a Team allows to keep the number of tasks in check, but may lead to less parallelism.
     * This is an optimizer choice.
     * 
     * The lists of non expanded, i.e., "shared", Team are dependencies for every "pipeline", so there might be many! 
     * 
     */
    std::tuple<std::span<IDType>, ExecutionStatistics>
    TeamIndexExecutor::run(std::vector<std::vector<RequestInfo>> &request_infos,
                            std::vector<TeamMetaInfo> &team_workload_infos,
                            const PlanConfig& pcfg,
                            const ExecutorConfig& ecfg,
                            const Storage::StorageConfig& scfg)
    {
        constexpr bool verbose_tasks = false;
        constexpr unsigned id_mod = 2;  // module for cluster_id, to control verbosity

        constexpr bool performance_tracking = true;

        constexpr size_t max_bitmaps = 64; // Maximum expected Bitmaps. Determins stack pointer array size for sequential task implementations 
        

        // track start time as ISO date time string to be used in filenames:
        auto start_time = std::chrono::system_clock::now();



        if (max_bitmaps < pcfg.leaf_union_list_parallel_threshold) {
            throw std::runtime_error("pcfg.leaf_union_list_parallel_threshold parameter surpasses maximum stack-array size of "
                 + std::to_string(max_bitmaps) + " (Paramter: " + std::to_string(pcfg.leaf_union_list_parallel_threshold) + ")!");
        }

        using Bitmap = roaring::Roaring;
        
        bool do_io = ecfg.backend != StorageBackendID::DRAM;

        assert(scfg.queue_pair_count == request_infos.size() 
            or (request_infos.size() == 1 and !do_io));
        
        assert(team_workload_infos[PRIMARY_EXPANSION].is_included);


        /////////////////////// Fill in some meta data:
        std::vector<Path> team_files;
        std::unordered_map<TeamName, TeamID> team_ids;
        
        auto team_cnt = team_workload_infos.size();

        auto list_count = 0u;
        auto total_read_volume = 0u; 
        auto tota_request_cnt = 0u;
        auto tota_read_card = 0u;
        {
            TeamID id = 0;
            for (auto& team_info : team_workload_infos) {
                team_ids[team_info.team_name] = id++;
                list_count += team_info.list_cnt;
                total_read_volume += team_info.total_size_comp;
                tota_request_cnt += team_info.request_cnt;
                tota_read_card += team_info.total_cardinality;
                team_files.push_back(team_info.team_file_path);
            }
        }
        /////////////////////// Determine "Pipeline" parallelism and related information
                
        std::vector<TeamID> included_expanded_teams;
        std::vector<TeamID> excluded_expanded_teams;
        std::vector<TeamID> included_distributed_teams;
        std::vector<TeamID> excluded_distributed_teams;
        
        // note that the actual group sizes can vary greatly, we use this to track edge cases
        std::vector<unsigned> average_group_cardinalities;
        average_group_cardinalities.reserve(team_workload_infos.size());

        for (auto& team_info : team_workload_infos) {
            TeamID tid = team_ids[team_info.team_name];
            if (team_info.expand) {
                if (team_info.is_included) {
                    included_expanded_teams.emplace_back(tid);
                }
                else {
                    excluded_expanded_teams.emplace_back(tid);
                }
            }
            else {
                if (team_info.is_included) {
                    included_distributed_teams.emplace_back(tid);
                }
                else {
                    excluded_distributed_teams.emplace_back(tid);
                }
            }
            average_group_cardinalities.emplace_back((float) team_info.list_cnt / team_info.group_count);
            assert(average_group_cardinalities[tid] >= 1);
        }
        if (not excluded_expanded_teams.empty()) {
            assert(pcfg.ise_count > 0);
            assert(not included_expanded_teams.empty()); // we need something to subtract from! TODO: could subtract from a distributed, included Team..
        }
        if (pcfg.ise_count > 0) {
            assert(team_workload_infos[PRIMARY_EXPANSION].expand); // needs to be sorted "expanded" -> "distributed"
            assert(pcfg.outer_union_term_count > 0); // Distributing terms over an "expanded Team" implies there is an outer union
            assert(pcfg.outer_union_term_count * pcfg.outer_intersection_term_count == pcfg.ise_count);
            assert(pcfg.outer_intersection_term_count == 1 or pcfg.outer_intersection_term_count >= pcfg.outer_intersection_group_count); // otherwise makes no sense to group, no?
            assert(pcfg.outer_union_term_count == 1 or pcfg.outer_union_term_count >= pcfg.outer_union_group_count); // otherwise makes no sense to group, no?
        }
        else {
            assert(pcfg.outer_union_term_count == 0 and pcfg.outer_union_group_count == 0);
        }
        if (ecfg.verbose) {
            if (pcfg.ise_count)
                std::cout << "There are " << pcfg.ise_count << " ISEs to be executed." << std::endl;
            std::cout << "We will expand " << included_expanded_teams.size()+excluded_expanded_teams.size() << " Teams." << std::endl;
            std::cout << "We will distribute " << included_distributed_teams.size()+excluded_distributed_teams.size() << " Teams." << std::endl;
            // std::cout << "All ISEs process a unique combination of " << expansion_team_cnt << " leafs from expanded Teams." << std::endl;
            // std::cout << "All task clusters share " << shared_list_cnt << " leafs." << std::endl;
            // std::cout << "There are " << big_intersection_operand_count << " cluster-unions to be intersected." << std::endl;
            // std::cout << "Each cluster union has " << big_union_operand_count << " operands." << std::endl;
            // assert(big_union_operand_count * big_intersection_operand_count == task_cluster_cnt);
        }
        
        // we register tasks here, which allows us to attach a data structure for stat tracking at creation time
        TaskTracker tracker;

        // implements visitor pattern, which tracks worker id, start and end times in the stat data structure
        PerformanceObserver performanceobserver(tracker);



        /////////////////////// Allocate data structures for I/O and deserialization

        Storage::StorageAccessor storage(scfg); // use either this
        Storage::DRAMAccessor backend; // or this

        if (ecfg.backend == StorageBackendID::LIBURING) {
            if (ecfg.verbose)
                std::cout << "liburing based access, will load data during runtime..." << std::endl;
            storage.register_Teams(team_workload_infos);
        }
        else {
            if (ecfg.verbose)
                std::cout << "DRAM based access, preloading data into memory..." << std::endl;
            // already reads all team files into main memory buffers for later use - no IO involved during runtime!
            backend.register_Teams(team_files);
        }
        
        Timer plan_timer("PlanCreation");

        // pointers to I/O buffers, needs cleanup later
        std::vector<BufferType*> io_buffers; 
        io_buffers.reserve(list_count*do_io);
        
        //////////////////////////////////// Data structures that hold intermediate results

        // holds data after decompression (will not be used if we use Roaring)
        std::vector<IDType*> decompression_buffers; 
        decompression_buffers.reserve(list_count);


        // Intermediate "group" results for unions over leafs:
        using Group = std::vector<Bitmap>;
        using LeafUnionIterators = std::pair<Group::iterator,Group::iterator>;

        // holds input/"leaf data"
        // Each team has leafs. Those leafs are partitioned into 1 or more Groups. Each Group is assigned a potentially varying number of leafs
        std::vector<std::vector<Group>> leaf_data; // accessed via "tid" and "group_id"
        
        // stores leaf unions. Will simply contain leaf data, if there is no leaf union for a team
        std::vector<std::vector<Bitmap>> leaf_union_results; // indexed via [tid][group_id]
        
        // meta data for runtime access to the leaf bitmap without relying on potentially moving object addresses
        std::vector<std::vector<std::vector<unsigned>>> leaf_meta_data; // for each Team and group, we have one entry per leaf

        // to be initialized during runtime to properly refer to leaf groups (since their sizes are not yet known)
        std::vector<std::vector<LeafUnionIterators>> leaf_union_iterators;

        leaf_data.resize(team_cnt);
        leaf_union_iterators.resize(team_cnt);
        leaf_union_results.resize(team_cnt);
        leaf_meta_data.resize(team_cnt);
        for (auto& tinfo : team_workload_infos) {
            auto tid = team_ids[tinfo.team_name];
            // Note: We do not know exactly how large each group is, so we will append Bitmaps later (instead of allocating here)
            // Note: A group_id is already assigned in the request-meta data for each leaf/list
            leaf_data[tid].resize(tinfo.group_count);
            
            leaf_union_results[tid].resize(tinfo.group_count);
            
            // Also, one <begin(), end()> pair for each group:
            leaf_union_iterators[tid].resize(tinfo.group_count);
            
            leaf_meta_data[tid].resize(tinfo.group_count);
        }


        /////////////////////// The following is used in case there are EXPANDED teams:
        // Stores intermediate result of a independent sub expression (ISE)
        // Note that there may be NO ISE at all, i.e., no expanded Teams!
        std::vector<Bitmap> ise_results;
        ise_results.resize(pcfg.ise_count);
        
        // holds the result of combining groups of distributed Teams with the ISE intermediate result
        // We need need this additional intermediate, since they we need to intersect each group of the 
        // distributed Teams with the ISE result, followed by a union
        // we may skip using this, if there is no such included, distributed Team alongside expanded Teams
        std::vector<Bitmap> distributed_intersection_results;
        distributed_intersection_results.resize(pcfg.ise_count); 
        
        // one bitmap per outer/"big" union. There is one union FOR EACH COMBINATION of excluded, expanded list combination!
        std::vector<Group> big_union_group_results; // indexed by [big_union_id][big_union_group_id]
        // if there are intersections, we union only subsets (regardless of grouping)
        // i.e., there may be one than one grouped outer union (outer_intersection_term_count many, to be precise)
        big_union_group_results.resize(pcfg.outer_intersection_term_count);
        for (auto& bug: big_union_group_results) {
            // note: each of these unions has, on average, outer_union_term_count/outer_union_group_count many terms
            bug.resize(pcfg.outer_union_group_count);
        }

        // the second state of outer unions: per intersection, we have one set of groups that we need to union (before the outer intersection runs)
        std::vector<Bitmap> big_union_results;
        if (pcfg.ise_count > 0) {
            // if there are ISEs, we have one union term per ISE
            big_union_results.resize(pcfg.outer_intersection_term_count);
        }
        else {
            // if there is no ISE/no expanded Team, we implement a "UNION FIRST" approach. Then, we have one union per Team.
            // we store the results in this array
            big_union_results.resize(team_workload_infos.size());
        }
        // Holds results for the groups of the outer intersection
        // The outer_intersection_term_count many unions are group-aggregated first, if required
        std::vector<Bitmap> big_intersection_group_results;
        
        // count the number of subtractions. Only makes sense if we have no expanded Teams
        auto team_subtraction_count = 0u;
        if (pcfg.ise_count == 0)
            team_subtraction_count = team_workload_infos.size()-pcfg.outer_intersection_term_count;
        
        big_intersection_group_results.resize(pcfg.outer_intersection_group_count+team_subtraction_count);
        // for (auto& big_res : big_intersection_group_results) {
        //     big_res.flip(0,std::numeric_limits<IDType>::max());
        // }

        Bitmap final_intersection; // outer_intersection_group_count many intersections are reduced to 1
        // final_intersection.flip(0,std::numeric_limits<IDType>::max());

        /////////////////////// final result
        IDType* result = nullptr; // to be allocated later
        ListCardinality result_cardinality = 0;

        /////////////////////// Create the Task Graph
        tf::Taskflow taskflow;
        taskflow.name("Index Evaluation");

        ///////////////////// "Decompression" Tasks, that run directly on I/O results. One task per list!
        // a "leaf" represents the start of computation (after I/O)
        std::vector<std::vector<tf::Task>> leaf_tasks; // to be accessed via TeamID
        leaf_tasks.resize(team_cnt);
        for (auto& tinfo : team_workload_infos) {
            auto tid = team_ids[tinfo.team_name];
            leaf_tasks[tid].resize(tinfo.list_cnt);
        }
        
        // store a mapping from request_id to one or more tasks that are associated with each request
        // these "spans" (i.e., intervals) refer to tasks in "leaf_tasks[tid]"
        // note: a "span" always refers to a single team only
        std::vector<std::tuple<unsigned, unsigned>> to_be_scheduled;
        to_be_scheduled.resize(tota_request_cnt * do_io); // already create entries, we assign spans by "request_id"

        ///////////////////// Leaf union tasks ("GROUPS")
        using GroupTasks = std::vector<tf::Task>;

        std::vector<GroupTasks> leaf_union_prepare_tasks;
        std::vector<GroupTasks> leaf_union_tasks;
        leaf_union_prepare_tasks.resize(team_cnt);
        leaf_union_tasks.resize(team_cnt);
        for (auto& tinfo : team_workload_infos) {
            auto tid = team_ids[tinfo.team_name];
            leaf_union_prepare_tasks[tid].resize(tinfo.group_count);
            leaf_union_tasks[tid].resize(tinfo.group_count);
        }


        ///////////////////// ISE tasks
        std::vector<std::vector<tf::Task>> ise_tasks; // for each vector per team; one task per ISE
        if (pcfg.ise_count > 0) {
            ise_tasks.resize(team_cnt);
            for (auto& vec : ise_tasks)
                // note that there are likely more than one task for distributed Teams. We store them somewhere else
                vec.resize(pcfg.ise_count);
        }

        std::vector<tf::Task> ise_done_tasks;
        ise_done_tasks.resize(pcfg.ise_count);
        
        ///////////////////// outer/big union/intersection tasks
        // The number of terms in the big union(s) is given by the product of group counts of all expanded, included Teams
        //
        // With no expanded, excluded Teams, there is only a single intersection term, i.e., no big intersection.
        // The product of the group count for all expanded, excluded Team gives the number of intersections.
        //
        // Notably, with NO grouping of big-union terms, there would be only a single union.
        // The most notable case is if ALL Teams are expanded and included.
        //
        // Lastly, with NO EXPANDED Teams, there is a big outer intersection but no big outer union.
        // The big outer intersection has as many terms as there are relevant Teams. We may still want to group the intersection!
        
        std::vector<GroupTasks> inner_big_union_tasks; // indexed by [big_union_id][big_union_group_id]
        std::vector<tf::Task> outer_big_union_tasks; // indexed by [big_union_id]
        if (pcfg.ise_count > 0) {
            inner_big_union_tasks.resize(pcfg.outer_intersection_term_count);
            for (auto& ibu_tasks : inner_big_union_tasks) {
                // note: each of these unions has, on average, outer_union_term_count/outer_union_group_count many terms
                ibu_tasks.resize(pcfg.outer_union_group_count);
            }

            outer_big_union_tasks.resize(pcfg.outer_intersection_term_count);
        }
        else {
            // if there is no ISE/no expanded Team, we implement a "UNION FIRST" approach. Then, we have one union per Team.
            // we store the results in this array
            outer_big_union_tasks.resize(team_workload_infos.size());
        }
        
        ///////// Simple loop that determines the group size of all leaf groups. Requires iterating over all requests and their decompression infos

        std::vector<std::vector<std::size_t>> group_sizes; // one vector per Team
        group_sizes.resize(team_cnt);
        for (auto& tinfo : team_workload_infos) {
            auto tid = team_ids[tinfo.team_name];
            group_sizes[tid].resize(tinfo.group_count, 0);
        }

        for (auto& rinfo : request_infos) {
            for (auto& request : rinfo) {
                auto tid = team_ids[request.team_name];
                auto& tinfo = team_workload_infos[tid];
                
                for (auto& [off, card, codec_id, size, group_id] : request.decomp_info) {
                    // one entry per list in this request (StartBlock, ListCardinality, CodecID, ListSizeCompressed, Group ID)
                    group_sizes[tid][group_id]++;
                    assert(group_sizes[tid][group_id] <= tinfo.max_group_size);
                }
            }
        }
        
        std::vector<std::unique_ptr<std::atomic<unsigned>>> finished_leafs;
        for (size_t i = 0; i < team_cnt; ++i) {
            finished_leafs.emplace_back(std::make_unique<std::atomic<unsigned>>(0));
        }


        ////////////////////// actually create leaf union tasks
        // we do this first, so we can assign the leafs to their respective leaf group union, this requires the task to already properly exist
        for (auto& tinfo : team_workload_infos) {
            auto tid = team_ids[tinfo.team_name];

            // we choose different physical implementations for the union, especially for edge cases or very small unions
            // case tinfo.group_count = 1: there is just one big union! handled as either of the 2 following cases
            // case tinfo.group_count < tinfo.list_cnt: every leaf is in its own group -> no union -> simply pass through data
            // further:
            // case max_group_size < x: we use a sequential implementation
            // case max_group_size >= x: we use a parallel tree-reduction (taskflow reduce)
            // with x = pcfg.leaf_union_list_parallel_threshold

            for (auto group_id = 0u; group_id < tinfo.group_count; group_id++) {
                auto gsize = group_sizes[tid][group_id];
                assert(tinfo.min_group_size <= gsize);
                assert(tinfo.max_group_size >= gsize);

                tf::Task leaf_union;
                // we need to create a task for each group, even if there is no union
                if (tinfo.group_count < tinfo.list_cnt) {
                    assert(tinfo.max_group_size > 1); // just a parameter consistency check
                    // Task to initialize iterators to be used by the followup group union
                    // Note: This task will be proceeded by the corresponding leafs
                    tf::Task leaf_union_prepare_task = taskflow.emplace([tid, group_id, &leaf_union_iterators, &leaf_data] ()
                        {
                            // will be called during runtime, i.e., after the vector's actual size is determined
                            LeafUnionIterators iter_pair = {leaf_data[tid][group_id].begin(), leaf_data[tid][group_id].end()};
                            leaf_union_iterators[tid][group_id] = iter_pair;
                        });
                    
                    // performance tracking:
                    // leaf_union_prepare_task.data(tracker.register_task(TaskTypes::LeafUnionPrepare));

                    leaf_union_prepare_task.name("LEAF_UNION.PREPARE."+tinfo.team_name+"."+std::to_string(group_id));
                    leaf_union_prepare_tasks[tid][group_id] = leaf_union_prepare_task;   

                    if (tinfo.max_group_size > pcfg.leaf_union_list_parallel_threshold) {

                        // a tree reduction
                        leaf_union = taskflow.reduce(std::ref(leaf_union_iterators[tid][group_id].first),
                                                     std::ref(leaf_union_iterators[tid][group_id].second),
                                                     leaf_union_results[tid][group_id] /** init value, i.e., an empty bitmap */,
                                                     [tid, group_id]
                                (roaring::Roaring& a, const roaring::Roaring& b) -> roaring::Roaring
                            {
                                if constexpr (verbose_tasks)
                                    std::cout << "Leaf union " << tid << "/"<< group_id << ": "
                                        << a.cardinality() 
                                        << " += " << b.cardinality() << std::endl;
                                return a | b;
                            });
                        
                        leaf_union.name("LEAF_UNION." + tinfo.team_name + "." + std::to_string(group_id) +".PARALLEL");
                        leaf_union.data(tracker.register_task(TaskTypes::LeafUnion));
                    }
                    else {

                        leaf_union = taskflow.emplace([tid, group_id, &leaf_union_results, &leaf_union_iterators, &max_bitmaps] ()
                            {
                                auto& [begin, end] = leaf_union_iterators[tid][group_id];
                                
                                assert(begin->cardinality() > 0);
                                
                                std::size_t dist = std::distance(begin, end);

                                assert(dist <= max_bitmaps);
                                
                                    // quick case, this is a trivial leaf union, i.e., pass through
                                if (dist == 1) {
                                    leaf_union_results[tid][group_id] = *begin;
                                    return;
                                }
                                
                                // roaring reaaaallly wants a Bitmap**, so we have to convert the vector of vectors :(
                                std::span<Bitmap> span{begin, dist};
                                assert(span.size() <= max_bitmaps);
                                
                                const Bitmap* bitmap_array[max_bitmaps];
                                for (size_t i = 0; i < span.size(); ++i) {
                                    bitmap_array[i] = &span[i];
                                }

                                // actual union operation
                                leaf_union_results[tid][group_id] = roaring::Roaring::fastunion(span.size(), bitmap_array);
                                
                                if constexpr (verbose_tasks)
                                    std::cout << "Sequential Leaf union " << tid << "/"<< group_id << " over " << span.size() 
                                        << " bitmaps: " << leaf_union_results[tid][group_id].cardinality() << std::endl;
                                if (leaf_union_results[tid][group_id].cardinality() == 0) {
                                    std::cerr << "Empty leaf union! TID: " << tid << " group_id: " << group_id  << std::endl;
                                }

                            });
                        leaf_union.name("LEAF_UNION." + tinfo.team_name + "." + std::to_string(group_id) +".SEQUENTIAL");
                        leaf_union.data(tracker.register_task(TaskTypes::LeafUnion));
                    }
                    leaf_union.succeed(leaf_union_prepare_task);
                    // leaf_union_tasks[tid][group_id] = leaf_union;
                }
                
                else {
                    assert(tinfo.group_count == tinfo.list_cnt and tinfo.max_group_size == 1);

                    // pass through and simply swap data
                    leaf_union = taskflow.emplace([&leaf_union_results, &leaf_data, tid, group_id, &finished_leafs]()
                        {
                            if (leaf_data.at(tid).at(group_id).size() != 1) {
                                std::cerr << "Leaf union: " << tid << "/"<< group_id << " has " 
                                    << leaf_data[tid][group_id].size() << " bitmaps!" << std::endl;
                            }
                            auto ingoing_card = leaf_data[tid][group_id][0].cardinality();

                            leaf_union_results.at(tid).at(group_id).swap(leaf_data.at(tid).at(group_id).at(0));

                            assert(leaf_union_results[tid][group_id].cardinality() == ingoing_card);
                            
                        });
                    leaf_union.name("LEAF_UNION."+tinfo.team_name+"."+std::to_string(group_id)+".PASS");
                    // we set dependencies below
                    leaf_union.data(tracker.register_task(TaskTypes::LeafUnion));
                }
                leaf_union_tasks[tid][group_id] = leaf_union;
            }
        }

        ///////////////////// IO Tasks, may be fewer than lists (due to request merging):
        std::vector<tf::Task> submission_tasks;
        std::vector<tf::Task> await_tasks;

        std::vector<std::vector<Storage::IORequest*>> arrival_vectors; // temporarily holds pointers to finished requests
        arrival_vectors.resize(scfg.queue_pair_count * do_io);


        // Note: The following 3 tasks have no meaning with "!do_io"
        // "conditional task", only the first dependent will be triggered, the rest "dangles" (until manually started)
        // ALL tasks are directly or indirectly dependent on this task, so we can better control when they start
        auto stopper_task = taskflow.emplace([]() {return 0 /* i.e. only I/O tasks are directly started*/;});
        stopper_task.name("START");

        auto io_start_task = taskflow.emplace([](){});
        io_start_task.succeed(stopper_task);
        io_start_task.name("I/O."+std::to_string(tota_request_cnt)+"[requests]");

        auto io_end_task = taskflow.emplace([tota_request_cnt](){
            if constexpr (verbose_tasks)
                std::cout << "I/O Done! (" << tota_request_cnt << " requests)" << std::endl;
        });
        io_end_task.name("I/O_DONE");
        
        std::vector<tf::Task> io_queue_done_tasks;
        io_queue_done_tasks.reserve(request_infos.size());
        for (auto queue_id = 0u; queue_id < request_infos.size(); queue_id++) {
            auto q_request_cnt = request_infos[queue_id].size();
            tf::Task qdone_task = taskflow.emplace([queue_id,q_request_cnt](){
                if constexpr (verbose_tasks)
                    std::cout << "I/O queue "+ std::to_string(queue_id) + " done! (" << q_request_cnt << " requests)" << std::endl;
            });
            qdone_task.name("I/O_DONE.Q" + std::to_string(queue_id) + "." + std::to_string(q_request_cnt) + "[requests].");

            qdone_task.precede(io_end_task);
            io_queue_done_tasks.emplace_back(qdone_task);
        }
        
        std::vector<unsigned> leaf_ids;
        leaf_ids.resize(team_cnt, 0);


        
        for (auto request_group = 0u; request_group < request_infos.size(); request_group++) {
            if (do_io) {
                // define actual I/O tasks:
                // one pair of "submit" and "await" for each ring
                auto queue_pair_id = request_group;
                auto submit_task = taskflow.placeholder();
                submit_task.name("SUBMIT.Q"+std::to_string(queue_pair_id)+".LOOP");
                submit_task.work([&storage, queue_pair_id]() {
                    
                    auto i = storage.submit_batch(queue_pair_id); // my not do anything, if request queue is full atm
                    // std::cout << queue_pair_id << ": Called submit_batch, " << i << " remaining..." << std::endl;
                });
                
                auto await_task = taskflow.placeholder();
                await_task.name("AWAIT.Q"+std::to_string(queue_pair_id)+".LOOP");
                await_task.work([&storage, &arrival_vectors, &to_be_scheduled, &leaf_tasks, queue_pair_id] (tf::Runtime& rt) {

                    /// quickly peak into the arrival queue and take out a batch of requests
                    auto cnt = storage.await_batch(queue_pair_id, arrival_vectors[queue_pair_id]);
                    
                    assert(arrival_vectors[queue_pair_id].size() == cnt);
                    /// process arrivals, if any are available
                    for (Storage::IORequest* arrival: arrival_vectors[queue_pair_id]) {
                        assert(arrival);
                        auto [start, count] = to_be_scheduled[arrival->req_id];
                        std::span<tf::Task> tasks{leaf_tasks[arrival->tid].data()+start,count};
                        for (auto& task: tasks) {
                            rt.schedule(task); // the callback starts tasks processing the data for this request
                        }
                    }

                    arrival_vectors[queue_pair_id].clear();                    
                });

                auto loop_task = taskflow.placeholder();
                loop_task.name("LOOP."+std::to_string(request_infos[queue_pair_id].size())+"[requests]");
                loop_task.work([&storage, queue_pair_id]() {

                    /// queue up submit_task again, if there is still some I/O to submit left 
                    if (storage.remaining_submissions(queue_pair_id) > 0) {
                        return 0;
                    }
                    // // no submissions, but in-flight I/Os?
                    if (storage.remaining_arrivals(queue_pair_id) > 0) {
                        return 1; // loop back
                    }
                    // all done!
                    return 2; // break out
                });

                submit_task.succeed(io_start_task);
                await_task.succeed(submit_task);
                loop_task.succeed(await_task);
                loop_task.precede(submit_task, await_task, io_queue_done_tasks[queue_pair_id]); // loop back to submission, just wait again or we are done!

                submission_tasks.emplace_back(submit_task);
                await_tasks.emplace_back(await_task);
            } // if(do_io) END
            
            for (auto& request_info : request_infos.at(request_group)) {
                TeamID tid = team_ids[request_info.team_name];
                
                if (team_workload_infos[tid].group_count > team_workload_infos[tid].list_cnt)
                    throw std::runtime_error("Group count is larger than list count! This should not happen!");

                assert(leaf_tasks[tid].size() == team_workload_infos[tid].list_cnt);
                
                auto& leaf_id = leaf_ids[tid];

                /// first, create one or more tasks for decompression and bitmap creation
                auto task_range_start = leaf_id; // TODO this is a little shaky, as it relies on us allocating the vector beforehand
                BufferType* io_buffer_ptr = nullptr;
                if (do_io) {
                    io_buffer_ptr = get_new_io_buffer(request_info.total_block_cnt); // where to store the I/O data
                    io_buffers.emplace_back(io_buffer_ptr);
                }

                for (auto& [off, card, codec_id, size, group_id] : request_info.decomp_info) { // one entry per list in this request (StartBlock, ListCardinality, CodecID, ListSizeCompressed)
                    BufferType* list_ptr = nullptr;
                    if (ecfg.backend == StorageBackendID::LIBURING) {
                        // get physical address of the list within the buffer
                        // Note: "off" is relative to the buffer begin and in "blocks", not bytes
                        list_ptr = io_buffer_ptr+(off*PAGESIZE);
                    }
                    else if (ecfg.backend == StorageBackendID::DRAM) {                        

                        list_ptr = backend.get_list_ptr(tid, request_info.start_block+off); // set pointer to actual list data (potentially compressed)
                    }

                    // remember position BEFORE adding an element
                    auto bitmap_position = leaf_data[tid][group_id].size();
                    // add empty bitmap; to be overwritten later
                    leaf_data[tid][group_id].emplace_back();
                    
                    tf::Task leaf_task = taskflow.placeholder();

                    // store result reference to be recovered when creating follow-up tasks
                    // leaf_task.data(&(leaf_data[tid][group_id][bitmap_position]));
                    leaf_meta_data[tid][group_id].emplace_back(bitmap_position);

                    if (codec_id != CodecID::ROARING) {
                        IDType* output_ptr = get_new_decompression_buffer(card);
                        decompression_buffers.emplace_back(output_ptr); // remember pointer for later clean-up
                        
                        leaf_task.name("DECOMPRESS."+to_string(codec_id)+"."+request_info.team_name+".R"+std::to_string(request_info.rid)+"."+std::to_string(card)+"[tids]");
                        leaf_task.work([&leaf_data, tid, group_id, leaf_id, bitmap_position, codec_id, list_ptr, size, output_ptr, card] () {
                            // first decompress....
                            auto result = decode(codec_id, list_ptr, size, output_ptr, card);
                            assert(result.size() == card);

                            // ... then convert to a Roaring bitmap
                            leaf_data[tid][group_id][bitmap_position] = roaring::Roaring(result.size(), result.data());
                            // free(output_ptr);
                            // free(list_ptr);
                            // if constexpr (verbose_tasks)
                            //     std::cout << "Decompressed " << leaf_data[tid][group_id][bitmap_position].cardinality() << " values!" << std::endl;
                        });
                        leaf_task.data(tracker.register_task(TaskTypes::Leaf));
                    }
                    else {
                        leaf_task.name("DESERIALIZE."+to_string(codec_id)+"."+request_info.team_name+".R"+std::to_string(request_info.rid)+"."+std::to_string(card)+"[tids]");
                        leaf_task.work([&leaf_data, tid, group_id, leaf_id, bitmap_position, codec_id, list_ptr, size, card] () {
                            static_assert(sizeof(uint32_t) == sizeof(IDType));
                            auto fbm = roaring::Roaring::frozenView(list_ptr, size);
                            assert(fbm.cardinality() == card);
                            leaf_data[tid][group_id][bitmap_position] = std::move(fbm);
                            // free(output_ptr);
                            // free(list_ptr);
                            // if constexpr (verbose_tasks)
                            //     std::cout << "Deserialized bitmap with " << leaf_data[tid][group_id][bitmap_position].cardinality() 
                            //         << " values! (TID " << tid << ", group_id " << group_id << ")" << std::endl;
                        });
                        leaf_task.data(tracker.register_task(TaskTypes::Leaf));
                    }
                    if (do_io) {
                        leaf_task.succeed(stopper_task); // this prevents immediate execution, may need to wait for I/O first
                    } // else: we do not need to wait for I/O, as we have all data in memory already
                    
                    if (team_workload_infos[tid].group_count < team_workload_infos[tid].list_cnt)
                        leaf_task.precede(leaf_union_prepare_tasks[tid][group_id]);
                    else {
                        // in this case, there are no leaf unions and the union tasks actually does not do any work
                        leaf_task.precede(leaf_union_tasks[tid][group_id]);
                    }
                    assert(leaf_id < leaf_tasks[tid].size());
                    // std::cout << "Created Leaf: " << leaf_id << " tid(" << tid << ")" << std::endl;
                    leaf_tasks[tid][leaf_id] = leaf_task;
                    leaf_id++;
                }
                
                if (do_io) {
                    // store a range of the tasks to be triggered later
                    to_be_scheduled[request_info.rid] = {task_range_start, leaf_id-task_range_start};
                    // std::cout << "Request " << request_info.rid << " will schedule leaf range [" << task_range_start << ", " << leaf_id << ") TID: " << tid << std::endl;
                    // create corresponding request:
                    assert(request_group < scfg.queue_pair_count);
                    TeamIndex::Storage::IORequest request{request_info.rid, io_buffer_ptr, [] () {}, request_info.start_block, request_info.total_block_cnt, tid};
                    storage.register_request(request_group, std::move(request));
                }
            }
        }
        
        for (int tid = 0; tid < team_cnt; tid++) {
            assert(leaf_ids[tid] == team_workload_infos[tid].list_cnt);
        }

        for (int tid = 0; tid < team_cnt; tid++) {
            assert(team_workload_infos[tid].group_count == leaf_data[tid].size());
            auto cnt = 0u;
            for (auto group_id = 0u; group_id < team_workload_infos[tid].group_count; group_id++)
                cnt += leaf_data[tid][group_id].size();
            assert(cnt == team_workload_infos[tid].list_cnt);
        }
        ////////////////////// create ISE tasks

        // these partial products of cardinalities help us determine the which values take part in a specific ISE (i.e., tuple in the cross product)  
        std::vector<unsigned> product_suffix(team_cnt + 1, 1u);
        // we rely on the sorting order of "expanded" first, "distributed" last in the team_workload_info vector
        for (int tid = team_cnt - 1; tid >= 0; --tid) {
            const auto& tinfo = team_workload_infos[tid];
            if (team_workload_infos[tid].expand) {
                product_suffix[tid] = product_suffix[tid + 1] * tinfo.group_count;
            }
        }

        // one outer iteration per task cluster, which is a "tuple in the cross product of all expanded Teams"
        assert(pcfg.ise_count == (pcfg.outer_intersection_term_count * pcfg.outer_union_term_count));
        for (auto ise_id = 0u; ise_id < pcfg.ise_count; ise_id++) {
            
            // determine at which position this ISE places it's results
            // Note: We do not place it at position ise_id, because we would like to store results for later aggregation into a contiguous region
            // For the outer INTERSECTION, this requires storing results together that belong to the same combination of expanded, excluded values
            unsigned result_position = (ise_id % pcfg.outer_intersection_term_count) * pcfg.outer_union_term_count + ise_id / pcfg.outer_intersection_term_count;

            assert(result_position < pcfg.ise_count);

            // first, we add a task that marks the end of an ISE
            tf::Task ise_done_task = taskflow.placeholder();
            ise_done_task.name("ISE"+std::to_string(ise_id)+".DONE");
            ise_done_task.work([ise_id, &ise_results]() {
                if constexpr (verbose_tasks)
                    std::cout << "ISE " << ise_id << " done! (" << ise_results[ise_id].cardinality() << " ids)"  << std::endl;
            });

            ise_done_tasks[result_position] = ise_done_task;

            // this task is used to form dependencies for tasks within the ISE
            // tf::Task last_task;
            // last_task = expanded_cluster_task;

            for (TeamID tid = PRIMARY_EXPANSION; tid <  team_cnt; tid++) {
                const auto& tinfo = team_workload_infos[tid];
                // in this loop, we will emit one (actual work-) task per team, but:
                // expanded team: each leaf takes part in only a subset of all clusters
                // shared team: all leafs take part in every clusters

                if (tinfo.expand) {                    
                    // calculate which element takes part in the current tuple of the cross product
                    auto operand_position = (ise_id / product_suffix[tid+1]) % tinfo.group_count; // the "i-value in A_i" (with A_i being potentially a union)

                    tf::Task expanded_ise_task = taskflow.placeholder(); // either intersection or subtraction (and either expanded or shared)
                    
                    // Obtain a pointer to the operand (either leaf or leaf-union result)
                    // Note: leaf union results are stored in the "first leaf"
                    // We always access the first entry of a group. In case of no leaf union, this is also the only entry
                                        
                    // the task that makes the right-hand-side operand available
                    tf::Task rhs_task = leaf_union_tasks[tid][operand_position];
                    // we pass this pointer to the tasks
                    
                    roaring::Roaring* rhs = nullptr;
                    // if (group_sizes[tid][operand_position] > 1) {
                    //     // we have a union of operands
                        rhs = &(leaf_union_results[tid][operand_position]);
                    // }
                    // else {
                    //     // we have a single operand, take data directly from the leaf
                    //     rhs = &(leaf_data[tid][operand_position][0]);
                    // }

                    expanded_ise_task.succeed(rhs_task); // SECOND (rhs) OPERAND task
                                        
                    // primary expansion task can start right away and initializes all followup tasks.
                    if (PRIMARY_EXPANSION < tid) {
                        assert(team_workload_infos[tid-1].expand); // make sure the Teams are sorted properly
                        
                        // first Team's task only initializes the ISE, rest leads to actual merges. Rest is executed in sequence
                        expanded_ise_task.succeed(ise_tasks[tid-1][result_position]); // FIRST (lhs) OPERAND task
                    }
                    
                    if (tinfo.group_count == tinfo.list_cnt)
                        assert(leaf_data[tid][operand_position].size() == 1);

                    // add actual work tasks:
                    if (tid == PRIMARY_EXPANSION) {
                        assert(tinfo.is_included);
                        expanded_ise_task.name("EXPANDED.INIT."+tinfo.team_name+".ISE"+std::to_string(ise_id)+".Op"+std::to_string(operand_position));
                        expanded_ise_task.work([&ise_results, ise_id, result_position, rhs] ()
                            {
                                assert(rhs);
                                assert(rhs->cardinality() > 0); // we initialize an ISE always with leaf data, otherwise we wouldn't bother...
                                ise_results[result_position] = *rhs; // need copy, source will used multiple times
                                if constexpr (verbose_tasks) {
                                    if (ise_id % id_mod == 0)
                                        std::cout << ise_id << ": Bitmap initialized to "
                                            << ise_results[result_position].cardinality() << " values! (EI)" << std::endl;
                                }
                            });
                        expanded_ise_task.data(tracker.register_task(TaskTypes::ExpandedInit));
                    }
                    else if (tinfo.is_included) {
                        // expanded inclusive -> will lead to UNIONS across either all or a subset of clusters
                        expanded_ise_task.name("EXPANDED.INTERSECT."+tinfo.team_name+".ISE"+std::to_string(ise_id)+".Op"+std::to_string(operand_position));
                        expanded_ise_task.work([&ise_results, ise_id, result_position, rhs,   tid, &ise_tasks, &tracker] ()
                            {
                                auto before = ise_results[result_position].cardinality();
                                void* data = ise_tasks[tid][result_position].data();
                                if (data != nullptr) {
                                    auto& stats = tracker.get_stats(reinterpret_cast<TaskTracker::StatPosition>(data));
                                    stats.input_volume = before + rhs->cardinality(); // combination of both arguments!
                                }
                                assert(rhs);
                                // intersect the leaf bitmap with the partial result bitmap
                                ise_results[result_position] &= *rhs;
                                if constexpr (verbose_tasks) {
                                    if (ise_id % id_mod == 0)
                                        std::cout << ise_id << ": Bitmap reduced from " << before 
                                            << " to " << ise_results[result_position].cardinality() << " values! (EI)" << std::endl;
                                }
                                if (data != nullptr) {
                                    auto& stats = tracker.get_stats(reinterpret_cast<TaskTracker::StatPosition>(data));
                                    stats.output_volume = ise_results[result_position].cardinality();
                                }
                            });
                        expanded_ise_task.data(tracker.register_task(TaskTypes::ExpandedIntersection));
                    }
                    else {
                        // expanded NEGATIVE -> will lead to big INTERSECTION across the big union of clusters
                        expanded_ise_task.name("EXPANDED.SUBTRACT."+tinfo.team_name+".ISE"+std::to_string(ise_id)+".Op"+std::to_string(operand_position));
                        expanded_ise_task.work([&ise_results, ise_id, result_position, rhs] ()
                        {   
                            auto before = ise_results[result_position].cardinality();
                            // subtract the leaf bitmap from the partial result bitmap                                    
                            ise_results[result_position] -= *rhs; // was already initialized
                            if constexpr (verbose_tasks) {
                                if (ise_id % id_mod == 0)
                                    std::cout << ise_id << ": Bitmap reduced from " << before 
                                        << " to " << ise_results[result_position].cardinality() << " values! (EE)" << std::endl;
                            }
                        });
                        expanded_ise_task.data(tracker.register_task(TaskTypes::ExpandedSubtraction));
                    }
                    ise_tasks[tid][result_position] = expanded_ise_task;
                }
                else {
                    // DISTRIBUTED Team, so we add one task for every operand of this Team to each ISE
                    assert(tinfo.group_count == leaf_union_results[tid].size() and tinfo.group_count == leaf_union_tasks[tid].size());
                    
                    if (tinfo.is_included) {
                        // DISTRIBUTED INCLUDED
                        // need to do a union over intersections, which may be run in parallel:
                        if (tinfo.group_count >= pcfg.distributed_intersection_parallel_threshold) {
                            // union over all groups/operands of this Team. vector already fully filled, so no std::ref(iter) necessary
                            tf::Task distributed_ise_task = taskflow.reduce(leaf_union_results[tid].begin(),
                                                                            leaf_union_results[tid].end(),
                                                                            distributed_intersection_results[result_position],
                                    [&ise_results, &distributed_intersection_results, tid, ise_id, result_position] (roaring::Roaring a, const roaring::Roaring& b) -> roaring::Roaring
                                {
                                    if constexpr (verbose_tasks)
                                        std::cout << "DIST.INT union (DISTRIBUTED) " << tid << "/"<< ise_id << ": "
                                            << a.cardinality() 
                                            << " += (" << ise_results[result_position].cardinality()
                                            << " & " << b.cardinality() << ")" << std::endl;
                                    return a | (ise_results[result_position] & b);
                                });
                            distributed_ise_task.name("DISTRIBUTED.INTERSECT."+tinfo.team_name+".ISE"+std::to_string(ise_id)+".PARALLEL");
                            
                            distributed_ise_task.data(tracker.register_task(TaskTypes::DistributedIntersection));
                            
                            // preceeding team tasks (LHS) are executed first
                            distributed_ise_task.succeed(ise_tasks[tid-1][result_position]);
                            
                            // RHS dependency to all operands of this DISTRIBUTED INCLUDED Team task
                            for (auto leaf_union_task : leaf_union_tasks[tid]) {
                                distributed_ise_task.succeed(leaf_union_task);
                            }

                            // additional cleanup task, since union result of this ISE operand is not yet stored in ise_results
                            tf::Task DI_parallel_union_done = taskflow.emplace([&distributed_intersection_results, &ise_results, result_position] () {
                                // pass intermediate result (which is a union over intersections) to the next Team's task
                                ise_results[result_position] = distributed_intersection_results[result_position];
                                distributed_intersection_results[result_position] = Bitmap(); // reset to empty for next DISTRIBUTED INCLUDED
                            });

                            DI_parallel_union_done.name("DISTRIBUTED.INTERSECT."+tinfo.team_name+".ISE"+std::to_string(ise_id)+".DONE");
                            DI_parallel_union_done.succeed(distributed_ise_task);

                            
                            ise_tasks[tid][result_position] = DI_parallel_union_done;
                        }
                        else {
                            tf::Task distributed_ise_task = taskflow.emplace([&distributed_intersection_results,
                                    &ise_results, &leaf_union_results, tid, ise_id, result_position, &ise_tasks, &tracker]() {
                                assert(distributed_intersection_results[result_position].cardinality() == 0);

                                void* data = ise_tasks[tid][result_position].data();
                                if (data != nullptr) {
                                    auto& stats = tracker.get_stats(reinterpret_cast<TaskTracker::StatPosition>(data));
                                    stats.input_volume = ise_results[result_position].cardinality(); // combination of both arguments!
                                }
                                
                                for (auto& rhs : leaf_union_results[tid]) {
                                    if constexpr (verbose_tasks)
                                        std::cout << "DIST.INT union " << tid << "/"<< result_position << ": "
                                            << distributed_intersection_results[result_position].cardinality() 
                                            << " += (" << ise_results[result_position].cardinality()
                                            << " & " << rhs.cardinality() << ")" << std::endl;
                                        distributed_intersection_results[result_position] |= (ise_results[result_position] & rhs);
                                        if (data != nullptr) {
                                            auto& stats = tracker.get_stats(reinterpret_cast<TaskTracker::StatPosition>(data));
                                            auto new_vol = stats.input_volume.value();
                                            new_vol += rhs.cardinality(); // sum of all arguments!
                                            stats.input_volume = new_vol;
                                        }
                                }

                                ise_results[result_position] = distributed_intersection_results[result_position];
                                distributed_intersection_results[result_position] = Bitmap(); // reset to empty for any next DISTRIBUTED INCLUDED Team
                                if (data != nullptr) {
                                    auto& stats = tracker.get_stats(reinterpret_cast<TaskTracker::StatPosition>(data));
                                    stats.output_volume = ise_results[result_position].cardinality();
                                }
                            });
                            distributed_ise_task.name("DISTRIBUTED.INTERSECT."+tinfo.team_name+".ISE"+std::to_string(ise_id)+".SEQUENTIAL");
                            
                            distributed_ise_task.data(tracker.register_task(TaskTypes::DistributedIntersection));
                            
                            // preceeding team tasks are executed first
                            distributed_ise_task.succeed(ise_tasks[tid-1][result_position]);
                            // incoming rhs dependency to all operands of this DISTRIBUTED INCLUDED Team task
                            for (auto leaf_union_task : leaf_union_tasks[tid]) {
                                distributed_ise_task.succeed(leaf_union_task);
                            }
                            
                            ise_tasks[tid][result_position] = distributed_ise_task;
                        }

                    }
                    else {
                        // DISTRIBUTED EXLCLUDED
                        // we queue one task per operand for every single ISE!

                        // temporarily add task from preceeding Team as first dependency
                        // current task will proceede the next task for this Team, so we will overwrite this in the loop
                        // note that we do not keep all tasks in the array, only the last one
                        ise_tasks[tid][result_position] = ise_tasks[tid-1][result_position];

                        auto operand_id = 0u;
                        for (auto leaf_union_task : leaf_union_tasks[tid]) {

                            // input data to be used by the spawned task
                            roaring::Roaring* rhs = &(leaf_union_results[tid][operand_id]);

                            // DISTRIBUTED EXCLUDED -> can be subtracted right here with no further (global) aggregation
                            tf::Task distributed_ise_task = taskflow.emplace([&ise_results, ise_id, operand_id, result_position, rhs] ()
                                {
                                    assert(rhs);
                                    auto before = ise_results[result_position].cardinality();
                                    // subtract the leaf bitmap from the partial result bitmap
                                    // Note that we directly alter the expansion result and do not write to the "cluster_result"                                    
                                    ise_results[result_position] -= *rhs; // was already initialized
                                    if constexpr (verbose_tasks) {
                                        if (ise_id % id_mod == 0) 
                                            std::cout << ise_id << "/" << operand_id << ": Bitmap reduced from " << before 
                                                << " to " << ise_results[result_position].cardinality() << " values! (DE)" << std::endl;
                                    }
                                });
                            distributed_ise_task.name("DISTRIBUTED.SUBTRACT."+tinfo.team_name+".ISE"+std::to_string(ise_id)+".Op"+std::to_string(operand_id));
                            
                            distributed_ise_task.data(tracker.register_task(TaskTypes::DistributedSubtraction));
                            
                            // we depend on the partial result of the expansion term to be complete, before we can start
                            distributed_ise_task.succeed(ise_tasks[tid][result_position]); // FIRST (lhs) OPERAND

                            // add dependencies and store task:
                            distributed_ise_task.succeed(leaf_union_task); // SECOND (rhs) OPERAND
                            
                            // the last task is what the next Team will refer to!
                            ise_tasks[tid][result_position] = distributed_ise_task; 
                            operand_id++;
                        }
                    }

                    // all Team tasks always end in the ISE_DONE task, which makes it easier to track dependencies with variying types of Teams
                }
                ise_tasks[tid][result_position].precede(ise_done_tasks[result_position]);
            }
        }
    
        
        ///////////////// final aggregations
        if (pcfg.ise_count > 0) {
            assert(big_union_results.size() == pcfg.outer_intersection_term_count);
            auto union_base_group_size = pcfg.outer_union_term_count / pcfg.outer_union_group_count; // how many terms to include into each group
            auto union_remainder = pcfg.outer_union_term_count % pcfg.outer_union_group_count; // how many groups get one extra

            /// ISEs results are unified in a 2-stage process:
            /// 1. The inner union of all ISEs that share the same outer union
            /// 2. The outer union of all inner unions that share the same outer intersection
            for (auto union_id = 0u, ise_start = 0u; union_id < pcfg.outer_intersection_term_count; union_id++) {
                // first the outer most union. Note that there are more than one  2-stage union, if there is an outer
                // intersection, i.e., an expanded excluded Team
                tf::Task outer_big_union_task;

                if (big_union_group_results[union_id].size() == 1) {
                    // no need to union, just swap the result
                    outer_big_union_task = taskflow.emplace([union_id, &big_union_group_results, &big_union_results] ()
                        {
                            big_union_results[union_id].swap(big_union_group_results[union_id][0]);
                        });
                    outer_big_union_task.name("BIG_OUTER_UNION."+std::to_string(union_id)+".PASS");
                    outer_big_union_task.data(tracker.register_task(TaskTypes::BigOuterUnion));
                }
                else if (big_union_group_results.size() < pcfg.leaf_union_list_parallel_threshold) {
                    outer_big_union_task = taskflow.emplace([&big_union_group_results, &big_union_results, union_id, &max_bitmaps] ()
                        {
                            auto begin = big_union_group_results[union_id].begin();
                            auto end = big_union_group_results[union_id].end();
                            
                            // assert(begin->cardinality() > 0);
                            assert(std::distance(begin, end) > 1);
                            
                            // roaring reaaaallly wants a Bitmap**, so we have to convert the vector of vectors :(
                            std::span<Bitmap> span{begin, static_cast<std::size_t>(std::distance(begin, end))};
                            

                            assert(span.size() <= max_bitmaps);
                            // if (span.size() > max_bitmaps)
                            //     throw std::runtime_error("pcfg.leaf_union_list_parallel_threshold parameter surpasses maximum stack-array size!");
                            
                            const Bitmap* bitmap_array[max_bitmaps];
                            for (size_t i = 0; i < span.size(); ++i) {
                                bitmap_array[i] = &span[i];
                            }

                            // actual union operation
                            big_union_results[union_id] = roaring::Roaring::fastunion(span.size(), bitmap_array);
                            
                            if constexpr (verbose_tasks)
                                std::cout << "Big outer union " << union_id << " over " << span.size() 
                                    << " bitmaps: " << big_union_results[union_id].cardinality() << std::endl;
                        });
                    outer_big_union_task.name("BIG_OUTER_UNION."+std::to_string(union_id)+".SEQUENTIAL");
                    outer_big_union_task.data(tracker.register_task(TaskTypes::BigOuterUnion));
                }
                else {

                    outer_big_union_task = taskflow.reduce(big_union_group_results[union_id].begin(),
                                                                big_union_group_results[union_id].end(),
                                                                big_union_results[union_id],
                                                                [union_id]
                            (roaring::Roaring& a, const roaring::Roaring& b) -> roaring::Roaring
                        {                       
                            if constexpr (verbose_tasks)
                                std::cout << "U"<< union_id << ": "
                                    << a.cardinality() 
                                    << " += " << b.cardinality() << std::endl;
                            return a | b;
                    });
                    outer_big_union_task.name("BIG_OUTER_UNION."+std::to_string(union_id)+".PARALLEL");
                    outer_big_union_task.data(tracker.register_task(TaskTypes::BigOuterUnion));
                }
                outer_big_union_tasks[union_id] = outer_big_union_task;


                auto ise_end = ise_start+pcfg.outer_union_term_count; // each outer union has covers term_count many ISEs
                assert(ise_end <= ise_results.size());
                
                for (auto union_group_id = 0u, sub_start = ise_start; union_group_id < pcfg.outer_union_group_count; union_group_id++) {
                    auto group_size = (union_group_id < union_remainder) ? (union_base_group_size + 1) : union_base_group_size;
                    // Calculate the start index for the group.
                    auto sub_end = sub_start + group_size;
                    assert(sub_end<=ise_end);

                    // define big group union task:
                    tf::Task inner_big_union_task;
                    if (group_size == 1) {
                        // no need to union, just copy the result
                        inner_big_union_task = taskflow.emplace([union_id, union_group_id, sub_start, &ise_results, &big_union_group_results] ()
                            {
                                big_union_group_results[union_id][union_group_id].swap(ise_results[sub_start]);
                            });
                        inner_big_union_task.name("BIG_INNER_UNION.Union"+std::to_string(union_id)+".Gr"+std::to_string(union_group_id)+".PASS");
                        inner_big_union_task.data(tracker.register_task(TaskTypes::BigInnerUnion));
                    }
                    else if (group_size < pcfg.leaf_union_list_parallel_threshold) {
                        inner_big_union_task = taskflow.emplace([&big_union_group_results,
                                &ecfg,
                                &ise_results,
                                sub_start,
                                sub_end,
                                union_id,
                                union_group_id,
                                &max_bitmaps] ()
                            {
                                auto begin = ise_results.begin()+sub_start;
                                auto end = ise_results.begin()+sub_end;
                                
                                auto dist = static_cast<std::size_t>(std::distance(begin, end));
                                assert(dist > 1);
                                std::span<Bitmap> span{begin, dist};
                                
                                if constexpr (verbose_tasks)
                                    std::cout << "Big inner union " << union_id << " over " << span.size() 
                                            << " bitmaps: " << std::flush;

                                // assert(begin->cardinality() > 0);
                                
                                // roaring reaaaallly wants a Bitmap**, so we have to convert the vector of vectors :(

                                if (span.size() > max_bitmaps)
                                    throw std::runtime_error("pcfg.leaf_union_list_parallel_threshold parameter surpasses maximum stack-array size!");
                                
                                const Bitmap* bitmap_array[max_bitmaps];
                                for (size_t i = 0; i < span.size(); ++i) {
                                    bitmap_array[i] = &span[i];
                                }

                                // actual union operation
                                big_union_group_results[union_id][union_group_id] = roaring::Roaring::fastunion(span.size(), bitmap_array);
                                
                                if constexpr (verbose_tasks)
                                    std::cout << big_union_group_results[union_id][union_group_id].cardinality() << std::endl;
                            });
                        inner_big_union_task.name("BIG_INNER_UNION.Union"+std::to_string(union_id)+".Gr"+std::to_string(union_group_id)+".SEQUENTIAL");
                        inner_big_union_task.data(tracker.register_task(TaskTypes::BigInnerUnion));
                    }
                    else {
                        inner_big_union_task = taskflow.reduce(ise_results.begin()+sub_start,
                                                                ise_results.begin()+sub_end,
                                                                big_union_group_results[union_id][union_group_id],
                                                                [union_id,union_group_id]
                                (roaring::Roaring& a, const roaring::Roaring& b) -> roaring::Roaring
                            {                       
                                if constexpr (verbose_tasks)
                                    std::cout << "U" << union_id << "/Gr"<< union_group_id << ": "
                                        << a.cardinality() 
                                        << " += " << b.cardinality() << std::endl;
                                
                                return a | b;
                            });
                        inner_big_union_task.name("BIG_INNER_UNION.Union"+std::to_string(union_id)+".Gr"+std::to_string(union_group_id)+".PARALLEL");
                        inner_big_union_task.data(tracker.register_task(TaskTypes::BigInnerUnion));
                    }
                    
                    for (auto i = sub_start; i < sub_end; i++) {
                        inner_big_union_task.succeed(ise_done_tasks.at(i)); 
                    }
                    
                    // Update sub_start for the next group.
                    sub_start = sub_end;
                    inner_big_union_task.precede(outer_big_union_task);
                    inner_big_union_tasks[union_id][union_group_id] = inner_big_union_task;
                }
                ise_start = ise_end; 
            }
        }
        else {
            /// TEAM UNION SITUATION, i.e., just one union per team and a final intersection between Teams
            // We do not do any expansion/have no ISEs in this case

            // we still need to union, but this time we aggregate Team-wise, not over ISE results
            assert(pcfg.outer_union_term_count == 0);
            assert(big_union_results.size() == team_workload_infos.size());


            for (auto tinfo: team_workload_infos) {
                auto tid = team_ids[tinfo.team_name];
                tf::Task team_union_task;
                if (leaf_union_results[tid].size() == 1) {
                    // no need to union, just copy the result
                    team_union_task = taskflow.emplace([tid, &leaf_union_results, &leaf_data, &big_union_results, &team_workload_infos] ()
                        {
                            assert(team_workload_infos[tid].total_cardinality == leaf_union_results[tid][0].cardinality());
                            // big_union_results[tid] = leaf_union_results[tid][0];
                            big_union_results[tid].swap(leaf_union_results[tid][0]);
                            
                        });
                    team_union_task.name("TEAM.UNION."+tinfo.team_name+".PASS");
                    team_union_task.data(tracker.register_task(TaskTypes::TeamUnion));
                }
                else if (leaf_union_results[tid].size() < pcfg.leaf_union_list_parallel_threshold) {
                // else {
                    team_union_task = taskflow.emplace([tid,
                            &leaf_union_results,
                            &big_union_results,
                            &team_workload_infos,
                            &max_bitmaps,
                            &ecfg] ()
                        {
                            
                            auto begin = leaf_union_results[tid].begin();
                            auto end = leaf_union_results[tid].end();
                            
                            std::size_t dist = std::distance(begin, end);
                            // assert(begin->cardinality() > 0);
                            assert(dist > 1);
                            assert(dist <= max_bitmaps);
                            
                            // roaring reaaaallly wants a Bitmap**, so we have to convert the vector of vectors :(
                            std::span<Bitmap> span{begin, dist};

                            const Bitmap* bitmap_array[max_bitmaps];
                            for (size_t i = 0; i < span.size(); ++i) {
                                bitmap_array[i] = &span[i];
                                
                                // we can expect that we ALWAYS have something to intersect, otherwise we would not consider the Team
                                assert(bitmap_array[i]->cardinality() > 0);
                            }

                            // actual union operation
                            big_union_results[tid] = roaring::Roaring::fastunion(span.size(), bitmap_array);
                            
                            assert(team_workload_infos[tid].total_cardinality == big_union_results[tid].cardinality());
                            if (ecfg.verbose)
                                std::cout << "Sequential Team union " << tid << " over " << span.size() 
                                    << " bitmaps: " << big_union_results[tid].cardinality() << std::endl;

                        });
                    team_union_task.name("TEAM.UNION."+tinfo.team_name+".SEQUENTIAL");
                    team_union_task.data(tracker.register_task(TaskTypes::TeamUnion));
                }
                else {
                    team_union_task = taskflow.reduce(leaf_union_results[tid].begin(),
                                                    leaf_union_results[tid].end(),
                                                    big_union_results[tid],
                                                    [&tinfo]
                            (roaring::Roaring& a, const roaring::Roaring& b) -> roaring::Roaring
                        {                       
                            if constexpr (verbose_tasks)
                                std::cout << "Team Union " << tinfo.team_name << ": "
                                    << a.cardinality() 
                                    << " += " << b.cardinality() << std::endl;
                            
                            return a | b;
                        });
                    team_union_task.name("TEAM.UNION."+tinfo.team_name+".PARALLEL");
                    team_union_task.data(tracker.register_task(TaskTypes::TeamUnion));
                }

                for (auto leaf_union_group_id = 0u; leaf_union_group_id < tinfo.group_count; leaf_union_group_id++) {
                    leaf_union_tasks[tid][leaf_union_group_id].precede(team_union_task);
                }

                outer_big_union_tasks[tid] = team_union_task;
            }
        }


        ///////// BIG OUTER INTERSECTION TASK (there is ever only 1 or none)
        // tf::Task big_intersection_task = taskflow.reduce(big_intersection_group_results.begin(),
        //                                         big_intersection_group_results.end(),
        //                                         final_intersection,
        //                                         [] (roaring::Roaring& a, const roaring::Roaring& b) -> roaring::Roaring
        //     {
        //         if constexpr (verbose_tasks)
        //             std::cout << a.cardinality() << " |= " << b.cardinality() << std::endl;
        //         // a &= b;
        //         // big intersection, all unions are done. Outermost operation
        //         return a & b; // should become a std::move
        //         // return a;
        //     });
        // we only use a sequential version, as grouping will reduce the number of terms to a manageable size for a single thread
        // For just one term, we simply pass the result
        tf::Task big_intersection_task = taskflow.emplace([&big_intersection_group_results, &final_intersection, &pcfg, &ecfg] ()
            {
                assert(not big_intersection_group_results.empty());

                if constexpr (verbose_tasks)
                    std::cout << "Big outer intersection with " << pcfg.outer_intersection_group_count << " terms" << std::endl;
                
                final_intersection = big_intersection_group_results[0];
                // final_intersection.swap(big_intersection_group_results[0]);

                if constexpr (verbose_tasks)
                    std::cout << "\tInit: " <<  final_intersection.cardinality() << std::endl;

                for (auto big_res_id = 1; big_res_id < pcfg.outer_intersection_group_count; big_res_id++) {
                    if constexpr (verbose_tasks)
                        std::cout << "\t" <<  final_intersection.cardinality() << " &= " << big_intersection_group_results[big_res_id].cardinality() << std::endl;

                        final_intersection &= big_intersection_group_results[big_res_id];

                }
            });
        big_intersection_task.name("BIG_INTERSECT.OUTER");
        big_intersection_task.data(tracker.register_task(TaskTypes::BigOuterIntersection));

        /////// BIG INNER INTERSECTIONS (if we decide to group, which makes only sense for many big unions!)
        auto inters_base_group_size = pcfg.outer_intersection_term_count / pcfg.outer_intersection_group_count; // how many terms to include into each group
        auto inters_remainder = pcfg.outer_intersection_term_count % pcfg.outer_intersection_group_count; // how many groups get one extra
        

        for (auto big_intersection_group_id = 0u, obu_start = 0u; big_intersection_group_id < pcfg.outer_intersection_group_count; big_intersection_group_id++) {
            auto group_size = (big_intersection_group_id < inters_remainder) ? (inters_base_group_size + 1) : inters_base_group_size;
            
            auto obu_end = obu_start + group_size;
            assert(obu_end <= big_union_results.size());

            // big_intersection_group_results[big_intersection_group_id].flip(0,std::numeric_limits<IDType>::max());
            tf::Task inner_big_intersection_task;
            // Inner intersection of "grouped"/2-stage intersection. Does nothing ("PASS") if there is no grouping, i.e., only 1 term!
            // if (group_size == 1 or (group_size < pcfg.leaf_union_list_parallel_threshold)) {
            inner_big_intersection_task = taskflow.emplace(
                    [&big_union_results, obu_start, obu_end, &big_intersection_group_results, big_intersection_group_id, &ecfg] ()
                {
                    if constexpr (verbose_tasks)
                        std::cout << "Big inner intersection with " << obu_end-obu_start << " terms" << std::endl;
                    assert(obu_end-obu_start > 0);

                    // big_intersection_group_results[big_intersection_group_id] = big_union_results[obu_start];
                    big_intersection_group_results[big_intersection_group_id].swap(big_union_results[obu_start]);
                    if constexpr (verbose_tasks)
                        std::cout << "\tInit: " <<  big_intersection_group_results[big_intersection_group_id].cardinality() << std::endl;

                    for (auto group_id = obu_start+1; group_id < obu_end; group_id++) {
                        if constexpr (verbose_tasks)
                            std::cout << "\t" <<  big_intersection_group_results[big_intersection_group_id].cardinality() << " &= " << big_union_results[group_id].cardinality() << std::endl;
                        big_intersection_group_results[big_intersection_group_id] &= big_union_results[group_id];
                    }
                });
            if (group_size == 1) {
                inner_big_intersection_task.name("BIG_INTERSECTION.INNER."+std::to_string(big_intersection_group_id)+".PASS");
                inner_big_intersection_task.data(tracker.register_task(TaskTypes::BigInnerIntersection));
            }
            else {
                inner_big_intersection_task.name("BIG_INTERSECTION.INNER."+std::to_string(big_intersection_group_id)+".SEQUENTIAL");
                inner_big_intersection_task.data(tracker.register_task(TaskTypes::BigInnerIntersection));
            }

            big_intersection_task.succeed(inner_big_intersection_task);
            
            for (auto big_union_id = obu_start; big_union_id < obu_end; big_union_id++) {
                outer_big_union_tasks[big_union_id].precede(inner_big_intersection_task);
            }

            obu_start = obu_end;
        }
        
        tf::Task subtractions_task;

        if (pcfg.ise_count == 0 and team_subtraction_count > 0) {
            // add a task for subtraction, in case we not only have intersections
            // Will be executed after Team-wise intersections 
            assert(big_intersection_group_results.size() == (team_subtraction_count+pcfg.outer_intersection_term_count));
            subtractions_task = taskflow.emplace([&big_union_results, &final_intersection, team_subtraction_count] ()
            {
                if constexpr (verbose_tasks) {
                    std::cout << "Subtracting " << team_subtraction_count << " team unions" << std::endl;
                }
                auto start_id = big_union_results.size()-team_subtraction_count;
                assert(start_id > 0);

                for (auto team_id = start_id; team_id < big_union_results.size(); team_id++) {
                    if constexpr (verbose_tasks)
                        std::cout << "\t" <<  final_intersection.cardinality() << " -= " << big_union_results[team_id].cardinality() << " " << team_id << std::endl;

                        final_intersection -= big_union_results[team_id];

                }
            });
            subtractions_task.name("TEAM.SUBTRACTION");

            subtractions_task.data(tracker.register_task(TaskTypes::TeamSubtraction));
            // lhs argument:
            subtractions_task.succeed(big_intersection_task);
            
            // rhs arguments:
            auto start_id = team_workload_infos.size()-team_subtraction_count;
            for (auto tid = start_id; tid < team_workload_infos.size(); tid++) {
                outer_big_union_tasks[tid].precede(subtractions_task);
            }
        }


        // this task materializes a Roaring Bitmap holding the final result into a continous array
        tf::Task finalize_task = taskflow.placeholder();
        finalize_task.work([&final_intersection,
                &ecfg,
                result_ptr_ref = std::reference_wrapper(result),
                result_card_ref = std::reference_wrapper(result_cardinality)] () 
            {
                

                if constexpr (verbose_tasks) {
                    if (ecfg.return_result) {
                        std::cout << "Materializing " <<  final_intersection.cardinality() << " IDs..." << std::endl;
                    }
                }
                
                // materialize final result bitmap into an integers array to be returned
                result_card_ref.get() = final_intersection.cardinality();
                
                
                
                if (result_card_ref.get() == 0 or not ecfg.return_result) {
                    result_ptr_ref.get() = nullptr;
                }
                else {
                    result_ptr_ref.get() = get_new_decompression_buffer(result_card_ref.get());
                    final_intersection.toUint32Array(result_ptr_ref.get());
                }
            });

        finalize_task.name("Materialize");
        
        finalize_task.data(tracker.register_task(TaskTypes::Materialize));
        
        finalize_task.succeed(io_end_task);

        if (pcfg.ise_count == 0 and team_subtraction_count > 0) {
            finalize_task.succeed(subtractions_task);
        }
        else {
            finalize_task.succeed(big_intersection_task);
        }

        plan_timer.stop();
        
        ///////////////////////////////////////////////////////////

        auto start_time_str = std::chrono::system_clock::to_time_t(start_time);
        // construct a string looks like this: "2025_04_14-11_41_00"
        std::string start_time_str_formatted = getCurrentDateTime();

        // task-graph created, dump as .dot:
        if (ecfg.print_execution_plan.has_value()) {
            // expand the provided path with the timestamp at the end:
            std::string path = ecfg.print_execution_plan.value();
            if (path.find(".dot") == std::string::npos) {
                path += ".dot";
            }
            // replace the file extension with the timestamp
            path.replace(path.find(".dot"), 4, "-" + start_time_str_formatted + ".dot");

            std::ofstream fout(path);
            taskflow.dump(fout);
            if (ecfg.verbose)
                std::cout << "Execution plan written to " << path << std::endl;
        }
        if (ecfg.verbose)
            std::cout << "Plan created! Task count: " << taskflow.num_tasks() << std::endl;
        
        
        /////////////////////// Run graph

        // need to define executor early because we want to interact with it during runtime (to schedule leaf tasks after async I/O)
        auto exec_ptr = std::make_shared<tf::Executor>(ecfg.worker_count, tf::make_worker_interface<PinnedWorker>());

        // TaskflowDebugger dbg;
        if constexpr (performance_tracking) {
            auto observer = exec_ptr->make_observer<PerformanceObserver>(performanceobserver);
        }

        // std::cout << "Executor pending tasks: " << exec_ptr->num_topologies() << "\n";

        // auto dummy = taskflow.emplace([](){ std::cout << "Dummy task\n"; });
        // dummy.name("DUMMY TASK");
        // auto final_join = taskflow.emplace([](){ std::cout << "FINAL JOIN!\n"; });
        // final_join.name("FINAL JOIN");
        // dummy.precede(final_join);

        // std::cout << "Final task created" << std::endl;

        // taskflow.for_each_task([&dbg](tf::Task task) {
        //     std::cout << "Task in graph: " << task.name() 
        //             << " - successors: " << task.num_successors() 
        //             << " - dependents: " << task.num_dependents() << "\n";
        // //     dbg.register_task(task);
        // });

        Timer execution_timer("Execution", total_read_volume, true);
        if (ecfg.verbose) {
            execution_timer.set_verbose_on_death();
        }

        ///////////////////////////////////////////////////////////////////////////////
        ///// EXECUTION
        auto execution_start_time = execution_timer.start();

        // Run the flow (which may not be altered anymore)
        auto f = exec_ptr->run(taskflow);

        // wait for a fixed number of seconds at most:
        auto t = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        auto status = f.wait_until(t);
        auto execution_stop_time = execution_timer.stop();
        ///////////////////////////////////////////////////////////////////////////////

        if (status == std::future_status::timeout) {
            std::cerr << "Execution timed out!" << std::endl;
        }
        else if (status == std::future_status::deferred) {
            std::cerr << "Execution deferred!" << std::endl;
        }
        if (ecfg.verbose) {
            std::cout << "Expected to read " << tota_read_card << " ids in total." << std::endl;
            std::cout << "Final result has " << result_cardinality << " ids." << std::endl;
        }

        std::optional<std::string> task_stats_json;
        if (ecfg.print_task_stats) {
            // expand the provided path with the timestamp at the end:
            std::string path = ecfg.print_task_stats.value();

            // check if the path ends with ".json"
            if (path.find(".json") == std::string::npos) {
                path += ".json";
            }

            // replace the file extension with the timestamp
            path.replace(path.find(".json"), 5, "-" + start_time_str_formatted + ".json");

            task_stats_json = tracker.dump_stats(execution_start_time, execution_stop_time, path);

            if (ecfg.verbose)
                std::cout << "Task stats written to " << task_stats_json.value() << std::endl;
        }

        // collect some information on the execution
        ExecutionStatistics stats;
        stats.input_cardinality = tota_read_card;
        stats.result_cardinality = result_cardinality;
        stats.executor_runtime = execution_timer.duration();
        stats.plan_construction_runtime = plan_timer.duration();
        stats.task_stats_path = task_stats_json;


        // dump result stats to a json, if desired
        if (ecfg.print_result_stats) {
            // expand the provided path with the timestamp at the end:
            std::string path = ecfg.print_result_stats.value();

            // check if the path ends with ".json"
            if (path.find(".json") == std::string::npos) {
                path += ".json";
            }

            // replace the file extension with the timestamp
            path.replace(path.find(".json"), 5, "-" + start_time_str_formatted + ".json");
            
            // we create a new file at <output.json>
            std::ofstream output_file(path);
            if (!output_file) {
                std::cerr << "Failed to open file for runtime stats: " << path << std::endl;
            }
            else {
                nlohmann::json output_json;
                output_json["table_cardinality"] = pcfg.table_cardinality;
                output_json["ise_count"] = pcfg.ise_count;
                output_json["input_cardinality"] = stats.input_cardinality;
                output_json["team_count"] = team_workload_infos.size();
                output_json["expanded_team_count"] = included_expanded_teams.size()+excluded_expanded_teams.size();
        
                if (stats.result_cardinality.has_value()) {
                    output_json["result_cardinality"] = stats.result_cardinality.value();
                } else {
                    output_json["result_cardinality"] = nlohmann::json::value_t::null;
                }
        
                if (stats.executor_runtime.has_value())
                    output_json["executor_runtime"] = stats.executor_runtime.value();
                else 
                    output_json["executor_runtime"] = nlohmann::json::value_t::null;
                
                if (stats.plan_construction_runtime.has_value())
                    output_json["plan_construction_runtime"] = stats.plan_construction_runtime.value();
                else 
                    output_json["plan_construction_runtime"] = nlohmann::json::value_t::null;
                
                if (stats.task_stats_path.has_value())
                    output_json["task_stats_path"] = stats.task_stats_path.value();
                else
                    output_json["task_stats_path"] = nlohmann::json::value_t::null;
                
                output_file << output_json.dump(4); // pretty print with 4 spaces
                output_file.close();
                std::cout << "Execution statistics written to: " << path << std::endl;
            }
        }


        /////////////////////// Cleanup 

        for (auto buff: io_buffers)
            free(buff);
        
        for (auto ptr: decompression_buffers) {
            free(ptr);
        }

        return {{result, result_cardinality}, stats};
    };


    /***
     * IO only, no computation.
     */
    std::tuple<std::span<IDType>, ExecutionStatistics>
    TeamIndexExecutor::run_read_all(std::vector<std::vector<RequestInfo>> &request_infos,
                                    std::vector<TeamMetaInfo> &team_workload_infos,
                                    const ExecutorConfig& ecfg,
                                    const Storage::StorageConfig& scfg)
    {
        assert(scfg.queue_pair_count == request_infos.size());
        
        std::unordered_map<TeamName, TeamID> team_ids;
        // counts number of ids loaded for non-smallest teams
        auto list_count = 0u;
        auto total_read_volume = 0u; 
        auto tota_request_cnt = 0u;
        auto tota_read_card = 0u;
        {
            TeamID id = 0;
            for (auto& team_info : team_workload_infos) {
                team_ids[team_info.team_name] = id++;
                list_count += team_info.list_cnt;
                total_read_volume += team_info.total_size_comp;
                tota_request_cnt += team_info.request_cnt;
                tota_read_card += team_info.total_cardinality;
            }
        }
        

        /////////////////////// Allocate Data Structures for Runtime
        // we store buffer pointers for later cleanup
        std::vector<BufferType*> io_buffers; 
        io_buffers.reserve(list_count);

        Storage::StorageAccessor storage(scfg);
        storage.register_Teams(team_workload_infos);
        

        /////////////////////// Create the Task Graph
        Timer plan_timer("PlanCreation");

        tf::Taskflow taskflow;

        // need to define executor early because we want to interact with it during runtime (to schedule leaf tasks)
        // tf::Executor executor();
        auto exec_ptr = std::make_shared<tf::Executor>(ecfg.worker_count, tf::make_worker_interface<PinnedWorker>());

        //// IO Tasks:
        std::vector<tf::Task> submission_tasks;
        std::vector<tf::Task> await_tasks;

        std::vector<std::vector<Storage::IORequest*>> arrival_vectors; // temporarily holds pointers to finished requests
        arrival_vectors.resize(scfg.queue_pair_count);

        std::atomic<std::size_t> accumulated = 0;

        std::vector<std::span<tf::Task>> to_be_scheduled;
        to_be_scheduled.resize(tota_request_cnt);

        // dummy work task (since this is I/O only):
        std::vector<tf::Task> leaf_tasks; // one per relevant inverted list (all Teams)
        leaf_tasks.reserve(list_count);
        
        // "conditional task", only the first dependent will be triggered, the rest "dangles" (until manually started)
        // ALL tasks are directly or indirectly dependent on this task, so we can better control when they start
        auto stopper_task = taskflow.emplace([]() {return 0 /* i.e. only I/O tasks are directly started*/;});
        stopper_task.name("START");

        auto io_start_task = taskflow.emplace([]() {});
        io_start_task.succeed(stopper_task);
        io_start_task.name("I/O");

        auto io_end_task = taskflow.emplace([]() {});
        io_end_task.name("I/O_DONE");

        // define I/O tasks:
        for (auto queue_pair_id = 0u; queue_pair_id < scfg.queue_pair_count; queue_pair_id++) {
            
            // one pair for each ring
            auto submit_task = taskflow.placeholder();
            submit_task.name("SUBMIT["+std::to_string(queue_pair_id)+"].LOOP");
            submit_task.work([&storage, queue_pair_id]() {
                
                auto i = storage.submit_batch(queue_pair_id); // my not do anything, if request queue is full atm
                // std::cout << queue_pair_id << ": Called submit_batch, " << i << " remaining..." << std::endl;
            });
            
            auto await_task = taskflow.placeholder();
            await_task.name("AWAIT["+std::to_string(queue_pair_id)+"].LOOP");
            await_task.work([&storage, &arrival_vectors, &to_be_scheduled, queue_pair_id] (tf::Runtime& rt) {

                /// quickly peak into the arrival queue and take out a batch of requests
                auto cnt = storage.await_batch(queue_pair_id, arrival_vectors[queue_pair_id]);
                
                assert(arrival_vectors[queue_pair_id].size() == cnt);
                /// process arrivals, if any are available
                for (Storage::IORequest* arrival: arrival_vectors[queue_pair_id]) {
                    assert(arrival);
                    for (auto& task: to_be_scheduled[arrival->req_id])
                        rt.schedule(task); // the callback starts tasks processing the data for this request
                }

                arrival_vectors[queue_pair_id].clear();                    
            });

            auto loop_task = taskflow.placeholder();
            loop_task.name("LOOP");
            loop_task.work([&storage, queue_pair_id]() {

                /// queue up submit_task again, if there is still some I/O to submit left 
                if (storage.remaining_submissions(queue_pair_id) > 0) {
                    return 0;
                }
                // // no submissions, but in-flight I/Os?
                if (storage.remaining_arrivals(queue_pair_id) > 0) {
                    return 1; // loop back
                }
                // all done!
                return 2; // break out
            });

            submit_task.succeed(io_start_task);
            await_task.succeed(submit_task);
            loop_task.succeed(await_task);
            loop_task.precede(submit_task, await_task, io_end_task); // loop back to submission, just wait again or we are done!

            submission_tasks.emplace_back(std::move(submit_task));
            await_tasks.emplace_back(std::move(await_task));

            // define dummy "work" tasks and actual I/O requests
            for (auto& request_info : request_infos[queue_pair_id]) {
                auto tid = team_ids[request_info.team_name];
                auto task_range_start = leaf_tasks.size();
                auto ptr = get_new_io_buffer(request_info.total_block_cnt); // where to store the I/O data
                io_buffers.emplace_back(ptr);

                for (auto& [off,card,codec,size,group_id] : request_info.decomp_info) { // one entry per list in this request (StartBlock, ListCardinality, CodecID, ListSizeCompressed)
                    // define "dummy work" task
                    tf::Task leaf_task = taskflow.emplace([&accumulated, ptr, off, card, codec, size]() {
                        accumulated.fetch_add(card);
                    });
                    leaf_task.name("WORK."+std::to_string(request_info.rid));

                    leaf_task.succeed(stopper_task); // this prevents immediate execution, need to wait for async I/O first!

                    leaf_tasks.emplace_back(leaf_task);
                }
                
                // define callback, that triggers computation by queueing the corresponding set of tasks (one or more per I/O)
                 
                std::span<tf::Task> task_range(leaf_tasks.data()+task_range_start, leaf_tasks.size()-task_range_start);
                auto rid = request_info.rid;
                to_be_scheduled[request_info.rid] = std::move(task_range);

                // create corresponding request:

                TeamIndex::Storage::IORequest request{request_info.rid, ptr, [] () {}, request_info.start_block, request_info.total_block_cnt, tid};
                storage.register_request(queue_pair_id, std::move(request));
    
                // std::cout << "Added request to " << queue_pair_id << std::endl;
            }
            
        }

        plan_timer.stop();

        // task-graph created, dump as .dot:
        if (ecfg.print_execution_plan.has_value()) {
            // expand the name in this to also have the start of the execution at the end
            auto name = ecfg.print_execution_plan.value();
            

            name = name.substr(0, name.find_last_of('.')) + "_start.dot";

            std::ofstream fout(ecfg.print_execution_plan.value());
            taskflow.dump(fout);
        }
        
        /////////////////////// Run graph


        ExecutionStatistics stats;
        
        Timer execution_timer("Execution", total_read_volume);
        execution_timer.set_verbose_on_death();

        // Run the flow (which may not be altered anymore)
        exec_ptr->run(taskflow);
        
        exec_ptr->wait_for_all();
        std::cout << "Read " << accumulated << " values." << std::endl;
        std::cout << "Expected to read " << tota_read_card << " values." << std::endl;
        stats.executor_runtime = execution_timer.duration();
        stats.plan_construction_runtime = plan_timer.duration();
        /////////////////////// Cleanup 

        for (auto buff: io_buffers)
            free(buff);

        return {std::span<IDType>{}, stats};
    };
} // namespace