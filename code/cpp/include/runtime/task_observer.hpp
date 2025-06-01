#pragma once


#include "benchmark/performance_tracking.hpp"
#include <taskflow/taskflow.hpp>
#include <iostream>
#include <unordered_map>
#include <string>
#include <mutex>


namespace TeamIndex {
    struct PerformanceObserver : public tf::ObserverInterface {

        PerformanceObserver(TaskTracker& tracker): _tracker(tracker) {
            // std::cout << "PerformanceObserver initialized." << '\n';
        }

        void set_up(size_t) override final {
        }

        void on_entry(tf::WorkerView w, tf::TaskView tv) override final {

            /** Hack for task.hpp:
                    template<typename T>
                    T* TaskView::data() const {
                        return static_cast<T*>(_node._data);
                    }
             */
            void* data_ptr = tv.data();
            unsigned pos = reinterpret_cast<std::size_t>(data_ptr);
            TaskTracker::TaskStats& stats = _tracker.get_stats(pos);
            stats.start();
            stats.worker_id = w.id();
            
        }

        void on_exit(tf::WorkerView w, tf::TaskView tv) override final {
            void* data_ptr = tv.data();
            unsigned pos = reinterpret_cast<std::size_t>(data_ptr);
            TaskTracker::TaskStats& stats = _tracker.get_stats(pos);
            stats.stop();
            
        }

        private:
        TaskTracker& _tracker;
    };
   
    class TaskflowDebugger {
    public:
        TaskflowDebugger() = default;
        
        struct TaskInfo {
            int creation_predecessors = 0;
            int creation_successors = 0;
            std::string name;
        };

        void register_task(tf::Task task) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& info = task_map_[task.name()];
            info.name = task.name();
            info.creation_predecessors = task.num_dependents();
            info.creation_successors = task.num_successors();
        }

        void validate_task(tf::TaskView& task) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = task_map_.find(task.name());
            if (it == task_map_.end()) {
                std::cerr << "ERROR: Task " << task.name()
                        << " not found in task map!\n";
                return;
            }

            const TaskInfo& info = it->second;

            if (info.creation_predecessors != task.num_dependents() ||
                info.creation_successors != task.num_successors()) {
                std::cerr << "TASKFLOW DEBUGGER WARNING: Task "
                        << task.name() << " changed:\n"
                        << "  Predecessors (created): " << info.creation_predecessors << "\n"
                        << "  Predecessors (runtime): " << task.num_dependents() << "\n"
                        << "  Successors (created): " << info.creation_successors << "\n"
                        << "  Successors (runtime): " << task.num_successors() << "\n";
            }
        }

        void dump_tasks() {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& [task, info] : task_map_) {
                std::cout << "Task: " << info.name
                        << ", Pred: " << info.creation_predecessors
                        << ", Succ: " << info.creation_successors << "\n";
            }
        }

    private:
        std::unordered_map<std::string, TaskInfo> task_map_;
        std::mutex mutex_;
    };


    struct GraphChecker : public tf::ObserverInterface {

        GraphChecker(TaskflowDebugger& debugger) : debugger_(debugger) {
            std::cout << "GraphChecker initialized." << '\n';
        }

        void set_up(size_t num_workers) override final {
            std::cout << "setting up observer with " << num_workers << " workers\n";
        }

        void on_entry(tf::WorkerView w, tf::TaskView tv) override final {
            // debugger_.validate_task(tv);
            std::cout << "[RUNNING] " << tv.name() 
              << " (Dependents: " << tv.num_successors()
              << ", Predecessors: " << tv.num_dependents() << ")\n";
        }

        void on_exit(tf::WorkerView w, tf::TaskView tv) override final {
            // std::ostringstream oss;
            // oss << "worker " << w.id() << " finished running " << tv.name() << '\n';
            // std::cout << oss.str();
        }

        private:
        TaskflowDebugger& debugger_;

    };

}