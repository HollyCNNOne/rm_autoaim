#pragma once

#include "rm_autoaim/BallisticSolver.hpp"
#include "rm_autoaim/Detector.hpp"
#include "rm_autoaim/Predictor.hpp"
#include "rm_autoaim/Reader.hpp"
#include "rm_autoaim/Tracker.hpp"
#include "rm_autoaim/Types.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace rm_autoaim {

// ============================================================================
// Pipeline — Module 6: Multi-Threaded Pipeline Orchestration
//
// Connects 5 modules via lock-free atomic<shared_ptr> slots:
//
//   Reader → [FrameData] → Detector → [Armor2D[]] → Tracker
//     → [TrackedArmor[]] → Predictor → [PredictedState[]] → Ballistic
//     → [AimAngle[]]
//
// Each module runs in its own std::jthread.
// Data passes through SPSC (single-producer single-consumer) slots.
// ============================================================================

class Pipeline {
public:
  Pipeline(const std::string& video_path);

  Pipeline(const Pipeline&) = delete;
  auto operator=(const Pipeline&) -> Pipeline& = delete;
  Pipeline(Pipeline&&) = delete;
  auto operator=(Pipeline&&) -> Pipeline& = delete;

  // Start all pipeline threads
  auto start() -> void;

  // Stop all threads gracefully
  auto stop() -> void;

  // Get latest aim angles (for display / logging)
  [[nodiscard]] auto latest_aim_angles() -> std::vector<AimAngle>;

  // Get pipeline statistics
  [[nodiscard]] auto stats() const -> PipelineStats;

  // Check if pipeline is done (video ended)
  [[nodiscard]] auto is_done() const -> bool;

private:
  // Thread entry points
  auto reader_thread_fn(std::stop_token st) -> void;
  auto detector_thread_fn(std::stop_token st) -> void;
  auto tracker_thread_fn(std::stop_token st) -> void;
  auto predictor_thread_fn(std::stop_token st) -> void;
  auto ballistic_thread_fn(std::stop_token st) -> void;

  // Performance measurement helpers
  struct PerfCounter {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    auto record_start() -> void;
    auto record_end() -> double;  // returns latency in microseconds
  };

  // Modules
  Reader reader_;
  Detector detector_;
  Tracker tracker_;
  Predictor predictor_;
  BallisticSolver ballistic_;

  // Lock-free slots between stages
  // Slot 1: Reader → Detector
  std::shared_ptr<std::atomic<std::shared_ptr<FrameData>>> frame_slot_;
  // Slot 2: Detector → Tracker
  std::shared_ptr<std::atomic<std::shared_ptr<std::vector<Armor2D>>>> det_slot_;
  // Slot 3: Tracker → Predictor
  std::shared_ptr<std::atomic<std::shared_ptr<std::vector<TrackedArmor>>>>
      track_slot_;
  // Slot 4: Predictor → Ballistic
  std::shared_ptr<std::atomic<std::shared_ptr<std::vector<PredictedState>>>>
      pred_slot_;
  // Slot 5: Ballistic → Output
  std::shared_ptr<std::atomic<std::shared_ptr<std::vector<AimAngle>>>>
      aim_slot_;

  // Version counters for deduplication
  // Each producer increments its version when storing new data.
  // Each consumer reads the version and skips if unchanged.
  std::atomic<uint64_t> frame_version_{0};
  std::atomic<uint64_t> det_version_{0};
  std::atomic<uint64_t> track_version_{0};
  std::atomic<uint64_t> pred_version_{0};

  // Threads
  std::jthread reader_thread_;
  std::jthread detector_thread_;
  std::jthread tracker_thread_;
  std::jthread predictor_thread_;
  std::jthread ballistic_thread_;

  // Statistics
  PipelineStats stats_;
  std::atomic<bool> done_{false};
  std::atomic<bool> started_{false};
};

}  // namespace rm_autoaim