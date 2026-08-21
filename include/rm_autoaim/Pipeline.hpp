#pragma once

#include "rm_autoaim/BallisticSolver.hpp"
#include "rm_autoaim/Detector.hpp"
#include "rm_autoaim/Predictor.hpp"
#include "rm_autoaim/Reader.hpp"
#include "rm_autoaim/Tracker.hpp"
#include "rm_autoaim/Types.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/videoio.hpp>

namespace rm_autoaim {

// 5-stage lock-free pipeline: Reader → Detector → Tracker → Predictor → Ballistic
class Pipeline {
public:
  Pipeline(const std::string& video_path);

  Pipeline(const Pipeline&) = delete;
  auto operator=(const Pipeline&) -> Pipeline& = delete;
  Pipeline(Pipeline&&) = delete;
  auto operator=(Pipeline&&) -> Pipeline& = delete;

  auto start() -> void;
  auto stop() -> void;

  auto detector() -> Detector& { return detector_; }

  [[nodiscard]] auto latest_aim_angles() -> std::vector<AimAngle>;
  [[nodiscard]] auto stats() const -> PipelineStats;
  [[nodiscard]] auto is_done() const -> bool;

  auto enable_debug_viz(const std::string& output_path) -> void;

private:
  auto reader_thread_fn(std::stop_token st) -> void;
  auto detector_thread_fn(std::stop_token st) -> void;
  auto tracker_thread_fn(std::stop_token st) -> void;
  auto predictor_thread_fn(std::stop_token st) -> void;
  auto ballistic_thread_fn(std::stop_token st) -> void;

  struct PerfCounter {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    auto record_start() -> void;
    auto record_end() -> double;
  };

  Reader reader_;
  Detector detector_;
  Tracker tracker_;
  Predictor predictor_;
  BallisticSolver ballistic_;

  // V7.1: Bounded frame queue — replaces single-slot atomic for Reader→Detector
  // Absorbs transient jitter: even if Detector runs 6ms, the 5-frame buffer
  // prevents version-skip induced frame loss. Reader drops oldest when full.
  static constexpr size_t kMaxFrameQueueSize{5};
  std::queue<std::shared_ptr<FrameData>> frame_queue_;
  std::mutex frame_queue_mutex_;
  std::condition_variable frame_queue_cv_;

  // Lock-free slots: atomic<shared_ptr<T>> with version-based deduplication
  std::shared_ptr<std::atomic<std::shared_ptr<std::vector<Armor2D>>>> det_slot_;
  std::shared_ptr<std::atomic<std::shared_ptr<std::vector<TrackedArmor>>>>
      track_slot_;
  std::shared_ptr<std::atomic<std::shared_ptr<std::vector<PredictedState>>>>
      pred_slot_;
  std::shared_ptr<std::atomic<std::shared_ptr<std::vector<AimAngle>>>>
      aim_slot_;

  std::atomic<uint64_t> det_version_{0};
  std::atomic<uint64_t> track_version_{0};
  std::atomic<uint64_t> pred_version_{0};

  std::jthread reader_thread_;
  std::jthread detector_thread_;
  std::jthread tracker_thread_;
  std::jthread predictor_thread_;
  std::jthread ballistic_thread_;

  PipelineStats stats_;
  std::atomic<bool> done_{false};
  std::atomic<bool> started_{false};

  auto draw_debug_frame(const cv::Mat& frame,
                        const Detector::DetectorDebugInfo& det_debug,
                        const std::vector<Armor2D>& detections,
                        const std::vector<TrackedArmor>& tracks,
                        const std::vector<AimAngle>& aims) -> cv::Mat;

  auto render_thread_fn(std::stop_token st) -> void;

  std::shared_ptr<std::atomic<std::shared_ptr<cv::Mat>>> debug_frame_slot_;
  std::shared_ptr<std::atomic<std::shared_ptr<Detector::DetectorDebugInfo>>>
      debug_det_slot_;
  std::string debug_viz_path_;
  bool debug_viz_enabled_{false};

  // V5.1: Independent render thread — drawing fully decoupled from pipeline
  std::jthread render_thread_;
  std::atomic<bool> render_running_{false};
  int render_frame_counter_{0};
  static constexpr int kVizDecimate{3};

  // V6/V7.1: Render signal queue — mutex+cv, Detector pushes frame_index,
  // Render thread blocks on cv instead of polling.
  std::queue<uint64_t> render_signal_queue_;
  std::mutex render_signal_mutex_;
  std::condition_variable render_signal_cv_;
};

}  // namespace rm_autoaim