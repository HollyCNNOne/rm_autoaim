#include "rm_autoaim/Pipeline.hpp"

#include <chrono>

#include <spdlog/spdlog.h>

namespace rm_autoaim {

// ============================================================================
// PerfCounter
// ============================================================================

auto Pipeline::PerfCounter::record_start() -> void {
  start = std::chrono::steady_clock::now();
}

auto Pipeline::PerfCounter::record_end() -> double {
  end = std::chrono::steady_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  return static_cast<double>(dur.count());
}

// ============================================================================
// Construction
// ============================================================================

Pipeline::Pipeline(const std::string& video_path)
    : reader_(video_path)
    , frame_slot_(
          std::make_shared<std::atomic<std::shared_ptr<FrameData>>>())
    , det_slot_(
          std::make_shared<std::atomic<std::shared_ptr<std::vector<Armor2D>>>>())
    , track_slot_(
          std::make_shared<
              std::atomic<std::shared_ptr<std::vector<TrackedArmor>>>>())
    , pred_slot_(
          std::make_shared<
              std::atomic<std::shared_ptr<std::vector<PredictedState>>>>())
    , aim_slot_(
          std::make_shared<
              std::atomic<std::shared_ptr<std::vector<AimAngle>>>>()) {}

// ============================================================================
// Start / Stop
// ============================================================================

auto Pipeline::start() -> void {
  if (started_.exchange(true)) {
    spdlog::warn("Pipeline already started");
    return;
  }

  spdlog::info("Pipeline: starting all modules...");

  // Start reader thread first
  reader_.start();

  // Start processing threads
  reader_thread_ = std::jthread([this](std::stop_token st) {
    reader_thread_fn(st);
  });
  detector_thread_ = std::jthread([this](std::stop_token st) {
    detector_thread_fn(st);
  });
  tracker_thread_ = std::jthread([this](std::stop_token st) {
    tracker_thread_fn(st);
  });
  predictor_thread_ = std::jthread([this](std::stop_token st) {
    predictor_thread_fn(st);
  });
  ballistic_thread_ = std::jthread([this](std::stop_token st) {
    ballistic_thread_fn(st);
  });

  spdlog::info("Pipeline: all {} threads started", 5);
}

auto Pipeline::stop() -> void {
  spdlog::info("Pipeline: stopping all threads...");

  reader_.stop();

  reader_thread_.request_stop();
  detector_thread_.request_stop();
  tracker_thread_.request_stop();
  predictor_thread_.request_stop();
  ballistic_thread_.request_stop();

  // jthread destructors will auto-join
  spdlog::info("Pipeline: all threads stopped");

  // Print stats
  spdlog::info("=== Pipeline Statistics ===");
  spdlog::info("Reader:    avg={:.1f}us, max={:.1f}us, frames={}",
               stats_.reader.avg_latency_us, stats_.reader.max_latency_us,
               stats_.reader.frames_processed);
  spdlog::info("Detector:  avg={:.1f}us, max={:.1f}us, frames={}",
               stats_.detector.avg_latency_us, stats_.detector.max_latency_us,
               stats_.detector.frames_processed);
  spdlog::info("Tracker:   avg={:.1f}us, max={:.1f}us, frames={}",
               stats_.tracker.avg_latency_us, stats_.tracker.max_latency_us,
               stats_.tracker.frames_processed);
  spdlog::info("Predictor: avg={:.1f}us, max={:.1f}us, frames={}",
               stats_.predictor.avg_latency_us, stats_.predictor.max_latency_us,
               stats_.predictor.frames_processed);
  spdlog::info("Ballistic: avg={:.1f}us, max={:.1f}us, frames={}",
               stats_.ballistic.avg_latency_us, stats_.ballistic.max_latency_us,
               stats_.ballistic.frames_processed);
  spdlog::info("Total frames processed: {}", stats_.total_frames);
}

auto Pipeline::latest_aim_angles() -> std::vector<AimAngle> {
  auto aim = aim_slot_->load();
  if (aim) {
    return *aim;
  }
  return {};
}

auto Pipeline::stats() const -> PipelineStats { return stats_; }

auto Pipeline::is_done() const -> bool { return done_.load(); }

// ============================================================================
// Thread Functions
//
// Version-based deduplication strategy:
//   Each producer increments a version counter AFTER storing new data.
//   Each consumer reads the version and skips processing if unchanged.
//   This avoids the infinite spin-loop where downstream threads
//   re-process the same data millions of times.
// ============================================================================

auto Pipeline::reader_thread_fn(std::stop_token st) -> void {
  spdlog::info("[Reader] thread started");
  PerfCounter perf;
  const FrameData* last_frame = nullptr;

  while (!st.stop_requested()) {
    perf.record_start();
    auto frame = reader_.latest_frame();
    double latency = perf.record_end();

    if (frame) {
      // Skip if the same frame (no new frame decoded yet)
      if (frame.get() == last_frame) {
        if (reader_.is_done()) break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        continue;
      }
      last_frame = frame.get();

      frame_slot_->store(frame);
      frame_version_.fetch_add(1, std::memory_order_release);
      frame_slot_->notify_all();

      stats_.reader.frames_processed++;
      stats_.reader.min_latency_us =
          std::min(stats_.reader.min_latency_us, latency);
      stats_.reader.max_latency_us =
          std::max(stats_.reader.max_latency_us, latency);
      stats_.reader.avg_latency_us =
          (stats_.reader.avg_latency_us * (stats_.reader.frames_processed - 1) +
           latency) /
          stats_.reader.frames_processed;
    } else if (reader_.is_done()) {
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }

  done_.store(true);
  spdlog::info("[Reader] thread finished");
}

auto Pipeline::detector_thread_fn(std::stop_token st) -> void {
  spdlog::info("[Detector] thread started");
  PerfCounter perf;
  uint64_t last_version = 0;

  while (!st.stop_requested()) {
    auto frame = frame_slot_->load();
    if (!frame) {
      if (done_.load()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    // Skip if frame version hasn't changed
    uint64_t cur_version = frame_version_.load(std::memory_order_acquire);
    if (cur_version == last_version) {
      if (done_.load()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    last_version = cur_version;

    perf.record_start();
    auto detections = detector_.detect(frame->image);
    double latency = perf.record_end();

    auto det_ptr = std::make_shared<std::vector<Armor2D>>(std::move(detections));
    det_slot_->store(det_ptr);
    det_version_.fetch_add(1, std::memory_order_release);
    det_slot_->notify_all();

    stats_.detector.frames_processed++;
    stats_.detector.min_latency_us =
        std::min(stats_.detector.min_latency_us, latency);
    stats_.detector.max_latency_us =
        std::max(stats_.detector.max_latency_us, latency);
    stats_.detector.avg_latency_us =
        (stats_.detector.avg_latency_us *
             (stats_.detector.frames_processed - 1) +
         latency) /
        stats_.detector.frames_processed;
  }

  spdlog::info("[Detector] thread finished");
}

auto Pipeline::tracker_thread_fn(std::stop_token st) -> void {
  spdlog::info("[Tracker] thread started");
  PerfCounter perf;
  uint64_t last_version = 0;

  while (!st.stop_requested()) {
    auto detections = det_slot_->load();
    if (!detections) {
      if (done_.load()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    // Skip if detection version hasn't changed
    uint64_t cur_version = det_version_.load(std::memory_order_acquire);
    if (cur_version == last_version) {
      if (done_.load()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    last_version = cur_version;

    perf.record_start();
    auto tracks = tracker_.update(*detections);
    double latency = perf.record_end();

    auto track_ptr =
        std::make_shared<std::vector<TrackedArmor>>(std::move(tracks));
    track_slot_->store(track_ptr);
    track_version_.fetch_add(1, std::memory_order_release);
    track_slot_->notify_all();

    stats_.tracker.frames_processed++;
    stats_.tracker.min_latency_us =
        std::min(stats_.tracker.min_latency_us, latency);
    stats_.tracker.max_latency_us =
        std::max(stats_.tracker.max_latency_us, latency);
    stats_.tracker.avg_latency_us =
        (stats_.tracker.avg_latency_us *
             (stats_.tracker.frames_processed - 1) +
         latency) /
        stats_.tracker.frames_processed;
  }

  spdlog::info("[Tracker] thread finished");
}

auto Pipeline::predictor_thread_fn(std::stop_token st) -> void {
  spdlog::info("[Predictor] thread started");
  PerfCounter perf;

  // Frame interval for outpost video: ~7ms at 142 FPS
  constexpr double kDefaultDt = 1.0 / 142.0;
  uint64_t last_version = 0;

  while (!st.stop_requested()) {
    auto tracks = track_slot_->load();
    if (!tracks) {
      if (done_.load()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    // Skip if track version hasn't changed
    uint64_t cur_version = track_version_.load(std::memory_order_acquire);
    if (cur_version == last_version) {
      if (done_.load()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    last_version = cur_version;

    perf.record_start();
    auto predictions = predictor_.predict(*tracks, kDefaultDt);
    double latency = perf.record_end();

    auto pred_ptr =
        std::make_shared<std::vector<PredictedState>>(std::move(predictions));
    pred_slot_->store(pred_ptr);
    pred_version_.fetch_add(1, std::memory_order_release);
    pred_slot_->notify_all();

    stats_.predictor.frames_processed++;
    stats_.predictor.min_latency_us =
        std::min(stats_.predictor.min_latency_us, latency);
    stats_.predictor.max_latency_us =
        std::max(stats_.predictor.max_latency_us, latency);
    stats_.predictor.avg_latency_us =
        (stats_.predictor.avg_latency_us *
             (stats_.predictor.frames_processed - 1) +
         latency) /
        stats_.predictor.frames_processed;
  }

  spdlog::info("[Predictor] thread finished");
}

auto Pipeline::ballistic_thread_fn(std::stop_token st) -> void {
  spdlog::info("[Ballistic] thread started");
  PerfCounter perf;

  // Identity quaternion for outpost (fixed camera)
  Quaternion identity_imu{};
  uint64_t last_version = 0;

  while (!st.stop_requested()) {
    auto predictions = pred_slot_->load();
    if (!predictions) {
      if (done_.load()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    // Skip if prediction version hasn't changed
    uint64_t cur_version = pred_version_.load(std::memory_order_acquire);
    if (cur_version == last_version) {
      if (done_.load()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    last_version = cur_version;

    perf.record_start();
    auto aims = ballistic_.solve(*predictions, identity_imu);
    double latency = perf.record_end();

    auto aim_ptr = std::make_shared<std::vector<AimAngle>>(std::move(aims));
    aim_slot_->store(aim_ptr);
    aim_slot_->notify_all();

    stats_.ballistic.frames_processed++;
    stats_.ballistic.min_latency_us =
        std::min(stats_.ballistic.min_latency_us, latency);
    stats_.ballistic.max_latency_us =
        std::max(stats_.ballistic.max_latency_us, latency);
    stats_.ballistic.avg_latency_us =
        (stats_.ballistic.avg_latency_us *
             (stats_.ballistic.frames_processed - 1) +
         latency) /
        stats_.ballistic.frames_processed;

    stats_.total_frames = stats_.ballistic.frames_processed;
  }

  spdlog::info("[Ballistic] thread finished");
}

}  // namespace rm_autoaim