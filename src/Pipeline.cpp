#include "rm_autoaim/Pipeline.hpp"

#include <chrono>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace rm_autoaim {

auto Pipeline::PerfCounter::record_start() -> void {
  start = std::chrono::steady_clock::now();
}

auto Pipeline::PerfCounter::record_end() -> double {
  end = std::chrono::steady_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  return static_cast<double>(dur.count());
}

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
              std::atomic<std::shared_ptr<std::vector<AimAngle>>>>())
    , debug_frame_slot_(
          std::make_shared<std::atomic<std::shared_ptr<cv::Mat>>>())
    , debug_det_slot_(
          std::make_shared<std::atomic<std::shared_ptr<Detector::DetectorDebugInfo>>>()) {}

auto Pipeline::start() -> void {
  if (started_.exchange(true)) {
    spdlog::warn("Pipeline already started");
    return;
  }

  spdlog::info("Pipeline: starting all modules...");

  reader_.start();

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

  spdlog::info("Pipeline: all threads stopped");

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

// Version-based deduplication: each producer increments a version counter
// after storing new data. Each consumer reads the version and skips
// processing if unchanged, avoiding infinite re-processing of stale data.
auto Pipeline::reader_thread_fn(std::stop_token st) -> void {
  spdlog::info("[Reader] thread started");
  PerfCounter perf;
  const FrameData* last_frame = nullptr;

  while (!st.stop_requested()) {
    perf.record_start();
    auto frame = reader_.latest_frame();
    double latency = perf.record_end();

    if (frame) {
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

    uint64_t cur_version = frame_version_.load(std::memory_order_acquire);
    if (cur_version == last_version) {
      if (done_.load()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    last_version = cur_version;

    perf.record_start();
    auto [detections, det_debug] = detector_.detect_debug(frame->image);
    double latency = perf.record_end();

    auto det_ptr = std::make_shared<std::vector<Armor2D>>(std::move(detections));
    det_slot_->store(det_ptr);
    det_version_.fetch_add(1, std::memory_order_release);
    det_slot_->notify_all();

    if (debug_viz_enabled_) {
      auto debug_frame = std::make_shared<cv::Mat>(frame->image.clone());
      debug_frame_slot_->store(debug_frame);
      auto debug_info =
          std::make_shared<Detector::DetectorDebugInfo>(std::move(det_debug));
      debug_det_slot_->store(debug_info);
    }

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

    if (debug_viz_enabled_) {
      auto debug_frame = debug_frame_slot_->load();
      auto debug_det = debug_det_slot_->load();
      auto dets = det_slot_->load();
      auto aims = aim_slot_->load();
      if (debug_frame && debug_det && dets) {
        draw_debug_frame(*debug_frame, *debug_det, *dets,
                         *track_ptr,
                         aims ? *aims : std::vector<AimAngle>{});
      }
    }

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

  uint64_t last_version = 0;
  auto last_process_time = std::chrono::steady_clock::now();
  bool first_frame = true;

  while (!st.stop_requested()) {
    auto tracks = track_slot_->load();
    if (!tracks) {
      if (done_.load()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    uint64_t cur_version = track_version_.load(std::memory_order_acquire);
    if (cur_version == last_version) {
      if (done_.load()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    last_version = cur_version;

    // V4: dt from actual wall-clock elapsed time, not a hardcoded constant
    auto now = std::chrono::steady_clock::now();
    double dt = first_frame
        ? (1.0 / 166.7)
        : std::chrono::duration<double>(now - last_process_time).count();
    last_process_time = now;
    first_frame = false;

    perf.record_start();
    auto predictions = predictor_.predict(*tracks, dt);
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

auto Pipeline::enable_debug_viz(const std::string& output_path) -> void {
  debug_viz_path_ = output_path;
  debug_viz_enabled_ = true;
}

auto Pipeline::draw_debug_frame(
    const cv::Mat& frame,
    const Detector::DetectorDebugInfo& det_debug,
    const std::vector<Armor2D>& detections,
    const std::vector<TrackedArmor>& tracks,
    const std::vector<AimAngle>& aims) -> void {
  cv::Mat viz = frame.clone();

  // Blue: light bar candidates (Step 4 output)
  for (const auto& lb : det_debug.light_bars) {
    cv::Point2f vertices[4];
    lb.rect.points(vertices);
    for (int i = 0; i < 4; ++i) {
      cv::line(viz, vertices[i], vertices[(i + 1) % 4],
               cv::Scalar(255, 0, 0), 2);
    }
  }

  // Green: armor pairs (Step 5, before Step 6 filtering)
  for (const auto& pair : det_debug.pairs) {
    auto corners = Detector::extract_corners(pair);
    for (int i = 0; i < 4; ++i) {
      cv::line(viz, corners[i], corners[(i + 1) % 4],
               cv::Scalar(0, 255, 0), 2);
    }
  }

  // Red: tracked armors with ID, status, depth, and aim angles
  for (const auto& t : tracks) {
    for (int i = 0; i < 4; ++i) {
      cv::line(viz, t.detection.corners[i],
               t.detection.corners[(i + 1) % 4],
               cv::Scalar(0, 0, 255), 2);
    }

    auto cx = (t.detection.corners[0].x + t.detection.corners[1].x +
               t.detection.corners[2].x + t.detection.corners[3].x) / 4.0F;
    auto cy = (t.detection.corners[0].y + t.detection.corners[1].y +
               t.detection.corners[2].y + t.detection.corners[3].y) / 4.0F;

    cv::putText(viz, "ID:" + std::to_string(t.id),
                cv::Point2f(cx - 30.0F, cy - 30.0F),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 255), 2);

    const char* status_str =
        (t.status == TrackedArmor::Status::kConfirmed) ? "CONFIRMED" :
        (t.status == TrackedArmor::Status::kLost)      ? "LOST" : "TENTATIVE";
    cv::putText(viz, status_str,
                cv::Point2f(cx - 30.0F, cy - 8.0F),
                cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(0, 0, 255), 1);

    auto depth_str = cv::format("Z:%.2fm", t.pose.depth);
    cv::putText(viz, depth_str,
                cv::Point2f(cx - 30.0F, cy + 12.0F),
                cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(0, 0, 255), 1);

    for (const auto& aim : aims) {
      if (aim.target_id == t.id) {
        auto aim_str = cv::format("yaw:%.1f pit:%.1f t:%.0fms",
                                  aim.yaw * 180.0 / CV_PI,
                                  aim.pitch * 180.0 / CV_PI,
                                  aim.flight_time * 1000.0);
        cv::putText(viz, aim_str,
                    cv::Point2f(cx - 30.0F, cy + 32.0F),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(0, 0, 255), 1);
        break;
      }
    }
  }

  if (!debug_writer_.isOpened()) {
    debug_writer_.open(debug_viz_path_,
                       cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                       142.0,
                       cv::Size(frame.cols, frame.rows));
  }

  debug_writer_.write(viz);
}

}  // namespace rm_autoaim