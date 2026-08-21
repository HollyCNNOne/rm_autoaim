#include "rm_autoaim/Pipeline.hpp"

#include <chrono>

#ifdef __linux__
#include <pthread.h>
#include <unistd.h>
#endif

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace rm_autoaim {

// ============================================================================
// V7.1: Real-time helpers (Linux only, graceful fallback on other platforms)
// ============================================================================

#ifdef __linux__
namespace {

[[maybe_unused]] auto promote_to_realtime() -> void {
  struct sched_param param;
  param.sched_priority = 99;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) == 0) {
    spdlog::info("[V7.1] Thread promoted to SCHED_FIFO (prio=99)");
  } else if (nice(-20) != -1) {
    spdlog::warn("[V7.1] SCHED_FIFO failed, fell back to nice(-20)");
  } else {
    spdlog::warn("[V7.1] Realtime promotion failed, using default scheduler");
  }
}

[[maybe_unused]] auto bind_to_core(int core) -> void {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0) {
    spdlog::info("[V7.1] Thread bound to core {}", core);
  } else {
    spdlog::warn("[V7.1] Failed to bind thread to core {}", core);
  }
}

}  // anonymous namespace
#else
namespace {
[[maybe_unused]] auto promote_to_realtime() -> void {
  spdlog::info("[V7.1] Real-time priority not available on this platform");
}
[[maybe_unused]] auto bind_to_core(int /*core*/) -> void {
  // no-op on non-Linux platforms
}
}  // anonymous namespace
#endif

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

  // V7.1: Wake all threads blocked on queues
  frame_queue_cv_.notify_all();
  render_signal_cv_.notify_all();

  if (render_running_.load()) {
    render_thread_.request_stop();
  }

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

  while (!st.stop_requested()) {
    perf.record_start();
    auto frame = reader_.latest_frame();
    double latency = perf.record_end();

    if (frame) {
      // V7.1: Bounded queue — drop oldest if full, non-blocking push
      {
        std::lock_guard<std::mutex> lock(frame_queue_mutex_);
        if (frame_queue_.size() >= kMaxFrameQueueSize) {
          frame_queue_.pop();  // drop oldest
        }
        frame_queue_.push(frame);
      }
      frame_queue_cv_.notify_one();

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
  frame_queue_cv_.notify_all();  // wake detector to exit
  spdlog::info("[Reader] thread finished");
}

auto Pipeline::detector_thread_fn(std::stop_token st) -> void {
  spdlog::info("[Detector] thread started");

  // V7.1: Realtime priority — SCHED_FIFO prio=99, fallback nice(-20)
  promote_to_realtime();
  // V7.1: CPU affinity — isolate Detector on core 0, eliminate cache thrashing
  bind_to_core(0);

  PerfCounter perf;

  while (!st.stop_requested()) {
    // V7.1: Blocking dequeue with timeout — absorbs transient jitter
    std::shared_ptr<FrameData> frame;
    {
      std::unique_lock<std::mutex> lock(frame_queue_mutex_);
      bool got = frame_queue_cv_.wait_for(
          lock, std::chrono::milliseconds(5),
          [this, &st] { return !frame_queue_.empty() || st.stop_requested(); });
      if (!got) {
        if (done_.load()) break;
        continue;
      }
      if (st.stop_requested() && frame_queue_.empty()) break;
      frame = frame_queue_.front();
      frame_queue_.pop();
    }

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

      // V7.1: Signal render thread via mutex+cv queue
      {
        std::lock_guard<std::mutex> lock(render_signal_mutex_);
        if (render_signal_queue_.size() < 60) {
          render_signal_queue_.push(frame->frame_index);
          render_signal_cv_.notify_one();
        }
      }
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

  auto last_process_time = std::chrono::steady_clock::now();
  bool first_frame = true;

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

    // V7: dt from actual wall-clock elapsed time for EKF predict
    auto now = std::chrono::steady_clock::now();
    double dt = first_frame
        ? (1.0 / 166.7)
        : std::chrono::duration<double>(now - last_process_time).count();
    last_process_time = now;
    first_frame = false;

    perf.record_start();
    auto tracks = tracker_.update(*detections, dt);
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
  render_running_.store(true);
  render_thread_ = std::jthread([this](std::stop_token st) {
    render_thread_fn(st);
  });
}

auto Pipeline::draw_debug_frame(
    const cv::Mat& frame,
    const Detector::DetectorDebugInfo& det_debug,
    const std::vector<Armor2D>& detections,
    const std::vector<TrackedArmor>& tracks,
    const std::vector<AimAngle>& aims) -> cv::Mat {
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

  return viz;
}

auto Pipeline::render_thread_fn(std::stop_token st) -> void {
  spdlog::info("[Render] thread started");

  // V7.1: CPU affinity — isolate Render on core 1 (if available)
  // Keeps Render's memory traffic off Detector's L3 cache lines.
  bind_to_core(1);

  cv::VideoWriter writer;
  writer.open(debug_viz_path_,
              cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
              30.0,
              cv::Size(960, 600));

  if (!writer.isOpened()) {
    spdlog::error("[Render] failed to open '{}'", debug_viz_path_);
    return;
  }

  while (!st.stop_requested()) {
    // V7.1: Block on signal cv instead of polling — no spin-wait
    {
      std::unique_lock<std::mutex> lock(render_signal_mutex_);
      render_signal_cv_.wait(lock, [this, &st] {
        return !render_signal_queue_.empty() || st.stop_requested();
      });
      if (st.stop_requested() && render_signal_queue_.empty()) {
        break;
      }
      // Drain all accumulated signals
      while (!render_signal_queue_.empty()) {
        render_signal_queue_.pop();
      }
    }

    // Decimate: draw only 1 frame every kVizDecimate
    if (++render_frame_counter_ % kVizDecimate != 0) continue;

    auto debug_frame = debug_frame_slot_->load();
    auto debug_det = debug_det_slot_->load();
    auto dets = det_slot_->load();
    auto tracks = track_slot_->load();
    auto aims = aim_slot_->load();

    if (debug_frame && debug_det && dets && tracks) {
      auto viz = draw_debug_frame(*debug_frame, *debug_det, *dets, *tracks,
                                  aims ? *aims : std::vector<AimAngle>{});
      cv::Mat viz_small;
      cv::resize(viz, viz_small, cv::Size(960, 600));
      writer.write(viz_small);
    }
  }

  // V7.2: Final flush — write last frame to disk before releasing VideoWriter.
  // Ensures debug_viz.avi contains all frames, not truncated.
  {
    auto debug_frame = debug_frame_slot_->load();
    auto debug_det = debug_det_slot_->load();
    auto dets = det_slot_->load();
    auto tracks = track_slot_->load();
    auto aims = aim_slot_->load();
    if (debug_frame && debug_det && dets && tracks) {
      auto viz = draw_debug_frame(*debug_frame, *debug_det, *dets, *tracks,
                                  aims ? *aims : std::vector<AimAngle>{});
      cv::Mat viz_small;
      cv::resize(viz, viz_small, cv::Size(960, 600));
      writer.write(viz_small);
    }
  }

  writer.release();
  spdlog::info("[Render] thread finished");
}

}  // namespace rm_autoaim