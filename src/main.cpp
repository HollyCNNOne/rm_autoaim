#include "rm_autoaim/Pipeline.hpp"

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <spdlog/spdlog.h>

namespace {

// Global pipeline pointer for signal handler
rm_autoaim::Pipeline* g_pipeline{nullptr};
std::atomic<bool> g_running{true};

auto signal_handler(int /*sig*/) -> void {
  spdlog::info("Received interrupt signal, shutting down...");
  g_running.store(false);
  if (g_pipeline) {
    g_pipeline->stop();
  }
}

auto print_usage(const char* prog) -> void {
  std::cout << "Usage: " << prog << " <video_path>\n"
            << "  video_path: path to outpost.mkv (or armor.mkv, energy.mkv)\n"
            << "\n"
            << "Example:\n"
            << "  " << prog << " outpost.mkv\n"
            << "  " << prog << " /path/to/27赛季视觉组特招考核试题/图片和附件/outpost.mkv\n";
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  // Configure logging
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

  if (argc < 2) {
    spdlog::error("Missing video path argument");
    print_usage(argv[0]);
    return 1;
  }

  std::string video_path = argv[1];

  spdlog::info("============================================");
  spdlog::info(" RoboMaster Autoaim System — Scenario C: Outpost");
  spdlog::info(" Video: {}", video_path);
  spdlog::info("============================================");

  // Register signal handlers for graceful shutdown
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  // Create and start pipeline
  rm_autoaim::Pipeline pipeline(video_path);
  g_pipeline = &pipeline;

  pipeline.start();

  // Main loop: periodically print aim angles
  int frame_count = 0;
  while (g_running.load() && !pipeline.is_done()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto aims = pipeline.latest_aim_angles();
    for (const auto& aim : aims) {
      spdlog::info("Target #{} | pitch={:+.2f}° yaw={:+.2f}° t_fly={:.1f}ms",
                   aim.target_id, aim.pitch * 180.0 / M_PI,
                   aim.yaw * 180.0 / M_PI, aim.flight_time * 1000.0);
    }

    frame_count++;
    if (frame_count % 50 == 0) {
      auto stats = pipeline.stats();
      spdlog::info("--- Frame {} ---", stats.total_frames);
      spdlog::info("  Reader:    {:.0f}us avg", stats.reader.avg_latency_us);
      spdlog::info("  Detector:  {:.0f}us avg", stats.detector.avg_latency_us);
      spdlog::info("  Tracker:   {:.0f}us avg", stats.tracker.avg_latency_us);
      spdlog::info("  Predictor: {:.0f}us avg", stats.predictor.avg_latency_us);
      spdlog::info("  Ballistic: {:.0f}us avg", stats.ballistic.avg_latency_us);
    }
  }

  pipeline.stop();

  // Final statistics
  auto stats = pipeline.stats();
  spdlog::info("============================================");
  spdlog::info(" Pipeline Complete");
  spdlog::info(" Total frames: {}", stats.total_frames);
  spdlog::info(" Reader:    avg={:.0f}us max={:.0f}us",
               stats.reader.avg_latency_us, stats.reader.max_latency_us);
  spdlog::info(" Detector:  avg={:.0f}us max={:.0f}us",
               stats.detector.avg_latency_us, stats.detector.max_latency_us);
  spdlog::info(" Tracker:   avg={:.0f}us max={:.0f}us",
               stats.tracker.avg_latency_us, stats.tracker.max_latency_us);
  spdlog::info(" Predictor: avg={:.0f}us max={:.0f}us",
               stats.predictor.avg_latency_us, stats.predictor.max_latency_us);
  spdlog::info(" Ballistic: avg={:.0f}us max={:.0f}us",
               stats.ballistic.avg_latency_us, stats.ballistic.max_latency_us);
  spdlog::info("============================================");

  return 0;
}