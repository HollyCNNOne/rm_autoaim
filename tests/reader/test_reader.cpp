// Module 1: Reader standalone test
// Usage: ./test_reader <video_path> [--baseline baseline.json]
// Outputs JSON test report to stdout

#include "rm_autoaim/Reader.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// Simple MD5 of first frame (header + first 1024 bytes of pixel data)
auto frame_hash(const cv::Mat& img) -> std::string {
  std::ostringstream oss;
  oss << img.cols << "x" << img.rows << "x" << img.channels() << ":";
  auto total = img.total() * img.elemSize();
  auto sample = std::min(total, size_t{4096});
  auto* data = img.ptr<uint8_t>();
  for (size_t i = 0; i < sample; ++i) {
    char buf[3];
    std::snprintf(buf, sizeof(buf), "%02x", data[i]);
    oss << buf;
  }
  return oss.str();
}

auto json_escape(const std::string& s) -> std::string {
  std::string out;
  for (auto c : s) {
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else out += c;
  }
  return out;
}

} // anonymous namespace

auto main(int argc, char* argv[]) -> int {
  if (argc < 2) {
    std::cerr << "Usage: test_reader <video_path> [--baseline baseline.json]\n";
    return 1;
  }

  std::string video_path = argv[1];
  std::string baseline_path;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--baseline") == 0 && i + 1 < argc) {
      baseline_path = argv[++i];
    }
  }

  rm_autoaim::Reader reader(video_path);

  // --- Test 1: Init ---
  reader.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  int width = reader.width();
  int height = reader.height();
  double fps = reader.fps();

  // --- Test 2: Decode all frames ---
  std::vector<double> inter_arrival_us;
  inter_arrival_us.reserve(6000);
  std::string first_frame_hash;
  int frame_count = 0;

  auto last_frame = reader.latest_frame();
  auto last_arrival = std::chrono::steady_clock::now();
  auto t_start = last_arrival;

  for (;;) {
    auto frame = reader.latest_frame();

    if (frame.get() == last_frame.get()) {
      if (reader.is_done()) break;  // no more frames coming
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    auto now = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(now - last_arrival);
    inter_arrival_us.push_back(static_cast<double>(dur.count()));
    last_arrival = now;
    last_frame = frame;

    if (frame && !frame->image.empty()) {
      frame_count++;
      if (frame_count == 1) {
        first_frame_hash = frame_hash(frame->image);
      }
    }
  }

  // Drain the final frame after done_
  {
    auto frame = reader.latest_frame();
    if (frame.get() != last_frame.get() && frame && !frame->image.empty()) {
      frame_count++;
      if (frame_count == 1) {
        first_frame_hash = frame_hash(frame->image);
      }
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  auto total_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

  reader.stop();

  // --- Compute statistics ---
  if (!inter_arrival_us.empty()) {
    std::sort(inter_arrival_us.begin(), inter_arrival_us.end());
  }
  double avg_inter = 0.0;
  for (auto l : inter_arrival_us) avg_inter += l;
  avg_inter /= inter_arrival_us.empty() ? 1.0 : static_cast<double>(inter_arrival_us.size());

  double p50 = inter_arrival_us.empty() ? 0.0 : inter_arrival_us[inter_arrival_us.size() / 2];
  double p99 = inter_arrival_us.empty() ? 0.0 : inter_arrival_us[inter_arrival_us.size() * 99 / 100];
  double p_max = inter_arrival_us.empty() ? 0.0 : inter_arrival_us.back();

  // --- IMU validation ---
  auto frame = reader.latest_frame();
  double imu_norm = 0.0;
  if (frame) {
    auto& q = frame->imu;
    imu_norm = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
  }

  // --- Output JSON report ---
  std::cout << "{\n";
  std::cout << "  \"module\": \"Reader\",\n";
  std::cout << "  \"video\": \"" << json_escape(video_path) << "\",\n";
  std::cout << "  \"tests\": {\n";

  // TC1: Frame count
  std::cout << "    \"tc1_frame_count\": {\n";
  std::cout << "      \"value\": " << frame_count << ",\n";
  std::cout << "      \"pass\": " << (frame_count == 5195 ? "true" : "false") << "\n";
  std::cout << "    },\n";

  // TC2: Resolution
  std::cout << "    \"tc2_resolution\": {\n";
  std::cout << "      \"width\": " << width << ",\n";
  std::cout << "      \"height\": " << height << ",\n";
  std::cout << "      \"pass\": " << (width == 1920 && height == 1200 ? "true" : "false") << "\n";
  std::cout << "    },\n";

  // TC3: IMU norm
  std::cout << "    \"tc3_imu\": {\n";
  std::cout << "      \"norm\": " << std::fixed << std::setprecision(6) << imu_norm << ",\n";
  std::cout << "      \"pass\": " << (std::abs(imu_norm - 1.0) < 1e-6 ? "true" : "false") << "\n";
  std::cout << "    },\n";

  // TC4: First frame hash
  std::cout << "    \"tc4_first_frame_hash\": {\n";
  std::cout << "      \"hash\": \"" << first_frame_hash << "\"\n";
  std::cout << "    },\n";

  // TC5: Performance
  std::cout << "    \"tc5_performance\": {\n";
  std::cout << "      \"total_wall_ms\": " << total_wall_ms << ",\n";
  std::cout << "      \"avg_inter_arrival_us\": " << std::fixed << std::setprecision(1) << avg_inter << ",\n";
  std::cout << "      \"p50_inter_us\": " << p50 << ",\n";
  std::cout << "      \"p99_inter_us\": " << p99 << ",\n";
  std::cout << "      \"max_inter_us\": " << p_max << ",\n";
  std::cout << "      \"throughput_fps\": " << std::fixed << std::setprecision(1)
            << (total_wall_ms > 0 ? frame_count * 1000.0 / total_wall_ms : 0.0) << ",\n";
  std::cout << "      \"video_fps\": " << std::fixed << std::setprecision(1) << fps << "\n";
  std::cout << "    }\n";

  std::cout << "  }\n";
  std::cout << "}\n";

  // --- Save baseline if requested ---
  if (!baseline_path.empty()) {
    std::freopen(baseline_path.c_str(), "w", stdout);
    std::cout << "{\n";
    std::cout << "  \"first_frame_hash\": \"" << first_frame_hash << "\",\n";
    std::cout << "  \"frame_count\": " << frame_count << ",\n";
    std::cout << "  \"width\": " << width << ",\n";
    std::cout << "  \"height\": " << height << ",\n";
    std::cout << "  \"fps\": " << std::fixed << std::setprecision(1) << fps << ",\n";
    std::cout << "  \"imu_norm\": " << std::fixed << std::setprecision(6) << imu_norm << ",\n";
    std::cout << "  \"latency_avg_inter_us\": " << std::fixed << std::setprecision(1) << avg_inter << ",\n";
    std::cout << "  \"latency_p99_inter_us\": " << std::fixed << std::setprecision(1) << p99 << ",\n";
    std::cout << "  \"throughput_fps\": " << std::fixed << std::setprecision(1)
              << (total_wall_ms > 0 ? frame_count * 1000.0 / total_wall_ms : 0.0) << "\n";
    std::cout << "}\n";
    std::fclose(stdout);
    std::cout << "Baseline saved to " << baseline_path << "\n";
  }

  return 0;
}