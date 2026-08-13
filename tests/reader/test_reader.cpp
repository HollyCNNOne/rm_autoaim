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
  // Wall-clock timing: measure total decode time inside Reader, not
  // the consumer-side polling latency (which is ~0us because
  // latest_frame() is just an atomic load).
  auto t_start = std::chrono::steady_clock::now();
  std::string first_frame_hash;
  int frame_count = 0;

  auto last_frame = reader.latest_frame();

  // Same consumption pattern as Pipeline::reader_thread_fn:
  // is_done() is checked ONLY when we get a duplicate frame, not in
  // the loop condition. This ensures the last frame(s) are not
  // missed due to a race between the final store() and done_=true.
  for (;;) {
    auto frame = reader.latest_frame();

    if (frame.get() == last_frame.get()) {
      if (reader.is_done()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    last_frame = frame;

    if (frame && !frame->image.empty()) {
      frame_count++;
      if (frame_count == 1) {
        first_frame_hash = frame_hash(frame->image);
      }
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  auto total_decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
  double throughput_fps = (total_decode_ms.count() > 0)
    ? (frame_count * 1000.0 / total_decode_ms.count()) : 0.0;

  reader.stop();

  // --- Compute statistics ---
  // Performance is measured by wall-clock throughput (total frames / total time),
  // not by consumer-side polling latency (which is meaningless ~0us).

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

  // TC5: Performance (wall-clock throughput, not polling latency)
  std::cout << "    \"tc5_performance\": {\n";
  std::cout << "      \"total_decode_ms\": " << total_decode_ms.count() << ",\n";
  std::cout << "      \"throughput_fps\": " << std::fixed << std::setprecision(1) << throughput_fps << ",\n";
  std::cout << "      \"video_fps\": " << std::fixed << std::setprecision(1) << fps << ",\n";
  std::cout << "      \"realtime_factor\": " << std::fixed << std::setprecision(1)
            << (throughput_fps / fps) << ",\n";
  std::cout << "      \"total_frames\": " << frame_count << "\n";
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
    std::cout << "  \"total_decode_ms\": " << total_decode_ms.count() << ",\n";
    std::cout << "  \"throughput_fps\": " << std::fixed << std::setprecision(1) << throughput_fps << "\n";
    std::cout << "}\n";
    std::fclose(stdout);
    std::cout << "Baseline saved to " << baseline_path << "\n";
  }

  return 0;
}