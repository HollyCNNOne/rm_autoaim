#pragma once

#include "rm_autoaim/Types.hpp"

#include <atomic>
#include <memory>
#include <stop_token>
#include <string>
#include <thread>

struct AVFormatContext;
struct AVCodecContext;
struct SwsContext;

namespace rm_autoaim {

// ============================================================================
// Reader — Module 1: Video Decoding & IMU Parsing
//
// Reads MKV video files, decodes H.264 frames via FFmpeg,
// extracts IMU quaternion from subtitle track,
// and publishes FrameData to downstream via lock-free atomic slot.
// ============================================================================

class Reader {
public:
  explicit Reader(const std::string& video_path);
  ~Reader();

  Reader(const Reader&) = delete;
  auto operator=(const Reader&) -> Reader& = delete;
  Reader(Reader&&) = delete;
  auto operator=(Reader&&) -> Reader& = delete;

  // Start the decoding thread
  auto start() -> void;

  // Request stop and wait for thread to join
  auto stop() -> void;

  // Get the latest decoded frame (lock-free, safe to call from any thread)
  [[nodiscard]] auto latest_frame() -> std::shared_ptr<FrameData>;

  // Check if the reader has reached end of file
  [[nodiscard]] auto is_done() const -> bool;

  // Get video properties
  [[nodiscard]] auto width() const -> int;
  [[nodiscard]] auto height() const -> int;
  [[nodiscard]] auto fps() const -> double;

private:
  auto init_decoder() -> bool;
  auto decode_loop(std::stop_token st) -> void;
  auto decode_one_frame() -> std::shared_ptr<FrameData>;
  auto cleanup() -> void;

  // IMU parsing
  auto preload_imu() -> void;
  auto parse_srt_subtitles() -> std::vector<Quaternion>;
  static auto decode_base64_to_quaternion(const std::string& b64) -> Quaternion;

  std::string video_path_;

  // FFmpeg handles
  AVFormatContext* fmt_ctx_{nullptr};
  AVCodecContext* codec_ctx_{nullptr};
  SwsContext* sws_ctx_{nullptr};
  int video_stream_idx_{-1};
  int subtitle_stream_idx_{-1};

  // Video properties
  int width_{0};
  int height_{0};
  double fps_{0.0};

  // IMU data (preloaded from subtitle track)
  std::vector<Quaternion> imu_data_{};
  Quaternion fixed_imu_{};  // for outpost: constant IMU

  // Lock-free output slot
  std::shared_ptr<std::atomic<std::shared_ptr<FrameData>>> output_frame_;

  // Decoding thread
  std::jthread decode_thread_;

  // State
  std::atomic<bool> done_{false};
  std::atomic<bool> started_{false};
  int64_t frame_count_{0};
};

}  // namespace rm_autoaim