#include "rm_autoaim/Reader.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <string_view>

#include <spdlog/spdlog.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace rm_autoaim {

// ============================================================================
// Construction / Destruction
// ============================================================================

Reader::Reader(const std::string& video_path)
    : video_path_(video_path)
    , output_frame_(
          std::make_shared<std::atomic<std::shared_ptr<FrameData>>>()) {}

Reader::~Reader() {
  stop();
  cleanup();
}

// ============================================================================
// Public API
// ============================================================================

auto Reader::start() -> void {
  if (started_.exchange(true)) {
    spdlog::warn("Reader already started");
    return;
  }

  // Preload IMU data (need to open file separately for subtitle parsing)
  preload_imu();

  // Start decoding thread
  decode_thread_ = std::jthread([this](std::stop_token st) {
    spdlog::info("Reader thread started");
    if (!init_decoder()) {
      spdlog::error("Reader: failed to initialize decoder");
      done_.store(true);
      return;
    }
    decode_loop(st);
    spdlog::info("Reader thread finished, {} frames decoded", frame_count_);
  });
}

auto Reader::stop() -> void {
  decode_thread_.request_stop();
  if (decode_thread_.joinable()) {
    decode_thread_.join();
  }
}

auto Reader::latest_frame() -> std::shared_ptr<FrameData> {
  return output_frame_->load();
}

auto Reader::is_done() const -> bool { return done_.load(); }

auto Reader::width() const -> int { return width_; }

auto Reader::height() const -> int { return height_; }

auto Reader::fps() const -> double { return fps_; }

// ============================================================================
// Decoder Initialization
// ============================================================================

auto Reader::init_decoder() -> bool {
  // Open input file
  if (avformat_open_input(&fmt_ctx_, video_path_.c_str(), nullptr, nullptr) <
      0) {
    spdlog::error("Reader: failed to open '{}'", video_path_);
    return false;
  }

  if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
    spdlog::error("Reader: failed to find stream info");
    return false;
  }

  // Find video and subtitle streams
  video_stream_idx_ = -1;
  subtitle_stream_idx_ = -1;

  for (unsigned int i = 0; i < fmt_ctx_->nb_streams; ++i) {
    auto type = fmt_ctx_->streams[i]->codecpar->codec_type;
    if (type == AVMEDIA_TYPE_VIDEO && video_stream_idx_ < 0) {
      video_stream_idx_ = static_cast<int>(i);
    } else if (type == AVMEDIA_TYPE_SUBTITLE && subtitle_stream_idx_ < 0) {
      subtitle_stream_idx_ = static_cast<int>(i);
    }
  }

  if (video_stream_idx_ < 0) {
    spdlog::error("Reader: no video stream found");
    return false;
  }

  // Get video properties
  auto* video_stream = fmt_ctx_->streams[video_stream_idx_];
  width_ = video_stream->codecpar->width;
  height_ = video_stream->codecpar->height;

  if (video_stream->avg_frame_rate.num > 0) {
    fps_ = static_cast<double>(video_stream->avg_frame_rate.num) /
           static_cast<double>(video_stream->avg_frame_rate.den);
  }

  spdlog::info("Reader: {}x{} @ {:.1f} FPS", width_, height_, fps_);

  // Create decoder
  auto* codecpar = video_stream->codecpar;
  const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
  if (!codec) {
    spdlog::error("Reader: codec not found");
    return false;
  }

  codec_ctx_ = avcodec_alloc_context3(codec);
  if (!codec_ctx_) {
    spdlog::error("Reader: failed to allocate codec context");
    return false;
  }

  if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
    spdlog::error("Reader: failed to copy codec params");
    return false;
  }

  if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
    spdlog::error("Reader: failed to open codec");
    return false;
  }

  // Create scaler: YUV → BGR24
  sws_ctx_ = sws_getContext(width_, height_, codec_ctx_->pix_fmt, width_,
                            height_, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr,
                            nullptr, nullptr);
  if (!sws_ctx_) {
    spdlog::error("Reader: failed to create scaler");
    return false;
  }

  return true;
}

// ============================================================================
// Decode Loop
// ============================================================================

auto Reader::decode_loop(std::stop_token st) -> void {
  while (!st.stop_requested()) {
    auto frame = decode_one_frame();
    if (!frame) {
      done_.store(true);
      break;
    }

    output_frame_->store(std::move(frame));
    output_frame_->notify_all();
  }
}

auto Reader::decode_one_frame() -> std::shared_ptr<FrameData> {
  auto* packet = av_packet_alloc();
  auto* frame = av_frame_alloc();

  if (!packet || !frame) {
    av_packet_free(&packet);
    av_frame_free(&frame);
    return nullptr;
  }

  while (av_read_frame(fmt_ctx_, packet) >= 0) {
    if (packet->stream_index != video_stream_idx_) {
      av_packet_unref(packet);
      continue;
    }

    auto ret = avcodec_send_packet(codec_ctx_, packet);
    av_packet_unref(packet);

    if (ret < 0) {
      spdlog::warn("Reader: error sending packet: {}", ret);
      continue;
    }

    ret = avcodec_receive_frame(codec_ctx_, frame);
    if (ret == AVERROR(EAGAIN)) {
      continue;
    }
    if (ret < 0) {
      av_packet_free(&packet);
      av_frame_free(&frame);
      return nullptr;  // EOF or error
    }

    // YUV420p → BGR24
    auto bgr_size = static_cast<size_t>(width_ * height_ * 3);
    std::vector<uint8_t> bgr_buffer(bgr_size);

    uint8_t* dst_data[1] = {bgr_buffer.data()};
    int dst_linesize[1] = {width_ * 3};

    sws_scale(sws_ctx_, frame->data, frame->linesize, 0, height_, dst_data,
              dst_linesize);

    // Wrap in cv::Mat (deep copy for safety)
    cv::Mat image(height_, width_, CV_8UC3, bgr_buffer.data());
    cv::Mat image_copy = image.clone();

    auto result = std::make_shared<FrameData>();
    result->image = std::move(image_copy);
    result->timestamp_us = static_cast<uint64_t>(frame->pts);
    result->frame_index = frame_count_++;
    result->imu = fixed_imu_;

    av_frame_unref(frame);
    av_packet_free(&packet);
    av_frame_free(&frame);
    return result;
  }

  av_packet_free(&packet);
  av_frame_free(&frame);
  return nullptr;
}

// ============================================================================
// Resource Cleanup
// ============================================================================

auto Reader::cleanup() -> void {
  if (sws_ctx_) {
    sws_freeContext(sws_ctx_);
    sws_ctx_ = nullptr;
  }
  if (codec_ctx_) {
    avcodec_free_context(&codec_ctx_);
    codec_ctx_ = nullptr;
  }
  if (fmt_ctx_) {
    avformat_close_input(&fmt_ctx_);
    fmt_ctx_ = nullptr;
  }
}

// ============================================================================
// IMU / Subtitle Parsing
// ============================================================================

auto Reader::preload_imu() -> void {
  // Open a temporary context to read subtitle stream
  AVFormatContext* tmp_ctx = nullptr;
  if (avformat_open_input(&tmp_ctx, video_path_.c_str(), nullptr, nullptr) <
      0) {
    spdlog::warn("Reader: failed to open for IMU preload");
    return;
  }

  avformat_find_stream_info(tmp_ctx, nullptr);

  // Find subtitle stream index
  int sub_idx = -1;
  for (unsigned int i = 0; i < tmp_ctx->nb_streams; ++i) {
    if (tmp_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE) {
      sub_idx = static_cast<int>(i);
      break;
    }
  }

  if (sub_idx < 0) {
    spdlog::info("Reader: no subtitle stream, IMU will be identity");
    avformat_close_input(&tmp_ctx);
    return;
  }

  // Parse subtitle packets
  AVPacket* pkt = av_packet_alloc();
  std::string raw_text;

  while (av_read_frame(tmp_ctx, pkt) >= 0) {
    if (pkt->stream_index == sub_idx) {
      // SRT subtitle data is stored directly in the packet
      std::string_view raw(reinterpret_cast<const char*>(pkt->data),
                           static_cast<size_t>(pkt->size));

      // Extract base64 data from SRT format
      // SRT format: "N\nHH:MM:SS,mmm --> HH:MM:SS,mmm\n{base64}\n"
      // We need to find the base64 portion (last non-empty line)
      std::regex b64_regex(R"(([A-Za-z0-9+/=]{40,50}))");
      std::string raw_str(raw);
      std::smatch match;
      if (std::regex_search(raw_str, match, b64_regex)) {
        try {
          auto q = decode_base64_to_quaternion(match[1].str());
          if (q.is_unit()) {
            fixed_imu_ = q;
            spdlog::info("Reader: IMU quaternion loaded: [{:.4f}, {:.4f}, "
                         "{:.4f}, {:.4f}]",
                         q.w, q.x, q.y, q.z);
            break;  // outpost: only need first IMU value
          }
        } catch (const std::exception& e) {
          spdlog::warn("Reader: IMU parse error: {}", e.what());
        }
      }
    }
    av_packet_unref(pkt);
  }

  av_packet_free(&pkt);
  avformat_close_input(&tmp_ctx);
}

// Manual Base64 decode (av_base64_decode removed in FFmpeg 6.x)
namespace {
static const char kBase64Table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

auto b64_decode(const std::string& input) -> std::vector<uint8_t> {
  std::vector<uint8_t> output;
  int val = 0, valb = -8;
  for (unsigned char c : input) {
    if (c == '=') break;
    const char* p = std::strchr(kBase64Table, c);
    if (!p) continue;
    val = (val << 6) + static_cast<int>(p - kBase64Table);
    valb += 6;
    if (valb >= 0) {
      output.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return output;
}
}  // anonymous namespace

auto Reader::decode_base64_to_quaternion(const std::string& b64)
    -> Quaternion {
  // Decode Base64 → 32 bytes (manual implementation, FFmpeg 6.x compat)
  auto raw_bytes = b64_decode(b64);

  if (raw_bytes.size() != 32) {
    throw std::runtime_error(
        "IMU base64 decode: expected 32 bytes, got " + std::to_string(raw_bytes.size()));
  }

  // Little-endian: read 4 doubles
  Quaternion q{};
  std::memcpy(&q.w, raw_bytes.data() + 0, 8);
  std::memcpy(&q.x, raw_bytes.data() + 8, 8);
  std::memcpy(&q.y, raw_bytes.data() + 16, 8);
  std::memcpy(&q.z, raw_bytes.data() + 24, 8);

  return q;
}

// ============================================================================
// Quaternion implementation
// ============================================================================

auto Quaternion::to_rotation_matrix() const -> Eigen::Matrix3d {
  Eigen::Matrix3d R;

  double x2 = x * x, y2 = y * y, z2 = z * z;
  double wx = w * x, wy = w * y, wz = w * z;
  double xy = x * y, xz = x * z, yz = y * z;

  R(0, 0) = 1.0 - 2.0 * (y2 + z2);
  R(0, 1) = 2.0 * (xy - wz);
  R(0, 2) = 2.0 * (xz + wy);
  R(1, 0) = 2.0 * (xy + wz);
  R(1, 1) = 1.0 - 2.0 * (x2 + z2);
  R(1, 2) = 2.0 * (yz - wx);
  R(2, 0) = 2.0 * (xz - wy);
  R(2, 1) = 2.0 * (yz + wx);
  R(2, 2) = 1.0 - 2.0 * (x2 + y2);

  return R;
}

auto Quaternion::is_unit(double epsilon) const -> bool {
  double norm = w * w + x * x + y * y + z * z;
  return std::abs(norm - 1.0) < epsilon;
}

}  // namespace rm_autoaim