#pragma once

#include "rm_autoaim/Types.hpp"

#include <vector>

#include <opencv2/core.hpp>

namespace rm_autoaim {

// Outpost armor plate detection — 6-step pipeline with soft-scoring pairing
class Detector {
public:
  Detector() = default;

  [[nodiscard]] auto detect(const cv::Mat& bgr_image)
      -> std::vector<Armor2D>;

  struct LightBar {
    cv::RotatedRect rect;
    cv::Point2f center;
    float height{0.0F};
    float width{0.0F};
    float angle{0.0F};
    float area{0.0F};
    float convexity{0.0F};
  };

  struct ArmorPair {
    LightBar left;
    LightBar right;
    double cost{0.0};
  };

  struct DetectorDebugInfo {
    std::vector<LightBar> light_bars;
    std::vector<ArmorPair> pairs;
  };

  [[nodiscard]] auto detect_debug(const cv::Mat& bgr_image)
      -> std::pair<std::vector<Armor2D>, DetectorDebugInfo>;

  auto set_diff_threshold(int threshold) -> void;
  auto set_target_color(bool is_red) -> void;

  [[nodiscard]] static auto extract_corners(const ArmorPair& pair)
      -> std::array<cv::Point2f, 4>;

private:
  [[nodiscard]] auto extract_color(const cv::Mat& bgr) const -> cv::Mat;

  // Adapts closing kernel to distance: near(9×9) → mid(7×7) → far(3×3)
  [[nodiscard]] auto apply_morphology(const cv::Mat& binary) -> cv::Mat;

  [[nodiscard]] static auto extract_contours(const cv::Mat& binary)
      -> std::vector<std::vector<cv::Point>>;

  // Thresholds widened ~20% vs. original; passes suspicious candidates
  // to Step 5 so the cost function makes the final decision
  [[nodiscard]] static auto filter_light_bars(
      const std::vector<std::vector<cv::Point>>& contours)
      -> std::vector<LightBar>;

  // Soft-scoring replaces hard thresholds: cost = w1·f(Height) + w2·f(Angle)
  //   + w3·f(Y_offset) + w4·f(X_ratio)
  // Sliding-window + greedy conflict resolution prevents cross-panel pairing
  [[nodiscard]] static auto pair_light_bars(
      const std::vector<LightBar>& candidates) -> std::vector<ArmorPair>;

  // Weights: height(0.35) + angle(0.25) + y-offset(0.20) + x-ratio(0.20)
  [[nodiscard]] static auto compute_pair_cost(const LightBar& a,
                                              const LightBar& b) -> double;

  [[nodiscard]] static auto get_endpoints(const cv::RotatedRect& rect)
      -> std::pair<cv::Point2f, cv::Point2f>;

  int diff_threshold_{80};
  bool detect_red_{true};

  float prev_avg_lightbar_height_{80.0F};
  float prev_avg_armor_long_edge_{0.0F};
};

}  // namespace rm_autoaim