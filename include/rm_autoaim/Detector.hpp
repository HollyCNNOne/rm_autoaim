#pragma once

#include "rm_autoaim/Types.hpp"

#include <vector>

#include <opencv2/core.hpp>

namespace rm_autoaim {

// ============================================================================
// Detector — Module 2: Armor Plate Detection
//
// Detects outpost armor plates from a single BGR image frame.
//
// Pipeline:
//   1. Color separation (HSV thresholding for orange/amber light bars)
//   2. Morphological operations (open + close)
//   3. Contour extraction
//   4. Light bar filtering (aspect ratio, area, convexity)
//   5. Light bar pairing (5 geometric constraints)
//   6. Corner extraction (4 armor corners)
// ============================================================================

class Detector {
public:
  Detector() = default;

  // Detect armor plates in a single frame
  // Returns list of detected armors (4 corners each)
  [[nodiscard]] auto detect(const cv::Mat& bgr_image)
      -> std::vector<Armor2D>;

  // Set detection parameters (useful for tuning)
  auto set_diff_threshold(int threshold) -> void;
  auto set_target_color(bool is_red) -> void;  // true = detect red, false = blue

private:
  // ========================================================================
  // Sub-steps
  // ========================================================================

  // Step 1: Color separation via HSV thresholding (orange/amber light bars)
  [[nodiscard]] auto extract_color(const cv::Mat& bgr) const -> cv::Mat;

  // Step 2: Morphological operations
  [[nodiscard]] static auto apply_morphology(const cv::Mat& binary) -> cv::Mat;

  // Step 3: Contour extraction
  [[nodiscard]] static auto extract_contours(const cv::Mat& binary)
      -> std::vector<std::vector<cv::Point>>;

  // Step 4: Filter light bar candidates
  struct LightBar {
    cv::RotatedRect rect;
    cv::Point2f center;
    float height{0.0F};
    float width{0.0F};
    float angle{0.0F};
    float area{0.0F};
    float convexity{0.0F};
  };

  [[nodiscard]] static auto filter_light_bars(
      const std::vector<std::vector<cv::Point>>& contours)
      -> std::vector<LightBar>;

  // Step 5: Pair light bars into armor plates
  struct ArmorPair {
    LightBar left;
    LightBar right;
  };

  [[nodiscard]] static auto pair_light_bars(
      const std::vector<LightBar>& candidates) -> std::vector<ArmorPair>;

  // Step 6: Extract 4 corners from paired light bars
  [[nodiscard]] static auto extract_corners(const ArmorPair& pair)
      -> std::array<cv::Point2f, 4>;

  // Helper: get light bar endpoints (top-center, bottom-center)
  [[nodiscard]] static auto get_endpoints(const cv::RotatedRect& rect)
      -> std::pair<cv::Point2f, cv::Point2f>;

  // Parameters
  int diff_threshold_{80};
  bool detect_red_{true};  // default: detect red armor (enemy)
};

}  // namespace rm_autoaim