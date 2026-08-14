#pragma once

#include "rm_autoaim/Types.hpp"

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
namespace rm_autoaim {

// ============================================================================
// Detector — Module 2: Armor Plate Detection
//
// Detects outpost armor plates from a single BGR image frame.
//
// Optimized Pipeline (v2.1):
//   1. CLAHE-enhanced HSV thresholding with dynamic V lower-bound
//   2. Distance-adaptive morphological operations (temporal feedback)
//   3. Contour extraction
//   4. Relaxed light bar filtering → feature encoder (passes more
//      candidates to Step 5 with per-bar feature vectors)
//   5. Soft-scoring + sliding-window pruning + greedy conflict resolution
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
// Debug visualization: output annotated video
  auto enable_debug_viz(const std::string& output_path) -> void;
  auto disable_debug_viz() -> void;
private:
  // ========================================================================
  // Sub-steps
  // ========================================================================

  // Step 1: CLAHE-enhanced color separation with dynamic V threshold
  [[nodiscard]] auto extract_color(const cv::Mat& bgr) const -> cv::Mat;

  // Step 2: Distance-adaptive morphological operations
  // Uses prev_avg_lightbar_height_ to adapt the closing kernel size:
  //   near (h > 100px) → 9×9;  mid (h > 50px) → 7×7;  far (h < 20px) → 3×3
  [[nodiscard]] auto apply_morphology(const cv::Mat& binary) -> cv::Mat;

  // Step 3: Contour extraction
  [[nodiscard]] static auto extract_contours(const cv::Mat& binary)
      -> std::vector<std::vector<cv::Point>>;

  // Step 4: Relaxed light bar filtering → feature encoder
  // Thresholds widened ~20% vs. original; passes "suspicious" candidates
  // to Step 5 so the cost function can make the final decision.
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

  // Step 5: Soft-scoring + sliding-window + greedy conflict resolution
  //
  // Replaces hard if-thresholds with a continuous cost function:
  //   Score = w1·f(Height) + w2·f(Angle) + w3·f(Y_offset) + w4·f(X_ratio)
  //
  // Algorithm:
  //   A. Sort candidates by center.x
  //   B. Sliding window: for each i, examine j in [i+1, i+max_search_range),
  //      break early when x-distance exceeds max_allowed_distance
  //   C. Sort all scored pairs by cost (ascending)
  //   D. Greedy assignment with conflict resolution (used[] flag array)
  struct ArmorPair {
    LightBar left;
    LightBar right;
    double cost{0.0};
  };

[[nodiscard]] static auto pair_light_bars(
      const std::vector<LightBar>& candidates) -> std::vector<ArmorPair>;

  // Cost function: lower = better pairing
  // Weights: height(0.35) + angle(0.25) + y-offset(0.20) + x-ratio(0.20)
  [[nodiscard]] static auto compute_pair_cost(const LightBar& a,
                                              const LightBar& b) -> double;

  // Step 6: Extract 4 corners from paired light bars
  [[nodiscard]] static auto extract_corners(const ArmorPair& pair)
      -> std::array<cv::Point2f, 4>;

  // Helper: get light bar endpoints (top-center, bottom-center)
  [[nodiscard]] static auto get_endpoints(const cv::RotatedRect& rect)
      -> std::pair<cv::Point2f, cv::Point2f>;

// Debug visualization  ← 放这里，在 ArmorPair 定义之后
  auto draw_debug_frame(const cv::Mat& bgr,
                        const std::vector<LightBar>& light_bars,
                        const std::vector<ArmorPair>& pairs) -> void;

  // Parameters
  int diff_threshold_{80};
  bool detect_red_{true};  // default: detect red armor (enemy)

  // Temporal feedback for adaptive morphology (Step 2)
  // Updated every frame from the average height of filtered light bars.
  float prev_avg_lightbar_height_{80.0F};  // conservative init: default 5×5 kernel
 float prev_avg_armor_long_edge_{0.0F}; 
// Debug members
  cv::VideoWriter debug_writer_;
  std::string debug_viz_path_;
  bool debug_viz_enabled_{false};
};

}  // namespace rm_autoaim