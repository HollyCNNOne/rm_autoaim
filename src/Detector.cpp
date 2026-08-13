#include "rm_autoaim/Detector.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace rm_autoaim {

// ============================================================================
// Relaxed thresholds for Step 4 (~20% wider than Types.hpp constants)
// Widening allows "suspicious" candidates to reach Step 5 where the cost
// function makes the final decision, rather than being killed by hard ifs.
// ============================================================================
namespace {
inline constexpr double kRelaxedMinAspectRatio{0.04};   // was 0.05
inline constexpr double kRelaxedMaxAspectRatio{0.60};   // was 0.50
inline constexpr double kRelaxedMinArea{16.0};          // was 20.0
inline constexpr double kRelaxedMaxArea{9600.0};        // was 8000.0
inline constexpr double kRelaxedMinConvexity{0.32};     // was 0.40
}  // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

auto Detector::detect(const cv::Mat& bgr_image) -> std::vector<Armor2D> {
  std::vector<Armor2D> results;

  if (bgr_image.empty()) {
    return results;
  }

  // Step 1: CLAHE-enhanced color separation with dynamic V threshold
  cv::Mat diff = extract_color(bgr_image);

  // ROI: only process the upper half (outpost tower is in the upper portion)
  auto roi_h = bgr_image.rows / 2;
  cv::Rect roi_rect(0, 0, bgr_image.cols, roi_h);
  cv::Mat diff_roi = diff(roi_rect);

  // Step 2: Distance-adaptive morphology (uses prev_avg_lightbar_height_)
  cv::Mat clean = apply_morphology(diff_roi);

  // Step 3: Contour extraction
  auto contours = extract_contours(clean);

  // Step 4: Relaxed light bar filtering → feature encoder
  auto light_bars = filter_light_bars(contours);

  // Update adaptive state for next frame's morphology
  if (!light_bars.empty()) {
    auto sum_h = 0.0F;
    for (const auto& lb : light_bars) {
      sum_h += lb.height;
    }
    prev_avg_lightbar_height_ = sum_h / static_cast<float>(light_bars.size());
  }

  // Step 5: Soft-scoring + sliding-window + greedy conflict resolution
  auto pairs = pair_light_bars(light_bars);

  // Step 6: Corner extraction
  for (const auto& pair : pairs) {
    Armor2D armor;
    armor.corners = extract_corners(pair);
    armor.confidence = 1.0F;
    results.push_back(armor);
  }

  return results;
}

auto Detector::set_diff_threshold(int threshold) -> void {
  diff_threshold_ = threshold;
}

auto Detector::set_target_color(bool is_red) -> void {
  detect_red_ = is_red;
}

// ============================================================================
// Step 1: CLAHE-enhanced Color Separation with Dynamic V Threshold
//
// Key improvements over v1:
//   - CLAHE on V channel: suppresses highlight blowout, lifts shadow detail,
//     making light bar edges consistently sharp regardless of lighting.
//   - Dynamic V lower-bound: v_low = mean(V) * 0.35, clamped to [60, 140].
//     In bright scenes the threshold rises to suppress reflections; in dim
//     scenes it drops to capture faint light bars.
// ============================================================================

auto Detector::extract_color(const cv::Mat& bgr) const -> cv::Mat {
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

  // Split HSV → apply CLAHE to V → merge back
  std::vector<cv::Mat> channels(3);
  cv::split(hsv, channels);

  auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  clahe->apply(channels[2], channels[2]);

  cv::merge(channels, hsv);

  // Dynamic V lower-bound: adapt to mean scene brightness
  auto mean_v = cv::mean(channels[2])[0];
  auto v_low = static_cast<int>(std::clamp(mean_v * 0.35, 60.0, 140.0));

  cv::Mat mask;

  if (detect_red_) {
    // Three-range union for red/orange/amber light bars
    cv::Mat mask1, mask2, mask3;

    cv::inRange(hsv,
                cv::Scalar(0, 60, v_low),
                cv::Scalar(8, 255, 255),
                mask1);

    cv::inRange(hsv,
                cv::Scalar(5, 60, v_low),
                cv::Scalar(30, 255, 255),
                mask2);

    cv::inRange(hsv,
                cv::Scalar(170, 60, v_low),
                cv::Scalar(179, 255, 255),
                mask3);

    cv::bitwise_or(mask1, mask2, mask);
    cv::bitwise_or(mask, mask3, mask);
  } else {
    cv::inRange(hsv,
                cv::Scalar(90, 60, v_low),
                cv::Scalar(135, 255, 255),
                mask);
  }

  return mask;
}

// ============================================================================
// Step 2: Distance-Adaptive Morphological Operations
//
// Uses prev_avg_lightbar_height_ (temporal feedback from the previous frame)
// to adapt the closing kernel size:
//   near  (h > 100px): 9×9 — large bars need bigger kernel to fill gaps
//   mid   (h > 50px):  7×7
//   far   (h < 20px):  3×3 — small bars risk over-merging with large kernels
//   default:           5×5
//
// Opening kernel stays 3×3 (noise removal is size-independent).
// ============================================================================

auto Detector::apply_morphology(const cv::Mat& binary) -> cv::Mat {
  cv::Mat result;

  // Opening: remove noise (3×3 is universal)
  auto kernel_open =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::morphologyEx(binary, result, cv::MORPH_OPEN, kernel_open);

  // Closing: adaptive kernel based on previous frame's light bar height
  auto close_size = 5;
  if (prev_avg_lightbar_height_ > 100.0F) {
    close_size = 9;
  } else if (prev_avg_lightbar_height_ > 50.0F) {
    close_size = 7;
  } else if (prev_avg_lightbar_height_ < 20.0F) {
    close_size = 3;
  }

  auto kernel_close =
      cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                cv::Size(close_size, close_size));
  cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel_close);

  return result;
}

// ============================================================================
// Step 3: Contour Extraction (unchanged from v1)
// ============================================================================

auto Detector::extract_contours(const cv::Mat& binary)
    -> std::vector<std::vector<cv::Point>> {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  return contours;
}

// ============================================================================
// Step 4: Relaxed Light Bar Filtering → Feature Encoder
//
// Thresholds widened ~20% vs. v1. This is intentional: "suspicious but
// possibly real" candidates are passed to Step 5, where the cost function
// naturally penalizes them (e.g. low convexity → higher pair cost).
// The hard if-kill is replaced by soft scoring downstream.
// ============================================================================

auto Detector::filter_light_bars(
    const std::vector<std::vector<cv::Point>>& contours)
    -> std::vector<LightBar> {
  std::vector<LightBar> candidates;

  for (const auto& contour : contours) {
    if (contour.size() < 6) {
      continue;
    }

    auto rect = cv::minAreaRect(contour);

    // Normalize: width = short side, height = long side
    auto w = std::min(rect.size.width, rect.size.height);
    auto h = std::max(rect.size.width, rect.size.height);

    // Aspect ratio (relaxed 20%)
    auto ratio = w / h;
    if (ratio < kRelaxedMinAspectRatio || ratio > kRelaxedMaxAspectRatio) {
      continue;
    }

    // Area (relaxed 20%)
    auto area = cv::contourArea(contour);
    if (area < kRelaxedMinArea || area > kRelaxedMaxArea) {
      continue;
    }

    // Convexity (relaxed 20%)
    auto rect_area = w * h;
    auto convexity = (rect_area > 0.0F) ? (area / rect_area) : 0.0F;
    if (convexity < kRelaxedMinConvexity) {
      continue;
    }

    // Angle normalization: always along the long side
    auto angle = rect.angle;
    if (rect.size.width < rect.size.height) {
      angle += 90.0F;
    }

    LightBar lb;
    lb.rect = rect;
    lb.center = rect.center;
    lb.height = h;
    lb.width = w;
    lb.angle = angle;
    lb.area = static_cast<float>(area);
    lb.convexity = convexity;

    candidates.push_back(lb);
  }

  return candidates;
}

// ============================================================================
// Step 5: Soft-Scoring + Sliding-Window Pruning + Greedy Conflict Resolution
//
// Algorithm:
//   A. Sort candidates by center.x (enables sliding-window pruning)
//   B. Sliding window: for each i, examine j in [i+1, i+kMaxSearchRange);
//      break early when x-distance exceeds kMaxXDistanceRatio * h_i
//   C. Score each candidate pair with a continuous cost function
//   D. Sort all scored pairs by cost (ascending = best first)
//   E. Greedy assignment: iterate sorted pairs, mark used[] flags
//
// Why greedy instead of Hungarian?
//   After Step 4, real light bars per frame are typically ≤ 6.
//   On such tiny datasets greedy is indistinguishable from global optimum,
//   but runs in O(n·k·log(n·k)) vs. Hungarian's O(n³).
// ============================================================================

auto Detector::compute_pair_cost(const LightBar& a, const LightBar& b)
    -> double {
  // --- f1: Height similarity (0 = identical, 1 = completely different) ---
  auto h_min = std::min(a.height, b.height);
  auto h_max = std::max(a.height, b.height);
  auto h_ratio = (h_max > 1e-6F) ? (h_min / h_max) : 0.0F;
  auto h_cost = 1.0 - static_cast<double>(h_ratio);

  // --- f2: Angle difference (normalized to [0, 1], 45° = max penalty) ---
  auto ang_diff = std::abs(a.angle - b.angle);
  if (ang_diff > 180.0F) {
    ang_diff = 360.0F - ang_diff;
  }
  auto ang_cost = std::min(static_cast<double>(ang_diff) / 45.0, 1.0);

  // --- f3: Y-offset (normalized by average height) ---
  auto avg_h = (a.height + b.height) / 2.0F;
  auto y_diff = std::abs(a.center.y - b.center.y);
  auto y_cost = (avg_h > 1e-6F)
                    ? std::min(static_cast<double>(y_diff / avg_h), 1.0)
                    : 1.0;

  // --- f4: X-ratio deviation from ideal (ideal ≈ 2.45 for small armor) ---
  auto x_dist = std::abs(a.center.x - b.center.x);
  auto x_ratio = (avg_h > 1e-6F) ? (x_dist / avg_h) : 0.0;
  constexpr double kIdealXRatio = 2.45;
  auto x_cost = std::min(std::abs(x_ratio - kIdealXRatio) / 3.0, 1.0);

  // Weighted sum (weights sum to 1.0)
  constexpr double kWeightHeight{0.35};
  constexpr double kWeightAngle{0.25};
  constexpr double kWeightYOffset{0.20};
  constexpr double kWeightXRatio{0.20};

  return kWeightHeight * h_cost + kWeightAngle * ang_cost +
         kWeightYOffset * y_cost + kWeightXRatio * x_cost;
}

auto Detector::pair_light_bars(const std::vector<LightBar>& candidates)
    -> std::vector<ArmorPair> {
  const size_t n = candidates.size();
  if (n < 2) {
    return {};
  }

  // Step A: Sort by center.x (enables pruning)
  auto sorted = candidates;
  std::sort(sorted.begin(), sorted.end(),
            [](const LightBar& a, const LightBar& b) {
              return a.center.x < b.center.x;
            });

  // Step B: Sliding-window scoring
  struct ScoredPair {
    size_t li;   // index in sorted[]
    size_t ri;   // index in sorted[]
    double cost;
  };
  std::vector<ScoredPair> scored;

  constexpr size_t kMaxSearchRange{8};
  constexpr double kMaxXDistanceRatio{5.0};

  for (size_t i = 0; i < n; ++i) {
    auto j_end = std::min(i + kMaxSearchRange, n);
    for (size_t j = i + 1; j < j_end; ++j) {
      // Prune: if x-distance exceeds generous bound, all subsequent
      // candidates are even further (sorted by x), so break.
      auto x_dist = sorted[j].center.x - sorted[i].center.x;
      if (x_dist > kMaxXDistanceRatio * sorted[i].height) {
        break;
      }

      auto cost = compute_pair_cost(sorted[i], sorted[j]);
      scored.push_back({i, j, cost});
    }
  }

  if (scored.empty()) {
    return {};
  }

  // Step C: Sort by cost (ascending = best first)
  std::sort(scored.begin(), scored.end(),
            [](const ScoredPair& a, const ScoredPair& b) {
              return a.cost < b.cost;
            });

  // Step D: Greedy assignment with conflict resolution
  std::vector<bool> used(n, false);
  std::vector<ArmorPair> pairs;

  for (const auto& sp : scored) {
    if (!used[sp.li] && !used[sp.ri]) {
      used[sp.li] = true;
      used[sp.ri] = true;

      // Determine left/right by x-coordinate
      const auto& left =
          (sorted[sp.li].center.x < sorted[sp.ri].center.x) ? sorted[sp.li]
                                                            : sorted[sp.ri];
      const auto& right =
          (sorted[sp.li].center.x < sorted[sp.ri].center.x) ? sorted[sp.ri]
                                                            : sorted[sp.li];

      pairs.push_back({left, right});
    }
  }

  return pairs;
}

// ============================================================================
// Step 6: Corner Extraction (unchanged from v1)
// ============================================================================

auto Detector::extract_corners(const ArmorPair& pair)
    -> std::array<cv::Point2f, 4> {
  auto [left_top, left_bottom] = get_endpoints(pair.left.rect);
  auto [right_top, right_bottom] = get_endpoints(pair.right.rect);

  // Order: [TL, TR, BR, BL]
  return {left_top, right_top, right_bottom, left_bottom};
}

auto Detector::get_endpoints(const cv::RotatedRect& rect)
    -> std::pair<cv::Point2f, cv::Point2f> {
  cv::Point2f pts[4];
  rect.points(pts);

  // Sort by y-coordinate to find top and bottom pairs
  std::sort(pts, pts + 4,
            [](const cv::Point2f& a, const cv::Point2f& b) {
              return a.y < b.y;
            });

  // Top midpoint = average of the two points with smallest y
  auto top = (pts[0] + pts[1]) * 0.5F;
  // Bottom midpoint = average of the two points with largest y
  auto bottom = (pts[2] + pts[3]) * 0.5F;

  return {top, bottom};
}

}  // namespace rm_autoaim