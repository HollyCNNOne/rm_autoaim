#include "rm_autoaim/Detector.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace rm_autoaim {

// Relaxed thresholds for Step 4 (~5% wider than Types.hpp constants)
namespace {
inline constexpr double kRelaxedMinAspectRatio{0.04};
inline constexpr double kRelaxedMaxAspectRatio{0.60};
inline constexpr double kRelaxedMinArea{48.0};
inline constexpr double kRelaxedMaxArea{9600.0};
inline constexpr double kRelaxedMinConvexity{0.50};
}  // anonymous namespace

auto Detector::detect(const cv::Mat& bgr_image) -> std::vector<Armor2D> {
  std::vector<Armor2D> results;

  if (bgr_image.empty()) {
    return results;
  }

  auto roi_h = bgr_image.rows / 2;
  cv::Rect roi_rect(0, 0, bgr_image.cols, roi_h);
  cv::Mat bgr_roi = bgr_image(roi_rect);
  cv::Mat diff = extract_color(bgr_roi);

  cv::Mat clean = apply_morphology(diff);

  auto contours = extract_contours(clean);

  auto light_bars = filter_light_bars(contours);

  if (!light_bars.empty()) {
    auto sum_h = 0.0F;
    for (const auto& lb : light_bars) {
      sum_h += lb.height;
    }
    prev_avg_lightbar_height_ = sum_h / static_cast<float>(light_bars.size());
  }

  auto pairs = pair_light_bars(light_bars);

  // Temporal long-edge validation: reject sudden size jumps caused by
  // cross-panel mis-pairing (yellow stripes etc.)
  constexpr float kEmaAlpha{0.15F};

  for (const auto& pair : pairs) {
    auto corners = extract_corners(pair);

    auto v1 = corners[1] - corners[0];
    auto v2 = corners[3] - corners[0];
    auto w = static_cast<float>(cv::norm(v1));
    auto h = static_cast<float>(cv::norm(v2));
    auto aspect = (h > 1e-4F) ? (w / h) : 0.0F;
    if (aspect < 1.2F || aspect > 4.5F) continue;

    auto top_edge = static_cast<float>(cv::norm(corners[1] - corners[0]));
    auto bot_edge = static_cast<float>(cv::norm(corners[2] - corners[3]));
    auto long_edge = std::max(top_edge, bot_edge);

    if (prev_avg_armor_long_edge_ > 1e-4F) {
      auto ratio = long_edge / prev_avg_armor_long_edge_;
      if (ratio > 1.5F) continue;
    }

    if (prev_avg_armor_long_edge_ < 1e-4F) {
      prev_avg_armor_long_edge_ = long_edge;
    } else {
      prev_avg_armor_long_edge_ = kEmaAlpha * long_edge
        + (1.0F - kEmaAlpha) * prev_avg_armor_long_edge_;
    }

    Armor2D armor;
    armor.corners = corners;
    armor.confidence = 1.0F;
    results.push_back(armor);
  }

  return results;
}

auto Detector::detect_debug(const cv::Mat& bgr_image)
    -> std::pair<std::vector<Armor2D>, DetectorDebugInfo> {
  std::vector<Armor2D> results;
  DetectorDebugInfo debug;

  if (bgr_image.empty()) {
    return {results, debug};
  }

  auto roi_h = bgr_image.rows / 2;
  cv::Rect roi_rect(0, 0, bgr_image.cols, roi_h);
  cv::Mat bgr_roi = bgr_image(roi_rect);
  cv::Mat diff = extract_color(bgr_roi);

  cv::Mat clean = apply_morphology(diff);

  auto contours = extract_contours(clean);

  auto light_bars = filter_light_bars(contours);

  if (!light_bars.empty()) {
    auto sum_h = 0.0F;
    for (const auto& lb : light_bars) {
      sum_h += lb.height;
    }
    prev_avg_lightbar_height_ = sum_h / static_cast<float>(light_bars.size());
  }

  auto pairs = pair_light_bars(light_bars);

  constexpr float kEmaAlpha{0.15F};

  for (const auto& pair : pairs) {
    auto corners = extract_corners(pair);

    auto v1 = corners[1] - corners[0];
    auto v2 = corners[3] - corners[0];
    auto w = static_cast<float>(cv::norm(v1));
    auto h = static_cast<float>(cv::norm(v2));
    auto aspect = (h > 1e-4F) ? (w / h) : 0.0F;
    if (aspect < 1.2F || aspect > 4.5F) continue;

    auto top_edge = static_cast<float>(cv::norm(corners[1] - corners[0]));
    auto bot_edge = static_cast<float>(cv::norm(corners[2] - corners[3]));
    auto long_edge = std::max(top_edge, bot_edge);

    if (prev_avg_armor_long_edge_ > 1e-4F) {
      auto ratio = long_edge / prev_avg_armor_long_edge_;
      if (ratio > 1.5F) continue;
    }

    if (prev_avg_armor_long_edge_ < 1e-4F) {
      prev_avg_armor_long_edge_ = long_edge;
    } else {
      prev_avg_armor_long_edge_ = kEmaAlpha * long_edge
        + (1.0F - kEmaAlpha) * prev_avg_armor_long_edge_;
    }

    Armor2D armor;
    armor.corners = corners;
    armor.confidence = 1.0F;
    results.push_back(armor);
  }

  debug.light_bars = std::move(light_bars);
  debug.pairs = std::move(pairs);

  return {results, debug};
}

auto Detector::set_diff_threshold(int threshold) -> void {
  diff_threshold_ = threshold;
}

auto Detector::set_target_color(bool is_red) -> void {
  detect_red_ = is_red;
}

// ============================================================================
// Step 1: CLAHE-enhanced Color Separation with Dynamic V Threshold
// ============================================================================

auto Detector::extract_color(const cv::Mat& bgr) const -> cv::Mat {
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

  std::vector<cv::Mat> channels(3);
  cv::split(hsv, channels);

  // Compute V-channel mean BEFORE CLAHE (raw lighting)
  auto mean_v = cv::mean(channels[2])[0];
  auto v_low = static_cast<int>(std::clamp(mean_v * 0.35, 60.0, 140.0));

  // Apply CLAHE to V channel for local contrast enhancement
  auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  clahe->apply(channels[2], channels[2]);

  cv::merge(channels, hsv);

  cv::Mat mask;

  if (detect_red_) {
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
// ============================================================================

auto Detector::apply_morphology(const cv::Mat& binary) -> cv::Mat {
  cv::Mat result;

  auto kernel_open =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::morphologyEx(binary, result, cv::MORPH_OPEN, kernel_open);

  auto close_size = 5;
  auto effective_h = prev_avg_lightbar_height_;
  if (effective_h > 150.0F) {
    effective_h = 100.0F;
  }

  if (effective_h > 100.0F) {
    close_size = 9;
  } else if (effective_h > 50.0F) {
    close_size = 7;
  } else if (effective_h < 20.0F) {
    close_size = 3;
  }

  auto kernel_close =
      cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                cv::Size(close_size, close_size));
  cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel_close);

  return result;
}

// ============================================================================
// Step 3: Contour Extraction
// ============================================================================

auto Detector::extract_contours(const cv::Mat& binary)
    -> std::vector<std::vector<cv::Point>> {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  return contours;
}

// ============================================================================
// Step 4: Relaxed Light Bar Filtering
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

    auto w = std::min(rect.size.width, rect.size.height);
    auto h = std::max(rect.size.width, rect.size.height);

    auto ratio = w / h;
    if (ratio < kRelaxedMinAspectRatio || ratio > kRelaxedMaxAspectRatio) {
      continue;
    }

    auto area = cv::contourArea(contour);
    if (area < kRelaxedMinArea || area > kRelaxedMaxArea) {
      continue;
    }

    auto rect_area = w * h;
    auto convexity = (rect_area > 0.0F) ? (area / rect_area) : 0.0F;
    if (convexity < kRelaxedMinConvexity) {
      continue;
    }

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
// Step 5: Soft-Scoring + Sliding-Window + Greedy Conflict Resolution
// ============================================================================

auto Detector::compute_pair_cost(const LightBar& a, const LightBar& b)
    -> double {
  // Hard reject: cross-plate pairs have very different bar heights
  auto h_max = std::max(a.height, b.height);
  auto h_min = std::min(a.height, b.height);
  if (h_max > 1e-6F && (h_min / h_max) < 0.65F) {
    return 1.0;
  }

  // Hard reject: bars from same armor plate must have vertical overlap >= 50%
  auto a_top = a.center.y - a.height * 0.5F;
  auto a_bot = a.center.y + a.height * 0.5F;
  auto b_top = b.center.y - b.height * 0.5F;
  auto b_bot = b.center.y + b.height * 0.5F;
  auto overlap_top = std::max(a_top, b_top);
  auto overlap_bot = std::min(a_bot, b_bot);
  auto overlap = overlap_bot - overlap_top;
  if (overlap <= 0.0F) return 1.0;
  auto shorter_h = std::min(a.height, b.height);
  if (overlap / shorter_h < 0.5F) return 1.0;

  auto h_ratio = (h_max > 1e-6F) ? (h_min / h_max) : 0.0F;
  auto h_cost = 1.0 - static_cast<double>(h_ratio);

  auto ang_diff = std::abs(a.angle - b.angle);
  if (ang_diff > 180.0F) {
    ang_diff = 360.0F - ang_diff;
  }
  auto ang_cost = std::min(static_cast<double>(ang_diff) / 45.0, 1.0);

  auto avg_h = (a.height + b.height) / 2.0F;
  auto y_diff = std::abs(a.center.y - b.center.y);
  auto y_cost = (avg_h > 1e-6F)
                    ? std::min(static_cast<double>(y_diff / avg_h), 1.0)
                    : 1.0;

  auto x_dist = std::abs(a.center.x - b.center.x);
  auto x_ratio = (avg_h > 1e-6F) ? (x_dist / avg_h) : 0.0;
  constexpr double kIdealXRatioSmall = 2.45;
  constexpr double kIdealXRatioLarge = 3.0;
  const double ideal_xr =
      (avg_h > 60.0F) ? kIdealXRatioLarge : kIdealXRatioSmall;
  auto x_cost = std::min(std::abs(x_ratio - ideal_xr) / 3.0, 1.0);

  constexpr double kWeightHeight{0.45};
  constexpr double kWeightAngle{0.25};
  constexpr double kWeightYOffset{0.10};
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

  auto sorted = candidates;
  std::sort(sorted.begin(), sorted.end(),
            [](const LightBar& a, const LightBar& b) {
              return a.center.x < b.center.x;
            });

  struct ScoredPair {
    size_t li;
    size_t ri;
    double cost;
  };
  std::vector<ScoredPair> scored;

  constexpr size_t kMaxSearchRange{8};
  constexpr double kMaxXDistanceRatio{3.0};

  for (size_t i = 0; i < n; ++i) {
    auto j_end = std::min(i + kMaxSearchRange, n);
    for (size_t j = i + 1; j < j_end; ++j) {
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

  std::sort(scored.begin(), scored.end(),
            [](const ScoredPair& a, const ScoredPair& b) {
              return a.cost < b.cost;
            });

  std::vector<bool> used(n, false);
  std::vector<ArmorPair> pairs;

  constexpr double kMaxPairCost{0.35};
  for (const auto& sp : scored) {
    if (sp.cost > kMaxPairCost) break;
    if (!used[sp.li] && !used[sp.ri]) {
      used[sp.li] = true;
      used[sp.ri] = true;

      const auto& left =
          (sorted[sp.li].center.x < sorted[sp.ri].center.x) ? sorted[sp.li]
                                                            : sorted[sp.ri];
      const auto& right =
          (sorted[sp.li].center.x < sorted[sp.ri].center.x) ? sorted[sp.ri]
                                                            : sorted[sp.li];

      pairs.push_back({left, right, sp.cost});

      if (pairs.size() >= n / 2) {
        break;
      }
    }
  }

  return pairs;
}

// ============================================================================
// Step 6: Corner Extraction
// ============================================================================

auto Detector::extract_corners(const ArmorPair& pair)
    -> std::array<cv::Point2f, 4> {
  auto [left_top, left_bottom] = get_endpoints(pair.left.rect);
  auto [right_top, right_bottom] = get_endpoints(pair.right.rect);

  return {left_top, right_top, right_bottom, left_bottom};
}

auto Detector::get_endpoints(const cv::RotatedRect& rect)
    -> std::pair<cv::Point2f, cv::Point2f> {
  cv::Point2f pts[4];
  rect.points(pts);

  std::sort(pts, pts + 4,
            [](const cv::Point2f& a, const cv::Point2f& b) {
              return a.y < b.y;
            });

  auto top = (pts[0] + pts[1]) * 0.5F;
  auto bottom = (pts[2] + pts[3]) * 0.5F;

  return {top, bottom};
}

}  // namespace rm_autoaim