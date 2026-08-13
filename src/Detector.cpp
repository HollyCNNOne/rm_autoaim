#include "rm_autoaim/Detector.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace rm_autoaim {

// ============================================================================
// Public API
// ============================================================================

auto Detector::detect(const cv::Mat& bgr_image) -> std::vector<Armor2D> {
  std::vector<Armor2D> results;

  if (bgr_image.empty()) {
    return results;
  }

  // Step 1: Color separation (HSV)
  cv::Mat diff = extract_color(bgr_image);

  // Simple ROI: only process the upper half of the image
  // The outpost tower is in the upper portion; ignoring the ground
  // significantly reduces false positives
  auto roi_h = bgr_image.rows / 2;
  cv::Rect roi_rect(0, 0, bgr_image.cols, roi_h);
  cv::Mat diff_roi = diff(roi_rect);

  // Step 2: Morphology
  cv::Mat clean = apply_morphology(diff_roi);

  // Step 3: Contours
  auto contours = extract_contours(clean);

  // Step 4: Light bar filtering
  auto light_bars = filter_light_bars(contours);

  // Step 5: Pairing
  auto pairs = pair_light_bars(light_bars);

  // Step 6: Corner extraction
  for (const auto& pair : pairs) {
    Armor2D armor;
    armor.corners = extract_corners(pair);
    armor.confidence = 1.0F;  // base confidence, can be refined
    results.push_back(armor);
  }

  // Debug: log every 30 frames
  {
    static int64_t debug_counter = 0;
    if (++debug_counter % 30 == 0) {
      auto non_zero = cv::countNonZero(diff_roi);
      spdlog::info("Frame #{}: HSV non-zero={}, contours={}, light_bars={}, pairs={}, armors={}",
                    debug_counter, non_zero, contours.size(), light_bars.size(),
                    pairs.size(), results.size());
    }
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
// Step 1: Color separation via HSV thresholding
// ============================================================================

auto Detector::extract_color(const cv::Mat& bgr) const -> cv::Mat {
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

  cv::Mat mask;

  if (detect_red_) {
    // Orange / amber light bars (outpost)
    // OpenCV H range: 0-179; red wraps around 0/180
    // Strategy: union of red wrap-around (low H) + orange (mid H) + red wrap-around (high H)

    // Range 1: Red wrap-around — low H end (H: 0-8)
    cv::Mat mask1;
    cv::inRange(hsv,
                cv::Scalar(0, 60, 100),
                cv::Scalar(8, 255, 255),
                mask1);

    // Range 2: Orange / amber core (H: 5-30)
    cv::Mat mask2;
    cv::inRange(hsv,
                cv::Scalar(5, 60, 100),
                cv::Scalar(30, 255, 255),
                mask2);

    // Range 3: Red wrap-around — high H end (H: 170-179)
    cv::Mat mask3;
    cv::inRange(hsv,
                cv::Scalar(170, 60, 100),
                cv::Scalar(179, 255, 255),
                mask3);

    // Union of all three ranges
    cv::bitwise_or(mask1, mask2, mask);
    cv::bitwise_or(mask, mask3, mask);
  } else {
    // Blue light bars (friendly)
    cv::inRange(hsv,
                cv::Scalar(90, 60, 100),
                cv::Scalar(135, 255, 255),
                mask);
  }

  return mask;  // already binary
}

// ============================================================================
// Step 2: Morphological Operations
// ============================================================================

auto Detector::apply_morphology(const cv::Mat& binary) -> cv::Mat {
  cv::Mat result;

  // Step 3a: Opening (remove noise)
  auto kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::morphologyEx(binary, result, cv::MORPH_OPEN, kernel_open);

  // Step 3b: Closing (connect broken light bars)
  auto kernel_close =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
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
// Step 4: Light Bar Filtering
// ============================================================================

auto Detector::filter_light_bars(
    const std::vector<std::vector<cv::Point>>& contours)
    -> std::vector<LightBar> {
  std::vector<LightBar> candidates;

  for (const auto& contour : contours) {
    // Skip contours with too few points (can't fit a rotated rect)
    if (contour.size() < 6) {
      continue;
    }

    auto rect = cv::minAreaRect(contour);

    // Normalize: width = short side, height = long side
    auto w = std::min(rect.size.width, rect.size.height);
    auto h = std::max(rect.size.width, rect.size.height);

    // Aspect ratio check
    auto ratio = w / h;
    if (ratio < constants::kMinAspectRatio || ratio > constants::kMaxAspectRatio) {
      continue;
    }

    // Area check
    auto area = cv::contourArea(contour);
    if (area < constants::kMinArea || area > constants::kMaxArea) {
      continue;
    }

    // Convexity check
    auto rect_area = w * h;
    auto convexity = (rect_area > 0.0F) ? (area / rect_area) : 0.0F;
    if (convexity < constants::kMinConvexity) {
      continue;
    }

    // Adjust angle: minAreaRect returns angle in [-90, 0)
    // Normalize to long-side orientation:
    //   - If original width < height (tall bar), angle += 90
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
// Step 5: Light Bar Pairing
// ============================================================================

auto Detector::pair_light_bars(const std::vector<LightBar>& candidates)
    -> std::vector<ArmorPair> {
  std::vector<ArmorPair> pairs;

  const size_t n = candidates.size();
  if (n < 2) {
    return pairs;
  }

  // Diagnostic: log light bar details when pairing is attempted
  static int pair_diag_counter = 0;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      const auto& a = candidates[i];
      const auto& b = candidates[j];

      // Constraint 1: Determine left/right by x-coordinate
      const auto& left = (a.center.x < b.center.x) ? a : b;
      const auto& right = (a.center.x < b.center.x) ? b : a;

      // Capture all constraint values for diagnostics
      auto h_min = std::min(left.height, right.height);
      auto h_max = std::max(left.height, right.height);
      auto h_ratio = (h_max > 1e-6F) ? (h_min / h_max) : 0.0F;

      auto ang_diff = std::abs(left.angle - right.angle);
      if (ang_diff > 180.0F) {
        ang_diff = 360.0F - ang_diff;
      }
      auto avg_h = (left.height + right.height) / 2.0F;
      auto dist = right.center.x - left.center.x;
      auto x_ratio = (avg_h > 1e-6F) ? (dist / avg_h) : 0.0F;
      auto y_diff = std::abs(left.center.y - right.center.y);
      auto y_ratio = (avg_h > 1e-6F) ? (y_diff / avg_h) : 0.0F;

      // Log diagnostics every 30 pair attempts
      if (++pair_diag_counter % 30 == 0) {
        spdlog::info("Pair diag: L(h={:.1f},w={:.1f},ang={:.1f},a={:.1f}) "
                      "R(h={:.1f},w={:.1f},ang={:.1f},a={:.1f}) "
                      "h_ratio={:.2f} ang_diff={:.1f} x_ratio={:.2f} y_ratio={:.2f}",
                      left.height, left.width, left.angle, left.area,
                      right.height, right.width, right.angle, right.area,
                      h_ratio, ang_diff, x_ratio, y_ratio);
      }

      // Constraint 2: Height similarity
      if (h_max < 1e-6F) {
        continue;
      }
      if (h_ratio < constants::kMinHeightRatio) {
        continue;
      }

      // Constraint 3: Parallelism
      if (ang_diff > constants::kMaxAngleDiff &&
          std::abs(ang_diff - 180.0F) > constants::kMaxAngleDiff) {
        continue;
      }

      // Constraint 4: Horizontal spacing
      if (avg_h < 1e-6F) {
        continue;
      }
      if (x_ratio < constants::kMinXDistanceRatio ||
          x_ratio > constants::kMaxXDistanceRatio) {
        continue;
      }

      // Constraint 5: Vertical offset
      if (y_diff > constants::kMaxYOffsetRatio * avg_h) {
        continue;
      }

      // All constraints passed
      static int pass_counter = 0;
      if (++pass_counter % 5 == 0) {
        spdlog::info("Pair PASS: L(h={:.1f},w={:.1f},ang={:.1f}) R(h={:.1f},w={:.1f},ang={:.1f}) "
                      "h_ratio={:.2f} ang_diff={:.1f} x_ratio={:.2f} y_ratio={:.2f}",
                      left.height, left.width, left.angle,
                      right.height, right.width, right.angle,
                      h_ratio, ang_diff, x_ratio, y_ratio);
      }
      pairs.push_back({left, right});
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

  // Order: [TL, TR, BR, BL]
  return {left_top, right_top, right_bottom, left_bottom};
}

auto Detector::get_endpoints(const cv::RotatedRect& rect)
    -> std::pair<cv::Point2f, cv::Point2f> {
  cv::Point2f pts[4];
  rect.points(pts);

  // Sort by y-coordinate to find top and bottom pairs
  std::sort(pts, pts + 4,
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

  // Top midpoint = average of the two points with smallest y
  auto top = (pts[0] + pts[1]) * 0.5F;
  // Bottom midpoint = average of the two points with largest y
  auto bottom = (pts[2] + pts[3]) * 0.5F;

  return {top, bottom};
}

}  // namespace rm_autoaim