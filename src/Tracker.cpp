#include "rm_autoaim/Tracker.hpp"
#include "rm_autoaim/internal/Hungarian.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>

namespace rm_autoaim {

// ============================================================================
// Construction
// ============================================================================

Tracker::Tracker()
    : camera_matrix_(make_camera_matrix())
    , dist_coeffs_(make_dist_coeffs())
    , model_points_(make_armor_model_points()) {}

// ============================================================================
// Public API
// ============================================================================

auto Tracker::update(const std::vector<Armor2D>& detections)
    -> std::vector<TrackedArmor> {
  // Associate detections with existing tracks
  auto [matches, unmatched_det, unmatched_trk] =
      associate(detections, tracks_);

  // Update lifecycle
  update_tracks(detections, matches, unmatched_det, unmatched_trk);

  // Return all confirmed tracks
  std::vector<TrackedArmor> active;
  for (auto& t : tracks_) {
    if (t.status == TrackedArmor::Status::kConfirmed ||
        t.status == TrackedArmor::Status::kLost) {
      active.push_back(t);
    }
  }
  return active;
}

auto Tracker::tracks() const -> const std::vector<TrackedArmor>& {
  return tracks_;
}

auto Tracker::reset() -> void {
  tracks_.clear();
  next_track_id_ = 0;
}

// ============================================================================
// PnP Pose Estimation
// ============================================================================

auto Tracker::solve_pnp(const Armor2D& detection) -> ArmorPose {
  ArmorPose pose;

  // Convert 2D corners to vector<Point2f>
  std::vector<cv::Point2f> image_points(detection.corners.begin(),
                                        detection.corners.end());

  cv::Mat rvec, tvec;

  bool success = cv::solvePnP(model_points_, image_points, camera_matrix_,
                              dist_coeffs_, rvec, tvec, false,
                              cv::SOLVEPNP_IPPE);

  if (!success) {
    // Fallback: try EPnP
    cv::solvePnP(model_points_, image_points, camera_matrix_, dist_coeffs_,
                 rvec, tvec, false, cv::SOLVEPNP_EPNP);
  }

  // Convert rotation vector to matrix
  cv::Mat rot_mat;
  cv::Rodrigues(rvec, rot_mat);

  // Copy to Eigen
  pose.translation =
      Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      pose.rotation(i, j) = rot_mat.at<double>(i, j);
    }
  }

  pose.depth = pose.translation.z();

  // Convert to Eigen quaternion
  Eigen::Quaterniond q(pose.rotation);
  pose.quaternion = q;

  return pose;
}

// ============================================================================
// Data Association (Hungarian Algorithm)
// ============================================================================

auto Tracker::associate(
    const std::vector<Armor2D>& detections,
    const std::vector<TrackedArmor>& tracks)
    -> std::tuple<std::vector<std::pair<int, int>>, std::vector<int>,
                  std::vector<int>> {
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatched_det;
  std::vector<int> unmatched_trk;

  const int n_det = static_cast<int>(detections.size());
  const int n_trk = static_cast<int>(tracks.size());

  if (n_det == 0) {
    for (int i = 0; i < n_trk; ++i) {
      unmatched_trk.push_back(i);
    }
    return {matches, unmatched_det, unmatched_trk};
  }

  if (n_trk == 0) {
    for (int i = 0; i < n_det; ++i) {
      unmatched_det.push_back(i);
    }
    return {matches, unmatched_det, unmatched_trk};
  }

  // Build cost matrix: cost = 1 - IoU
  internal::Hungarian::CostMatrix cost(n_det, n_trk);
  for (int i = 0; i < n_det; ++i) {
    for (int j = 0; j < n_trk; ++j) {
      double iou = compute_iou(detections[i], tracks[j].detection);
      cost(i, j) = 1.0 - iou;
    }
  }

  // Solve assignment
  auto [total_cost, assignments] =
      internal::Hungarian::solve_with_threshold(cost, 1.0 - constants::kIoUMin);

  // Separate matched and unmatched
  std::vector<bool> det_matched(n_det, false);
  std::vector<bool> trk_matched(n_trk, false);

  for (const auto& a : assignments) {
    if (a.row >= 0 && a.col >= 0 && a.row < n_det && a.col < n_trk) {
      matches.emplace_back(a.row, a.col);
      det_matched[a.row] = true;
      trk_matched[a.col] = true;
    }
  }

  for (int i = 0; i < n_det; ++i) {
    if (!det_matched[i]) {
      unmatched_det.push_back(i);
    }
  }
  for (int j = 0; j < n_trk; ++j) {
    if (!trk_matched[j]) {
      unmatched_trk.push_back(j);
    }
  }

  return {matches, unmatched_det, unmatched_trk};
}

// ============================================================================
// Lifecycle Management
// ============================================================================

auto Tracker::update_tracks(
    const std::vector<Armor2D>& detections,
    const std::vector<std::pair<int, int>>& matches,
    const std::vector<int>& unmatched_det,
    const std::vector<int>& unmatched_trk) -> void {
  // Update matched tracks
  for (const auto& [det_idx, trk_idx] : matches) {
    auto& track = tracks_[trk_idx];
    track.detection = detections[det_idx];
    track.pose = solve_pnp(detections[det_idx]);
    track.consecutive_detections++;
    track.consecutive_misses = 0;
    track.age++;

    if (track.consecutive_detections >= constants::kConfirmThreshold) {
      track.status = TrackedArmor::Status::kConfirmed;
    }
  }

  // Create new tracks for unmatched detections
  for (int det_idx : unmatched_det) {
    TrackedArmor new_track;
    new_track.id = next_track_id_++;
    new_track.detection = detections[det_idx];
    new_track.pose = solve_pnp(detections[det_idx]);
    new_track.status = TrackedArmor::Status::kTentative;
    new_track.consecutive_detections = 1;
    new_track.consecutive_misses = 0;
    new_track.age = 0;
    tracks_.push_back(new_track);
  }

  // Update unmatched tracks (lost)
  for (int trk_idx : unmatched_trk) {
    auto& track = tracks_[trk_idx];
    track.consecutive_misses++;
    track.consecutive_detections = 0;
    track.age++;

    if (track.status == TrackedArmor::Status::kTentative) {
      track.status = TrackedArmor::Status::kDead;
    } else if (track.consecutive_misses >= constants::kLostThreshold) {
      track.status = TrackedArmor::Status::kDead;
    } else {
      track.status = TrackedArmor::Status::kLost;
    }
  }

  // Remove dead tracks
  tracks_.erase(
      std::remove_if(tracks_.begin(), tracks_.end(),
                     [](const TrackedArmor& t) {
                       return t.status == TrackedArmor::Status::kDead;
                     }),
      tracks_.end());
}

// ============================================================================
// IoU Computation
// ============================================================================

auto Tracker::compute_iou(const Armor2D& a, const Armor2D& b) -> double {
  // Compute axis-aligned bounding boxes from corners
  auto get_bbox = [](const Armor2D& armor) -> cv::Rect2f {
    float x_min = std::numeric_limits<float>::max();
    float y_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_max = std::numeric_limits<float>::lowest();

    for (const auto& c : armor.corners) {
      x_min = std::min(x_min, c.x);
      y_min = std::min(y_min, c.y);
      x_max = std::max(x_max, c.x);
      y_max = std::max(y_max, c.y);
    }

    return cv::Rect2f(x_min, y_min, x_max - x_min, y_max - y_min);
  };

  auto box_a = get_bbox(a);
  auto box_b = get_bbox(b);

  // Intersection
  float x_inter = std::max(0.0F,
      std::min(box_a.x + box_a.width, box_b.x + box_b.width) -
      std::max(box_a.x, box_b.x));
  float y_inter = std::max(0.0F,
      std::min(box_a.y + box_a.height, box_b.y + box_b.height) -
      std::max(box_a.y, box_b.y));

  float inter_area = x_inter * y_inter;
  float union_area = box_a.area() + box_b.area() - inter_area;

  if (union_area < 1e-6F) {
    return 0.0;
  }

  return static_cast<double>(inter_area / union_area);
}

}  // namespace rm_autoaim