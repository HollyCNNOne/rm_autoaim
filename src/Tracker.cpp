#include "rm_autoaim/Tracker.hpp"
#include "rm_autoaim/internal/Hungarian.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>

namespace rm_autoaim {

// ============================================================================
// Construction
// ============================================================================

Tracker::Tracker()
    : camera_matrix_(make_camera_matrix())
    , dist_coeffs_(make_dist_coeffs())
    , model_points_(make_armor_model_points()) {
  for (int i = 0; i < kMaxSlots; ++i) {
    free_slots_.push(i);
  }
}

// ============================================================================
// Public API
// ============================================================================

auto Tracker::update(const std::vector<Armor2D>& detections)
    -> std::vector<TrackedArmor> {
  // ============================================================
  // Phase 1: Countdown — decrement lease for all active slots
  // ============================================================
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_active_[i]) {
      slot_timeout_[i]--;
      if (slot_timeout_[i] <= 0) {
        slot_active_[i] = false;
        free_slots_.push(i);
      }
    }
  }

  // ============================================================
  // Phase 2: Build active slot list for Hungarian input
  // ============================================================
  std::vector<TrackedArmor> active_tracks;
  std::vector<int> active_to_slot;
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_active_[i]) {
      active_tracks.push_back(slots_[i]);
      active_to_slot.push_back(i);
    }
  }

  // ============================================================
  // Phase 3: Hungarian matching (pure IoU)
  // ============================================================
  auto [matches, unmatched_det, unmatched_trk] =
      associate(detections, active_tracks);

  // ============================================================
  // Phase 4: Execute matches — renew lease, update PnP
  // ============================================================
  for (const auto& [det_idx, trk_idx] : matches) {
    int slot_idx = active_to_slot[trk_idx];
    auto& slot = slots_[slot_idx];
    slot.detection = detections[det_idx];
    slot.pose = solve_pnp(detections[det_idx]);
    slot.status = TrackedArmor::Status::kConfirmed;
    slot_timeout_[slot_idx] = kTimeoutFrames;
  }

  // ============================================================
  // Phase 5: New detections → assign free slots from queue
  // ============================================================
  for (int det_idx : unmatched_det) {
    if (free_slots_.empty()) {
      continue;
    }
    int new_slot = free_slots_.front();
    free_slots_.pop();

    auto& slot = slots_[new_slot];
    slot.id = new_slot;
    slot.detection = detections[det_idx];
    slot.pose = solve_pnp(detections[det_idx]);
    slot.status = TrackedArmor::Status::kConfirmed;
    slot_active_[new_slot] = true;
    slot_timeout_[new_slot] = kTimeoutFrames;
  }

  // ============================================================
  // Phase 6: Unmatched tracks → Phase 1 countdown handles timeout
  // ============================================================

  // Return all active slots
  std::vector<TrackedArmor> active;
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_active_[i]) {
      active.push_back(slots_[i]);
    }
  }
  return active;
}

auto Tracker::tracks() const -> std::vector<TrackedArmor> {
  std::vector<TrackedArmor> result;
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_active_[i]) {
      result.push_back(slots_[i]);
    }
  }
  return result;
}

auto Tracker::reset() -> void {
  for (int i = 0; i < kMaxSlots; ++i) {
    slot_active_[i] = false;
    slot_timeout_[i] = 0;
  }
  free_slots_ = std::queue<int>();
  for (int i = 0; i < kMaxSlots; ++i) {
    free_slots_.push(i);
  }
}

// ============================================================================
// PnP Pose Estimation (UNCHANGED)
// ============================================================================

auto Tracker::solve_pnp(const Armor2D& detection) -> ArmorPose {
  ArmorPose pose;

  std::vector<cv::Point2f> image_points(detection.corners.begin(),
                                        detection.corners.end());

  cv::Mat rvec, tvec;

  bool success = cv::solvePnP(model_points_, image_points, camera_matrix_,
                              dist_coeffs_, rvec, tvec, false,
                              cv::SOLVEPNP_IPPE);

  if (!success) {
    cv::solvePnP(model_points_, image_points, camera_matrix_, dist_coeffs_,
                 rvec, tvec, false, cv::SOLVEPNP_EPNP);
  }

  cv::Mat rot_mat;
  cv::Rodrigues(rvec, rot_mat);

  pose.translation =
      Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      pose.rotation(i, j) = rot_mat.at<double>(i, j);
    }
  }

  pose.depth = pose.translation.z();

  Eigen::Quaterniond q(pose.rotation);
  pose.quaternion = q;

  return pose;
}

// ============================================================================
// Data Association — Hungarian Algorithm (mixed cost: 60% IoU + 40% center distance)
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

  // Pre-compute detection centers
  std::vector<cv::Point2f> det_centers(n_det);
  for (int i = 0; i < n_det; ++i) {
    const auto& c = detections[i].corners;
    det_centers[i] = {(c[0].x + c[1].x + c[2].x + c[3].x) / 4.0F,
                      (c[0].y + c[1].y + c[2].y + c[3].y) / 4.0F};
  }

  // Pre-compute track centers
  std::vector<cv::Point2f> trk_centers(n_trk);
  for (int j = 0; j < n_trk; ++j) {
    const auto& c = tracks[j].detection.corners;
    trk_centers[j] = {(c[0].x + c[1].x + c[2].x + c[3].x) / 4.0F,
                      (c[0].y + c[1].y + c[2].y + c[3].y) / 4.0F};
  }

  // Build mixed cost matrix: 60% IoU + 40% center distance
  // Center distance helps maintain track identity during rotation
  // when IoU drops due to armor plate aspect ratio change
  internal::Hungarian::CostMatrix cost(n_det, n_trk);
  constexpr double kMaxCenterDist{150.0};
  for (int i = 0; i < n_det; ++i) {
    for (int j = 0; j < n_trk; ++j) {
      double iou = compute_iou(detections[i], tracks[j].detection);
      double iou_cost = 1.0 - iou;
      double dx = static_cast<double>(det_centers[i].x - trk_centers[j].x);
      double dy = static_cast<double>(det_centers[i].y - trk_centers[j].y);
      double center_dist = std::sqrt(dx * dx + dy * dy);
      double center_cost = std::min(center_dist / kMaxCenterDist, 1.0);
      cost(i, j) = 0.6 * iou_cost + 0.4 * center_cost;
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
// IoU Computation (UNCHANGED)
// ============================================================================

auto Tracker::compute_iou(const Armor2D& a, const Armor2D& b) -> double {
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