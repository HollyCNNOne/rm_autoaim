#include "rm_autoaim/Tracker.hpp"
#include "rm_autoaim/internal/Hungarian.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>

namespace rm_autoaim {

Tracker::Tracker()
    : camera_matrix_(make_camera_matrix())
    , dist_coeffs_(make_dist_coeffs())
    , model_points_(make_armor_model_points()) {
  for (int i = 0; i < kMaxSlots; ++i) {
    free_slots_.push(i);
  }
}

auto Tracker::update(const std::vector<Armor2D>& detections)
    -> std::vector<TrackedArmor> {
  frame_counter_++;

  // Record first-seen frame for phase origin
  if (first_seen_frame_ < 0 && !detections.empty()) {
    first_seen_frame_ = frame_counter_;
  }

  // Phase 1: Countdown — decrement lease for all active slots
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_active_[i]) {
      slot_timeout_[i]--;
      if (slot_timeout_[i] <= 0) {
        slot_active_[i] = false;
        slot_release_frame_[i] = frame_counter_;
        free_slots_.push(i);
      }
    }
  }

  // Phase 2: Build active slot list for Hungarian input
  std::vector<TrackedArmor> active_tracks;
  std::vector<int> active_to_slot;
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_active_[i]) {
      active_tracks.push_back(slots_[i]);
      active_to_slot.push_back(i);
    }
  }

  // Phase 3: Hungarian matching (V3.2: phase bonus injected into cost)
  auto [matches, unmatched_det, unmatched_trk] =
      associate(detections, active_tracks);

  // Phase 4: Execute matches — renew lease, update PnP
  for (const auto& [det_idx, trk_idx] : matches) {
    int slot_idx = active_to_slot[trk_idx];
    auto& slot = slots_[slot_idx];
    slot.detection = detections[det_idx];
    slot.pose = solve_pnp(detections[det_idx]);
    slot.status = TrackedArmor::Status::kConfirmed;
    slot_timeout_[slot_idx] = kTimeoutFrames;
  }

  // V3.2: Save current frame's pairings for next frame's inertia bonus
  prev_detections_ = detections;
  prev_det_to_slot_.assign(detections.size(), -1);
  for (const auto& [det_idx, trk_idx] : matches) {
    int slot_idx = active_to_slot[trk_idx];
    prev_det_to_slot_[det_idx] = slot_idx;
  }

  // Phase 5: New detections → assign slots
  // V3.2: if period is locked, try preferred slot first (when idle),
  //       then fall back to queue
  constexpr int kMaxVisibleTargets{2};
  constexpr double kMinArmorAspect{1.2};
  constexpr double kMaxArmorAspect{4.5};

  int active_count = 0;
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_active_[i]) active_count++;
  }

  for (int det_idx : unmatched_det) {
    if (active_count >= kMaxVisibleTargets) continue;

    const auto& corners = detections[det_idx].corners;
    auto w = cv::norm(corners[1] - corners[0]);
    auto h = cv::norm(corners[3] - corners[0]);
    auto aspect = (h > 1e-4F) ? (w / h) : 0.0F;
    if (aspect < kMinArmorAspect || aspect > kMaxArmorAspect) continue;

    int new_slot = -1;

    // V3.2 Layer 2: preferred slot reuse (only when idle)
    if (period_initialized_) {
      auto phase = (frame_counter_ - first_seen_frame_) % measured_period_;
      auto preferred = phase_to_slot(phase);
      if (!slot_active_[preferred]) {
        new_slot = preferred;
        std::queue<int> filtered;
        while (!free_slots_.empty()) {
          auto s = free_slots_.front();
          free_slots_.pop();
          if (s != preferred) {
            filtered.push(s);
          }
        }
        free_slots_ = std::move(filtered);
      }
    }

    // Fallback: queue-based assignment
    if (new_slot == -1) {
      if (free_slots_.empty()) continue;
      new_slot = free_slots_.front();
      free_slots_.pop();
    }

    // V3.2: Period auto-calibration sample
    if (slot_release_frame_[new_slot] > 0) {
      auto sample = frame_counter_ - slot_release_frame_[new_slot];
      slot_release_frame_[new_slot] = 0;

      if (!period_initialized_) {
        period_samples_.push_back(sample);
        if (static_cast<int>(period_samples_.size()) > kMaxPeriodSamples) {
          period_samples_.pop_front();
        }
        if (static_cast<int>(period_samples_.size()) >= kMinSamplesForLock) {
          std::vector<int> sorted(period_samples_.begin(),
                                  period_samples_.end());
          std::sort(sorted.begin(), sorted.end());
          measured_period_ = sorted[sorted.size() / 2];
          period_initialized_ = true;
          spdlog::info("[V3.2] Period locked: {} frames (~{:.2f}s)",
                       measured_period_,
                       measured_period_ / 166.7);
        }
      } else {
        auto ratio = static_cast<double>(sample) / measured_period_;
        if (ratio > 1.0 - kPeriodRejectRatio &&
            ratio < 1.0 + kPeriodRejectRatio) {
          measured_period_ = static_cast<int>(
              kPeriodEmaAlpha * sample +
              (1.0 - kPeriodEmaAlpha) * measured_period_);
          rejected_sample_count_ = 0;
        } else {
          rejected_sample_count_++;
          if (rejected_sample_count_ > 10) {
            spdlog::warn("[V3.2] Period calibration reset "
                         "(10 consecutive rejections)");
            period_initialized_ = false;
            period_samples_.clear();
            rejected_sample_count_ = 0;
          }
        }
      }
    }

    auto& slot = slots_[new_slot];
    slot.id = new_slot;
    slot.detection = detections[det_idx];
    slot.pose = solve_pnp(detections[det_idx]);
    slot.status = TrackedArmor::Status::kConfirmed;
    slot_active_[new_slot] = true;
    slot_timeout_[new_slot] = kTimeoutFrames;
    active_count++;
  }

  // Phase 6: Unmatched tracks → Phase 1 countdown handles timeout & recycle

  prev_active_count_ = active_count;

  // V3.2+: Pose EMA smoothing — blend raw PnP output with historical trend
  for (int i = 0; i < kMaxSlots; ++i) {
    if (!slot_active_[i]) continue;
    auto& pose = slots_[i].pose;

    if (!smooth_state_[i].initialized) {
      smooth_state_[i].pitch = pose.pitch;
      smooth_state_[i].yaw = pose.yaw;
      smooth_state_[i].depth = pose.depth;
      smooth_state_[i].initialized = true;
    } else {
      smooth_state_[i].pitch = kSmoothAlpha * pose.pitch
                             + (1.0 - kSmoothAlpha) * smooth_state_[i].pitch;
      smooth_state_[i].yaw = kSmoothAlpha * pose.yaw
                           + (1.0 - kSmoothAlpha) * smooth_state_[i].yaw;
      smooth_state_[i].depth = kSmoothAlpha * pose.depth
                             + (1.0 - kSmoothAlpha) * smooth_state_[i].depth;

      pose.pitch = smooth_state_[i].pitch;
      pose.yaw = smooth_state_[i].yaw;
      pose.depth = smooth_state_[i].depth;

      double roll = std::atan2(pose.rotation(2, 1), pose.rotation(2, 2));
      Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
      Eigen::AngleAxisd ry(pose.pitch, Eigen::Vector3d::UnitY());
      Eigen::AngleAxisd rz(pose.yaw, Eigen::Vector3d::UnitZ());
      pose.rotation = (rz * ry * rx).toRotationMatrix();
      pose.quaternion = Eigen::Quaterniond(pose.rotation);
    }
  }

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
    slot_release_frame_[i] = 0;
  }
  while (!free_slots_.empty()) {
    free_slots_.pop();
  }
  for (int i = 0; i < kMaxSlots; ++i) {
    free_slots_.push(i);
  }
  frame_counter_ = 0;
  first_seen_frame_ = -1;
  measured_period_ = kInitialPeriod;
  period_initialized_ = false;
  prev_active_count_ = 0;
  period_samples_.clear();
  rejected_sample_count_ = 0;
  prev_detections_.clear();
  prev_det_to_slot_.clear();
  smooth_state_ = {};
}

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

  // Extract pitch/yaw from rotation matrix (Rz·Ry·Rx convention)
  pose.pitch = -std::asin(std::clamp(pose.rotation(2, 0), -1.0, 1.0));
  pose.yaw = std::atan2(pose.rotation(1, 0), pose.rotation(0, 0));

  Eigen::Quaterniond q(pose.rotation);
  pose.quaternion = q;

  return pose;
}

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

  internal::Hungarian::CostMatrix cost(n_det, n_trk);
  constexpr double kMaxCenterDist{300.0};

  // V3.2: Pre-compute expected phase ID for Layer 1 bonus
  int expected_id = -1;
  if (period_initialized_ && first_seen_frame_ >= 0) {
    auto phase = (frame_counter_ - first_seen_frame_) % measured_period_;
    expected_id = phase_to_slot(phase);
  }

  // V3.2: Match inertia — map each current detection to its previous track
  std::vector<int> det_to_prev_track(n_det, -1);
  if (!prev_detections_.empty() && !prev_det_to_slot_.empty()) {
    for (int i = 0; i < n_det; ++i) {
      double best_iou = 0.0;
      int best_prev = -1;
      for (size_t p = 0; p < prev_detections_.size(); ++p) {
        auto iou = compute_iou(detections[i], prev_detections_[p]);
        if (iou > best_iou && iou > kInertiaMinIoU) {
          best_iou = iou;
          best_prev = static_cast<int>(p);
        }
      }
      if (best_prev >= 0 &&
          best_prev < static_cast<int>(prev_det_to_slot_.size())) {
        det_to_prev_track[i] = prev_det_to_slot_[best_prev];
      }
    }
  }

  for (int i = 0; i < n_det; ++i) {
    for (int j = 0; j < n_trk; ++j) {
      const auto& dc = detections[i].corners;
      const auto& tc = tracks[j].detection.corners;
      double d_cx = (dc[0].x + dc[1].x + dc[2].x + dc[3].x) / 4.0
                  - (tc[0].x + tc[1].x + tc[2].x + tc[3].x) / 4.0;
      double d_cy = (dc[0].y + dc[1].y + dc[2].y + dc[3].y) / 4.0
                  - (tc[0].y + tc[1].y + tc[2].y + tc[3].y) / 4.0;
      double center_dist = std::sqrt(d_cx * d_cx + d_cy * d_cy);
      if (center_dist > kMaxCenterDist) {
        cost(i, j) = 1.0;
        continue;
      }
      double iou = compute_iou(detections[i], tracks[j].detection);
      double base_cost = 1.0 - iou;

      // V3.2 Layer 1: Phase bonus — soft preference for phase-expected ID
      if (expected_id >= 0 && expected_id < kMaxSlots) {
        if (tracks[j].id == expected_id) {
          base_cost += kPhaseBonusMatch;    // -0.10 discount
        } else {
          base_cost += kPhaseBonusMismatch; // +0.02 penalty
        }
      }

      // V3.2: Match inertia — if this (det, track) was paired last frame,
      // give a strong discount to prevent oscillation
      if (det_to_prev_track[i] >= 0 && tracks[j].id == det_to_prev_track[i]) {
        base_cost += kInertiaBonus;  // -0.30
      }

      cost(i, j) = base_cost;
    }
  }

  auto [total_cost, assignments] =
      internal::Hungarian::solve_with_threshold(cost, 1.0 - constants::kIoUMin);

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

auto Tracker::phase_to_slot(int phase) const -> int {
  auto seg = measured_period_ / 3;
  if (phase < seg) {
    return 0;
  } else if (phase < 2 * seg) {
    return 1;
  } else {
    return 2;
  }
}

}  // namespace rm_autoaim