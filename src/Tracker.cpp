#include "rm_autoaim/Tracker.hpp"
#include "rm_autoaim/internal/Hungarian.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>

namespace rm_autoaim {

// ============================================================================
// Helper: compute median of a deque<double>
// ============================================================================
namespace {
[[nodiscard]] auto median_of(std::deque<double>& window) -> double {
  if (window.empty()) return 0.0;
  std::vector<double> sorted(window.begin(), window.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted[sorted.size() / 2];
}
}  // anonymous namespace

// ============================================================================
// Constructor
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
// V7: Dual-loop update — EKF predict → matching → EKF update → state machine
// ============================================================================

auto Tracker::update(const std::vector<Armor2D>& detections, double dt)
    -> std::vector<TrackedArmor> {
  // Sanity: clamp dt to avoid filter divergence
  dt = std::clamp(dt, 0.001, 0.050);  // 1ms ~ 50ms

  // ---- Loop 1: EKF predict (always, even without measurements) ----
  // This is the key to solving "numeric freeze on frame drop":
  // the EKF keeps the turret center state moving even when no
  // detections are available.
  if (ekf_initialized_) {
    ekf_.predict(dt);
  }

  // ---- Phase 1: Countdown lease for all active slots ----
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_status_[i] != SlotStatus::kInactive) {
      slot_timeout_[i]--;
      if (slot_timeout_[i] <= 0) {
        slot_status_[i] = SlotStatus::kInactive;
        slot_hit_count_[i] = 0;
        slot_miss_count_[i] = 0;
        free_slots_.push(i);
      }
    }
  }

  // ---- Handle empty detections ----
  if (detections.empty()) {
    // All active slots get a miss — state machine may transition
    for (int i = 0; i < kMaxSlots; ++i) {
      if (slot_status_[i] == SlotStatus::kInactive) continue;
      slot_miss_count_[i]++;
      slot_hit_count_[i] = 0;

      // State transitions on miss
      switch (slot_status_[i]) {
      case SlotStatus::kTentative:
        if (slot_miss_count_[i] >= kTentativeMissThreshold) {
          slot_status_[i] = SlotStatus::kInactive;
          free_slots_.push(i);
        }
        break;
      case SlotStatus::kConfirmed:
        if (slot_miss_count_[i] >= kLostMissThreshold) {
          slot_status_[i] = SlotStatus::kLost;
          spdlog::info("[V7] Slot {} CONFIRMED → LOST ({} misses)",
                       i, slot_miss_count_[i]);
        }
        break;
      case SlotStatus::kLost:
        if (slot_miss_count_[i] >= kInactiveMissThreshold) {
          slot_status_[i] = SlotStatus::kInactive;
          free_slots_.push(i);
          spdlog::info("[V7] Slot {} LOST → INACTIVE ({} misses)",
                       i, slot_miss_count_[i]);
        }
        break;
      default:
        break;
      }
    }
    return tracks();
  }

  // ---- EKF initialization (first detection) ----
  if (!ekf_initialized_) {
    // Use first detection to initialize EKF at phase 0
    auto raw_pose = solve_pnp(detections[0]);
    Eigen::Vector3d armor_pos(raw_pose.translation.x(),
                              raw_pose.translation.y(),
                              raw_pose.translation.z());
    ekf_.init(armor_pos, 0.0);  // phase 0
    ekf_initialized_ = true;

    // Assign to slot 0 (phase 0) as TENTATIVE
    if (!free_slots_.empty()) {
      int slot = free_slots_.front();
      free_slots_.pop();
      slots_[slot].id = slot;
      slots_[slot].detection = detections[0];
      slots_[slot].pose = raw_pose;
      slots_[slot].status = TrackedArmor::Status::kTentative;
      slot_status_[slot] = SlotStatus::kTentative;
      slot_timeout_[slot] = kTimeoutFrames;
      slot_hit_count_[slot] = 1;
      slot_miss_count_[slot] = 0;
      slot_just_activated_[slot] = true;
    }
    return tracks();
  }

  // ---- Build active slot list ----
  std::vector<int> active_slots;
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_status_[i] != SlotStatus::kInactive) {
      active_slots.push_back(i);
    }
  }

  // ---- Phase 2: Hungarian matching (prediction-based cost) ----
  auto [matches, unmatched_det, unmatched_trk] =
      associate(detections, active_slots);

  // ---- Phase 3: EKF update for matched slots ----
  for (const auto& [det_idx, active_idx] : matches) {
    int slot_idx = active_slots[active_idx];
    auto& slot = slots_[slot_idx];

    auto raw_pose = solve_pnp(detections[det_idx]);

    // Physics consistency check
    if (raw_pose.depth < kMinDepthM || raw_pose.depth > kMaxDepthM) {
      raw_pose.depth = last_valid_[slot_idx].valid
          ? last_valid_[slot_idx].depth
          : std::clamp(raw_pose.depth, kMinDepthM, kMaxDepthM);
    }
    if (std::abs(raw_pose.pitch) > kMaxPitchRad) {
      raw_pose.pitch = last_valid_[slot_idx].valid
          ? last_valid_[slot_idx].pitch
          : std::clamp(raw_pose.pitch, -kMaxPitchRad, kMaxPitchRad);
    }

    if (last_valid_[slot_idx].valid) {
      double delta_yaw = std::abs(raw_pose.yaw - last_valid_[slot_idx].yaw);
      if (delta_yaw > M_PI) delta_yaw = 2.0 * M_PI - delta_yaw;
      if (delta_yaw > kMaxDeltaYawRad) {
        raw_pose.yaw = last_valid_[slot_idx].yaw;
        recovery_counter_[slot_idx] = kRecoveryFrames;
      }
      double delta_depth = std::abs(raw_pose.depth - last_valid_[slot_idx].depth);
      if (delta_depth > kMaxDeltaDepthM) {
        raw_pose.depth = last_valid_[slot_idx].depth;
        recovery_counter_[slot_idx] = kRecoveryFrames;
      }
      double delta_pitch = std::abs(raw_pose.pitch - last_valid_[slot_idx].pitch);
      if (delta_pitch > kMaxDeltaPitchRad) {
        raw_pose.pitch = last_valid_[slot_idx].pitch;
        recovery_counter_[slot_idx] = kRecoveryFrames;
      }
    }

    last_valid_[slot_idx].pitch = raw_pose.pitch;
    last_valid_[slot_idx].yaw = raw_pose.yaw;
    last_valid_[slot_idx].depth = raw_pose.depth;
    last_valid_[slot_idx].valid = true;

    // EKF update: measurement = armor plate position from PnP
    Eigen::Vector3d armor_pos(raw_pose.translation.x(),
                              raw_pose.translation.y(),
                              raw_pose.translation.z());
    ekf_.update(armor_pos, kPhaseOffsets[slot_idx]);

    slot.detection = detections[det_idx];
    slot.pose = raw_pose;
    slot.status = TrackedArmor::Status::kConfirmed;
    slot_timeout_[slot_idx] = kTimeoutFrames;
    slot_hit_count_[slot_idx]++;
    slot_miss_count_[slot_idx] = 0;

    // State transitions on hit
    if (slot_status_[slot_idx] == SlotStatus::kTentative) {
      if (slot_hit_count_[slot_idx] >= kTentativeHitThreshold) {
        slot_status_[slot_idx] = SlotStatus::kConfirmed;
        spdlog::info("[V7] Slot {} TENTATIVE → CONFIRMED ({} hits)",
                     slot_idx, slot_hit_count_[slot_idx]);
      }
    } else if (slot_status_[slot_idx] == SlotStatus::kLost) {
      slot_status_[slot_idx] = SlotStatus::kConfirmed;
      spdlog::info("[V7] Slot {} LOST → CONFIRMED (recovered)",
                   slot_idx);
    }
  }

  // ---- Phase 4: Handle unmatched tracks (miss) ----
  for (int active_idx : unmatched_trk) {
    int slot_idx = active_slots[active_idx];
    slot_miss_count_[slot_idx]++;
    slot_hit_count_[slot_idx] = 0;

    switch (slot_status_[slot_idx]) {
    case SlotStatus::kTentative:
      if (slot_miss_count_[slot_idx] >= kTentativeMissThreshold) {
        slot_status_[slot_idx] = SlotStatus::kInactive;
        free_slots_.push(slot_idx);
      }
      break;
    case SlotStatus::kConfirmed:
      if (slot_miss_count_[slot_idx] >= kLostMissThreshold) {
        slot_status_[slot_idx] = SlotStatus::kLost;
        spdlog::info("[V7] Slot {} CONFIRMED → LOST ({} misses)",
                     slot_idx, slot_miss_count_[slot_idx]);
      }
      break;
    case SlotStatus::kLost:
      if (slot_miss_count_[slot_idx] >= kInactiveMissThreshold) {
        slot_status_[slot_idx] = SlotStatus::kInactive;
        free_slots_.push(slot_idx);
        spdlog::info("[V7] Slot {} LOST → INACTIVE ({} misses)",
                     slot_idx, slot_miss_count_[slot_idx]);
      }
      break;
    default:
      break;
    }
  }

  // ---- Phase 5: New detections → assign free slots ----
  for (int det_idx : unmatched_det) {
    if (free_slots_.empty()) continue;

    const auto& corners = detections[det_idx].corners;
    auto w = cv::norm(corners[1] - corners[0]);
    auto h = cv::norm(corners[3] - corners[0]);
    auto aspect = (h > 1e-4F) ? (w / h) : 0.0F;
    if (aspect < 1.2 || aspect > 4.5) continue;

    int new_slot = free_slots_.front();
    free_slots_.pop();

    auto raw_pose = solve_pnp(detections[det_idx]);

    // Physics check for new slot
    if (raw_pose.depth < kMinDepthM || raw_pose.depth > kMaxDepthM) {
      raw_pose.depth = std::clamp(raw_pose.depth, kMinDepthM, kMaxDepthM);
    }
    if (std::abs(raw_pose.pitch) > kMaxPitchRad) {
      raw_pose.pitch = std::clamp(raw_pose.pitch, -kMaxPitchRad, kMaxPitchRad);
    }

    // New slot: freeze first frame pose from historical values
    if (last_valid_[new_slot].valid) {
      raw_pose.depth = last_valid_[new_slot].depth;
      raw_pose.pitch = last_valid_[new_slot].pitch;
      raw_pose.yaw = last_valid_[new_slot].yaw;
    }

    // Update last_valid with raw PnP (not frozen) for next frame comparison
    auto raw_pose_for_history = solve_pnp(detections[det_idx]);
    last_valid_[new_slot].pitch = raw_pose_for_history.pitch;
    last_valid_[new_slot].yaw = raw_pose_for_history.yaw;
    last_valid_[new_slot].depth = raw_pose_for_history.depth;
    last_valid_[new_slot].valid = true;

    slots_[new_slot].id = new_slot;
    slots_[new_slot].detection = detections[det_idx];
    slots_[new_slot].pose = raw_pose;
    slots_[new_slot].status = TrackedArmor::Status::kTentative;
    slot_status_[new_slot] = SlotStatus::kTentative;
    slot_timeout_[new_slot] = kTimeoutFrames;
    slot_hit_count_[new_slot] = 1;
    slot_miss_count_[new_slot] = 0;
    slot_just_activated_[new_slot] = true;
  }

  // =========================================================================
  // Post-processing: Median-anchored smoothing + Quaternion SLERP
  // =========================================================================

  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_status_[i] == SlotStatus::kInactive) continue;
    auto& pose = slots_[i].pose;

    double depth_alpha = (recovery_counter_[i] > 0) ? kRecoveryAlpha : kDepthEmaAlpha;
    double pitch_alpha = (recovery_counter_[i] > 0) ? kRecoveryAlpha : kDepthEmaAlpha;
    bool skip_ema = slot_just_activated_[i];

    // --- Depth smoothing ---
    {
      depth_history_[i].push_back(pose.depth);
      if (static_cast<int>(depth_history_[i].size()) > kHistoryWindowSize) {
        depth_history_[i].pop_front();
      }
      double baseline = median_of(depth_history_[i]);
      double deviation = pose.depth - baseline;
      double clamp_bound = baseline * kDeviationClampRatio;
      if (clamp_bound < 0.3) clamp_bound = 0.3;
      deviation = std::clamp(deviation, -clamp_bound, clamp_bound);
      double clamped = baseline + deviation;

      if (!ema_initialized_[i] || skip_ema) {
        ema_depth_[i] = skip_ema ? pose.depth : clamped;
      } else {
        ema_depth_[i] = depth_alpha * clamped
                      + (1.0 - depth_alpha) * ema_depth_[i];
      }
      pose.depth = ema_depth_[i];
    }

    // --- Pitch smoothing ---
    {
      pitch_history_[i].push_back(pose.pitch);
      if (static_cast<int>(pitch_history_[i].size()) > kHistoryWindowSize) {
        pitch_history_[i].pop_front();
      }
      double baseline = median_of(pitch_history_[i]);
      double deviation = pose.pitch - baseline;
      double clamp_bound = std::abs(baseline) * kDeviationClampRatio;
      if (clamp_bound < 0.0175) clamp_bound = 0.0175;
      deviation = std::clamp(deviation, -clamp_bound, clamp_bound);
      double clamped = baseline + deviation;

      if (!ema_initialized_[i] || skip_ema) {
        ema_pitch_[i] = skip_ema ? pose.pitch : clamped;
      } else {
        ema_pitch_[i] = pitch_alpha * clamped
                      + (1.0 - pitch_alpha) * ema_pitch_[i];
      }
      pose.pitch = ema_pitch_[i];
    }

    // --- Quaternion SLERP ---
    {
      double roll = std::atan2(pose.rotation(2, 1), pose.rotation(2, 2));
      Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
      Eigen::AngleAxisd ry(pose.pitch, Eigen::Vector3d::UnitY());
      Eigen::AngleAxisd rz(pose.yaw, Eigen::Vector3d::UnitZ());
      Eigen::Quaterniond q_raw = Eigen::Quaterniond((rz * ry * rx).toRotationMatrix());

      if (!quat_initialized_[i]) {
        prev_quat_[i] = q_raw;
        quat_initialized_[i] = true;
      } else {
        prev_quat_[i] = prev_quat_[i].slerp(kSlerpAlpha, q_raw);
      }

      pose.quaternion = prev_quat_[i];
      pose.rotation = prev_quat_[i].toRotationMatrix();
      Eigen::Matrix3d R = pose.rotation;
      pose.yaw = std::atan2(R(1, 0), R(0, 0));
    }

    // --- Frozen detection ---
    {
      bool depth_frozen = (ema_initialized_[i] &&
          std::abs(ema_depth_[i] - last_ema_depth_[i]) < 1e-6);
      bool pitch_frozen = (ema_initialized_[i] &&
          std::abs(ema_pitch_[i] - last_ema_pitch_[i]) < 1e-6);

      consecutive_frozen_[i] = (depth_frozen && pitch_frozen)
          ? consecutive_frozen_[i] + 1 : 0;

      last_ema_depth_[i] = ema_depth_[i];
      last_ema_pitch_[i] = ema_pitch_[i];

      if (consecutive_frozen_[i] >= kFrozenResetThreshold) {
        spdlog::warn("[V7] Slot {} frozen for {} frames — force reset",
                     i, consecutive_frozen_[i]);
        consecutive_frozen_[i] = 0;
        ema_initialized_[i] = false;
        quat_initialized_[i] = false;
        depth_history_[i].clear();
        pitch_history_[i].clear();
        recovery_counter_[i] = 0;
      }
    }

    if (recovery_counter_[i] > 0) recovery_counter_[i]--;
    slot_just_activated_[i] = false;
    ema_initialized_[i] = true;
  }

  return tracks();
}

// ============================================================================
// tracks() — snapshot of all non-INACTIVE slots
// ============================================================================

auto Tracker::tracks() const -> std::vector<TrackedArmor> {
  std::vector<TrackedArmor> result;
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_status_[i] != SlotStatus::kInactive) {
      auto t = slots_[i];
      // Map SlotStatus to TrackedArmor::Status for downstream visualization
      switch (slot_status_[i]) {
      case SlotStatus::kTentative: t.status = TrackedArmor::Status::kTentative; break;
      case SlotStatus::kConfirmed: t.status = TrackedArmor::Status::kConfirmed; break;
      case SlotStatus::kLost:      t.status = TrackedArmor::Status::kLost; break;
      default: break;
      }
      result.push_back(t);
    }
  }
  return result;
}

// ============================================================================
// reset()
// ============================================================================

auto Tracker::reset() -> void {
  for (int i = 0; i < kMaxSlots; ++i) {
    slot_status_[i] = SlotStatus::kInactive;
    slot_timeout_[i] = 0;
    slot_hit_count_[i] = 0;
    slot_miss_count_[i] = 0;
  }
  while (!free_slots_.empty()) free_slots_.pop();
  for (int i = 0; i < kMaxSlots; ++i) free_slots_.push(i);

  ekf_initialized_ = false;
  ekf_ = internal::TurretEKF{};

  last_valid_ = {};
  depth_history_ = {};
  pitch_history_ = {};
  ema_depth_ = {};
  ema_pitch_ = {};
  ema_initialized_ = {};
  prev_quat_ = {};
  quat_initialized_ = {};
  recovery_counter_ = {};
  slot_just_activated_ = {};
  consecutive_frozen_ = {};
  last_ema_depth_ = {};
  last_ema_pitch_ = {};
}

// ============================================================================
// solve_pnp()
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
  pose.pitch = -std::asin(std::clamp(pose.rotation(2, 0), -1.0, 1.0));
  pose.yaw = std::atan2(pose.rotation(1, 0), pose.rotation(0, 0));

  Eigen::Quaterniond q(pose.rotation);
  pose.quaternion = q;

  return pose;
}

// ============================================================================
// V7: associate() — prediction-based cost matrix (HKUST approach)
// ============================================================================
// cost = w_iou * (1 - iou) + w_pred * pred_error_norm + w_center * center_dist_norm
//
// No hard inertia bonus. No expected_id. The EKF's predicted armor position
// provides the matching reference. A detection far from the predicted position
// gets a high cost — naturally preventing "old ID hijacking".
// ============================================================================

auto Tracker::associate(
    const std::vector<Armor2D>& detections,
    const std::vector<int>& active_slots)
    -> std::tuple<std::vector<std::pair<int, int>>,
                  std::vector<int>,
                  std::vector<int>> {
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatched_det;
  std::vector<int> unmatched_trk;

  const int n_det = static_cast<int>(detections.size());
  const int n_trk = static_cast<int>(active_slots.size());

  if (n_det == 0) {
    for (int i = 0; i < n_trk; ++i) unmatched_trk.push_back(i);
    return {matches, unmatched_det, unmatched_trk};
  }

  if (n_trk == 0) {
    for (int i = 0; i < n_det; ++i) unmatched_det.push_back(i);
    return {matches, unmatched_det, unmatched_trk};
  }

  internal::Hungarian::CostMatrix cost(n_det, n_trk);

  for (int i = 0; i < n_det; ++i) {
    for (int j = 0; j < n_trk; ++j) {
      int slot_idx = active_slots[j];

      // ---- Component 1: IoU cost ----
      double iou = compute_iou(detections[i], slots_[slot_idx].detection);
      double iou_cost = kWeightIoU * (1.0 - iou);

      // ---- Component 2: Prediction error ----
      // Compare detection center with EKF-predicted armor position
      // Projected back to image space using camera model
      double pred_cost = 0.0;
      if (ekf_initialized_) {
        Eigen::Vector3d pred_armor = ekf_.predictArmorPos(kPhaseOffsets[slot_idx]);
        // Project predicted 3D position to 2D image
        double px = constants::kFx * pred_armor(0) / pred_armor(2) + constants::kCx;
        double py = constants::kFy * pred_armor(1) / pred_armor(2) + constants::kCy;

        double det_cx = (detections[i].corners[0].x + detections[i].corners[1].x +
                         detections[i].corners[2].x + detections[i].corners[3].x) / 4.0;
        double det_cy = (detections[i].corners[0].y + detections[i].corners[1].y +
                         detections[i].corners[2].y + detections[i].corners[3].y) / 4.0;

        double pred_err = std::sqrt((det_cx - px) * (det_cx - px) +
                                    (det_cy - py) * (det_cy - py));
        pred_cost = kWeightPred * std::min(pred_err / kMaxPredError, 1.0);
      }

      // ---- Component 3: Center distance ----
      double det_cx = (detections[i].corners[0].x + detections[i].corners[1].x +
                       detections[i].corners[2].x + detections[i].corners[3].x) / 4.0;
      double det_cy = (detections[i].corners[0].y + detections[i].corners[1].y +
                       detections[i].corners[2].y + detections[i].corners[3].y) / 4.0;
      double trk_cx = (slots_[slot_idx].detection.corners[0].x +
                       slots_[slot_idx].detection.corners[1].x +
                       slots_[slot_idx].detection.corners[2].x +
                       slots_[slot_idx].detection.corners[3].x) / 4.0;
      double trk_cy = (slots_[slot_idx].detection.corners[0].y +
                       slots_[slot_idx].detection.corners[1].y +
                       slots_[slot_idx].detection.corners[2].y +
                       slots_[slot_idx].detection.corners[3].y) / 4.0;

      double center_dist = std::sqrt((det_cx - trk_cx) * (det_cx - trk_cx) +
                                     (det_cy - trk_cy) * (det_cy - trk_cy));
      if (center_dist > kMaxCenterDist) {
        cost(i, j) = 1.0;  // impossible match
        continue;
      }
      double center_cost = kWeightCenter * (center_dist / kMaxCenterDist);

      cost(i, j) = iou_cost + pred_cost + center_cost;
    }
  }

  auto [total_cost, assignments] =
      internal::Hungarian::solve_with_threshold(cost, 1.0 - kIoUMin);

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
    if (!det_matched[i]) unmatched_det.push_back(i);
  }
  for (int j = 0; j < n_trk; ++j) {
    if (!trk_matched[j]) unmatched_trk.push_back(j);
  }

  return {matches, unmatched_det, unmatched_trk};
}

// ============================================================================
// compute_iou()
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

  if (union_area < 1e-6F) return 0.0;

  return static_cast<double>(inter_area / union_area);
}

}  // namespace rm_autoaim