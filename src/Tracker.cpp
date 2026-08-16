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
// Main update loop
// ============================================================================

auto Tracker::update(const std::vector<Armor2D>& detections)
    -> std::vector<TrackedArmor> {
  frame_counter_++;

  // Phase 1: Countdown — decrement lease for all active slots
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_active_[i]) {
      slot_timeout_[i]--;
      if (slot_timeout_[i] <= 0) {
        slot_active_[i] = false;
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

  // Phase 3: Hungarian matching (V3.3: expected_id state machine drives bonus)
  auto [matches, unmatched_det, unmatched_trk] =
      associate(detections, active_tracks, expected_id_);

  // Phase 4: Execute matches — renew lease, update PnP, physics check
  for (const auto& [det_idx, trk_idx] : matches) {
    int slot_idx = active_to_slot[trk_idx];
    auto& slot = slots_[slot_idx];
    slot.detection = detections[det_idx];
    auto raw_pose = solve_pnp(detections[det_idx]);

    // V3.3 Problem 2: Physics consistency check
    // Reject or clamp values that violate physical constraints.
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
      if (delta_yaw > M_PI) delta_yaw = 2.0 * M_PI - delta_yaw;  // circular
      if (delta_yaw > kMaxDeltaYawRad) {
        raw_pose.yaw = last_valid_[slot_idx].yaw;
        recovery_counter_[slot_idx] = kRecoveryFrames;  // P5
      }
      double delta_depth = std::abs(raw_pose.depth - last_valid_[slot_idx].depth);
      if (delta_depth > kMaxDeltaDepthM) {
        raw_pose.depth = last_valid_[slot_idx].depth;
        recovery_counter_[slot_idx] = kRecoveryFrames;  // P5
      }
      double delta_pitch = std::abs(raw_pose.pitch - last_valid_[slot_idx].pitch);
      if (delta_pitch > kMaxDeltaPitchRad) {
        raw_pose.pitch = last_valid_[slot_idx].pitch;
        recovery_counter_[slot_idx] = kRecoveryFrames;  // P5
      }
    }

    last_valid_[slot_idx].pitch = raw_pose.pitch;
    last_valid_[slot_idx].yaw = raw_pose.yaw;
    last_valid_[slot_idx].depth = raw_pose.depth;
    last_valid_[slot_idx].valid = true;

    slot.pose = raw_pose;
    slot.status = TrackedArmor::Status::kConfirmed;
    slot_timeout_[slot_idx] = kTimeoutFrames;
  }

  // Tuning 1: Release brake when real detections are matched
  if (!matches.empty()) {
    skip_brake_ = false;
    consecutive_skips_ = 0;
  }

  // V3.3pro Expected-ID state machine update
  // Adjust 1: Max dwell time — force advance after 58 frames even if matched.
  // Adjust 2: Skip threshold reduced to 8 frames (was 20).
  // Adjust 3: Fast scan mode after skip — shorter threshold for rapid recovery.
  // Adjust 4: Skip storm suppression — 2+ consecutive skips → reset to detected.
  if (!detections.empty() && id_advance_cooldown_ == 0) {
    bool expected_matched = slot_active_[expected_id_];

    // Adjust 1: Track dwell time — force advance if exceeded
    if (expected_matched) {
      id_dwell_counter_++;
      if (id_dwell_counter_ >= kMaxDwellFrames) {
        int old_id = expected_id_;
        expected_id_ = (expected_id_ + 1) % 3;
        id_dwell_counter_ = 0;
        expected_id_miss_counter_ = 0;
        expected_id_skip_counter_ = 0;
        id_advance_cooldown_ = kAdvanceCooldownFrames;
        force_switched_ = true;  // Tuning 3: first frame after dwell → reuse history
        spdlog::info("[V3.3pro+] ID {} dwell timeout ({} frames) → forced advance to {}",
                     old_id, kMaxDwellFrames, expected_id_);
      }
    } else {
      id_dwell_counter_ = 0;
    }

    // Adjust 3: Use shorter threshold in fast scan mode
    int miss_threshold = fast_scan_mode_ ? kFastScanThreshold : kMissAdvanceThreshold;

    if (expected_matched) {
      expected_id_miss_counter_ = 0;
      expected_id_skip_counter_ = 0;
    } else {
      expected_id_miss_counter_++;
      if (expected_id_miss_counter_ >= miss_threshold) {
        int old_id = expected_id_;
        expected_id_ = (expected_id_ + 1) % 3;
        expected_id_miss_counter_ = 0;
        expected_id_skip_counter_ = 0;
        id_advance_cooldown_ = kAdvanceCooldownFrames;
        spdlog::info("[V3.3pro] Expected ID advanced: {} → {} ({} misses, "
                     "fast_scan={})",
                     old_id, expected_id_, miss_threshold, fast_scan_mode_);
      }
    }
  }
  if (id_advance_cooldown_ > 0) {
    id_advance_cooldown_--;
  }

  // Adjust 3: Decrement fast scan counter
  if (fast_scan_mode_) {
    fast_scan_counter_--;
    if (fast_scan_counter_ <= 0) {
      fast_scan_mode_ = false;
      consecutive_skips_ = 0;
    }
  }

  // Save current frame's pairings for next frame's inertia bonus
  prev_detections_ = detections;
  prev_det_to_slot_.assign(detections.size(), -1);
  for (const auto& [det_idx, trk_idx] : matches) {
    int slot_idx = active_to_slot[trk_idx];
    prev_det_to_slot_[det_idx] = slot_idx;
  }

  // Phase 5: New detections → assign slots
  // V3.3: preferred slot = expected_id_ (state-machine-driven, not clock-driven)
  constexpr int kMaxVisibleTargets{2};
  constexpr double kMinArmorAspect{1.2};
  constexpr double kMaxArmorAspect{4.5};

  int active_count = 0;
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_active_[i]) active_count++;
  }

  bool new_det_is_expected{false};

  for (int det_idx : unmatched_det) {
    if (active_count >= kMaxVisibleTargets) continue;

    const auto& corners = detections[det_idx].corners;
    auto w = cv::norm(corners[1] - corners[0]);
    auto h = cv::norm(corners[3] - corners[0]);
    auto aspect = (h > 1e-4F) ? (w / h) : 0.0F;
    if (aspect < kMinArmorAspect || aspect > kMaxArmorAspect) continue;

    int new_slot = -1;

    // V3.3 Layer 2: preferred slot = expected_id_ (when idle)
    if (!slot_active_[expected_id_]) {
      new_slot = expected_id_;
      // Remove expected_id_ from queue if present
      std::queue<int> filtered;
      while (!free_slots_.empty()) {
        auto s = free_slots_.front();
        free_slots_.pop();
        if (s != expected_id_) {
          filtered.push(s);
        }
      }
      free_slots_ = std::move(filtered);
      new_det_is_expected = true;
    }

    // Fallback: queue-based assignment
    if (new_slot == -1) {
      if (free_slots_.empty()) continue;
      new_slot = free_slots_.front();
      free_slots_.pop();
    }

    auto& slot = slots_[new_slot];
    slot.id = new_slot;
    slot.detection = detections[det_idx];

    ArmorPose raw_pose;

    // Tuning 3: After force-switch, skip PnP — reuse historical pose
    if (force_switched_ && last_valid_[new_slot].valid) {
      raw_pose = solve_pnp(detections[det_idx]);  // still need rotation + translation
      raw_pose.depth = last_valid_[new_slot].depth;
      raw_pose.pitch = last_valid_[new_slot].pitch;
      raw_pose.yaw = last_valid_[new_slot].yaw;
    } else {
      raw_pose = solve_pnp(detections[det_idx]);

      // Tuning 1: Release brake when real detections are assigned
      if (skip_brake_) {
        skip_brake_ = false;
        consecutive_skips_ = 0;
        spdlog::info("[V3.3pro+] Skip brake released — real detection assigned");
      }
    }

    // P2: First frame after slot (re)activation has unstable PnP.
    // Use historical values for the POSE OUTPUT, but keep raw PnP in
    // last_valid_ so the physics check on the next frame compares against
    // reality (not the freeze). This prevents permanent data lock.
    auto raw_for_history = raw_pose;  // save before freeze

    if (last_valid_[new_slot].valid) {
      raw_pose.depth = last_valid_[new_slot].depth;
      raw_pose.pitch = last_valid_[new_slot].pitch;
      raw_pose.yaw = last_valid_[new_slot].yaw;
    } else {
      // Physics check for new slots too
      if (raw_pose.depth < kMinDepthM || raw_pose.depth > kMaxDepthM) {
        raw_pose.depth = std::clamp(raw_pose.depth, kMinDepthM, kMaxDepthM);
      }
      if (std::abs(raw_pose.pitch) > kMaxPitchRad) {
        raw_pose.pitch = std::clamp(raw_pose.pitch, -kMaxPitchRad, kMaxPitchRad);
      }
    }

    // Write RAW PnP to last_valid_ (not frozen), so next frame's physics
    // check compares against actual measurements, not historical freeze.
    last_valid_[new_slot].pitch = raw_for_history.pitch;
    last_valid_[new_slot].yaw = raw_for_history.yaw;
    last_valid_[new_slot].depth = raw_for_history.depth;
    last_valid_[new_slot].valid = true;

    slot.pose = raw_pose;
    slot.status = TrackedArmor::Status::kConfirmed;
    slot_active_[new_slot] = true;
    slot_timeout_[new_slot] = kTimeoutFrames;
    slot_just_activated_[new_slot] = true;
    active_count++;
  }

  // V3.3pro+ Skip tolerance with fast scan, brake, and storm suppression
  // Tuning 1: Brake after 2 consecutive skips — wait for real detection.
  if (skip_brake_) {
    goto skip_done;
  }

  if (!new_det_is_expected && !slot_active_[expected_id_]) {
    expected_id_skip_counter_++;
    if (expected_id_skip_counter_ >= kSkipToleranceThreshold) {
      int old_id = expected_id_;

      // Tuning 1: Brake after 2 consecutive skips — stop advancing
      consecutive_skips_++;
      if (consecutive_skips_ >= kMaxConsecutiveSkips) {
        skip_brake_ = true;
        consecutive_skips_ = 0;
        fast_scan_mode_ = false;
        spdlog::warn("[V3.3pro+] Skip brake engaged after {} consecutive skips "
                     "— waiting for real detection",
                     kMaxConsecutiveSkips);
        goto skip_done;
      }

      expected_id_ = (expected_id_ + 1) % 3;
      expected_id_skip_counter_ = 0;
      expected_id_miss_counter_ = 0;
      id_dwell_counter_ = 0;
      force_switched_ = true;  // Tuning 3: first frame after skip → reuse history

      // Tuning 3: Enter fast scan mode after a skip
      fast_scan_mode_ = true;
      fast_scan_counter_ = kFastScanFrames;

      spdlog::warn("[V3.3pro+] Expected ID {} skipped (absent {} frames) → {} "
                   "(fast_scan={} frames)",
                   old_id, kSkipToleranceThreshold, expected_id_,
                   kFastScanFrames);
    }
  }
  skip_done:

  // Phase 6: Unmatched tracks → Phase 1 countdown handles timeout & recycle

  prev_active_count_ = active_count;
  force_switched_ = false;  // Tuning 3: reset after Phase 5 consumed it

  // =========================================================================
  // V3.3 Post-processing: Median-anchored smoothing + Quaternion SLERP
  // =========================================================================

  for (int i = 0; i < kMaxSlots; ++i) {
    if (!slot_active_[i]) continue;
    auto& pose = slots_[i].pose;

    // P5: Use reduced alpha during recovery to prevent "snap-back" jitter
    double depth_alpha = (recovery_counter_[i] > 0) ? kRecoveryAlpha : kDepthEmaAlpha;
    double pitch_alpha = (recovery_counter_[i] > 0) ? kRecoveryAlpha : kDepthEmaAlpha;

    // P2: Skip EMA update for just-activated slots — first frame is frozen
    bool skip_ema = slot_just_activated_[i];

    // --- Problem 3: Median-anchored deviation clamp for depth ---
    {
      depth_history_[i].push_back(pose.depth);
      if (static_cast<int>(depth_history_[i].size()) > kHistoryWindowSize) {
        depth_history_[i].pop_front();
      }
      double baseline = median_of(depth_history_[i]);
      double deviation = pose.depth - baseline;
      double clamp_bound = baseline * kDeviationClampRatio;
      if (clamp_bound < 0.3) clamp_bound = 0.3;  // min 30cm clamp
      deviation = std::clamp(deviation, -clamp_bound, clamp_bound);
      double clamped = baseline + deviation;

      if (!ema_initialized_[i] || skip_ema) {
        ema_depth_[i] = clamped;
        if (skip_ema) {
          // P2: On first frame after activation, keep the frozen value
          ema_depth_[i] = pose.depth;
        }
      } else {
        ema_depth_[i] = depth_alpha * clamped
                      + (1.0 - depth_alpha) * ema_depth_[i];
      }
      pose.depth = ema_depth_[i];
    }

    // --- Problem 3: Median-anchored deviation clamp for pitch ---
    {
      pitch_history_[i].push_back(pose.pitch);
      if (static_cast<int>(pitch_history_[i].size()) > kHistoryWindowSize) {
        pitch_history_[i].pop_front();
      }
      double baseline = median_of(pitch_history_[i]);
      double deviation = pose.pitch - baseline;
      double clamp_bound = std::abs(baseline) * kDeviationClampRatio;
      if (clamp_bound < 0.0175) clamp_bound = 0.0175;  // min 1°
      deviation = std::clamp(deviation, -clamp_bound, clamp_bound);
      double clamped = baseline + deviation;

      if (!ema_initialized_[i] || skip_ema) {
        ema_pitch_[i] = clamped;
        if (skip_ema) {
          ema_pitch_[i] = pose.pitch;
        }
      } else {
        ema_pitch_[i] = pitch_alpha * clamped
                      + (1.0 - pitch_alpha) * ema_pitch_[i];
      }
      pose.pitch = ema_pitch_[i];
    }

    // --- Problem 4: Quaternion SLERP for yaw (±180° boundary) ---
    {
      // Reconstruct quaternion from smoothed pitch, raw yaw, raw roll
      double roll = std::atan2(pose.rotation(2, 1), pose.rotation(2, 2));
      Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
      Eigen::AngleAxisd ry(pose.pitch, Eigen::Vector3d::UnitY());
      Eigen::AngleAxisd rz(pose.yaw, Eigen::Vector3d::UnitZ());
      Eigen::Quaterniond q_raw = Eigen::Quaterniond((rz * ry * rx).toRotationMatrix());

      if (!quat_initialized_[i]) {
        prev_quat_[i] = q_raw;
        quat_initialized_[i] = true;
      } else {
        // SLERP: shortest arc on the 4D sphere, naturally handles ±180°
        prev_quat_[i] = prev_quat_[i].slerp(kSlerpAlpha, q_raw);
      }

      pose.quaternion = prev_quat_[i];
      pose.rotation = prev_quat_[i].toRotationMatrix();

      // Update yaw from smoothed quaternion for consistency
      Eigen::Matrix3d R = pose.rotation;
      pose.yaw = std::atan2(R(1, 0), R(0, 0));
    }

    // P0 heartbeat: detect frozen values and force-reset smoothing
    // If depth and pitch stay identical for kFrozenResetThreshold frames,
    // the EMA has been locked. Reset the state to unfreeze.
    {
      bool depth_frozen = (ema_initialized_[i] &&
          std::abs(ema_depth_[i] - last_ema_depth_[i]) < 1e-6);
      bool pitch_frozen = (ema_initialized_[i] &&
          std::abs(ema_pitch_[i] - last_ema_pitch_[i]) < 1e-6);

      if (depth_frozen && pitch_frozen) {
        consecutive_frozen_[i]++;
      } else {
        consecutive_frozen_[i] = 0;
      }

      last_ema_depth_[i] = ema_depth_[i];
      last_ema_pitch_[i] = ema_pitch_[i];

      if (consecutive_frozen_[i] >= kFrozenResetThreshold) {
        spdlog::warn("[V3.3+] Slot {} frozen for {} frames — force reset",
                     i, consecutive_frozen_[i]);
        consecutive_frozen_[i] = 0;
        ema_initialized_[i] = false;
        quat_initialized_[i] = false;
        depth_history_[i].clear();
        pitch_history_[i].clear();
        recovery_counter_[i] = 0;
      }
    }

    // P5: Decrement recovery counter
    if (recovery_counter_[i] > 0) {
      recovery_counter_[i]--;
    }
    // P2: Clear activation flag after first frame
    slot_just_activated_[i] = false;
    ema_initialized_[i] = true;
  }

  std::vector<TrackedArmor> active;
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_active_[i]) {
      active.push_back(slots_[i]);
    }
  }
  return active;
}

// ============================================================================
// tracks() — snapshot of active slots
// ============================================================================

auto Tracker::tracks() const -> std::vector<TrackedArmor> {
  std::vector<TrackedArmor> result;
  for (int i = 0; i < kMaxSlots; ++i) {
    if (slot_active_[i]) {
      result.push_back(slots_[i]);
    }
  }
  return result;
}

// ============================================================================
// reset()
// ============================================================================

auto Tracker::reset() -> void {
  for (int i = 0; i < kMaxSlots; ++i) {
    slot_active_[i] = false;
    slot_timeout_[i] = 0;
  }
  while (!free_slots_.empty()) {
    free_slots_.pop();
  }
  for (int i = 0; i < kMaxSlots; ++i) {
    free_slots_.push(i);
  }
  frame_counter_ = 0;
  prev_active_count_ = 0;
  expected_id_ = 0;
  expected_id_miss_counter_ = 0;
  expected_id_skip_counter_ = 0;
  id_advance_cooldown_ = 0;
  id_dwell_counter_ = 0;
  fast_scan_mode_ = false;
  fast_scan_counter_ = 0;
  consecutive_skips_ = 0;
  skip_brake_ = false;
  force_switched_ = false;
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
  prev_detections_.clear();
  prev_det_to_slot_.clear();
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

  // Extract pitch/yaw from rotation matrix (Rz·Ry·Rx convention)
  pose.pitch = -std::asin(std::clamp(pose.rotation(2, 0), -1.0, 1.0));
  pose.yaw = std::atan2(pose.rotation(1, 0), pose.rotation(0, 0));

  Eigen::Quaterniond q(pose.rotation);
  pose.quaternion = q;

  return pose;
}

// ============================================================================
// associate() — Hungarian matching with expected-ID bonus + match inertia
// ============================================================================

auto Tracker::associate(
    const std::vector<Armor2D>& detections,
    const std::vector<TrackedArmor>& tracks,
    int expected_id)
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

  // Match inertia: map each current detection to its previous track
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

      // V3.3: Expected-ID bonus (state-machine-driven, not period-driven)
      if (tracks[j].id == expected_id) {
        base_cost += kPhaseBonusMatch;    // -0.10 discount
      } else {
        base_cost += kPhaseBonusMismatch; // +0.02 penalty
      }

      // Match inertia: strong discount for same pairing as last frame
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

  if (union_area < 1e-6F) {
    return 0.0;
  }

  return static_cast<double>(inter_area / union_area);
}

}  // namespace rm_autoaim