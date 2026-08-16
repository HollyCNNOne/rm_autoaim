#pragma once

#include "rm_autoaim/Types.hpp"

#include <array>
#include <deque>
#include <queue>
#include <vector>

#include <opencv2/core.hpp>

namespace rm_autoaim {

// V3.3: Physics-Guided Tracking
// Four core upgrades over V3.2:
//   1. Expected-ID state machine — visual-evidence-driven, not clock-driven
//   2. Physics consistency check — rejects impossible PnP results
//   3. Median-anchored deviation clamp — prevents EMA baseline drift
//   4. Quaternion SLERP — handles ±180° yaw boundary naturally
class Tracker {
public:
  Tracker();

  [[nodiscard]] auto update(const std::vector<Armor2D>& detections)
      -> std::vector<TrackedArmor>;

  [[nodiscard]] auto tracks() const -> std::vector<TrackedArmor>;

  auto reset() -> void;

private:
  [[nodiscard]] auto solve_pnp(const Armor2D& detection)
      -> ArmorPose;

  [[nodiscard]] auto associate(
      const std::vector<Armor2D>& detections,
      const std::vector<TrackedArmor>& tracks,
      int expected_id)
      -> std::tuple<std::vector<std::pair<int, int>>,
                    std::vector<int>,
                    std::vector<int>>;

  [[nodiscard]] static auto compute_iou(const Armor2D& a, const Armor2D& b)
      -> double;

  // Fixed-slot architecture
  static constexpr int kMaxSlots{3};
  static constexpr int kTimeoutFrames{15};
  std::array<TrackedArmor, kMaxSlots> slots_;
  bool slot_active_[kMaxSlots]{};
  int slot_timeout_[kMaxSlots]{};
  std::queue<int> free_slots_;
  int frame_counter_{0};
  int prev_active_count_{0};

  // =========================================================================
  // V3.3 Problem 1: Expected-ID state machine (visual-evidence-driven)
  // "Tend toward 0→1→2 but allow breaking the order when physics demands it."
  // =========================================================================
  int expected_id_{0};
  int expected_id_miss_counter_{0};        // frames current ID not matched
  int expected_id_skip_counter_{0};        // frames NEXT ID not detected
  static constexpr int kMissAdvanceThreshold{5};   // 5 misses → advance
  static constexpr int kSkipToleranceThreshold{8};  // 8 absent → skip (adj2: was 20)
  static constexpr int kAdvanceCooldownFrames{5};   // cooldown after advance
  int id_advance_cooldown_{0};

  // Adjust 1: Max dwell time — force advance even if ID is still matched
  // Physical window ≈ 72 frames / ID. Dwell limit = 65 frames (~390ms).
  static constexpr int kMaxDwellFrames{65};  // Tuning 2: was 58, +7 frames buffer
  int id_dwell_counter_{0};

  // Adjust 3+4: Fast scan mode after skip, skip storm suppression
  static constexpr int kFastScanFrames{3};         // fast scan lasts 3 frames
  static constexpr int kFastScanThreshold{2};       // shorter miss threshold in scan
  static constexpr int kMaxConsecutiveSkips{2};     // Tuning 1: brake after 2 skips
  bool fast_scan_mode_{false};
  int fast_scan_counter_{0};
  int consecutive_skips_{0};
  bool skip_brake_{false};  // Tuning 1: lock skip logic, wait for real detection

  // Tuning 3: Force-switch first frame — reuse historical pose, not raw PnP
  bool force_switched_{false};  // set by dwell timeout / skip, cleared after Phase 5

  // =========================================================================
  // V3.3 Problem 2: Physics consistency check
  // "Reject physically impossible values regardless of PnP convergence."
  // =========================================================================
  static constexpr double kMaxDepthM{15.0};
  static constexpr double kMinDepthM{0.5};
  static constexpr double kMaxPitchRad{0.5236};   // ±30°
  static constexpr double kMaxDeltaYawRad{0.5236}; // 30°/frame
  static constexpr double kMaxDeltaDepthM{2.0};    // 2m/frame
  static constexpr double kMaxDeltaPitchRad{0.2618}; // 15°/frame

  // P5: Recovery protection after physics validation failure
  static constexpr int kRecoveryFrames{1};  // P0: was 3, 1 frame is enough to block spike
  static constexpr double kRecoveryAlpha{0.30};
  std::array<int, kMaxSlots> recovery_counter_{}; // 0 = normal, >0 = in recovery

  struct LastValidPose {
    double pitch{0.0};
    double yaw{0.0};
    double depth{0.0};
    bool valid{false};
  };
  std::array<LastValidPose, kMaxSlots> last_valid_;

  // =========================================================================
  // V3.3 Problem 3: Median-anchored deviation clamp
  // "Stable baseline never drifts; only local fluctuations are allowed."
  // =========================================================================
  static constexpr int kHistoryWindowSize{15};
  static constexpr double kDeviationClampRatio{0.30};
  static constexpr double kDepthEmaAlpha{0.25};
  std::array<std::deque<double>, kMaxSlots> depth_history_;
  std::array<std::deque<double>, kMaxSlots> pitch_history_;
  std::array<double, kMaxSlots> ema_depth_{};
  std::array<double, kMaxSlots> ema_pitch_{};
  std::array<bool, kMaxSlots> ema_initialized_{};
  std::array<bool, kMaxSlots> slot_just_activated_{};  // P2: first frame after (re)activation

  // P0 heartbeat: force-reset smoothing if values stay frozen too long
  static constexpr int kFrozenResetThreshold{20};
  std::array<int, kMaxSlots> consecutive_frozen_{};
  std::array<double, kMaxSlots> last_ema_depth_{};
  std::array<double, kMaxSlots> last_ema_pitch_{};

  // =========================================================================
  // V3.3 Problem 4: Quaternion SLERP
  // "Solve the ±180° singularity at the mathematical representation level."
  // =========================================================================
  static constexpr double kSlerpAlpha{0.25};
  std::array<Eigen::Quaterniond, kMaxSlots> prev_quat_;
  std::array<bool, kMaxSlots> quat_initialized_{};

  // =========================================================================
  // V3.2 Retained: Hungarian cost bonuses
  // =========================================================================
  static constexpr double kPhaseBonusMatch{-0.10};
  static constexpr double kPhaseBonusMismatch{0.02};
  static constexpr double kInertiaBonus{-0.30};
  static constexpr double kInertiaMinIoU{0.3};
  std::vector<Armor2D> prev_detections_;
  std::vector<int> prev_det_to_slot_;

  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  std::vector<cv::Point3f> model_points_;
};

}  // namespace rm_autoaim