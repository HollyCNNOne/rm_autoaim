#pragma once

#include "rm_autoaim/Types.hpp"
#include "rm_autoaim/internal/TurretEKF.hpp"

#include <array>
#include <chrono>
#include <deque>
#include <queue>
#include <vector>

#include <opencv2/core.hpp>

namespace rm_autoaim {

// ============================================================================
// V7: Dual-Loop Turret-Centric Tracker (SCUT × HKUST fusion)
// ============================================================================
// Architecture:
//   Loop 1 (EKF):    TurretEKF::predict() every frame → continuous state
//                     TurretEKF::update() when matched → measurement correction
//   Loop 2 (Matching): prediction-based cost matrix → Hungarian → assignments
//
// Three upgrades over V6:
//   1. TurretEKF replaces "three independent armor plates" with "turret center
//      + armor phase" tracking (SCUT approach)
//   2. Prediction-based soft matching replaces hard inertia bonus (HKUST approach)
//   3. Four-state lifecycle machine (INACTIVE→TENTATIVE→CONFIRMED↔LOST)
//      replaces binary active/inactive
// ============================================================================

class Tracker {
public:
  // HKUST-style fine-grained state machine
  enum class SlotStatus {
    kInactive,   // slot is free, available for new detection
    kTentative,  // newly detected, needs N consecutive hits to confirm
    kConfirmed,  // tracking normally
    kLost        // temporarily lost, still predicting via EKF
  };

  Tracker();

  // dt: actual frame interval from steady_clock (seconds)
  [[nodiscard]] auto update(const std::vector<Armor2D>& detections, double dt)
      -> std::vector<TrackedArmor>;

  [[nodiscard]] auto tracks() const -> std::vector<TrackedArmor>;

  auto reset() -> void;

private:
  [[nodiscard]] auto solve_pnp(const Armor2D& detection) -> ArmorPose;

  [[nodiscard]] auto associate(
      const std::vector<Armor2D>& detections,
      const std::vector<int>& active_slots)
      -> std::tuple<std::vector<std::pair<int, int>>,
                    std::vector<int>,
                    std::vector<int>>;

  [[nodiscard]] static auto compute_iou(const Armor2D& a, const Armor2D& b)
      -> double;

  // =========================================================================
  // Core: Turret-centric EKF (SCUT approach)
  // =========================================================================
  internal::TurretEKF ekf_;
  bool ekf_initialized_{false};

  // =========================================================================
  // Phase slots: 3 fixed armor plates, each with a fixed phase offset
  // =========================================================================
  static constexpr int kMaxSlots{3};
  static constexpr double kPhaseOffsets[3] = {0.0, 2.0 * M_PI / 3.0, 4.0 * M_PI / 3.0};
  static constexpr int kTimeoutFrames{15};   // lease: ~90ms @ 166.7FPS
  static constexpr double kNominalDt{1.0 / 166.7};

  TrackedArmor slots_[kMaxSlots];
  SlotStatus slot_status_[kMaxSlots]{};
  int slot_timeout_[kMaxSlots]{};       // lease countdown
  int slot_hit_count_[kMaxSlots]{};     // consecutive hits
  int slot_miss_count_[kMaxSlots]{};    // consecutive misses
  std::queue<int> free_slots_;          // INACTIVE slots available for assignment

  // =========================================================================
  // State machine thresholds (HKUST approach)
  // =========================================================================
  static constexpr int kTentativeHitThreshold{3};    // 3 hits → CONFIRMED
  static constexpr int kLostMissThreshold{5};         // 5 misses in CONFIRMED → LOST
  static constexpr int kInactiveMissThreshold{10};    // 10 misses in LOST → INACTIVE
  static constexpr int kTentativeMissThreshold{1};    // 1 miss in TENTATIVE → INACTIVE

  // =========================================================================
  // Prediction-based matching (HKUST approach)
  // Replaces hard inertia bonus with soft multi-dimensional cost
  // =========================================================================
  // cost = w_iou * (1 - iou) + w_pred * pred_error_norm + w_center * center_dist_norm
  static constexpr double kWeightIoU{0.50};
  static constexpr double kWeightPred{0.30};
  static constexpr double kWeightCenter{0.20};
  static constexpr double kMaxPredError{0.50};    // 50cm max prediction error (norm reference)
  static constexpr double kMaxCenterDist{300.0};  // 300px max center distance (norm reference)
  static constexpr double kIoUMin{0.45};           // minimum IoU for match

  // =========================================================================
  // V3.3 Retained: Physics consistency check + Smoothing
  // =========================================================================
  static constexpr double kMaxDepthM{15.0};
  static constexpr double kMinDepthM{0.5};
  static constexpr double kMaxPitchRad{0.5236};     // ±30°
  static constexpr double kMaxDeltaYawRad{0.5236};  // 30°/frame
  static constexpr double kMaxDeltaDepthM{2.0};     // 2m/frame
  static constexpr double kMaxDeltaPitchRad{0.2618}; // 15°/frame

  static constexpr int kRecoveryFrames{1};
  static constexpr double kRecoveryAlpha{0.30};
  std::array<int, kMaxSlots> recovery_counter_{};

  struct LastValidPose {
    double pitch{0.0};
    double yaw{0.0};
    double depth{0.0};
    bool valid{false};
  };
  std::array<LastValidPose, kMaxSlots> last_valid_;

  // Median-anchored smoothing
  static constexpr int kHistoryWindowSize{15};
  static constexpr double kDeviationClampRatio{0.30};
  static constexpr double kDepthEmaAlpha{0.25};
  std::array<std::deque<double>, kMaxSlots> depth_history_;
  std::array<std::deque<double>, kMaxSlots> pitch_history_;
  std::array<double, kMaxSlots> ema_depth_{};
  std::array<double, kMaxSlots> ema_pitch_{};
  std::array<bool, kMaxSlots> ema_initialized_{};
  std::array<bool, kMaxSlots> slot_just_activated_{};

  // Frozen detection
  static constexpr int kFrozenResetThreshold{20};
  std::array<int, kMaxSlots> consecutive_frozen_{};
  std::array<double, kMaxSlots> last_ema_depth_{};
  std::array<double, kMaxSlots> last_ema_pitch_{};

  // Quaternion SLERP
  static constexpr double kSlerpAlpha{0.25};
  std::array<Eigen::Quaterniond, kMaxSlots> prev_quat_;
  std::array<bool, kMaxSlots> quat_initialized_{};

  // Camera intrinsics
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  std::vector<cv::Point3f> model_points_;
};

}  // namespace rm_autoaim