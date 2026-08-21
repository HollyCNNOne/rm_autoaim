#pragma once

#include <Eigen/Dense>

namespace rm_autoaim {
namespace internal {

// ============================================================================
// TurretEKF — 9D Extended Kalman Filter tracking the outpost turret center
// ============================================================================
// State vector (9D): [xc, vxc, yc, vyc, za, vza, yaw, vyaw, r]
//   xc, yc     : turret center in camera frame (m)
//   vxc, vyc   : turret center velocity (m/s)
//   za         : armor plate depth/height (m)
//   vza        : depth velocity (m/s)
//   yaw        : turret yaw angle (rad), current visible armor phase
//   vyaw       : angular velocity (rad/s)
//   r          : armor plate rotation radius from turret center (m)
//
// Measurement (3D): armor plate position from PnP [xa, ya, za]
//   h(x, phi) = [xc - r*cos(yaw+phi), yc - r*sin(yaw+phi), za]
// where phi is the fixed phase offset of the matched slot.
//
// Design rationale (SCUT approach):
//   - Track the turret center, not individual armor plates
//   - Phase transitions (armor jumps) are handled by adding delta_yaw
//     to the state — no filter reset, continuous state propagation
//   - Even without measurements, EKF::predict() keeps output smooth
//     (solves the "numeric freeze on frame drop" problem)
// ============================================================================

class TurretEKF {
public:
  static constexpr int kStateDim{9};
  static constexpr int kMeasDim{3};

  using StateVec = Eigen::Matrix<double, kStateDim, 1>;
  using StateMat = Eigen::Matrix<double, kStateDim, kStateDim>;
  using MeasVec = Eigen::Matrix<double, kMeasDim, 1>;
  using MeasJacobian = Eigen::Matrix<double, kMeasDim, kStateDim>;

  TurretEKF();

  // Initialize EKF with first armor plate measurement
  // armor_pos: [xa, ya, za] from PnP
  // phase_offset: slot phase (0, 2π/3, or 4π/3)
  auto init(const Eigen::Vector3d& armor_pos, double phase_offset = 0.0) -> void;

  // Predict forward by dt seconds (CV model for all states)
  // Must be called every frame, even without measurements.
  auto predict(double dt) -> void;

  // Update with armor plate measurement at given phase offset
  // EKF linearizes h(x, phi) around current state.
  auto update(const Eigen::Vector3d& armor_pos, double phase_offset) -> void;

  // Handle armor plate jump (phase transition)
  // Adds delta_yaw to state yaw + increases yaw covariance.
  // Call when expected_id changes (e.g., +2π/3 for next plate).
  auto handleArmorJump(double delta_yaw) -> void;

  // Predict armor plate position for a given phase offset
  // Used in associate() to compute prediction error.
  [[nodiscard]] auto predictArmorPos(double phase_offset) const -> Eigen::Vector3d;

  [[nodiscard]] auto isInitialized() const -> bool { return initialized_; }

  // Accessors
  [[nodiscard]] auto turretCenter() const -> Eigen::Vector3d {
    return {x_(0), x_(2), x_(4)};  // [xc, yc, za]
  }
  [[nodiscard]] auto turretYaw() const -> double { return x_(6); }
  [[nodiscard]] auto turretRadius() const -> double { return x_(8); }

private:
  StateVec x_;  // state vector
  StateMat P_;  // covariance matrix
  StateMat Q_;  // process noise
  Eigen::Matrix<double, kMeasDim, kMeasDim> R_;  // measurement noise
  bool initialized_{false};

  // Tuning parameters (SCUT reference values)
  static constexpr double kInitRadius{0.30};          // initial guess: 30cm radius
  static constexpr double kInitPosUncertainty{1.0};   // 1m initial σ
  static constexpr double kInitVelUncertainty{1.0};   // 1m/s initial σ
  static constexpr double kInitYawUncertainty{0.5};   // ~30° initial σ
  static constexpr double kInitVyawUncertainty{1.0};  // 1rad/s initial σ
  static constexpr double kInitRUncertainty{0.1};     // 10cm initial σ

  static constexpr double kProcessNoisePos{0.005};     // 5mm/sqrt(s)
  static constexpr double kProcessNoiseVel{0.05};      // 5cm/s/sqrt(s)
  static constexpr double kProcessNoiseYaw{0.01};      // 0.01rad/sqrt(s)
  static constexpr double kProcessNoiseVyaw{0.05};     // 0.05rad/s/sqrt(s)
  static constexpr double kProcessNoiseR{0.001};       // 1mm/sqrt(s)

  static constexpr double kMeasNoiseXY{0.015};  // 1.5cm σ (PnP noise)
  static constexpr double kMeasNoiseZ{0.03};    // 3cm σ (depth from PnP is noisier)
};

}  // namespace internal
}  // namespace rm_autoaim