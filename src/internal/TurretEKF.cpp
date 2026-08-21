#include "rm_autoaim/internal/TurretEKF.hpp"

#include <cmath>

#include <spdlog/spdlog.h>

namespace rm_autoaim {
namespace internal {

// ============================================================================
// Constructor
// ============================================================================

TurretEKF::TurretEKF() {
  x_ = StateVec::Zero();
  P_ = StateMat::Zero();
  Q_ = StateMat::Zero();
  R_ = Eigen::Matrix<double, kMeasDim, kMeasDim>::Zero();
}

// ============================================================================
// init() — initialize from first measurement
// ============================================================================

auto TurretEKF::init(const Eigen::Vector3d& armor_pos, double phase_offset) -> void {
  double xa = armor_pos(0);
  double ya = armor_pos(1);
  double za = armor_pos(2);

  // Back-project turret center from armor position:
  //   xa = xc - r*cos(yaw+phi)  →  xc = xa + r*cos(yaw+phi)
  //   ya = yc - r*sin(yaw+phi)  →  yc = ya + r*sin(yaw+phi)
  // Initial yaw = 0, radius = kInitRadius
  double yaw0 = 0.0;
  double r0 = kInitRadius;

  x_(0) = xa + r0 * std::cos(yaw0 + phase_offset);  // xc
  x_(1) = 0.0;                                       // vxc
  x_(2) = ya + r0 * std::sin(yaw0 + phase_offset);  // yc
  x_(3) = 0.0;                                       // vyc
  x_(4) = za;                                        // za
  x_(5) = 0.0;                                       // vza
  x_(6) = yaw0;                                      // yaw
  x_(7) = 0.0;                                       // vyaw
  x_(8) = r0;                                        // r

  // Initial covariance: diagonal with large uncertainties
  P_.setZero();
  P_(0, 0) = kInitPosUncertainty * kInitPosUncertainty;
  P_(1, 1) = kInitVelUncertainty * kInitVelUncertainty;
  P_(2, 2) = kInitPosUncertainty * kInitPosUncertainty;
  P_(3, 3) = kInitVelUncertainty * kInitVelUncertainty;
  P_(4, 4) = kInitPosUncertainty * kInitPosUncertainty;
  P_(5, 5) = kInitVelUncertainty * kInitVelUncertainty;
  P_(6, 6) = kInitYawUncertainty * kInitYawUncertainty;
  P_(7, 7) = kInitVyawUncertainty * kInitVyawUncertainty;
  P_(8, 8) = kInitRUncertainty * kInitRUncertainty;

  // Process noise (continuous, scaled by dt in predict)
  Q_.setZero();
  Q_(0, 0) = kProcessNoisePos * kProcessNoisePos;
  Q_(1, 1) = kProcessNoiseVel * kProcessNoiseVel;
  Q_(2, 2) = kProcessNoisePos * kProcessNoisePos;
  Q_(3, 3) = kProcessNoiseVel * kProcessNoiseVel;
  Q_(4, 4) = kProcessNoisePos * kProcessNoisePos;
  Q_(5, 5) = kProcessNoiseVel * kProcessNoiseVel;
  Q_(6, 6) = kProcessNoiseYaw * kProcessNoiseYaw;
  Q_(7, 7) = kProcessNoiseVyaw * kProcessNoiseVyaw;
  Q_(8, 8) = kProcessNoiseR * kProcessNoiseR;

  // Measurement noise
  R_.setZero();
  R_(0, 0) = kMeasNoiseXY * kMeasNoiseXY;
  R_(1, 1) = kMeasNoiseXY * kMeasNoiseXY;
  R_(2, 2) = kMeasNoiseZ * kMeasNoiseZ;

  initialized_ = true;
  spdlog::info("[TurretEKF] Initialized: xc={:.3f}, yc={:.3f}, za={:.3f}, "
               "yaw={:.3f}°, r={:.3f}",
               x_(0), x_(2), x_(4), x_(6) * 180.0 / M_PI, x_(8));
}

// ============================================================================
// predict() — CV model forward propagation
// ============================================================================

auto TurretEKF::predict(double dt) -> void {
  if (!initialized_) return;

  // Build state transition matrix F
  StateMat F = StateMat::Identity();
  F(0, 1) = dt;   // xc = xc + vxc*dt
  F(2, 3) = dt;   // yc = yc + vyc*dt
  F(4, 5) = dt;   // za = za + vza*dt
  F(6, 7) = dt;   // yaw = yaw + vyaw*dt
  // r is constant: F(8,8) = 1 (already identity)

  // Predict state
  x_ = F * x_;

  // Wrap yaw to [-π, π]
  x_(6) = std::fmod(x_(6) + M_PI, 2.0 * M_PI);
  if (x_(6) < 0.0) x_(6) += 2.0 * M_PI;
  x_(6) -= M_PI;

  // Predict covariance
  P_ = F * P_ * F.transpose() + Q_ * dt;
}

// ============================================================================
// update() — EKF measurement update with linearized observation model
// ============================================================================

auto TurretEKF::update(const Eigen::Vector3d& armor_pos, double phase_offset) -> void {
  if (!initialized_) return;

  double xc = x_(0);
  double yc = x_(2);
  double za = x_(4);
  double yaw = x_(6);
  double r = x_(8);

  double cos_term = std::cos(yaw + phase_offset);
  double sin_term = std::sin(yaw + phase_offset);

  // Predicted measurement: h(x, phi)
  MeasVec z_pred;
  z_pred(0) = xc - r * cos_term;  // xa
  z_pred(1) = yc - r * sin_term;  // ya
  z_pred(2) = za;                  // za

  // Innovation
  MeasVec y = armor_pos - z_pred;

  // Measurement Jacobian H (3×9)
  // h1 = xc - r*cos(yaw+phi) → dh1/dyaw = r*sin(yaw+phi), dh1/dr = -cos(yaw+phi)
  // h2 = yc - r*sin(yaw+phi) → dh2/dyaw = -r*cos(yaw+phi), dh2/dr = -sin(yaw+phi)
  // h3 = za → dh3/dza = 1
  MeasJacobian H = MeasJacobian::Zero();
  H(0, 0) = 1.0;                     // dh1/dxc
  H(0, 6) = r * sin_term;            // dh1/dyaw
  H(0, 8) = -cos_term;               // dh1/dr
  H(1, 2) = 1.0;                     // dh2/dyc
  H(1, 6) = -r * cos_term;           // dh2/dyaw
  H(1, 8) = -sin_term;               // dh2/dr
  H(2, 4) = 1.0;                     // dh3/dza

  // Innovation covariance
  Eigen::Matrix<double, kMeasDim, kMeasDim> S = H * P_ * H.transpose() + R_;

  // Kalman gain
  Eigen::Matrix<double, kStateDim, kMeasDim> K = P_ * H.transpose() * S.inverse();

  // State update
  x_ = x_ + K * y;

  // Wrap yaw
  x_(6) = std::fmod(x_(6) + M_PI, 2.0 * M_PI);
  if (x_(6) < 0.0) x_(6) += 2.0 * M_PI;
  x_(6) -= M_PI;

  // Covariance update (Joseph form for numerical stability)
  StateMat I = StateMat::Identity();
  StateMat I_KH = I - K * H;
  P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();
}

// ============================================================================
// handleArmorJump() — phase transition without filter reset
// ============================================================================

auto TurretEKF::handleArmorJump(double delta_yaw) -> void {
  if (!initialized_) return;

  x_(6) += delta_yaw;

  // Wrap yaw
  x_(6) = std::fmod(x_(6) + M_PI, 2.0 * M_PI);
  if (x_(6) < 0.0) x_(6) += 2.0 * M_PI;
  x_(6) -= M_PI;

  // Inflate yaw covariance to account for phase uncertainty
  // (exact phase offset may not be exactly 2π/3 due to mechanical tolerances)
  P_(6, 6) += 0.05 * 0.05;  // ~3° extra uncertainty

  spdlog::info("[TurretEKF] Armor jump: +{:.1f}° → yaw={:.1f}°",
               delta_yaw * 180.0 / M_PI, x_(6) * 180.0 / M_PI);
}

// ============================================================================
// predictArmorPos() — predicted armor plate position for a given phase
// ============================================================================

auto TurretEKF::predictArmorPos(double phase_offset) const -> Eigen::Vector3d {
  double xc = x_(0);
  double yc = x_(2);
  double za = x_(4);
  double yaw = x_(6);
  double r = x_(8);

  double cos_term = std::cos(yaw + phase_offset);
  double sin_term = std::sin(yaw + phase_offset);

  return {
    xc - r * cos_term,  // xa
    yc - r * sin_term,  // ya
    za                  // za
  };
}

}  // namespace internal
}  // namespace rm_autoaim