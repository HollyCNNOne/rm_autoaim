#pragma once

#include <Eigen/Dense>

#include <functional>

namespace rm_autoaim::internal {

// ============================================================================
// Generic Kalman Filter (Linear, Discrete-Time)
//
// State transition:  x_k = F * x_{k-1} + w_k    (w ~ N(0, Q))
// Measurement:       z_k = H * x_k + v_k         (v ~ N(0, R))
//
// Template parameters:
//   StateDim  — dimension of state vector x
//   MeasDim   — dimension of measurement vector z
// ============================================================================

template <int StateDim, int MeasDim>
class KalmanFilter {
public:
  using StateVec = Eigen::Matrix<double, StateDim, 1>;
  using StateMat = Eigen::Matrix<double, StateDim, StateDim>;
  using MeasVec = Eigen::Matrix<double, MeasDim, 1>;
  using MeasMat = Eigen::Matrix<double, MeasDim, MeasDim>;
  using GainMat = Eigen::Matrix<double, StateDim, MeasDim>;

  KalmanFilter() = default;

  // Initialize with initial state and covariance
  auto init(const StateVec& x0, const StateMat& p0) -> void;

  // Set model matrices (must be called before predict/update)
  auto set_transition(const StateMat& F) -> void;
  auto set_process_noise(const StateMat& Q) -> void;
  auto set_observation(const Eigen::Matrix<double, MeasDim, StateDim>& H)
      -> void;
  auto set_measurement_noise(const MeasMat& R) -> void;

  // Prediction step: x̂_k⁻ = F·x̂_{k-1},  P_k⁻ = F·P_{k-1}·Fᵀ + Q
  auto predict() -> void;

  // Update step: incorporate measurement z_k
  // K = P⁻·Hᵀ·(H·P⁻·Hᵀ + R)⁻¹
  // x̂ = x̂⁻ + K·(z - H·x̂⁻)
  // P = (I - K·H)·P⁻
  auto update(const MeasVec& z) -> void;

  // Accessors
  [[nodiscard]] auto state() const -> const StateVec&;
  [[nodiscard]] auto covariance() const -> const StateMat&;
  [[nodiscard]] auto kalman_gain() const -> const GainMat&;

  // Predict forward t seconds (constant velocity propagation)
  [[nodiscard]] auto predict_forward(double dt) const -> StateVec;

  // Custom angle normalization for rotational states
  using AngleNormalizer = std::function<double(double)>;
  auto set_angle_normalizer(int index, AngleNormalizer fn) -> void;

private:
  StateVec x_{StateVec::Zero()};
  StateMat P_{StateMat::Identity()};
  StateMat F_{StateMat::Identity()};
  StateMat Q_{StateMat::Identity()};
  Eigen::Matrix<double, MeasDim, StateDim> H_{
      Eigen::Matrix<double, MeasDim, StateDim>::Zero()};
  MeasMat R_{MeasMat::Identity()};
  GainMat K_{GainMat::Zero()};

  bool initialized_{false};

  // Angle normalizer per state index
  std::unordered_map<int, AngleNormalizer> angle_normalizers_;

  auto apply_angle_normalizers() -> void;
};

// ============================================================================
// Template implementation
// ============================================================================

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::init(const StateVec& x0,
                                           const StateMat& p0) -> void {
  x_ = x0;
  P_ = p0;
  initialized_ = true;
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::set_transition(const StateMat& F)
    -> void {
  F_ = F;
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::set_process_noise(const StateMat& Q)
    -> void {
  Q_ = Q;
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::set_observation(
    const Eigen::Matrix<double, MeasDim, StateDim>& H) -> void {
  H_ = H;
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::set_measurement_noise(const MeasMat& R)
    -> void {
  R_ = R;
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::predict() -> void {
  // x̂_k⁻ = F · x̂_{k-1}
  x_ = F_ * x_;

  // P_k⁻ = F · P_{k-1} · Fᵀ + Q
  P_ = F_ * P_ * F_.transpose() + Q_;

  apply_angle_normalizers();
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::update(const MeasVec& z) -> void {
  // Innovation: y = z - H·x̂⁻
  MeasVec y = z - H_ * x_;

  // Innovation covariance: S = H·P⁻·Hᵀ + R
  MeasMat S = H_ * P_ * H_.transpose() + R_;

  // Kalman gain: K = P⁻·Hᵀ·S⁻¹
  K_ = P_ * H_.transpose() * S.inverse();

  // State update: x̂ = x̂⁻ + K·y
  x_ += K_ * y;

  // Covariance update: P = (I - K·H)·P⁻
  StateMat I = StateMat::Identity();
  P_ = (I - K_ * H_) * P_;

  apply_angle_normalizers();
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::state() const -> const StateVec& {
  return x_;
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::covariance() const -> const StateMat& {
  return P_;
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::kalman_gain() const -> const GainMat& {
  return K_;
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::predict_forward(double dt) const
    -> StateVec {
  // Forward-predict state assuming constant velocity for dt seconds
  // Build a temporary F(dt) matrix
  StateMat F_dt = StateMat::Identity();
  for (int i = 0; i < StateDim / 2; ++i) {
    F_dt(i, i + StateDim / 2) = dt;
  }
  return F_dt * x_;
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::set_angle_normalizer(
    int index, AngleNormalizer fn) -> void {
  angle_normalizers_[index] = std::move(fn);
}

template <int StateDim, int MeasDim>
auto KalmanFilter<StateDim, MeasDim>::apply_angle_normalizers() -> void {
  for (const auto& [idx, fn] : angle_normalizers_) {
    if (idx < StateDim) {
      x_(idx) = fn(x_(idx));
    }
  }
}

// ============================================================================
// Common type aliases
// ============================================================================

using KalmanFilter6D = KalmanFilter<6, 3>;  // [x,y,z,vx,vy,vz], meas: [x,y,z]
using KalmanFilter3D = KalmanFilter<3, 1>;  // [θ,ω,α], meas: [θ]

}  // namespace rm_autoaim::internal