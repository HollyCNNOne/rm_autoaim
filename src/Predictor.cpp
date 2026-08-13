#include "rm_autoaim/Predictor.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace rm_autoaim {

// ============================================================================
// Construction
// ============================================================================

Predictor::Predictor() = default;

// ============================================================================
// Public API
// ============================================================================

auto Predictor::predict(const std::vector<TrackedArmor>& tracks, double dt)
    -> std::vector<PredictedState> {
  std::vector<PredictedState> results;

  // Mark all existing filters as not-matched
  std::vector<bool> filter_matched(filters_.size(), false);

  for (const auto& track : tracks) {
    // Skip non-confirmed tracks
    if (track.status != TrackedArmor::Status::kConfirmed &&
        track.status != TrackedArmor::Status::kLost) {
      continue;
    }

    // Find matching filter
    auto it = std::find_if(filters_.begin(), filters_.end(),
                           [&](const TargetFilter& f) {
                             return f.target_id == track.id;
                           });

    if (it == filters_.end()) {
      // New target: initialize filter
      auto new_filter = init_filter(track);
      filters_.push_back(new_filter);
      filter_matched.push_back(false);  // sync with newly added filter
      it = filters_.end() - 1;
    }

    auto& filter = *it;
    auto idx = std::distance(filters_.begin(), it);
    filter_matched[idx] = true;

    // Update position filter
    update_position_filter(filter, track.pose, dt);

    // Update rotation filter
    update_rotation_filter(filter, track.pose, dt);

    // Build predicted state
    PredictedState state;
    state.target_id = track.id;

    auto pos_state = filter.position_filter.state();
    state.position = pos_state.head<3>();
    state.velocity = pos_state.tail<3>();
    state.position_covariance = filter.position_filter.covariance();

    auto rot_state = filter.rotation_filter.state();
    state.rotation_angle = rot_state(0);
    state.rotation_velocity = rot_state(1);
    state.rotation_acceleration = rot_state(2);
    state.rotation_covariance = filter.rotation_filter.covariance();

    results.push_back(state);
  }

  // Remove filters for targets that are no longer tracked
  for (int i = static_cast<int>(filters_.size()) - 1; i >= 0; --i) {
    if (!filter_matched[i]) {
      filters_.erase(filters_.begin() + i);
    }
  }

  return results;
}

auto Predictor::reset() -> void { filters_.clear(); }

// ============================================================================
// Filter Initialization
// ============================================================================

auto Predictor::init_filter(const TrackedArmor& track) -> TargetFilter {
  TargetFilter filter;
  filter.target_id = track.id;

  // Position filter: 6D CV model
  // State: [x, y, z, vx, vy, vz]
  internal::KalmanFilter6D::StateVec x0 = internal::KalmanFilter6D::StateVec::Zero();
  x0(0) = track.pose.translation.x();
  x0(1) = track.pose.translation.y();
  x0(2) = track.pose.translation.z();
  // Velocity initialized to zero (no prior info)

  internal::KalmanFilter6D::StateMat p0 =
      internal::KalmanFilter6D::StateMat::Identity() * 10.0;
  p0(0, 0) = 0.01;  // x: moderately certain
  p0(1, 1) = 0.01;
  p0(2, 2) = 0.01;
  p0(3, 3) = 100.0;  // vx: very uncertain
  p0(4, 4) = 100.0;
  p0(5, 5) = 100.0;

  filter.position_filter.init(x0, p0);

  // Rotation filter: 3D Singer model
  // State: [θ, ω, α]
  double angle = extract_rotation_angle(track.pose);
  internal::KalmanFilter3D::StateVec r0 = internal::KalmanFilter3D::StateVec::Zero();
  r0(0) = angle;

  internal::KalmanFilter3D::StateMat rp0 =
      internal::KalmanFilter3D::StateMat::Identity();
  rp0(0, 0) = 0.01;    // θ: moderately certain
  rp0(1, 1) = 100.0;   // ω: very uncertain
  rp0(2, 2) = 1000.0;  // α: extremely uncertain

  filter.rotation_filter.init(r0, rp0);

  // Set angle normalizer for rotation state
  filter.rotation_filter.set_angle_normalizer(
      0, [](double a) { return normalize_angle(a); });

  filter.initialized = true;

  return filter;
}

// ============================================================================
// Position Filter Update (CV Model)
// ============================================================================

auto Predictor::update_position_filter(TargetFilter& filter,
                                       const ArmorPose& pose, double dt)
    -> void {
  // Build state transition matrix for CV model
  internal::KalmanFilter6D::StateMat F;
  F.setIdentity();
  F(0, 3) = dt;  // x += vx * dt
  F(1, 4) = dt;  // y += vy * dt
  F(2, 5) = dt;  // z += vz * dt

  // Build process noise matrix (Discrete White Noise Acceleration)
  double dt2 = dt * dt / 2.0;
  double dt3 = dt * dt * dt / 3.0;
  double q_p = process_noise_pos_;
  double q_v = process_noise_vel_;

  internal::KalmanFilter6D::StateMat Q = internal::KalmanFilter6D::StateMat::Zero();
  for (int i = 0; i < 3; ++i) {
    Q(i, i) = q_p * dt3;
    Q(i, i + 3) = q_p * dt2;
    Q(i + 3, i) = q_p * dt2;
    Q(i + 3, i + 3) = q_v * dt;
  }

  // Observation matrix: measure position only
  Eigen::Matrix<double, 3, 6> H;
  H.setZero();
  H(0, 0) = 1.0;
  H(1, 1) = 1.0;
  H(2, 2) = 1.0;

  // Measurement noise
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * measurement_noise_pos_;

  filter.position_filter.set_transition(F);
  filter.position_filter.set_process_noise(Q);
  filter.position_filter.set_observation(H);
  filter.position_filter.set_measurement_noise(R);

  // Predict
  filter.position_filter.predict();

  // Update with measurement
  Eigen::Vector3d z(pose.translation.x(), pose.translation.y(),
                    pose.translation.z());
  filter.position_filter.update(z);
}

// ============================================================================
// Rotation Filter Update (Singer Model)
// ============================================================================

auto Predictor::update_rotation_filter(TargetFilter& filter,
                                       const ArmorPose& pose, double dt)
    -> void {
  double angle = extract_rotation_angle(pose);

  double beta = singer_beta_;
  double exp_bt = std::exp(-beta * dt);
  double T = (beta * dt - 1.0 + exp_bt) / (beta * beta);

  // Singer model state transition matrix
  internal::KalmanFilter3D::StateMat F;
  F.setIdentity();
  F(0, 0) = 1.0;
  F(0, 1) = dt;
  F(0, 2) = T;
  F(1, 0) = 0.0;
  F(1, 1) = 1.0;
  F(1, 2) = (1.0 - exp_bt) / beta;
  F(2, 0) = 0.0;
  F(2, 1) = 0.0;
  F(2, 2) = exp_bt;

  // Process noise (simplified Singer model Q)
  internal::KalmanFilter3D::StateMat Q = internal::KalmanFilter3D::StateMat::Zero();
  Q(0, 0) = process_noise_rot_ * dt * dt * dt / 3.0;
  Q(1, 1) = process_noise_rot_ * dt;
  Q(2, 2) = process_noise_rot_ * (1.0 - exp_bt * exp_bt) / (2.0 * beta);

  // Observation: measure angle only
  Eigen::Matrix<double, 1, 3> H;
  H(0, 0) = 1.0;
  H(0, 1) = 0.0;
  H(0, 2) = 0.0;

  // Measurement noise
  Eigen::Matrix<double, 1, 1> R_mat;
  R_mat(0, 0) = measurement_noise_rot_;

  filter.rotation_filter.set_transition(F);
  filter.rotation_filter.set_process_noise(Q);
  filter.rotation_filter.set_observation(H);
  filter.rotation_filter.set_measurement_noise(R_mat);

  // Predict
  filter.rotation_filter.predict();

  // Update with measurement
  internal::KalmanFilter3D::MeasVec z;
  z(0) = angle;
  filter.rotation_filter.update(z);
}

// ============================================================================
// Rotation Angle Extraction
// ============================================================================

auto Predictor::extract_rotation_angle(const ArmorPose& pose) -> double {
  // Extract rotation around Z-axis from the rotation matrix
  // R = Rz(θ) * ... → θ = atan2(R(1,0), R(0,0))
  double angle = std::atan2(pose.rotation(1, 0), pose.rotation(0, 0));
  return normalize_angle(angle);
}

auto Predictor::normalize_angle(double angle) -> double {
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

}  // namespace rm_autoaim