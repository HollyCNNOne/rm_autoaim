#pragma once

#include "rm_autoaim/Types.hpp"
#include "rm_autoaim/internal/KalmanFilter.hpp"

#include <vector>

namespace rm_autoaim {

// ============================================================================
// Predictor — Module 4: Kalman Filter Prediction
//
// Tracks target motion using two Kalman filters:
//   1. 6D CV model: [x, y, z, vx, vy, vz] for position tracking
//   2. 3D Singer model: [θ, ω, α] for outpost rotation tracking
//
// Predicts future position by forward propagation.
// ============================================================================

class Predictor {
public:
  Predictor();

  // Process tracked targets and produce predicted states
  [[nodiscard]] auto predict(const std::vector<TrackedArmor>& tracks,
                             double dt) -> std::vector<PredictedState>;

  // Reset all filters
  auto reset() -> void;

private:
  // Per-target Kalman filter state
  struct TargetFilter {
    int target_id{-1};
    internal::KalmanFilter6D position_filter;  // CV model
    internal::KalmanFilter3D rotation_filter;  // Singer model
    bool initialized{false};
  };

  // Initialize a new filter for a target
  auto init_filter(const TrackedArmor& track) -> TargetFilter;

  // Update position filter with measurement
  auto update_position_filter(TargetFilter& filter,
                              const ArmorPose& pose, double dt) -> void;

  // Update rotation filter with measurement
  auto update_rotation_filter(TargetFilter& filter,
                              const ArmorPose& pose, double dt) -> void;

  // Extract rotation angle from pose
  [[nodiscard]] static auto extract_rotation_angle(const ArmorPose& pose)
      -> double;

  // Angle normalization: wrap to [-π, π]
  [[nodiscard]] static auto normalize_angle(double angle) -> double;

  // Active filters
  std::vector<TargetFilter> filters_;

  // Model parameters
  double process_noise_pos_{0.01};   // position process noise
  double process_noise_vel_{0.1};    // velocity process noise
  double process_noise_rot_{0.001};  // rotation process noise
  double measurement_noise_pos_{0.001};  // position measurement noise
  double measurement_noise_rot_{0.01};   // rotation measurement noise
  double singer_beta_{0.5};  // Singer model maneuver frequency
};

}  // namespace rm_autoaim