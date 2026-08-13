#include "rm_autoaim/BallisticSolver.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace rm_autoaim {

// ============================================================================
// Construction
// ============================================================================

BallisticSolver::BallisticSolver()
    : cross_section_(M_PI * bullet_radius_ * bullet_radius_) {}

// ============================================================================
// Public API
// ============================================================================

auto BallisticSolver::solve(const std::vector<PredictedState>& predictions,
                            const Quaternion& imu) -> std::vector<AimAngle> {
  std::vector<AimAngle> results;

  for (const auto& pred : predictions) {
    auto aim = solve_single(pred, imu);
    if (aim.flight_time > 0.0) {
      results.push_back(aim);
    }
  }

  return results;
}

auto BallisticSolver::set_bullet_speed(double v0) -> void {
  bullet_speed_ = v0;
}

auto BallisticSolver::set_air_resistance(bool enable) -> void {
  use_air_resistance_ = enable;
}

// ============================================================================
// Single Target Solver
// ============================================================================

auto BallisticSolver::solve_single(const PredictedState& target,
                                   const Quaternion& imu) -> AimAngle {
  AimAngle result;
  result.target_id = target.target_id;

  // Sanity check: reject implausible depth (PnP may fail on bad pairings)
  // Outpost is expected at 3–30 m; reject anything outside [0.5, 50]
  double depth = target.position.z();
  if (depth < 0.5 || depth > 50.0 || !std::isfinite(depth)) {
    return result;
  }
  double horizontal_dist_cam = std::sqrt(target.position.x() * target.position.x()
                                         + target.position.y() * target.position.y());
  if (horizontal_dist_cam < 1e-6 || horizontal_dist_cam > 50.0) {
    return result;
  }

  // Transform target position from camera frame to world frame (Z-up)
  // Camera: x=right, y=down, z=forward
  // World:   x=right, y=forward, z=up
  // Mapping: world_x = cam_x, world_y = cam_z, world_z = -cam_y
  Eigen::Vector3d world_pos;
  world_pos.x() = target.position.x();
  world_pos.y() = target.position.z();
  world_pos.z() = -target.position.y();

  // Apply IMU rotation (for moving platforms; identity for fixed outpost camera)
  auto R_imu = imu.to_rotation_matrix();
  world_pos = R_imu * world_pos;

  double dx = world_pos.x();
  double dy = world_pos.y();
  double dz = world_pos.z();

  // Initial flight time estimate (no motion compensation)
  double horizontal_dist = std::sqrt(dx * dx + dy * dy);
  if (horizontal_dist < 1e-6) {
    return result;  // target too close, can't compute
  }

  double t_fly = horizontal_dist / bullet_speed_;

  // Iterative refinement: compensate motion → recompute → repeat
  constexpr int kMaxIter = 5;
  double pitch = 0.0;
  double yaw = 0.0;

  for (int iter = 0; iter < kMaxIter; ++iter) {
    // Compensate for target motion during flight time
    Eigen::Vector3d aim_pos = compensate_motion(target, t_fly);

    // Transform aim_pos from camera to world frame (same mapping as world_pos)
    Eigen::Vector3d aim_world;
    aim_world.x() = aim_pos.x();
    aim_world.y() = aim_pos.z();
    aim_world.z() = -aim_pos.y();

    aim_world = R_imu * aim_world;

    dx = aim_world.x();
    dy = aim_world.y();
    dz = aim_world.z();

    horizontal_dist = std::sqrt(dx * dx + dy * dy);

    // Solve pitch (and new flight time)
    std::tie(pitch, t_fly) = use_air_resistance_
        ? solve_pitch_drag(dx, dy, dz, bullet_speed_)
        : solve_pitch_gravity(dx, dy, dz, bullet_speed_);

    if (t_fly <= 0.0) {
      return result;
    }
  }

  // Compute yaw
  yaw = std::atan2(dy, dx);

  result.pitch = pitch;
  result.yaw = yaw;
  result.flight_time = t_fly;

  return result;
}

// ============================================================================
// Gravity-Only Iterative Pitch Solver
// ============================================================================

auto BallisticSolver::solve_pitch_gravity(double dx, double dy, double dz,
                                          double v0)
    -> std::pair<double, double> {
  double horizontal_dist = std::sqrt(dx * dx + dy * dy);
  constexpr int kMaxIter = 10;
  constexpr double kTolerance = 1e-6;

  double pitch = 0.0;
  double t_fly = horizontal_dist / v0;

  for (int i = 0; i < kMaxIter; ++i) {
    // Compute required pitch to hit target at distance dz with flight time t_fly
    // dz = v0 * sin(pitch) * t_fly - 0.5 * g * t_fly²
    // → sin(pitch) = (dz + 0.5 * g * t_fly²) / (v0 * t_fly)
    double numerator = dz + 0.5 * constants::kGravity * t_fly * t_fly;
    double denominator = v0 * t_fly;

    if (denominator < 1e-10) {
      return {0.0, 0.0};
    }

    double sin_pitch = numerator / denominator;
    sin_pitch = std::clamp(sin_pitch, -1.0, 1.0);
    double new_pitch = std::asin(sin_pitch);

    // Recompute flight time with new pitch
    double cos_pitch = std::cos(new_pitch);
    if (cos_pitch < 1e-6) {
      return {0.0, 0.0};  // trajectory physically impossible (too steep)
    }
    double new_t_fly = horizontal_dist / (v0 * cos_pitch);

    if (new_t_fly <= 0.0 || !std::isfinite(new_t_fly)) {
      return {0.0, 0.0};
    }

    // Check convergence
    if (std::abs(new_pitch - pitch) < kTolerance &&
        std::abs(new_t_fly - t_fly) < kTolerance) {
      return {new_pitch, new_t_fly};
    }

    pitch = new_pitch;
    t_fly = new_t_fly;
  }

  return {pitch, t_fly};
}

// ============================================================================
// Air Resistance Pitch Solver (Binary Search)
// ============================================================================

auto BallisticSolver::solve_pitch_drag(double dx, double dy, double dz,
                                       double v0)
    -> std::pair<double, double> {
  double horizontal_dist = std::sqrt(dx * dx + dy * dy);

  // Binary search for pitch
  constexpr int kMaxIter = 15;
  constexpr double kTolerance = 1e-3;  // 1mm tolerance

  double pitch_low = -M_PI / 4.0;   // -45°
  double pitch_high = M_PI / 4.0;   // +45°
  double pitch = 0.0;
  double t_fly = horizontal_dist / v0;

  for (int i = 0; i < kMaxIter; ++i) {
    pitch = (pitch_low + pitch_high) / 2.0;
    double yaw = std::atan2(dy, dx);

    // Integrate trajectory
    auto state = integrate_drag(pitch, yaw, v0, t_fly * 2.0, 0.001);

    double hit_z = state.pos.z();
    double hit_dist = std::sqrt(state.pos.x() * state.pos.x() +
                                state.pos.y() * state.pos.y());

    if (std::abs(hit_z - dz) < kTolerance &&
        std::abs(hit_dist - horizontal_dist) / horizontal_dist < 0.01) {
      t_fly = std::sqrt(state.pos.x() * state.pos.x() +
                        state.pos.y() * state.pos.y() +
                        (state.pos.z() - dz) * (state.pos.z() - dz)) /
              v0;
      return {pitch, t_fly};
    }

    if (hit_z > dz) {
      pitch_high = pitch;
    } else {
      pitch_low = pitch;
    }
  }

  t_fly = horizontal_dist / (v0 * std::cos(pitch));
  return {pitch, t_fly};
}

// ============================================================================
// Motion Compensation
// ============================================================================

auto BallisticSolver::compensate_motion(const PredictedState& target,
                                        double t_fly) -> Eigen::Vector3d {
  // Forward-predict position using constant velocity
  Eigen::Vector3d aim_pos = target.position + target.velocity * t_fly;

  // Outpost rotation compensation: the armor plate rotates during flight
  double theta_aim = target.rotation_angle + target.rotation_velocity * t_fly;

  // Rotate the aim position by the additional rotation
  double cos_t = std::cos(theta_aim - target.rotation_angle);
  double sin_t = std::sin(theta_aim - target.rotation_angle);

  // Apply rotation around Z-axis
  Eigen::Vector3d rotated;
  rotated.x() = aim_pos.x() * cos_t - aim_pos.y() * sin_t;
  rotated.y() = aim_pos.x() * sin_t + aim_pos.y() * cos_t;
  rotated.z() = aim_pos.z();

  return rotated;
}

// ============================================================================
// Numerical Integration (RK4) for Drag Model
// ============================================================================

auto BallisticSolver::integrate_drag(double pitch, double yaw, double v0,
                                     double t_max, double dt_step) const
    -> BulletState {
  BulletState state;
  state.pos = Eigen::Vector3d::Zero();
  state.vel = Eigen::Vector3d(
      v0 * std::cos(pitch) * std::cos(yaw),
      v0 * std::cos(pitch) * std::sin(yaw),
      v0 * std::sin(pitch));

  // Drag coefficient: k = 0.5 * ρ * C_d * A / m
  double k = 0.5 * air_density_ * drag_coefficient_ * cross_section_ / bullet_mass_;

  for (double t = 0.0; t < t_max; t += dt_step) {
    // RK4 integration
    auto deriv = [&](const BulletState& s) -> BulletState {
      double v = s.vel.norm();
      double drag_acc = k * v * v;

      BulletState d;
      d.vel = -drag_acc * s.vel.normalized();
      d.vel.z() -= constants::kGravity;
      d.pos = s.vel;
      return d;
    };

    BulletState k1 = deriv(state);
    k1.vel *= dt_step;
    k1.pos *= dt_step;

    BulletState s2;
    s2.pos = state.pos + k1.pos * 0.5;
    s2.vel = state.vel + k1.vel * 0.5;
    BulletState k2 = deriv(s2);
    k2.vel *= dt_step;
    k2.pos *= dt_step;

    BulletState s3;
    s3.pos = state.pos + k2.pos * 0.5;
    s3.vel = state.vel + k2.vel * 0.5;
    BulletState k3 = deriv(s3);
    k3.vel *= dt_step;
    k3.pos *= dt_step;

    BulletState s4;
    s4.pos = state.pos + k3.pos;
    s4.vel = state.vel + k3.vel;
    BulletState k4 = deriv(s4);
    k4.vel *= dt_step;
    k4.pos *= dt_step;

    state.pos += (k1.pos + 2.0 * k2.pos + 2.0 * k3.pos + k4.pos) / 6.0;
    state.vel += (k1.vel + 2.0 * k2.vel + 2.0 * k3.vel + k4.vel) / 6.0;

    // Stop if bullet hits ground level (z < some threshold)
    if (state.pos.z() < -10.0) {
      break;
    }
  }

  return state;
}

}  // namespace rm_autoaim