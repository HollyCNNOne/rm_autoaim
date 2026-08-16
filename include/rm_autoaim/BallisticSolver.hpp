#pragma once

#include "rm_autoaim/Types.hpp"

#include <array>
#include <vector>

namespace rm_autoaim {

// ============================================================================
// BallisticSolver — Module 5: Ballistic Trajectory Computation
//
// Given predicted target states, computes the required gun pitch and yaw
// angles to hit the target, accounting for:
//   - Gravity (parabolic trajectory)
//   - Target motion compensation (forward prediction)
//   - Outpost rotation compensation
//   - Air resistance (optional, bonus)
//
// Iterative solver: pitch → flight time → pitch → ... until convergence
// ============================================================================

class BallisticSolver {
public:
  BallisticSolver();

  // Compute aim angles for all predicted targets
  [[nodiscard]] auto solve(const std::vector<PredictedState>& predictions,
                           const Quaternion& imu) -> std::vector<AimAngle>;

  // Set bullet speed (default: 15 m/s)
  auto set_bullet_speed(double v0) -> void;

  // Enable/disable air resistance model
  auto set_air_resistance(bool enable) -> void;

private:
  // Compute aim for a single target
  [[nodiscard]] auto solve_single(const PredictedState& target,
                                  const Quaternion& imu) -> AimAngle;

  // Gravity-only model: compute pitch iteratively
  [[nodiscard]] auto solve_pitch_gravity(double dx, double dy, double dz,
                                         double v0) -> std::pair<double, double>;

  // Air resistance model: compute pitch via binary search + numerical integration
  [[nodiscard]] auto solve_pitch_drag(double dx, double dy, double dz,
                                      double v0) -> std::pair<double, double>;

  // Forward-predict target position by flight time
  [[nodiscard]] static auto compensate_motion(const PredictedState& target,
                                              double t_fly) -> Eigen::Vector3d;

  // Numerical integration (RK4) for drag model
  struct BulletState {
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
  };
  [[nodiscard]] auto integrate_drag(double pitch, double yaw, double v0,
                                    double t_max, double dt_step) const
      -> BulletState;

  // V5: per-target last valid state for rate-of-change guard
  struct LastValid {
    double depth{0.0};
    AimAngle aim;
    bool valid{false};
  };
  static constexpr int kMaxTargets{3};
  static constexpr double kDepthMin{0.5};
  static constexpr double kDepthMax{15.0};
  static constexpr double kTflyMin{0.05};   // 50ms
  static constexpr double kTflyMax{0.6};    // 600ms
  static constexpr double kPitchMin{-25.0 * M_PI / 180.0};
  static constexpr double kPitchMax{+25.0 * M_PI / 180.0};
  static constexpr double kDepthChangeMax{0.40};  // 40% rate-of-change limit
  std::array<LastValid, kMaxTargets> last_valid_{};

  double bullet_speed_{15.0};  // m/s
  bool use_air_resistance_{false};

  // Drag parameters
  double drag_coefficient_{0.5};
  double air_density_{1.225};
  double bullet_radius_{0.0085};  // 17mm / 2
  double bullet_mass_{0.0032};
  double cross_section_{};  // π * r²
};

}  // namespace rm_autoaim