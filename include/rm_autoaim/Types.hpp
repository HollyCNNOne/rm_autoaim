#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace rm_autoaim {

// ============================================================================
// IMU / Quaternion types
// ============================================================================

struct Quaternion {
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};

  [[nodiscard]] auto to_rotation_matrix() const -> Eigen::Matrix3d;
  [[nodiscard]] auto is_unit(double epsilon = 1e-6) const -> bool;
};

// ============================================================================
// Frame data — output of Module 1 (Reader)
// ============================================================================

struct FrameData {
  cv::Mat image;
  uint64_t timestamp_us{0};
  int64_t frame_index{0};
  Quaternion imu;
};

// ============================================================================
// 2D Armor plate detection — output of Module 2 (Detector)
// ============================================================================

struct Armor2D {
  // 4 corners in pixel coordinates: [TL, TR, BR, BL]
  std::array<cv::Point2f, 4> corners;
  float confidence{0.0F};
};

// ============================================================================
// 3D Pose — output of Module 3 (Tracker) PnP step
// ============================================================================

struct ArmorPose {
  Eigen::Vector3d translation;  // [tx, ty, tz] in camera frame
  Eigen::Matrix3d rotation;     // 3x3 rotation matrix (Rz·Ry·Rx convention)
  Eigen::Quaterniond quaternion;
  double depth{0.0};   // tz for convenience
  double pitch{0.0};   // rad, extracted from rotation matrix
  double yaw{0.0};     // rad, extracted from rotation matrix
};

// ============================================================================
// Tracked target — output of Module 3 (Tracker)
// ============================================================================

struct TrackedArmor {
  int id{-1};
  ArmorPose pose;
  Armor2D detection;  // corresponding 2D detection

  enum class Status {
    kTentative,  // newly created, not yet confirmed
    kConfirmed,  // tracked for >= 3 frames
    kLost,       // temporarily lost
    kDead        // removed
  };
  Status status{Status::kTentative};

  int consecutive_detections{0};
  int consecutive_misses{0};
  int age{0};  // total frames since creation
};

// ============================================================================
// Predicted state — output of Module 4 (Predictor)
// ============================================================================

struct PredictedState {
  int target_id{-1};

  // Position and velocity in world frame
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};

  // Outpost rotation state
  double rotation_angle{0.0};       // rad
  double rotation_velocity{0.0};    // rad/s
  double rotation_acceleration{0.0};  // rad/s²

  // Prediction confidence
  Eigen::Matrix<double, 6, 6> position_covariance;
  Eigen::Matrix<double, 3, 3> rotation_covariance;
};

// ============================================================================
// Aim angle — output of Module 5 (BallisticSolver)
// ============================================================================

struct AimAngle {
  double pitch{0.0};  // rad, positive = aim up
  double yaw{0.0};    // rad, positive = aim right
  double flight_time{0.0};  // seconds
  int target_id{-1};
};

// ============================================================================
// Pipeline statistics
// ============================================================================

struct StageStats {
  std::string name;
  double avg_latency_us{0.0};
  double max_latency_us{0.0};
  double min_latency_us{1e9};
  int64_t frames_processed{0};
  int64_t frames_dropped{0};
};

struct PipelineStats {
  StageStats reader;
  StageStats detector;
  StageStats tracker;
  StageStats predictor;
  StageStats ballistic;
  double total_avg_latency_us{0.0};
  int64_t total_frames{0};
};

// ============================================================================
// Physical constants (RoboMaster official competition rules)
// ============================================================================

namespace constants {

// Small armor plate (Outpost uses small armor plates)
// Source: RoboMaster 2025 Official Rule Manual
inline constexpr double kArmorPlateWidth{0.135};   // 135 mm
inline constexpr double kArmorPlateHeight{0.125};  // 125 mm

// 17mm projectile
// Source: RoboMaster 2025 Official Rule Manual
inline constexpr double kBulletSpeed{15.0};      // m/s
inline constexpr double kBulletDiameter{0.017};  // 17 mm
inline constexpr double kBulletMass{0.0032};     // 3.2 g

// Gravity
inline constexpr double kGravity{9.81};  // m/s²

// Air density and drag (for air-resistance model, bonus)
inline constexpr double kAirDensity{1.225};   // kg/m³ at 20°C
inline constexpr double kDragCoefficient{0.5};  // sphere

// Camera intrinsics (reference values from assessment document)
// fx=2054.32, fy=2054.05, cx=951.53, cy=602.55
inline constexpr double kFx{2054.32};
inline constexpr double kFy{2054.05};
inline constexpr double kCx{951.53};
inline constexpr double kCy{602.55};

// Distortion coefficients
// k1=-0.0734, k2=0.1500, p1=0.0011, p2=0.0004, k3=0.0
inline constexpr double kK1{-0.0734};
inline constexpr double kK2{0.1500};
inline constexpr double kP1{0.0011};
inline constexpr double kP2{0.0004};
inline constexpr double kK3{0.0};

// Tracker parameters
inline constexpr int kConfirmThreshold{3};    // frames to confirm
inline constexpr int kLostThreshold{5};        // frames before removal
inline constexpr double kIoUMin{0.45};          // minimum IoU for match

// Detector parameters
// Relaxed for outpost scenario: small armor plates at a distance
inline constexpr int kDiffThreshold{80};       // channel difference
inline constexpr double kMinAspectRatio{0.05};  // w/h min (relaxed)
inline constexpr double kMaxAspectRatio{0.5};  // w/h max (relaxed)
inline constexpr double kMinArea{20.0};        // min contour area (relaxed)
inline constexpr double kMaxArea{8000.0};      // max contour area (relaxed)
inline constexpr double kMinConvexity{0.4};    // min convexity (relaxed)
inline constexpr double kMinHeightRatio{0.55};  // paired light bar height
inline constexpr double kMaxAngleDiff{30.0};   // max angle diff (degrees)
inline constexpr double kMinXDistanceRatio{1.0};  // min spacing
inline constexpr double kMaxXDistanceRatio{4.0};  // max spacing (tight: exclude cross-panel)
inline constexpr double kMaxYOffsetRatio{0.6};    // max vertical offset (tight: exclude cross-panel)

}  // namespace constants

// ============================================================================
// Camera intrinsics helper
// ============================================================================

[[nodiscard]] inline auto make_camera_matrix() -> cv::Mat {
  auto K = cv::Mat(cv::Mat::eye(3, 3, CV_64F));
  K.at<double>(0, 0) = constants::kFx;
  K.at<double>(1, 1) = constants::kFy;
  K.at<double>(0, 2) = constants::kCx;
  K.at<double>(1, 2) = constants::kCy;
  return K;
}

[[nodiscard]] inline auto make_dist_coeffs() -> cv::Mat {
  auto dist = cv::Mat(cv::Mat::zeros(5, 1, CV_64F));
  dist.at<double>(0) = constants::kK1;
  dist.at<double>(1) = constants::kK2;
  dist.at<double>(2) = constants::kP1;
  dist.at<double>(3) = constants::kP2;
  dist.at<double>(4) = constants::kK3;
  return dist;
}

// ============================================================================
// Armor plate 3D model points (centered at origin)
// ============================================================================

[[nodiscard]] inline auto make_armor_model_points() -> std::vector<cv::Point3f> {
  auto hw = static_cast<float>(constants::kArmorPlateWidth / 2.0);
  auto hh = static_cast<float>(constants::kArmorPlateHeight / 2.0);
  // OpenCV coordinate: Y-axis down
  // Order: [TL, TR, BR, BL]
  return {
      cv::Point3f(-hw, -hh, 0.0F),  // top-left
      cv::Point3f(hw, -hh, 0.0F),   // top-right
      cv::Point3f(hw, hh, 0.0F),    // bottom-right
      cv::Point3f(-hw, hh, 0.0F)    // bottom-left
  };
}

}  // namespace rm_autoaim