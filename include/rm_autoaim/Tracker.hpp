#pragma once

#include "rm_autoaim/Types.hpp"

#include <vector>

#include <opencv2/core.hpp>

namespace rm_autoaim {

// ============================================================================
// Tracker — Module 3: PnP Pose Estimation + Data Association
//
// For each frame:
//   1. SolvePnP: 2D corners → 3D pose (camera frame)
//   2. Hungarian algorithm: match detections to existing tracks
//   3. Target lifecycle: create, confirm, track, lose, delete
// ============================================================================

class Tracker {
public:
  Tracker();

  // Process one frame of detections
  // Returns all confirmed/lost tracks with updated poses
  [[nodiscard]] auto update(const std::vector<Armor2D>& detections)
      -> std::vector<TrackedArmor>;

  // Get all active tracks
  [[nodiscard]] auto tracks() const -> const std::vector<TrackedArmor>&;

  // Reset all tracks
  auto reset() -> void;

private:
  // PnP: solve 3D pose from 2D corners
  [[nodiscard]] auto solve_pnp(const Armor2D& detection)
      -> ArmorPose;

  // Data association: match detections to existing tracks
  [[nodiscard]] auto associate(
      const std::vector<Armor2D>& detections,
      const std::vector<TrackedArmor>& tracks)
      -> std::tuple<std::vector<std::pair<int, int>>,  // (det_idx, track_idx)
                    std::vector<int>,                   // unmatched det indices
                    std::vector<int>>;                  // unmatched track indices

  // Lifecycle management
  auto update_tracks(const std::vector<Armor2D>& detections,
                     const std::vector<std::pair<int, int>>& matches,
                     const std::vector<int>& unmatched_det,
                     const std::vector<int>& unmatched_trk) -> void;

  // Compute IoU between two sets of 2D corners
  [[nodiscard]] static auto compute_iou(const Armor2D& a, const Armor2D& b)
      -> double;

  // Camera intrinsics
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;

  // 3D model points
  std::vector<cv::Point3f> model_points_;

  // Active tracks
  std::vector<TrackedArmor> tracks_;
  int next_track_id_{0};
};

}  // namespace rm_autoaim