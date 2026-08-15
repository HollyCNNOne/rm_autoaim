#pragma once

#include "rm_autoaim/Types.hpp"

#include <array>
#include <queue>
#include <vector>

#include <opencv2/core.hpp>

namespace rm_autoaim {

// ============================================================================
// Tracker — Module 3: PnP Pose Estimation + Data Association
//
// V3: Fixed-slot cyclic queue architecture
//   - 3 fixed slots matching outpost's 3 physical armor plates
//   - Free-slot queue for ID recycling (0, 1, 2)
//   - Timeout lease (15 frames ≈ 90ms) for rotation cycle tolerance
//   - Hungarian algorithm unchanged (pure IoU matching)
// ============================================================================

class Tracker {
public:
  Tracker();

  [[nodiscard]] auto update(const std::vector<Armor2D>& detections)
      -> std::vector<TrackedArmor>;

  [[nodiscard]] auto tracks() const -> std::vector<TrackedArmor>;

  auto reset() -> void;

private:
  [[nodiscard]] auto solve_pnp(const Armor2D& detection)
      -> ArmorPose;

  // Data association: Hungarian (mixed cost: 60% IoU + 40% center distance)
  [[nodiscard]] auto associate(
      const std::vector<Armor2D>& detections,
      const std::vector<TrackedArmor>& tracks)
      -> std::tuple<std::vector<std::pair<int, int>>,
                    std::vector<int>,
                    std::vector<int>>;

  [[nodiscard]] static auto compute_iou(const Armor2D& a, const Armor2D& b)
      -> double;

  // Camera intrinsics
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;

  // 3D model points
  std::vector<cv::Point3f> model_points_;

  // Fixed-slot cyclic queue architecture
  static constexpr int kMaxSlots{3};           // outpost: 3 armor plates
  static constexpr int kTimeoutFrames{15};     // lease: ~90ms @ 166.7FPS
  std::array<TrackedArmor, kMaxSlots> slots_;  // fixed sandbox
  bool slot_active_[kMaxSlots]{};              // slot occupancy
  int slot_timeout_[kMaxSlots]{};              // lease countdown
  std::queue<int> free_slots_;                 // recyclable IDs {0,1,2}
};

}  // namespace rm_autoaim