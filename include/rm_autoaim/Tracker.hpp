#pragma once

#include "rm_autoaim/Types.hpp"

#include <array>
#include <deque>
#include <queue>
#include <vector>

#include <opencv2/core.hpp>

namespace rm_autoaim {

// PnP + data association with fixed-slot cyclic queue (3 slots for 3 physical armors)
// V3.2: Phase-aware soft constraints — cost bonus in Hungarian + preferred slot
//       reuse in Phase 5, backed by auto-calibrating period measurement
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

  [[nodiscard]] auto associate(
      const std::vector<Armor2D>& detections,
      const std::vector<TrackedArmor>& tracks)
      -> std::tuple<std::vector<std::pair<int, int>>,
                    std::vector<int>,
                    std::vector<int>>;

  [[nodiscard]] static auto compute_iou(const Armor2D& a, const Armor2D& b)
      -> double;

  // Fixed-slot architecture: 3 slots = 3 outpost armor plates, ID recycles
  static constexpr int kMaxSlots{3};
  static constexpr int kTimeoutFrames{15};
  std::array<TrackedArmor, kMaxSlots> slots_;
  bool slot_active_[kMaxSlots]{};
  int slot_timeout_[kMaxSlots]{};
  std::queue<int> free_slots_;

  // V3.2: Period auto-calibration
  // Samples frame-delta between slot release and re-assignment.
  // Sliding-window median → lock at 7+ samples → EMA tracking (±20% gate).
  static constexpr int kInitialPeriod{383};       // 2.3s @ 166.7FPS
  static constexpr int kMaxPeriodSamples{10};
  static constexpr int kMinSamplesForLock{7};
  static constexpr double kPeriodEmaAlpha{0.2};
  static constexpr double kPeriodRejectRatio{0.20};

  int frame_counter_{0};
  int first_seen_frame_{-1};
  int measured_period_{kInitialPeriod};
  bool period_initialized_{false};
  int prev_active_count_{0};
  int slot_release_frame_[kMaxSlots]{};    // frame# when slot was last recycled
  std::deque<int> period_samples_;         // max 10 raw observations
  int rejected_sample_count_{0};

  // Phase bonus: slight discount for "expected" ID, slight penalty otherwise
  // Magnitude is small enough that IoU still dominates the cost matrix
  static constexpr double kPhaseBonusMatch{-0.10};
  static constexpr double kPhaseBonusMismatch{0.02};

  // V3.2: Match inertia — prevents ID oscillation by remembering last frame's
  // pairings. When a detection-track pair existed in the previous frame,
  // the cost is reduced by 0.30, far outweighing small IoU fluctuations.
  static constexpr double kInertiaBonus{-0.30};
  static constexpr double kInertiaMinIoU{0.3};   // min IoU for cross-frame continuity
  std::vector<Armor2D> prev_detections_;          // detections from previous frame
  std::vector<int> prev_det_to_slot_;             // prev_det_idx → slot_idx (-1 = unmatched)

  // V3.2+: Pose EMA smoothing — suppresses PnP jitter by blending raw
  // measurements with historical trend. Alpha=0.25 means 25% new, 75% history.
  static constexpr double kSmoothAlpha{0.25};
  struct SmoothState {
    double pitch{0.0};
    double yaw{0.0};
    double depth{0.0};
    bool initialized{false};
  };
  std::array<SmoothState, kMaxSlots> smooth_state_;

  [[nodiscard]] auto phase_to_slot(int phase) const -> int;

  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  std::vector<cv::Point3f> model_points_;
};

}  // namespace rm_autoaim