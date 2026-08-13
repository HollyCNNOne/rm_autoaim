#pragma once

#include <Eigen/Dense>

#include <limits>
#include <vector>

namespace rm_autoaim::internal {

// ============================================================================
// Hungarian Algorithm (Kuhn-Munkres) for optimal assignment
//
// Given a cost matrix C[N][M], finds the assignment that minimizes total cost.
// Each row can be assigned to at most one column, and vice versa.
//
// Returns: vector of (row, col) assignments
//   Unassigned rows have col = -1
// ============================================================================

class Hungarian {
public:
  using CostMatrix = Eigen::MatrixXd;

  struct Assignment {
    int row;
    int col;
  };

  // Solve the assignment problem
  // Returns cost and assignments
  [[nodiscard]] static auto solve(const CostMatrix& cost)
      -> std::pair<double, std::vector<Assignment>>;

  // Solve with a cost threshold: assignments with cost > max_cost are rejected
  [[nodiscard]] static auto solve_with_threshold(const CostMatrix& cost,
                                                 double max_cost)
      -> std::pair<double, std::vector<Assignment>>;

private:
  // Internal implementation for square/sub-rectangular matrices
  static auto hungarian_impl(const CostMatrix& cost, int n, int m,
                             std::vector<int>& assignment) -> double;
};

}  // namespace rm_autoaim::internal