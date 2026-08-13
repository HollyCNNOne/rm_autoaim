#include "rm_autoaim/internal/Hungarian.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace rm_autoaim::internal {

// ============================================================================
// Hungarian Algorithm Implementation
//
// Classic O(n³) Kuhn-Munkres algorithm for the assignment problem.
// Adapted for rectangular matrices (n × m).
// ============================================================================

auto Hungarian::solve(const CostMatrix& cost)
    -> std::pair<double, std::vector<Assignment>> {
  int n = static_cast<int>(cost.rows());
  int m = static_cast<int>(cost.cols());

  // Pad to square if needed
  int size = std::max(n, m);
  CostMatrix padded = CostMatrix::Zero(size, size);
  padded.topLeftCorner(n, m) = cost;

  std::vector<int> assignment(size, -1);
  double total = hungarian_impl(padded, size, size, assignment);

  std::vector<Assignment> result;
  for (int i = 0; i < n; ++i) {
    if (assignment[i] >= 0 && assignment[i] < m) {
      result.push_back({i, assignment[i]});
    }
  }
  return {total, result};
}

auto Hungarian::solve_with_threshold(const CostMatrix& cost, double max_cost)
    -> std::pair<double, std::vector<Assignment>> {
  auto [total, all] = solve(cost);

  std::vector<Assignment> filtered;
  double filtered_cost = 0.0;
  for (const auto& a : all) {
    double c = cost(a.row, a.col);
    if (c <= max_cost) {
      filtered.push_back(a);
      filtered_cost += c;
    }
  }
  return {filtered_cost, filtered};
}

auto Hungarian::hungarian_impl(const CostMatrix& cost, int n, int m,
                               std::vector<int>& assignment) -> double {
  std::vector<double> u(n + 1, 0.0);
  std::vector<double> v(m + 1, 0.0);
  std::vector<int> p(m + 1, 0);
  std::vector<int> way(m + 1, 0);

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(m + 1, std::numeric_limits<double>::max());
    std::vector<bool> used(m + 1, false);

    do {
      used[j0] = true;
      int i0 = p[j0];
      double delta = std::numeric_limits<double>::max();
      int j1 = 0;

      for (int j = 1; j <= m; ++j) {
        if (!used[j]) {
          double cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      }

      for (int j = 0; j <= m; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  // Build assignment vector
  for (int j = 1; j <= m; ++j) {
    if (p[j] > 0) {
      assignment[p[j] - 1] = j - 1;
    }
  }

  // Compute total cost
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    if (assignment[i] >= 0) {
      total += cost(i, assignment[i]);
    }
  }
  return total;
}

}  // namespace rm_autoaim::internal