#pragma once

#include <tuple>

#include "eigen.hpp"

namespace arba
{
std::tuple<nd_array<double, -1, 3>, nd_array<double, -1, 4>>
  smooth_camera_trajectory(
    const nd_array<double, -1, 3>& camera_positions,
    const nd_array<double, -1, 4>& camera_orientations_xyzw,
    const nd_vector<int, -1>& fixed_knots,
    const bool loop                  = false,
    const std::size_t max_iterations = 100,
    const double displacement_lambda = 1.0,
    const double derotation_lambda   = 1.0,
    const double edge_lambda         = 10.0);
}