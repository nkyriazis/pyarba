#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace arba
{
/**
 * Eigen::Matrix with RowMajor storage order.
 *
 * Why? Because Eigen::Matrix is column-major by default,
 * and we want to be able to combine rowwise iteration with
 * pointer access.
 */
template <typename T, int Rows, int Cols>
using nd_array =
  Eigen::Matrix<T, Rows, Cols, Cols != 1 ? Eigen::RowMajor : Eigen::ColMajor>;

template <typename T, int D>
using nd_vector = Eigen::Vector<T, D>;
}