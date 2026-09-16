#include "camera_trajectory_smoother.hpp"
#include "cost_helpers.hpp"

#include <thread>

#include <ceres/ceres.h>

using namespace arba;
using namespace decl;
using namespace elems;

struct PositionSmoothnessCost
  : CostBase<PositionSmoothnessCost, decltype(vec3 = vec3, vec3, vec3)>
{
    template <typename Vec3, typename Residual>
    void compute(Vec3 x0, Vec3 x1, Vec3 x2, Residual r) const
    {
        using T = typename Residual::Scalar;

        r = (x1 - (x0 + x2) / T(2));
    }
};

struct OrientationSmoothnessCost
  : CostBase<OrientationSmoothnessCost, decltype(vec3 = quat, quat, quat)>
{
    template <typename Quat, typename Residual>
    void compute(Quat q0, Quat q1, Quat q2, Residual r) const
    {
        using T = typename Residual::Scalar;

        // compute the middle rotation
        auto middleRotation = q0.slerp(T(0.5), q2);

        // need a q1 that inverted cancels out middle exactly
        auto q1_inv = q1.inverse();

        // compute the residual
        Eigen::AngleAxis<T> aa(middleRotation * q1_inv);
        r = aa.angle() * aa.axis();
    }
};

struct DisplacementCost : CostBase<DisplacementCost, decltype(vec3 = vec3)>
{
    const Eigen::Vector3d originalPosition;

    DisplacementCost(const Eigen::Vector3d& originalPosition)
      : originalPosition(originalPosition)
    {
    }

    template <typename Vec3, typename Residual>
    void compute(Vec3 x, Residual r) const
    {
        using T = typename Residual::Scalar;

        r = (x - originalPosition.template cast<T>());
    }
};

struct DerotationCost : CostBase<DerotationCost, decltype(vec3 = quat)>
{
    const Eigen::Quaterniond originalRotation;

    DerotationCost(const Eigen::Quaterniond& originalRotation)
      : originalRotation(originalRotation)
    {
    }

    template <typename Quat, typename Residual>
    void compute(Quat q, Residual r) const
    {
        using T = typename Residual::Scalar;

        Eigen::AngleAxis<T> aa(originalRotation.template cast<T>() *
                               q.inverse());
        r = aa.angle() * aa.axis();
    }
};

struct EqualLengthsCost
  : CostBase<EqualLengthsCost, decltype(vec1 = vec3, vec3, vec3)>
{
    template <typename Vec3, typename Residual>
    void compute(Vec3 x0, Vec3 x1, Vec3 x2, Residual r) const
    {
        using T = typename Residual::Scalar;

        r.x() = (x0 - x1).squaredNorm() - (x1 - x2).squaredNorm();
    }
};

std::tuple<nd_array<double, -1, 3>, nd_array<double, -1, 4>>
  arba::smooth_camera_trajectory(
    const nd_array<double, -1, 3>& camera_positions,
    const nd_array<double, -1, 4>& camera_orientations_xyzw,
    const nd_vector<int, -1>& fixed_knots,
    const bool loop,
    const std::size_t max_iterations,
    const double displacement_lambda,
    const double derotation_lambda,
    const double edge_lambda)
{
    // Where we compute results
    nd_array<double, -1, 3> smoothed_positions    = camera_positions;
    nd_array<double, -1, 4> smoothed_orientations = camera_orientations_xyzw;

    if (camera_positions.rows() != camera_orientations_xyzw.rows())
    {
        throw std::runtime_error(
          "camera_positions and camera_orientations_xyzw "
          "must have the same number of rows");
    }

    if (camera_positions.rows() != fixed_knots.rows())
    {
        throw std::runtime_error(
          "camera_positions and fixed_knots must have the same number of rows");
    }

    const std::size_t N = smoothed_positions.rows();

    // Instantiate the problem
    ceres::Problem problem;

    // Register the parameters
    for (std::size_t i = 0; i < N; ++i)
    {
        auto&& pos_ptr = smoothed_positions.row(i).data();
        auto&& ori_ptr = smoothed_orientations.row(i).data();

        problem.AddParameterBlock(pos_ptr, 3);
        problem.AddParameterBlock(ori_ptr, 4);
        problem.SetManifold(ori_ptr, new ceres::EigenQuaternionManifold);

        if (fixed_knots(i) == 1)
        {
            problem.SetParameterBlockConstant(pos_ptr);
            problem.SetParameterBlockConstant(ori_ptr);
        }
    }

    // Add position smoothness
    for (std::size_t i = 1; i < N - 1; ++i)
    {
        auto&& x0        = smoothed_positions.row(i - 1);
        auto&& x1        = smoothed_positions.row(i);
        auto&& x2        = smoothed_positions.row(i + 1);
        auto smooth_cost = PositionSmoothnessCost::create();
        problem.AddResidualBlock(
          smooth_cost, nullptr, x0.data(), x1.data(), x2.data());
        auto edge_cost = EqualLengthsCost::create();
        problem.AddResidualBlock(
          edge_cost,
          new ceres::ScaledLoss(nullptr, edge_lambda, ceres::TAKE_OWNERSHIP),
          x0.data(),
          x1.data(),
          x2.data());
    }

    if (loop)
    {
        {
            auto&& x0 = smoothed_positions.row(N - 1);
            auto&& x1 = smoothed_positions.row(0);
            auto&& x2 = smoothed_positions.row(1);

            auto smooth_cost = PositionSmoothnessCost::create();
            problem.AddResidualBlock(
              smooth_cost, nullptr, x0.data(), x1.data(), x2.data());

            auto edge_cost = EqualLengthsCost::create();
            problem.AddResidualBlock(
              edge_cost,
              new ceres::ScaledLoss(
                nullptr, edge_lambda, ceres::TAKE_OWNERSHIP),
              x0.data(),
              x1.data(),
              x2.data());
        }

        {
            auto&& x0 = smoothed_positions.row(N - 2);
            auto&& x1 = smoothed_positions.row(N - 1);
            auto&& x2 = smoothed_positions.row(0);

            auto smooth_cost = PositionSmoothnessCost::create();
            problem.AddResidualBlock(
              smooth_cost, nullptr, x0.data(), x1.data(), x2.data());

            auto edge_cost = EqualLengthsCost::create();
            problem.AddResidualBlock(
              edge_cost,
              new ceres::ScaledLoss(
                nullptr, edge_lambda, ceres::TAKE_OWNERSHIP),
              x0.data(),
              x1.data(),
              x2.data());
        }
    }

    // Add orientation smoothness
    for (std::size_t i = 1; i < N - 1; ++i)
    {
        auto&& q0 = smoothed_orientations.row(i - 1);
        auto&& q1 = smoothed_orientations.row(i);
        auto&& q2 = smoothed_orientations.row(i + 1);
        auto cost = OrientationSmoothnessCost::create();
        problem.AddResidualBlock(
          cost, nullptr, q0.data(), q1.data(), q2.data());
    }

    if (loop)
    {
        {
            auto&& q0 = smoothed_orientations.row(N - 1);
            auto&& q1 = smoothed_orientations.row(0);
            auto&& q2 = smoothed_orientations.row(1);
            auto cost = OrientationSmoothnessCost::create();
            problem.AddResidualBlock(
              cost, nullptr, q0.data(), q1.data(), q2.data());
        }

        {
            auto&& q0 = smoothed_orientations.row(N - 2);
            auto&& q1 = smoothed_orientations.row(N - 1);
            auto&& q2 = smoothed_orientations.row(0);
            auto cost = OrientationSmoothnessCost::create();
            problem.AddResidualBlock(
              cost, nullptr, q0.data(), q1.data(), q2.data());
        }
    }

    // Add displacement cost
    for (std::size_t i = 0; i < N; ++i)
    {
        auto&& x  = smoothed_positions.row(i);
        auto cost = DisplacementCost::create(camera_positions.row(i));
        problem.AddResidualBlock(cost,
                                 new ceres::ScaledLoss(nullptr,
                                                       displacement_lambda,
                                                       ceres::TAKE_OWNERSHIP),
                                 x.data());
    }

    // Add derotation cost
    for (std::size_t i = 0; i < N; ++i)
    {
        auto&& q  = smoothed_orientations.row(i);
        auto cost = DerotationCost::create(Eigen::Map<const Eigen::Quaterniond>(
          camera_orientations_xyzw.row(i).data()));
        problem.AddResidualBlock(cost,
                                 new ceres::ScaledLoss(nullptr,
                                                       derotation_lambda,
                                                       ceres::TAKE_OWNERSHIP),
                                 q.data());
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations           = max_iterations;
    options.num_threads                  = std::thread::hardware_concurrency();
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.sparse_linear_algebra_library_type =
      ceres::SparseLinearAlgebraLibraryType::SUITE_SPARSE;
    options.preconditioner_type = ceres::PreconditionerType::SCHUR_JACOBI;
    options.update_state_every_iteration = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // return
    return std::make_tuple(smoothed_positions, smoothed_orientations);
}
