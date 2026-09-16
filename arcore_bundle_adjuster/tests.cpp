#define BOOST_TEST_MODULE "arcore_bundle_adjuster_tests"
#include "bundle_adjuster.hpp"
#include "cost_helpers.hpp"
#include "math.hpp"
#include <boost/test/unit_test.hpp>
#include <ceres/ceres.h>
#include <ceres/jet.h>

using namespace arba;
using namespace decl;
using namespace elems;

BOOST_AUTO_TEST_CASE(undistortion_gradient)
{
    constexpr std::size_t focal_length_dim            = 2;
    constexpr std::size_t principal_point_dim         = 2;
    constexpr std::size_t distortion_coefficients_dim = 4;
    constexpr std::size_t distorted_pixel_dim         = 2;
    constexpr std::size_t num_grads = focal_length_dim + principal_point_dim +
                                      distortion_coefficients_dim +
                                      distorted_pixel_dim;

    using Scalar = ceres::Jet<double, num_grads>;

    int index = 0;

    Eigen::Matrix<Scalar, distorted_pixel_dim, 1> distorted_pixel =
      Eigen::Matrix<Scalar, distorted_pixel_dim, 1>::Zero();
    distorted_pixel.x().v(index++) = 1;
    distorted_pixel.y().v(index++) = 1;

    Eigen::Matrix<Scalar, focal_length_dim, 1> focal_length =
      Eigen::Matrix<Scalar, focal_length_dim, 1>::Ones();
    focal_length.x().v(index++) = 1;
    focal_length.y().v(index++) = 1;

    Eigen::Matrix<Scalar, principal_point_dim, 1> principal_point =
      Eigen::Matrix<Scalar, principal_point_dim, 1>::Ones();
    principal_point.x().v(index++) = 1;
    principal_point.y().v(index++) = 1;

    Eigen::Matrix<Scalar, distortion_coefficients_dim, 1>
      distortion_coefficients =
        Eigen::Matrix<Scalar, distortion_coefficients_dim, 1>::Random() * 1e-3;
    distortion_coefficients.x().v(index++) = 1;
    distortion_coefficients.y().v(index++) = 1;
    distortion_coefficients.z().v(index++) = 1;
    distortion_coefficients.w().v(index++) = 1;

    BOOST_CHECK_EQUAL(index, num_grads);

    auto&& distorted_pixel_ndc =
      wnd2ndc(distorted_pixel, focal_length, principal_point);
    auto&& undistorted_pixel_ndc =
      undistort_ndc(distorted_pixel_ndc, distortion_coefficients);
    auto&& undistorted_pixel =
      ndc2wnd(undistorted_pixel_ndc, focal_length, principal_point);

    // Make sure that all grads are activated
    BOOST_CHECK(undistorted_pixel.x().v.all());
    BOOST_CHECK(undistorted_pixel.y().v.all());
}

struct SimpleCostFunction : CostBase<SimpleCostFunction, decltype(vec1 = vec1)>
{
    double observation;

    SimpleCostFunction(double observation) : observation(observation) {}

    template <typename U, typename Residual>
    void compute(U x, Residual residual) const
    {
        using T      = typename Residual::Scalar;
        residual.x() = x.x() - T(observation);
    }
};

BOOST_AUTO_TEST_CASE(understand_ceres_cost)
{
    double x = 15.0;

    double lambda1 = 0.1;
    double lambda2 = 0.05;
    double target1 = 10.0;
    double target2 = 20.0;

    double expected_objective = 0.0;

    // Define a single parameter array for the cost function evaluation
    const double* parameters[] = {&x};

    ceres::Problem problem;

    // Group 1
    constexpr int num_residuals_1 = 5;
    for (int i = 0; i < num_residuals_1; ++i)
    {
        auto cost_function1 = SimpleCostFunction::create(target1);

        double residual;
        cost_function1->Evaluate(parameters, &residual, nullptr);
        double scaled_residual = std::pow(residual, 2);

        auto loss = new ceres::ArctanLoss(1.0);
        double loss_values[3];
        loss->Evaluate(scaled_residual, loss_values);
        expected_objective += 0.5 * lambda1 / num_residuals_1 * loss_values[0];

        problem.AddResidualBlock(
          cost_function1,
          new ceres::ScaledLoss(
            loss, lambda1 / num_residuals_1, ceres::Ownership::TAKE_OWNERSHIP),
          &x);
    }

    // Group 2
    constexpr int num_residuals_2 = 10;
    for (int i = 0; i < num_residuals_2; ++i)
    {
        auto cost_function2 = SimpleCostFunction::create(target2);

        double residual;
        cost_function2->Evaluate(parameters, &residual, nullptr);
        double scaled_residual = std::pow(residual, 2);

        auto loss = new ceres::ArctanLoss(2.0);
        double loss_values[3];
        loss->Evaluate(scaled_residual, loss_values);
        expected_objective += 0.5 * lambda2 / num_residuals_2 * loss_values[0];

        problem.AddResidualBlock(
          cost_function2,
          new ceres::ScaledLoss(
            loss, lambda2 / num_residuals_2, ceres::Ownership::TAKE_OWNERSHIP),
          &x);
    }

    double total_cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(),
                     &total_cost,
                     nullptr,
                     nullptr,
                     nullptr);

    BOOST_CHECK_CLOSE(total_cost, expected_objective, 1e-10);
}

struct foo_residual
  : CostBase<foo_residual, decltype(vec1 = vec1, vec2, vec3, vec4)>
{
    template <typename Vec1,
              typename Vec2,
              typename Vec3,
              typename Vec4,
              typename Residual>
    void compute(Vec1 x, Vec2 y, Vec3 z, Vec4 w, Residual residual) const
    {
        using T = typename Residual::Scalar;

        if constexpr (!std::is_same_v<T, double>)
        {
            // If we stack all initialized v's into a matrix, it should be an
            // identity matrix
            Eigen::Matrix<double, 10, 10> J;
            J << x.x().v, y.x().v, y.y().v, z.x().v, z.y().v, z.z().v, w.x().v,
              w.y().v, w.z().v, w.w().v;

            BOOST_CHECK(J.isIdentity());
        }

        residual.x() =
          x.squaredNorm() + y.squaredNorm() + z.squaredNorm() + w.squaredNorm();
    }
};

BOOST_AUTO_TEST_CASE(understand_autodiff)
{
    auto res = foo_residual::create();

    double vec1_value   = 1.0;
    double vec2_value[] = {2.0, 3.0};
    double vec3_value[] = {4.0, 5.0, 6.0};
    double vec4_value[] = {7.0, 8.0, 9.0, 10.0};

    auto parameter_block_sizes = res->parameter_block_sizes();
    BOOST_CHECK_EQUAL(parameter_block_sizes.size(), 4);
    BOOST_CHECK_EQUAL(parameter_block_sizes[0], 1);
    BOOST_CHECK_EQUAL(parameter_block_sizes[1], 2);
    BOOST_CHECK_EQUAL(parameter_block_sizes[2], 3);
    BOOST_CHECK_EQUAL(parameter_block_sizes[3], 4);

    double* parameters[] = {&vec1_value, vec2_value, vec3_value, vec4_value};
    double jac1[1], jac2[2], jac3[3], jac4[4];
    double* jacobians[] = {jac1, jac2, jac3, jac4};
    double out;

    res->Evaluate(parameters, &out, jacobians);

    BOOST_CHECK_EQUAL(jac1[0], 2);
    BOOST_CHECK_EQUAL(jac2[0], 4);
    BOOST_CHECK_EQUAL(jac2[1], 6);
    BOOST_CHECK_EQUAL(jac3[0], 8);
    BOOST_CHECK_EQUAL(jac3[1], 10);
    BOOST_CHECK_EQUAL(jac3[2], 12);
    BOOST_CHECK_EQUAL(jac4[0], 14);
    BOOST_CHECK_EQUAL(jac4[1], 16);
    BOOST_CHECK_EQUAL(jac4[2], 18);
    BOOST_CHECK_EQUAL(jac4[3], 20);

    delete res;
}

BOOST_AUTO_TEST_CASE(undistortion_functional)
{
    for (int i = 0; i < 100; ++i)
    {
        Eigen::Vector2d undistorted_ndc   = Eigen::Vector2d::Random();
        Eigen::Vector4d distortion_coeffs = 0.01 * Eigen::Vector4d::Random();
        Eigen::Vector2d distorted_ndc =
          distort_ndc(undistorted_ndc, distortion_coeffs);
        Eigen::Vector2d undistorted_ndc_ =
          undistort_ndc(distorted_ndc, distortion_coeffs);

        BOOST_CHECK_LT(
          (undistorted_ndc - undistorted_ndc_).cwiseAbs().maxCoeff(), 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(unaryExpr_order)
{
    Eigen::Matrix<bool, 2, 3> m;
    int index       = 0;
    auto new_matrix = m.unaryExpr([&](auto&&) { return index++; });

    BOOST_CHECK_EQUAL(new_matrix(0, 0), 0);
    BOOST_CHECK_EQUAL(new_matrix(0, 1), 1);
    BOOST_CHECK_EQUAL(new_matrix(0, 2), 2);
    BOOST_CHECK_EQUAL(new_matrix(1, 0), 3);
    BOOST_CHECK_EQUAL(new_matrix(1, 1), 4);
    BOOST_CHECK_EQUAL(new_matrix(1, 2), 5);
}

struct SimpleCost : CostBase<SimpleCost, decltype(vec2 = vec1, vec1)>
{
    template <typename Vec1, typename Vec2, typename Residual>
    void compute(Vec1 x, Vec2 y, Residual residual) const
    {
        using T      = typename Residual::Scalar;
        residual.x() = x.x() - T(1.0);
        residual.y() = y.x() - T(2.0);

        if constexpr (!std::is_same_v<T, double>)
        {
            // Regardless of whether a variable has been marked as constant, the
            // gradient is still thoroughly computed
            BOOST_CHECK_EQUAL(x.x().v.size(), 2);
            BOOST_CHECK_EQUAL(y.x().v.size(), 2);
            BOOST_CHECK_EQUAL(x.x().v(0), 1);
            BOOST_CHECK_EQUAL(x.x().v(1), 0);
            BOOST_CHECK_EQUAL(y.x().v(0), 0);
            BOOST_CHECK_EQUAL(y.x().v(1), 1);
        }
    }
};

BOOST_AUTO_TEST_CASE(jets_of_constant_parameters)
{
    double x = 1.0;
    double y = 2.0;

    ceres::Problem problem;
    problem.AddResidualBlock(SimpleCost::create(), nullptr, &x, &y);

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    problem.SetParameterBlockConstant(&y);
    ceres::Solve(options, &problem, &summary);

    problem.SetParameterBlockVariable(&y);
    ceres::Solve(options, &problem, &summary);
}

BOOST_AUTO_TEST_CASE(integration)
{
    // Setup a trivial scene
    nd_vector<double, 2> focal_length    = {1, 1};
    nd_vector<double, 2> principal_point = {0, 0};
    nd_vector<double, 4> distortion_coefficients =
      1e-2 * nd_vector<double, 4>::Random();

    nd_vector<double, 3> camera0_position    = {0, 0, 0};
    nd_vector<double, 4> camera0_orientation = {0, 0, 0, 1};

    nd_vector<double, 3> camera1_position    = {0.2, 0, 0};
    nd_vector<double, 4> camera1_orientation = {0, 0, 0, 1};

    nd_array<double, -1, 3> point_cloud =
      0.5 * nd_array<double, -1, 3>::Random(100, 3);
    point_cloud.col(2).array() += 2;

    // project to get per camera feature locations
    const auto project =
      [&](auto&& camera_position, auto&& camera_orientation, auto&& point)
    {
        auto&& point_in_camera_space = to_camera_space(
          camera_position,
          Eigen::Map<const Eigen::Quaterniond>(camera_orientation.data()),
          point);

        // Make sure it's not behind the camera
        BOOST_CHECK_GE(point_in_camera_space.z(), 0);

        const auto ndc           = point_in_camera_space.hnormalized();
        const auto distorted_ndc = distort_ndc(ndc, distortion_coefficients);
        const auto wnd = ndc2wnd(distorted_ndc, focal_length, principal_point);

        return wnd.eval();
    };

    nd_array<double, -1, 2> features0(100, 2), features1(100, 2);
    for (int i = 0; i < 100; ++i)
    {
        auto&& p = point_cloud.row(i).transpose();
        features0.row(i) =
          project(camera0_position, camera0_orientation, p).transpose();
        features1.row(i) =
          project(camera1_position, camera1_orientation, p).transpose();
    }

    // Set up bundle adjustment problem
    bundle_adjuster ba(
      focal_length,
      principal_point,
      distortion_coefficients,
      {{0, features0}, {1, features1}},
      {{{0, 1}, nd_vector<int, -1>::LinSpaced(100, 0, 99).replicate(1, 2)}},
      {{0, {camera0_position, camera0_orientation}},
       {1, {camera1_position, camera1_orientation}}});

    for (auto iterations : {0, 100})
    {
        ba.solve(
          iterations, false, bundle_adjuster::optimization_inclusion::none);

        auto point_cloud_out = ba.get_point_cloud();

        // point_cloud_out has twice the rows as point_cloud, so pick every
        // second row
        for (int i = 0; i < 100; ++i)
        {
            if (iterations == 0) // unless we solve, the error is not small
            {
                BOOST_CHECK_GT((point_cloud.row(i) - point_cloud_out.row(2 * i))
                                 .cwiseAbs()
                                 .maxCoeff(),
                               1e-6);
            }
            else // if we solve, the error is small
            {
                BOOST_CHECK_LT((point_cloud.row(i) - point_cloud_out.row(2 * i))
                                 .cwiseAbs()
                                 .maxCoeff(),
                               1e-6);
            }
        }
    }
}