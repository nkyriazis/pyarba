#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unsupported/Eigen/AutoDiff>

namespace arba
{
namespace concepts
{
template <typename T>
concept is_eigen = std::is_base_of_v<Eigen::MatrixBase<T>, T>;

template <typename Derived, int Rows, int Cols>
concept is_sized_eigen =
  is_eigen<Derived> && Rows == Derived::RowsAtCompileTime &&
  Cols == Derived::ColsAtCompileTime;

template <typename Derived, int D>
concept is_sized_eigen_vector = is_sized_eigen<Derived, D, 1>;

template <typename First, typename... Rest>
concept have_same_scalar =
  (std::is_same_v<typename First::Scalar, typename Rest::Scalar> && ...);

} // namespace concepts

template <concepts::is_sized_eigen_vector<2> NDC,
          concepts::is_sized_eigen_vector<2> F,
          concepts::is_sized_eigen_vector<2> C>
    requires concepts::have_same_scalar<NDC, F, C>
auto ndc2wnd(const Eigen::MatrixBase<NDC>& ndc,
             const Eigen::MatrixBase<F>& f,
             const Eigen::MatrixBase<C>& c)
{
    return (ndc.cwiseProduct(f) + c).eval();
}

template <concepts::is_sized_eigen_vector<2> WND,
          concepts::is_sized_eigen_vector<2> F,
          concepts::is_sized_eigen_vector<2> C>
    requires concepts::have_same_scalar<WND, F, C>
auto wnd2ndc(const Eigen::MatrixBase<WND>& wnd,
             const Eigen::MatrixBase<F>& f,
             const Eigen::MatrixBase<C>& c)
{
    return ((wnd - c).cwiseProduct(f.cwiseInverse())).eval();
}

template <concepts::is_sized_eigen_vector<2> WND,
          concepts::is_sized_eigen_vector<2> F,
          concepts::is_sized_eigen_vector<2> C>
    requires concepts::have_same_scalar<WND, F, C>
auto wnd2ray_unnormalized(const Eigen::MatrixBase<WND>& wnd,
                          const Eigen::MatrixBase<F>& f,
                          const Eigen::MatrixBase<C>& c)
{
    return wnd2ndc(wnd, f, c).homogeneous().eval();
}

template <concepts::is_sized_eigen_vector<2> WND,
          concepts::is_sized_eigen_vector<2> F,
          concepts::is_sized_eigen_vector<2> C>
    requires concepts::have_same_scalar<WND, F, C>
auto wnd2ray(const Eigen::MatrixBase<WND>& wnd,
             const Eigen::MatrixBase<F>& f,
             const Eigen::MatrixBase<C>& c)
{
    return wnd2ray_unnormalized(wnd, f, c).normalized().eval();
}

template <typename T>
struct constant // It's difficult to get constants deep in AD
{
    static auto make(const double v) { return static_cast<T>(v); }
};

template <typename T, int N>
struct constant<ceres::Jet<T, N>>
{
    static auto make(const double v)
    {
        return ceres::Jet<T, N>(constant<T>::make(v));
    }
};

template <concepts::is_sized_eigen_vector<2> P,
          concepts::is_sized_eigen_vector<4> D>
    requires concepts::have_same_scalar<P, D>
auto distort_ndc(const Eigen::MatrixBase<P>& p,
                 const Eigen::MatrixBase<D>& d_coeffs)
{
    using Scalar           = typename P::Scalar;
    static const Scalar _1 = constant<Scalar>::make(1.0);
    static const Scalar _2 = constant<Scalar>::make(2.0);

    const Scalar r2 = p.x() * p.x() + p.y() * p.y();

    const Scalar k1 = d_coeffs.x();
    const Scalar k2 = d_coeffs.y();
    const Scalar p1 = d_coeffs.z();
    const Scalar p2 = d_coeffs.w();

    // Radial distortion
    const Eigen::Vector<Scalar, 2> radial = (_1 + k1 * r2 + k2 * r2 * r2) * p;

    // Tangential distortion
    const Eigen::Vector<Scalar, 2> tangential = {
      _2 * p1 * p.x() * p.y() + p2 * (r2 + _2 * p.x() * p.x()),
      p1 * (r2 + _2 * p.y() * p.y()) + _2 * p2 * p.x() * p.y()};

    return (radial + tangential).eval();
}

template <concepts::is_sized_eigen_vector<2> P,
          concepts::is_sized_eigen_vector<4> D>
    requires concepts::have_same_scalar<P, D>
Eigen::Matrix<typename P::Scalar, 2, 2> distort_ndc_jacobian(
  const Eigen::MatrixBase<P>& p, const Eigen::MatrixBase<D>& d_coeffs)
{
    using Scalar  = typename P::Scalar;
    using DScalar = ceres::Jet<Scalar, 2>;

    const Eigen::Vector<DScalar, 2> p_(DScalar(p.x(), 0), DScalar(p.y(), 1));
    const auto d_        = d_coeffs.template cast<DScalar>();
    const auto distorted = distort_ndc(p_, d_);

    Eigen::Matrix<Scalar, 2, 2> ret;
    ret(0, 0) = distorted.x().v(0);
    ret(0, 1) = distorted.x().v(1);
    ret(1, 0) = distorted.y().v(0);
    ret(1, 1) = distorted.y().v(1);

    return ret;
}

template <concepts::is_sized_eigen_vector<2> P,
          concepts::is_sized_eigen_vector<4> D>
    requires concepts::have_same_scalar<P, D>
auto undistort_ndc(const Eigen::MatrixBase<P>& p_distorted,
                   const Eigen::MatrixBase<D>& d_coeffs)
{
    using Scalar                    = typename P::Scalar;
    static const Scalar tolerance   = Scalar(1e-6);
    static const int max_iterations = 5;

    // Initialize the undistorted point with the distorted one (good initial
    // guess)
    P p_undistorted = p_distorted;

    for (int i = 0; i < max_iterations; ++i)
    {
        // Distort the current guess
        P p_current_distorted = distort_ndc(p_undistorted, d_coeffs);

        // Compute the residual (error vector)
        P residual = p_distorted - p_current_distorted;

        // Compute the Jacobian matrix of the distortion function
        Eigen::Matrix<Scalar, 2, 2> J =
          distort_ndc_jacobian(p_undistorted, d_coeffs);

        // Check if Jacobian is near singular
        Scalar det = J.determinant();
        if (abs(det) < 1e-10)
        {
            break; // Jacobian is singular, terminate the iterations
        }

        // Update rule: Newton-Raphson
        // p_new = p_old + J^(-1) * residual
        P update = J.inverse() * residual;
        p_undistorted += update;

        // Convergence check
        if (update.squaredNorm() < tolerance) { break; }
    }

    return p_undistorted;
}

template <concepts::is_sized_eigen_vector<3> Position,
          typename Orientation,
          concepts::is_sized_eigen_vector<3> Point>
    requires concepts::have_same_scalar<Position, Orientation, Point>
auto to_camera_space(
  const Eigen::MatrixBase<Position>& camera_world_position,
  const Eigen::QuaternionBase<Orientation>& camera_world_orientation,
  const Eigen::MatrixBase<Point>& point_world_location)
{
    return (camera_world_orientation.inverse() *
            (point_world_location - camera_world_position))
      .eval();
}

template <concepts::is_sized_eigen_vector<2> FocalLength,
          concepts::is_sized_eigen_vector<2> PrincipalPoint,
          concepts::is_sized_eigen_vector<3> Position,
          typename Orientation,
          concepts::is_sized_eigen_vector<3> Point>
    requires concepts::have_same_scalar<FocalLength,
                                        PrincipalPoint,
                                        Position,
                                        Orientation,
                                        Point>
auto project(const Eigen::MatrixBase<FocalLength>& focal_length,
             const Eigen::MatrixBase<PrincipalPoint>& principal_point,
             const Eigen::MatrixBase<Position>& camera_world_position,
             const Eigen::QuaternionBase<Orientation>& camera_world_orientation,
             const Eigen::MatrixBase<Point>& point_world_location)
{
    const auto point_camera_location = to_camera_space(
      camera_world_position, camera_world_orientation, point_world_location);

    const auto point_camera_location_dehomogenized =
      point_camera_location.hnormalized();

    const auto pixel =
      focal_length.cwiseProduct(point_camera_location_dehomogenized) +
      principal_point;

    return std::make_tuple(pixel.eval(), point_camera_location.z());
}

template <concepts::is_sized_eigen_vector<2> FocalLength,
          concepts::is_sized_eigen_vector<2> PrincipalPoint,
          concepts::is_sized_eigen_vector<3> CameraPosition,
          typename CameraOrientation,
          concepts::is_sized_eigen_vector<2> Pixel>
    requires concepts::have_same_scalar<FocalLength,
                                        PrincipalPoint,
                                        CameraPosition,
                                        CameraOrientation,
                                        Pixel>
auto unproject(
  const Eigen::MatrixBase<FocalLength>& focal_length,
  const Eigen::MatrixBase<PrincipalPoint>& principal_point,
  const Eigen::MatrixBase<CameraPosition>& camera_world_position,
  const Eigen::QuaternionBase<CameraOrientation>& camera_world_orientation,
  const Eigen::MatrixBase<Pixel>& pixel)
{
    const auto ray_direction = wnd2ray(pixel, focal_length, principal_point);

    const auto ray_direction_world = camera_world_orientation * ray_direction;

    return std::make_tuple(camera_world_position.eval(),
                           ray_direction_world.eval());
}

} // namespace arba