#include "bundle_adjuster.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>

#include <boost/functional/hash.hpp>
#include <ceres/ceres.h>
#include <tbb/tbb.h>

#include "cost_helpers.hpp"
#include "math.hpp"

using namespace arba;
using namespace decl;
using namespace elems;

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        std::size_t ret = 0;
        boost::hash_combine(ret, p.first);
        boost::hash_combine(ret, p.second);
        return ret;
    }
};

struct cache
{
    static constexpr std::size_t num_grads = 2 + 2 + 4 + 3 + 4 + 1;
    using scalar                           = ceres::Jet<double, num_grads>;
    template <int rows, int cols>
    using matrix = Eigen::Matrix<scalar, rows, cols>;
    template <int dim>
    using vector     = Eigen::Vector<scalar, dim>;
    using quaternion = Eigen::Quaternion<scalar>;
    using vec2       = vector<2>;
    using vec3       = vector<3>;
    using key        = std::pair<std::size_t, std::size_t>;
    struct data
    {
        vec2 undistorted_wnd;
        vec3 point;
    };

    std::unordered_map<key, std::size_t, pair_hash> indirection;
    std::vector<data> values;

    data& at(const key& k) { return values.at(indirection.at(k)); }
    const data& at(const key& k) const { return values.at(indirection.at(k)); }

    void populate(const key& k)
    {
        if (indirection.find(k) == indirection.end())
        {
            auto next_index = indirection.size();
            indirection[k]  = next_index;
        }
    }

    void realize() { values.resize(indirection.size()); }
};

struct back_projection_cost
  : CostBase<back_projection_cost,
             decltype(vec2 = vec2, vec2, vec4, vec3, quat, vec1, vec3, quat)>
{
    const cache::data &cache0, &cache1;

    back_projection_cost(const cache::data& cache0, const cache::data& cache1)
      : cache0(cache0), cache1(cache1)
    {
    }

    template <typename Vec2,
              typename Vec3,
              typename Vec4,
              typename Quat,
              typename Vec1,
              typename Residual>
    void compute(Vec2 focal_length,
                 Vec2 principal_point,
                 Vec4 dist_coeffs,
                 Vec3 camera0_world_position,
                 Quat camera0_world_orientation,
                 Vec1 t_0,
                 Vec3 camera1_world_position,
                 Quat camera1_world_orientation,
                 Residual residual) const
    {
        using T = typename Residual::Scalar;

        using Vec2_ = Eigen::Vector<T, 2>;
        using Vec3_ = Eigen::Vector<T, 3>;

        // These will be brought out of cache which has carefully been
        // computed to depend on the provided variables
        Vec3_ pt0;
        Vec2_ undistorted_px1;

        constexpr bool residuals_only = std::is_same_v<T, double>;

        if constexpr (residuals_only)
        {
            // bring in only the value part
            const auto fetch_value = [](auto&& x) { return x.a; };
            pt0                    = cache0.point.unaryExpr(fetch_value);
            undistorted_px1 = cache1.undistorted_wnd.unaryExpr(fetch_value);
        }
        else
        {
            // clang-format off
            // T has grads like this
            // fl_grad | pp_grad | dc_grad | cam0_pos_grad | cam0_quat_grad | t0_grad | cam1_pos_grad | cam1_quat_grad
            // 2       | 2       | 4       | 3             | 4              | 1       | 3             | 4
            //
            // cache, in general, has grads like this for each feature
            // fl_grad | pp_grad | dc_grad | cam_pos_grad | cam_quat_grad | t_grad
            // 2       | 2       | 4       | 3            | 4             | 1
            //
            // for cache 0 we copy the all gradients to head
            // for cache 1 we copy the intrinsics gradients
            // clang-format on

            pt0 = cache0.point.unaryExpr(
              [](auto&& x)
              {
                  T ret;
                  ret.a = x.a;
                  // the 3D point depends on all the parameters
                  ret.v.template head<cache::num_grads>() = x.v;
                  return ret;
              });

            undistorted_px1 = cache1.undistorted_wnd.unaryExpr(
              [](auto&& x)
              {
                  T ret;
                  ret.a = x.a;
                  // the undistorted pixel depends on intrinsics only
                  ret.v.template head<8>() = x.v.template head<8>();
                  return ret;
              });
        }

        // cross project 3d points to 2d
        auto&& [reproj0_on_1, _] = project(focal_length,
                                           principal_point,
                                           camera1_world_position,
                                           camera1_world_orientation,
                                           pt0);

        // residual
        residual = reproj0_on_1 - undistorted_px1;
    }
};

// Lifted-weights variant of the cross-projection residual: the source
// observation's block is [t, w] and the residual is scaled by the
// confidence w. The cached point's 16 gradients map to jet dims 0-15
// unchanged (w sits at dim 16, its gradient comes from the w-scaling
// itself); the target camera's gradients shift to dims 17-23 but are
// only intrinsics (head<8>) in the cache copy, so the surgery is
// identical to the plain cost.
struct back_projection_lifted_cost
  : CostBase<back_projection_lifted_cost,
             decltype(vec2 = vec2, vec2, vec4, vec3, quat, vec2, vec3, quat)>
{
    const cache::data &cache0, &cache1;

    back_projection_lifted_cost(const cache::data& cache0,
                                const cache::data& cache1)
      : cache0(cache0), cache1(cache1)
    {
    }

    template <typename Vec2,
              typename Vec3,
              typename Vec4,
              typename Quat,
              typename Vec2TW,
              typename Residual>
    void compute(Vec2 focal_length,
                 Vec2 principal_point,
                 Vec4 dist_coeffs,
                 Vec3 camera0_world_position,
                 Quat camera0_world_orientation,
                 Vec2TW tw_0,
                 Vec3 camera1_world_position,
                 Quat camera1_world_orientation,
                 Residual residual) const
    {
        using T = typename Residual::Scalar;

        using Vec2_ = Eigen::Vector<T, 2>;
        using Vec3_ = Eigen::Vector<T, 3>;

        Vec3_ pt0;
        Vec2_ undistorted_px1;

        constexpr bool residuals_only = std::is_same_v<T, double>;

        if constexpr (residuals_only)
        {
            const auto fetch_value = [](auto&& x) { return x.a; };
            pt0                    = cache0.point.unaryExpr(fetch_value);
            undistorted_px1 = cache1.undistorted_wnd.unaryExpr(fetch_value);
        }
        else
        {
            pt0 = cache0.point.unaryExpr(
              [](auto&& x)
              {
                  T ret;
                  ret.a                                   = x.a;
                  ret.v.template head<cache::num_grads>() = x.v;
                  return ret;
              });

            undistorted_px1 = cache1.undistorted_wnd.unaryExpr(
              [](auto&& x)
              {
                  T ret;
                  ret.a                    = x.a;
                  ret.v.template head<8>() = x.v.template head<8>();
                  return ret;
              });
        }

        auto&& [reproj0_on_1, _] = project(focal_length,
                                           principal_point,
                                           camera1_world_position,
                                           camera1_world_orientation,
                                           pt0);

        residual = (reproj0_on_1 - undistorted_px1) * tw_0(1);
    }
};

// Penalty tying a lifted confidence to 1: sigma * (w - 1). Together with
// the w-scaled residuals this is the Black-Rangarajan outlier process of
// a Geman-McClure kernel with pixel scale sigma; annealing sigma large ->
// small is GNC-GM.
struct lifted_weight_prior_cost
  : CostBase<lifted_weight_prior_cost, decltype(vec1 = vec2)>
{
    const double& sigma;

    explicit lifted_weight_prior_cost(const double& sigma) : sigma(sigma) {}

    template <typename Vec2TW, typename Residual>
    void compute(Vec2TW tw, Residual residual) const
    {
        using T     = typename Residual::Scalar;
        residual(0) = T(sigma) * (tw(1) - T(1));
    }
};

// Lifted track consensus (alternated form): pull an observation's ray point
// toward its track's consensus point, held fixed during a solve. The cached
// point jet already carries the full 16 gradients of this observation's
// local parameter set, so the gradient surgery is a straight copy.
struct consensus_cost
  : CostBase<consensus_cost, decltype(vec3 = vec2, vec2, vec4, vec3, quat, vec1)>
{
    const cache::data& c;
    const Eigen::Vector3d& target;
    const double& weight;
    const double& member_weight;

    consensus_cost(const cache::data& c,
                   const Eigen::Vector3d& target,
                   const double& weight,
                   const double& member_weight)
      : c(c), target(target), weight(weight), member_weight(member_weight)
    {
    }

    template <typename Vec2,
              typename Vec3,
              typename Vec4,
              typename Quat,
              typename Vec1,
              typename Residual>
    void compute(Vec2, Vec2, Vec4, Vec3, Quat, Vec1, Residual residual) const
    {
        using T = typename Residual::Scalar;

        Eigen::Vector<T, 3> pt;
        if constexpr (std::is_same_v<T, double>)
            pt = c.point.unaryExpr([](auto&& x) { return x.a; });
        else
            pt = c.point.unaryExpr(
              [](auto&& x)
              {
                  T ret;
                  ret.a                                   = x.a;
                  ret.v.template head<cache::num_grads>() = x.v;
                  return ret;
              });

        residual =
          (pt - target.cast<T>()) * T(std::sqrt(weight * member_weight));
    }
};

// Joint form of the consensus lift: the consensus point is a 3-dim
// parameter block optimized together with poses and ray times (it lands in
// the reduced system; the ray times stay Schur-eliminated). The cached
// point contributes the usual 16 gradients; the consensus point's arrive
// through autodiff on the mapped parameter.
struct consensus_joint_cost
  : CostBase<consensus_joint_cost,
             decltype(vec3 = vec2, vec2, vec4, vec3, quat, vec1, vec3)>
{
    const cache::data& c;
    const double& weight;
    const double& member_weight;

    consensus_joint_cost(const cache::data& c,
                         const double& weight,
                         const double& member_weight)
      : c(c), weight(weight), member_weight(member_weight)
    {
    }

    template <typename Vec2,
              typename Vec3,
              typename Vec4,
              typename Quat,
              typename Vec1,
              typename Vec3X,
              typename Residual>
    void compute(Vec2, Vec2, Vec4, Vec3, Quat, Vec1, Vec3X x, Residual residual)
      const
    {
        using T = typename Residual::Scalar;

        Eigen::Vector<T, 3> pt;
        if constexpr (std::is_same_v<T, double>)
            pt = c.point.unaryExpr([](auto&& v) { return v.a; });
        else
            pt = c.point.unaryExpr(
              [](auto&& v)
              {
                  T ret;
                  ret.a                                   = v.a;
                  ret.v.template head<cache::num_grads>() = v.v;
                  return ret;
              });

        residual = (pt - x) * T(std::sqrt(weight * member_weight));
    }
};

void bundle_adjuster::_initialize_cross_projection_residuals()
{
    auto optimization_cache = std::make_shared<cache>();
    this->cached_points     = optimization_cache;

    auto for_each_correspondence = [&](auto&& fn)
    {
        for (auto&& [camera_indices, matches] : this->matches)
        {
            auto&& [camera0_index, camera1_index] = camera_indices;
            for (std::size_t i = 0; i < matches.rows(); ++i)
            {
                auto&& match = matches.row(i);
                fn(cache::key{camera0_index, match.x()},
                   cache::key{camera1_index, match.y()});
            }
        }
    };

    if (!this->feature_tracks.empty() && !this->consensus)
    {
        _initialize_anchored_track_residuals();
        return;
    }

    for_each_correspondence(
      [&](const cache::key& key0, const cache::key& key1)
      {
          optimization_cache->populate(key0);
          optimization_cache->populate(key1);
      });

    optimization_cache->realize();

    for_each_correspondence(
      [&](const cache::key& key0, const cache::key& key1)
      {
          auto&& cache0 = optimization_cache->at(key0);
          auto&& cache1 = optimization_cache->at(key1);

          auto&& [camera0_index, feature0] = key0;
          auto&& [camera1_index, feature1] = key1;

          auto&& [camera0_position, camera0_quaternion] =
            this->camera_poses.at(camera0_index);
          auto&& [camera1_position, camera1_quaternion] =
            this->camera_poses.at(camera1_index);

          auto&& features0  = this->features.at(camera0_index);
          auto&& features1  = this->features.at(camera1_index);

          auto&& distorted_pixel0 = features0.row(feature0);
          auto&& distorted_pixel1 = features1.row(feature1);

          auto* ray_time0 =
            this->ray_time_of.at({(int)camera0_index, (int)feature0});
          auto* ray_time1 =
            this->ray_time_of.at({(int)camera1_index, (int)feature1});

          if (this->lifted_weights_sigma > 0.0)
          {
              // lifted mode: confidence-scaled residuals, no robust loss
              // (the lifted weights replace it)
              this->problem->AddResidualBlock(
                back_projection_lifted_cost::create(cache0, cache1),
                nullptr,
                this->focal_length.data(),
                this->principal_point.data(),
                this->distortion_coefficients.data(),
                camera0_position.data(),
                camera0_quaternion.data(),
                ray_time0,
                camera1_position.data(),
                camera1_quaternion.data());

              this->problem->AddResidualBlock(
                back_projection_lifted_cost::create(cache1, cache0),
                nullptr,
                this->focal_length.data(),
                this->principal_point.data(),
                this->distortion_coefficients.data(),
                camera1_position.data(),
                camera1_quaternion.data(),
                ray_time1,
                camera0_position.data(),
                camera0_quaternion.data());
              return;
          }

          this->problem->AddResidualBlock(
            // cost
            back_projection_cost::create(cache0, cache1),
            // loss
            this->scale_loss.get(),
            // params
            this->focal_length.data(),
            this->principal_point.data(),
            this->distortion_coefficients.data(),
            camera0_position.data(),
            camera0_quaternion.data(),
            ray_time0,
            camera1_position.data(),
            camera1_quaternion.data());

          this->problem->AddResidualBlock(
            // cost
            back_projection_cost::create(cache1, cache0),
            // loss
            this->scale_loss.get(),
            // params
            this->focal_length.data(),
            this->principal_point.data(),
            this->distortion_coefficients.data(),
            camera1_position.data(),
            camera1_quaternion.data(),
            ray_time1,
            camera0_position.data(),
            camera0_quaternion.data());
      });

    if (this->lifted_weights_sigma > 0.0)
    {
        std::size_t n_priors = 0;
        for (auto&& [key, _] : optimization_cache->indirection)
        {
            auto&& [camera_index, k] = key;
            this->problem->AddResidualBlock(
              lifted_weight_prior_cost::create(this->lifted_weights_sigma),
              nullptr,
              this->ray_time_of.at({(int)camera_index, (int)k}));
            ++n_priors;
        }
        std::cout << "Lifted weights: " << n_priors << " observations, sigma "
                  << this->lifted_weights_sigma << std::endl;
    }

    if (this->consensus && !this->feature_tracks.empty())
        _initialize_consensus_residuals();
}

void bundle_adjuster::_initialize_consensus_residuals()
{
    auto optimization_cache =
      std::static_pointer_cast<cache>(this->cached_points);

    // group matched observations by track id; unmatched features have no
    // cache entry and cannot participate
    std::map<int, std::vector<std::pair<int, int>>> members_of;
    for (auto&& [camera_index, track_ids] : this->feature_tracks)
        for (int k = 0; k < track_ids.rows(); ++k)
        {
            const int tid = track_ids(k);
            if (tid < 0) continue;
            const cache::key key{(std::size_t)camera_index, (std::size_t)k};
            if (optimization_cache->indirection.find(key) ==
                optimization_cache->indirection.end())
                continue;
            members_of[tid].push_back({camera_index, k});
        }

    this->consensus_groups.clear();
    for (auto& [tid, members] : members_of)
        if (members.size() >= 2)
            this->consensus_groups.push_back(std::move(members));

    // stable storage: the cost functors hold references into this vector,
    // so it must never be resized after this point
    this->consensus_targets.assign(this->consensus_groups.size(),
                                   Eigen::Vector3d::Zero());
    std::size_t n_members = 0;
    for (auto&& g : this->consensus_groups) n_members += g.size();
    this->consensus_member_weights.assign(n_members, 1.0);

    // The residual is pre-scaled by sqrt(weight), so a unit arctan loss
    // saturates at ||pt - X|| = 1/sqrt(weight): annealing the weight
    // tightens the consensus trust radius, and members far from a target
    // (including targets poisoned by near-infinity outlier rays)
    // contribute bounded cost instead of dominating the solve.
    this->consensus_scale_loss = std::make_shared<ceres::ArctanLoss>(1.0);

    std::size_t n_residuals = 0;
    for (std::size_t g = 0; g < this->consensus_groups.size(); ++g)
    {
        for (auto&& [camera_index, k] : this->consensus_groups[g])
        {
            auto&& [position, quaternion] =
              this->camera_poses.at(camera_index);
            auto&& cache_entry = optimization_cache->at(
              {(std::size_t)camera_index, (std::size_t)k});
            if (this->consensus_joint)
                this->problem->AddResidualBlock(
                  consensus_joint_cost::create(
                    cache_entry,
                    this->consensus_weight,
                    this->consensus_member_weights[n_residuals]),
                  this->consensus_scale_loss.get(),
                  this->focal_length.data(),
                  this->principal_point.data(),
                  this->distortion_coefficients.data(),
                  position.data(),
                  quaternion.data(),
                  this->ray_time_of.at({camera_index, k}),
                  this->consensus_targets[g].data());
            else
                this->problem->AddResidualBlock(
                  consensus_cost::create(
                    cache_entry,
                    this->consensus_targets[g],
                    this->consensus_weight,
                    this->consensus_member_weights[n_residuals]),
                  this->consensus_scale_loss.get(),
                  this->focal_length.data(),
                  this->principal_point.data(),
                  this->distortion_coefficients.data(),
                  position.data(),
                  quaternion.data(),
                  this->ray_time_of.at({camera_index, k}));
            ++n_residuals;
        }
        // consensus points stay frozen until a positive weight enables them
        if (this->consensus_joint)
            this->problem->SetParameterBlockConstant(
              this->consensus_targets[g].data());
    }

    std::cout << "Consensus: " << this->consensus_groups.size()
              << " tracks, " << n_residuals << " residuals" << std::endl;
}

void bundle_adjuster::set_consensus_weight(const double weight)
{
    this->consensus_weight = weight;
    if (this->consensus_joint)
        for (auto& x : this->consensus_targets)
        {
            if (weight > 0.0)
                this->problem->SetParameterBlockVariable(x.data());
            else
                this->problem->SetParameterBlockConstant(x.data());
        }
}

void bundle_adjuster::update_consensus_targets()
{
    if (this->consensus_groups.empty()) return;

    this->_update_cache();
    auto optimization_cache =
      std::static_pointer_cast<cache>(this->cached_points);

    // Robust targets: coordinate-wise median, then trim members far from
    // it before the pull. Ray points on outlier rays can sit arbitrarily
    // far away (their pixel residuals are saturated by the arctan loss);
    // a plain mean/quadratic pull toward them diverges the solve.
    std::size_t idx = 0;
    for (std::size_t g = 0; g < this->consensus_groups.size(); ++g)
    {
        const auto& members = this->consensus_groups[g];
        const std::size_t n = members.size();

        std::vector<Eigen::Vector3d> pts(n);
        for (std::size_t m = 0; m < n; ++m)
            pts[m] = optimization_cache
                       ->at({(std::size_t)members[m].first,
                             (std::size_t)members[m].second})
                       .point.unaryExpr([](auto&& x) { return x.a; })
                       .eval();

        Eigen::Vector3d med;
        std::vector<double> coord(n);
        for (int c = 0; c < 3; ++c)
        {
            for (std::size_t m = 0; m < n; ++m) coord[m] = pts[m](c);
            std::nth_element(
              coord.begin(), coord.begin() + n / 2, coord.end());
            med(c) = coord[n / 2];
        }

        std::vector<double> dist(n);
        for (std::size_t m = 0; m < n; ++m)
            dist[m] = (pts[m] - med).norm();
        std::vector<double> dist_sorted = dist;
        std::nth_element(
          dist_sorted.begin(), dist_sorted.begin() + n / 2, dist_sorted.end());
        const double gate = std::max(3.0 * dist_sorted[n / 2], 0.01);

        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        std::size_t n_in    = 0;
        for (std::size_t m = 0; m < n; ++m)
        {
            const bool in                        = dist[m] <= gate;
            this->consensus_member_weights[idx + m] = in ? 1.0 : 0.0;
            if (in)
            {
                sum += pts[m];
                ++n_in;
            }
        }
        this->consensus_targets[g] =
          n_in > 0 ? Eigen::Vector3d(sum / double(n_in)) : med;
        idx += n;
    }
}


void bundle_adjuster::_initialize_anchored_track_residuals()
{
    auto optimization_cache =
      std::static_pointer_cast<cache>(this->cached_points);

    // group observations by track id; anchor = first (lowest camera index)
    std::map<int, std::vector<cache::key>> members_of;
    for (auto&& [camera_index, track_ids] : this->feature_tracks)
    {
        for (int k = 0; k < track_ids.rows(); ++k)
        {
            const int tid = track_ids(k);
            if (tid >= 0)
                members_of[tid].push_back(cache::key{camera_index, k});
        }
    }

    for (auto& [tid, members] : members_of)
    {
        if (members.size() < 2) continue;
        std::sort(members.begin(), members.end());
        for (auto&& member : members) optimization_cache->populate(member);
    }

    optimization_cache->realize();

    std::size_t n_residuals = 0;
    for (auto& [tid, members] : members_of)
    {
        if (members.size() < 2) continue;
        const auto& anchor = members.front();

        auto&& [anchor_camera, anchor_feature] = anchor;
        auto&& [anchor_position, anchor_quaternion] =
          this->camera_poses.at(anchor_camera);
        auto* anchor_time = this->ray_time_of.at(
          {(int)anchor_camera, (int)anchor_feature});
        auto&& anchor_cache = optimization_cache->at(anchor);

        for (std::size_t m = 1; m < members.size(); ++m)
        {
            const auto& other                    = members[m];
            auto&& [other_camera, other_feature] = other;
            auto&& [other_position, other_quaternion] =
              this->camera_poses.at(other_camera);
            auto&& other_cache = optimization_cache->at(other);

            this->problem->AddResidualBlock(
              back_projection_cost::create(anchor_cache, other_cache),
              this->scale_loss.get(),
              this->focal_length.data(),
              this->principal_point.data(),
              this->distortion_coefficients.data(),
              anchor_position.data(),
              anchor_quaternion.data(),
              anchor_time,
              other_position.data(),
              other_quaternion.data());
            ++n_residuals;
        }
    }

    std::cout << "Anchored tracks: " << members_of.size() << " tracks, "
              << n_residuals << " residuals" << std::endl;
}

nd_array<double, -1, 3> bundle_adjuster::get_point_cloud() const
{
    auto cache_data = std::static_pointer_cast<cache>(this->cached_points);

    auto&& data = cache_data->values;

    nd_array<double, -1, 3> ret(data.size(), 3);

    tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, data.size()),
      [&](auto&& range)
      {
          for (std::size_t i = range.begin(); i != range.end(); ++i)
          {
              auto&& [_, point] = data.at(i);
              ret.row(i)        = point.unaryExpr([](auto&& x) { return x.a; });
          }
      });

    return ret;
}

auto to_jet(const double& value, int index)
{
    return std::make_tuple(cache::scalar(value, index), index + 1);
}

template <typename Derived>
auto to_jet(const Eigen::MatrixBase<Derived>& mat, int index)
{
    return std::make_tuple(
      mat.unaryExpr([&](auto&& value) { return cache::scalar(value, index++); })
        .eval(),
      index);
}

template <typename Derived>
auto to_jet(const Eigen::QuaternionBase<Derived>& quat, int index)
{
    return std::make_tuple(
      cache::quaternion(cache::scalar(quat.w(), index + 3),
                        cache::scalar(quat.x(), index + 0),
                        cache::scalar(quat.y(), index + 1),
                        cache::scalar(quat.z(), index + 2)),
      index + 4);
}

auto to_jets(int index, auto&& first, auto&&... rest)
{
    auto [first_jet, next_index] = to_jet(first, index);

    if constexpr (sizeof...(rest) == 0) { return std::make_tuple(first_jet); }
    else
    {
        auto rest_jets = to_jets(next_index, rest...);
        return std::tuple_cat(std::make_tuple(first_jet), rest_jets);
    }
}

void bundle_adjuster::_update_cache()
{
    auto optimization_cache =
      std::static_pointer_cast<cache>(this->cached_points);

    std::vector<cache::key> keys;
    keys.reserve(optimization_cache->indirection.size());
    for (auto&& [key, _] : optimization_cache->indirection)
    {
        keys.push_back(key);
    }

    tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, keys.size()),
      [&](auto&& range)
      {
          for (size_t i = range.begin(); i != range.end(); ++i)
          {
              // thread context
              auto&& key                   = keys.at(i);
              auto&& cache_out             = optimization_cache->at(key);
              auto&& [camera, feature_idx] = key;

              // global
              auto&& focal_length           = this->focal_length;
              auto&& principal_point        = this->principal_point;
              auto& distortion_coefficients = this->distortion_coefficients;

              // per camera
              auto&& [position, orientation] = this->camera_poses.at(camera);

              // per feature
              auto&& distorted_wnd = this->features.at(camera).row(feature_idx);
              auto&& ray_time =
                *this->ray_time_of.at({(int)camera, (int)feature_idx});

              // the order must be the same as in the back_projection_cost for
              // the gradient surgery to work
              auto&& [focal_length_jet,
                      principal_point_jet,
                      distortion_coefficients_jet,
                      position_jet,
                      orientation_jet,
                      ray_time_jet] =
                to_jets(
                  0,
                  focal_length,
                  principal_point,
                  distortion_coefficients,
                  position,
                  Eigen::Map<const Eigen::Quaterniond>(orientation.data()),
                  ray_time);

              auto&& distorted_wnd_jet =
                distorted_wnd.transpose().template cast<cache::scalar>();
              auto&& distorted_ndc_jet = wnd2ndc(
                distorted_wnd_jet, focal_length_jet, principal_point_jet);
              auto&& undistorted_ndc_jet =
                undistort_ndc(distorted_ndc_jet, distortion_coefficients_jet);
              cache_out.undistorted_wnd = ndc2wnd(
                undistorted_ndc_jet, focal_length_jet, principal_point_jet);

              auto&& [O0, D0] = unproject(focal_length_jet,
                                          principal_point_jet,
                                          position_jet,
                                          orientation_jet,
                                          cache_out.undistorted_wnd);

              cache_out.point = (O0 + ray_time_jet * D0).transpose();
          }
      });
}