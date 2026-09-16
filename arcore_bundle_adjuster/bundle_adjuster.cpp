#include "bundle_adjuster.hpp"

#include <set>
#include <thread>

#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <ceres/ceres.h>
#include <ceres/normal_prior.h>

#include "util.hpp"

using namespace arba;

namespace
{
// ceres::ArctanLoss with a scale that can be changed between solves
// (graduated non-convexity). Same formula as ceres/loss_function.cc.
struct mutable_arctan_loss : ceres::LossFunction
{
    double a;

    explicit mutable_arctan_loss(const double a) : a(a) {}

    void Evaluate(double s, double rho[3]) const override
    {
        const double b   = 1.0 / (a * a);
        const double sum = 1.0 + s * s * b;
        const double inv = 1.0 / sum;
        rho[0]           = a * std::atan2(s, a);
        rho[1]           = inv;
        rho[2]           = -2.0 * s * b * (inv * inv);
    }
};

// ceres::ScaledLoss (over an optional inner loss) with a mutable weight
// (prior homotopy).
struct mutable_scaled_loss : ceres::LossFunction
{
    const ceres::LossFunction* inner; // not owned
    double weight;

    mutable_scaled_loss(const ceres::LossFunction* inner, const double weight)
      : inner(inner), weight(weight)
    {
    }

    void Evaluate(double s, double rho[3]) const override
    {
        if (inner == nullptr)
        {
            rho[0] = weight * s;
            rho[1] = weight;
            rho[2] = 0.0;
            return;
        }
        inner->Evaluate(s, rho);
        rho[0] *= weight;
        rho[1] *= weight;
        rho[2] *= weight;
    }
};
} // namespace

void bundle_adjuster::set_arctan_loss(const double a)
{
    if (auto p = std::dynamic_pointer_cast<mutable_arctan_loss>(
          this->arctan_loss))
        p->a = a;
}

void bundle_adjuster::set_position_prior_lambda(const double lambda)
{
    if (auto p = std::dynamic_pointer_cast<mutable_scaled_loss>(
          this->prior_scale_loss))
        p->weight = lambda / this->camera_poses.size();
}

void bundle_adjuster::set_lifted_sigma(const double sigma)
{
    if (this->lifted_weights_sigma > 0.0) this->lifted_weights_sigma = sigma;
}

void bundle_adjuster::set_orientation_prior_lambda(const double lambda)
{
    if (auto p = std::dynamic_pointer_cast<mutable_scaled_loss>(
          this->orientation_prior_scale_loss))
        p->weight = lambda / this->camera_poses.size();
}

bundle_adjuster::bundle_adjuster(
  const nd_vector<double, 2>& focal_length,
  const nd_vector<double, 2>& principal_point,
  const nd_vector<double, 4>& distortion_coefficients,
  const camera_features_map& features,
  const camera_matches_map& matches,
  const camera_poses_map& camera_poses,
  const double arctan_loss,
  const double position_prior_lambda,
  const double orientation_prior_lambda,
  const camera_tracks_map& feature_tracks,
  const camera_ray_times_map& initial_ray_times,
  const bool consensus,
  const bool consensus_joint,
  const double lifted_weights_sigma)
  : focal_length(focal_length)
  , principal_point(principal_point)
  , distortion_coefficients(distortion_coefficients)
  , camera_poses(camera_poses)
  , features(features)
  , matches(matches)
  , feature_tracks(feature_tracks)
  , initial_ray_times(initial_ray_times)
  , consensus(consensus)
  , consensus_joint(consensus_joint)
  , lifted_weights_sigma(lifted_weights_sigma)
{
    scoped_timed_log _("Bundle adjuster initialization");

    {
        scoped_timed_log _("Creating ceres problem");
        _create_ceres_problem();
    }

    {
        scoped_timed_log _("Creating ceres helpers");
        _create_ceres_helpers(arctan_loss, position_prior_lambda);
        if (orientation_prior_lambda > 0.0)
            this->orientation_prior_scale_loss =
              std::make_shared<mutable_scaled_loss>(
                nullptr, orientation_prior_lambda / camera_poses.size());
    }

    {
        scoped_timed_log _("Initializing camera ray times");
        _initialize_camera_ray_times();
    }

    {
        scoped_timed_log _("Declaring ceres parameters");
        _declare_ceres_parameters();
    }

    {
        scoped_timed_log _("Initializing residuals");
        _initialize_cross_projection_residuals();
    }

    {
        scoped_timed_log _("Initializing position priors");
        _initialize_position_priors();
    }

    if (orientation_prior_lambda > 0.0)
    {
        scoped_timed_log _("Initializing orientation priors");
        _initialize_orientation_priors();
    }
}

namespace
{
// Small-angle rotation-vector distance to a fixed prior quaternion.
struct orientation_prior_cost
{
    const Eigen::Quaterniond prior;

    explicit orientation_prior_cost(const Eigen::Quaterniond& prior)
      : prior(prior)
    {
    }

    template <typename T>
    bool operator()(const T* const q_xyzw, T* residual) const
    {
        const Eigen::Map<const Eigen::Quaternion<T>> q(q_xyzw);
        const Eigen::Quaternion<T> delta =
          prior.template cast<T>().conjugate() * q;
        // 2 * vec(delta) ~ rotation vector for small angles; sign-correct
        // to the shorter arc
        const T sign = delta.w() < T(0) ? T(-1) : T(1);
        residual[0]  = T(2) * sign * delta.x();
        residual[1]  = T(2) * sign * delta.y();
        residual[2]  = T(2) * sign * delta.z();
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Quaterniond& prior)
    {
        return new ceres::AutoDiffCostFunction<orientation_prior_cost, 3, 4>(
          new orientation_prior_cost(prior));
    }
};
} // namespace

void bundle_adjuster::_initialize_orientation_priors()
{
    for (auto& [camera_index, pose] : this->camera_poses)
    {
        auto&& [_, camera_world_quaternion_xyzw] = pose;
        const Eigen::Quaterniond prior(
          Eigen::Map<const Eigen::Quaterniond>(
            camera_world_quaternion_xyzw.data()));
        this->problem->AddResidualBlock(
          orientation_prior_cost::create(prior),
          this->orientation_prior_scale_loss.get(),
          camera_world_quaternion_xyzw.data());
    }
}

struct DoOnEvaluationStart : ceres::EvaluationCallback
{
    std::function<void()> callback;
    bool last_new_evaluation_point =
      false; // Track the state of the last evaluation point

    explicit DoOnEvaluationStart(std::function<void()> callback)
      : callback(std::move(callback))
    {
    }

    void PrepareForEvaluation(bool evaluate_jacobians,
                              bool new_evaluation_point) override
    {
        BOOST_LOG_TRIVIAL(info)
          << boost::format(
               "PrepareForEvaluation: evaluate_jacobians=%1%, "
               "new_evaluation_point=%2%") %
               evaluate_jacobians % new_evaluation_point;

        // Trigger the callback if it is a new point, or if Jacobians are needed
        // but weren't computed since the last new point
        if (new_evaluation_point ||
            (evaluate_jacobians && !last_new_evaluation_point))
        {
            callback();
            last_new_evaluation_point = new_evaluation_point;
        }
    }
};

void bundle_adjuster::_create_ceres_problem()
{
    // create the problem
    ceres::Problem::Options problem_options;
    this->evalCallback = std::make_shared<DoOnEvaluationStart>(
      [this]()
      {
          scoped_timed_log _("Updating cache");
          this->_update_cache();
      });
    problem_options.loss_function_ownership   = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_options.manifold_ownership        = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_options.disable_all_safety_checks = true;
    problem_options.evaluation_callback       = this->evalCallback.get();
    this->problem = std::make_shared<ceres::Problem>(problem_options);
}

void bundle_adjuster::_create_ceres_helpers(const double arctan_loss,
                                            const double position_prior_lambda)
{
    auto quaternion_parameterization =
      std::make_shared<ceres::EigenQuaternionManifold>();
    this->quaternion_parameterization = quaternion_parameterization;

    if (arctan_loss > 0.0)
        this->arctan_loss = std::make_shared<mutable_arctan_loss>(arctan_loss);
    auto matches_count = std::accumulate(std::begin(matches),
                                         std::end(matches),
                                         0,
                                         [](auto&& sum, auto&& pair)
                                         { return sum + pair.second.rows(); });
    this->scale_loss   = std::make_shared<ceres::ScaledLoss>(
      this->arctan_loss.get(),
      // normalize by the amount of residual values. there are matches_count
      // matches with each one having 2 residuals with each residual being 2D
      1.0 / matches_count / 2 / 2,
      ceres::DO_NOT_TAKE_OWNERSHIP);
    this->prior_scale_loss = std::make_shared<mutable_scaled_loss>(
      nullptr, position_prior_lambda / camera_poses.size());
}

void bundle_adjuster::_declare_ceres_parameters()
{
    // parameter declaration shorthand
    auto add_parameter = [&](auto&& parameter)
    { this->problem->AddParameterBlock(parameter.data(), parameter.size()); };

    // intrinsics
    add_parameter(this->focal_length);

    // non-negative focal length
    this->problem->SetParameterLowerBound(this->focal_length.data(), 0, 1e-9);
    this->problem->SetParameterLowerBound(this->focal_length.data(), 1, 1e-9);

    add_parameter(this->principal_point);
    add_parameter(this->distortion_coefficients);

    // extrinsics positions
    for (auto&& [_, pose] : this->camera_poses)
    {
        auto&& [camera_world_position, camera_world_quaternion_xyzw] = pose;
        add_parameter(camera_world_position);
        add_parameter(camera_world_quaternion_xyzw);
        this->problem->SetManifold(camera_world_quaternion_xyzw.data(),
                                   std::static_pointer_cast<ceres::Manifold>(
                                     this->quaternion_parameterization)
                                     .get());
    }

    // ray times (unique variables only: track-shared times appear once);
    // lifted mode declares 2-dim [t, w] blocks, positivity bound on t only
    {
        const int block_size = this->lifted_weights_sigma > 0.0 ? 2 : 1;
        std::set<double*> declared;
        for (auto&& [_, ray_time_ptr] : this->ray_time_of)
        {
            if (!declared.insert(ray_time_ptr).second) continue;
            this->problem->AddParameterBlock(ray_time_ptr, block_size);
            this->problem->SetParameterLowerBound(ray_time_ptr, 0, 1e-9);
        }
    }
}

void bundle_adjuster::_initialize_camera_ray_times()
{
    // initialize camera_ray_times (default 1.0; external initial values,
    // e.g. AR-framework depth samples, override where positive)
    for (const auto& [camera_index, feature] : this->features)
    {
        auto& ray_times = this->camera_ray_times[camera_index];
        ray_times       = nd_array<double, -1, 1>::Ones(feature.rows());

        const auto init_it = this->initial_ray_times.find(camera_index);
        if (init_it == this->initial_ray_times.end()) continue;
        const auto& init = init_it->second;
        for (int k = 0; k < std::min(ray_times.rows(), init.rows()); ++k)
            if (init(k) > 0.0) ray_times(k) = init(k);
    }

    // resolve each observation's ray-time variable: shared per track when a
    // track id is available (anchored mode), per-observation otherwise.
    // Consensus mode keeps per-observation times: the coupling comes from
    // the consensus residuals, not from variable sharing.
    for (auto& [camera_index, ray_times] : this->camera_ray_times)
    {
        const auto tracks_it = this->feature_tracks.find(camera_index);
        for (int k = 0; k < ray_times.rows(); ++k)
        {
            int track_id = -1;
            if (!this->consensus &&
                tracks_it != this->feature_tracks.end() &&
                k < tracks_it->second.rows())
                track_id = tracks_it->second(k);

            if (track_id >= 0)
            {
                auto [it, _] = this->track_times.try_emplace(track_id, 1.0);
                this->ray_time_of[{camera_index, k}] = &it->second;
            }
            else if (this->lifted_weights_sigma > 0.0)
            {
                // [t, w] block; w initialized to full confidence
                auto& v = this->lifted_obs_params[camera_index];
                if (v.empty()) v.resize(ray_times.rows());
                v[k] = Eigen::Vector2d(ray_times(k), 1.0);
                this->ray_time_of[{camera_index, k}] = v[k].data();
            }
            else { this->ray_time_of[{camera_index, k}] = &ray_times(k); }
        }
    }
}

void bundle_adjuster::_initialize_position_priors()
{
    // initialize position priors
    for (auto& [camera_index, pose] : this->camera_poses)
    {
        auto&& [camera_world_position, _] = pose;
        this->problem->AddResidualBlock(
          new ceres::NormalPrior(Eigen::Matrix3d::Identity(),
                                 camera_world_position),
          this->prior_scale_loss.get(),
          camera_world_position.data());
    }
}

std::string bundle_adjuster::solve(const std::size_t max_iterations,
                                   const bool verbose,
                                   const optimization_inclusion inclusion,
                                   const double function_tolerance,
                                   const std::string& linear_solver)
{
    try
    {
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = verbose;
        options.max_num_iterations           = max_iterations;
        options.function_tolerance           = function_tolerance;
        options.num_threads        = std::thread::hardware_concurrency();
        options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type =
          ceres::SparseLinearAlgebraLibraryType::SUITE_SPARSE;
        if (linear_solver == "iterative_schur")
            options.linear_solver_type =
              ceres::LinearSolverType::ITERATIVE_SCHUR;
        else if (linear_solver == "dense_schur_cuda")
        {
            options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;
            options.dense_linear_algebra_library_type =
              ceres::DenseLinearAlgebraLibraryType::CUDA;
        }
        options.preconditioner_type = ceres::PreconditionerType::SCHUR_JACOBI;
        if (this->consensus_joint && !this->consensus_targets.empty())
        {
            // A consensus residual couples a ray time and a consensus
            // point; with safety checks disabled, automatic Schur ordering
            // may eliminate both (invalid). Pin the elimination group
            // explicitly: ray times eliminated, everything else reduced.
            auto ordering =
              std::make_shared<ceres::ParameterBlockOrdering>();
            std::set<double*> times;
            for (auto&& [_, p] : this->ray_time_of) times.insert(p);
            for (auto* p : times) ordering->AddElementToGroup(p, 0);
            ordering->AddElementToGroup(this->focal_length.data(), 1);
            ordering->AddElementToGroup(this->principal_point.data(), 1);
            ordering->AddElementToGroup(
              this->distortion_coefficients.data(), 1);
            for (auto& [_, pose] : this->camera_poses)
            {
                auto& [position, quaternion] = pose;
                ordering->AddElementToGroup(position.data(), 1);
                ordering->AddElementToGroup(quaternion.data(), 1);
            }
            for (auto& x : this->consensus_targets)
                ordering->AddElementToGroup(x.data(), 1);
            options.linear_solver_ordering = ordering;
        }
        // The positivity bounds (focal length, ray times) make the problem
        // constrained, which by default triggers a projected Armijo line
        // search costing one extra gradient evaluation per iteration. The
        // bounds are enforced by projection in ParameterBlock::Plus()
        // regardless, so skip the line search.
        options.max_num_line_search_step_size_iterations = 0;

        // update variability according to the inclusion
        const auto set_variability = [&](auto&& parameter, const bool value)
        {
            if (value)
                this->problem->SetParameterBlockVariable(parameter.data());
            else
                this->problem->SetParameterBlockConstant(parameter.data());
        };

        set_variability(this->focal_length,
                        inclusion & optimization_inclusion::focal_length);
        set_variability(this->principal_point,
                        inclusion & optimization_inclusion::principal_point);
        set_variability(this->distortion_coefficients,
                        inclusion &
                          optimization_inclusion::distortion_coefficients);

        for (auto&& [_, pose] : this->camera_poses)
        {
            auto&& [camera_world_position, camera_world_quaternion_xyzw] = pose;
            set_variability(
              camera_world_position,
              (inclusion & optimization_inclusion::camera_poses) ||
                (inclusion & optimization_inclusion::camera_positions));
            set_variability(
              camera_world_quaternion_xyzw,
              (inclusion & optimization_inclusion::camera_poses) ||
                (inclusion & optimization_inclusion::camera_orientations));
        }

        ceres::Solver::Summary summary;
        ceres::Solve(options, this->problem.get(), &summary);

        return summary.FullReport();
    }
    catch (const std::exception& e)
    {
        return e.what();
    }
    catch (...)
    {
        return "Unknown error";
    }
}