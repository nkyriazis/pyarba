#pragma once

#include <functional>
#include <map>
#include <memory>
#include <tuple>

#include "eigen.hpp"

// forward declare ceres::CostFunction
namespace ceres
{
class Problem;
class LossFunction;
class EvaluationCallback;
}; // namespace ceres

namespace arba
{

class bundle_adjuster
{
   public:
    enum class optimization_inclusion : int
    {
        none                    = 0,
        focal_length            = 1 << 0,
        principal_point         = 1 << 1,
        distortion_coefficients = 1 << 2,
        camera_poses            = 1 << 3,
        // finer-grained pose staging (camera_poses == both)
        camera_positions    = 1 << 4,
        camera_orientations = 1 << 5,
        all = focal_length | principal_point | distortion_coefficients |
              camera_poses
    };

    using camera_features_map = std::map<int, const nd_array<double, -1, 2>>;
    using camera_matches_map =
      std::map<std::pair<int, int>, const nd_array<int, -1, 2>>;
    using camera_poses_map =
      std::map<int, std::tuple<nd_vector<double, 3>, nd_vector<double, 4>>>;
    // per camera: track id per feature (-1 = untracked). When provided,
    // observations of the same track share one ray-time parameter anchored
    // on each observation's own ray.
    using camera_tracks_map = std::map<int, nd_array<int, -1, 1>>;
    // per camera: initial ray time per feature (<= 0 = use the default 1.0),
    // e.g. metric depth sampled from an AR framework's depth map
    using camera_ray_times_map = std::map<int, nd_array<double, -1, 1>>;

    bundle_adjuster(const nd_vector<double, 2>& focal_length,
                    const nd_vector<double, 2>& principal_point,
                    const nd_vector<double, 4>& distortion_coefficients,
                    const camera_features_map& features,
                    const camera_matches_map& matches,
                    const camera_poses_map& camera_poses,
                    const double arctan_loss              = 1e2,
                    const double position_prior_lambda    = 1e-2,
                    const double orientation_prior_lambda = 0.0,
                    const camera_tracks_map& feature_tracks = {},
                    const camera_ray_times_map& initial_ray_times = {},
                    const bool consensus                          = false,
                    const bool consensus_joint                    = false,
                    const double lifted_weights_sigma             = 0.0);

    // linear_solver: "sparse_schur" (default), "iterative_schur",
    // or "dense_schur_cuda" (CUDA dense algebra for the reduced system)
    std::string solve(
      const std::size_t max_iterations       = 100,
      const bool verbose                     = false,
      const optimization_inclusion inclusion = optimization_inclusion::all,
      const double function_tolerance        = 1e-6,
      const std::string& linear_solver       = "sparse_schur");

    // Mutable objective scales, for graduated non-convexity / prior
    // homotopy: adjust between successive solve() calls on the same
    // problem (parameters stay warm-started). No-ops when the respective
    // loss was not created at construction.
    void set_arctan_loss(const double a);
    void set_position_prior_lambda(const double lambda);
    void set_orientation_prior_lambda(const double lambda);

    // Lifted track consensus, alternated form: one 3D consensus point per
    // track, held fixed during a solve and recomputed closed-form (mean of
    // the track's current ray points) by update_consensus_targets().
    // Residuals sqrt(weight) * (ray_point - consensus) pull each member
    // observation toward it; weight 0 disables. Requires consensus=true and
    // feature_tracks at construction (per-observation cross-projection
    // residuals are kept, unlike anchored-track mode).
    void set_consensus_weight(const double weight);
    void update_consensus_targets();

    // Lifted robust weights (Geman-McClure via Black-Rangarajan): each
    // observation owns a confidence w multiplying its source-side
    // residuals, with penalty sigma*(w-1); sigma (pixels) is the scale
    // where down-weighting kicks in, mutable for annealing (GNC-GM).
    void set_lifted_sigma(const double sigma);

    camera_poses_map get_camera_poses() const { return camera_poses; }
    nd_vector<double, 2> get_focal_length() const { return focal_length; }
    nd_vector<double, 2> get_principal_point() const { return principal_point; }
    nd_vector<double, 4> get_distortion_coefficients() const
    {
        return distortion_coefficients;
    }

    nd_array<double, -1, 3> get_point_cloud() const;

   private:
    // problem configuration
    camera_features_map features;
    camera_matches_map matches;

    // problem parameters
    nd_vector<double, 2> focal_length;
    nd_vector<double, 2> principal_point;
    nd_vector<double, 4> distortion_coefficients;
    camera_poses_map camera_poses;
    camera_tracks_map feature_tracks;
    camera_ray_times_map initial_ray_times;
    std::map<int, nd_array<double, -1, 1>> camera_ray_times;
    std::map<int, double> track_times;
    // (camera, feature) -> the ray-time variable used by that observation
    std::map<std::pair<int, int>, double*> ray_time_of;

    // ceres stuff (the pointers need to be unique, but only shared_ptr can hide
    // deletion)
    std::shared_ptr<ceres::Problem> problem;
    std::shared_ptr<void> quaternion_parameterization;
    std::shared_ptr<ceres::LossFunction> arctan_loss, scale_loss,
      prior_scale_loss, orientation_prior_scale_loss, consensus_scale_loss;

    void _initialize_camera_ray_times();
    void _create_ceres_problem();
    void _create_ceres_helpers(const double arctan_loss,
                               const double position_prior_lambda);
    void _declare_ceres_parameters();
    void _initialize_cross_projection_residuals();
    void _initialize_anchored_track_residuals();
    void _initialize_consensus_residuals();

    // lifted-weights mode: per-observation [ray time, confidence] 2-dim
    // eliminated blocks; 0 = off (plain 1-dim ray-time blocks)
    double lifted_weights_sigma = 0.0;
    std::map<int, std::vector<Eigen::Vector2d>> lifted_obs_params;

    bool consensus          = false;
    // joint mode: consensus points are 3-dim parameter blocks optimized
    // with everything else (the targets vector doubles as their storage);
    // alternated mode holds them fixed per solve instead
    bool consensus_joint    = false;
    double consensus_weight = 0.0;
    // group g = observations (camera, feature) of one track; targets and
    // per-member weights are stable storage referenced by the consensus
    // cost functors. Member weights implement robust trimming: ray points
    // can sit near-infinity on outlier rays (the arctan loss saturates
    // them in pixel space), and a quadratic 3D pull toward a mean poisoned
    // by such a point diverges the whole solve.
    std::vector<std::vector<std::pair<int, int>>> consensus_groups;
    std::vector<Eigen::Vector3d> consensus_targets;
    std::vector<double> consensus_member_weights;
    void _initialize_position_priors();
    void _initialize_orientation_priors();

    std::shared_ptr<void> cached_points;
    std::shared_ptr<ceres::EvaluationCallback> evalCallback;
    void _update_cache();
};

inline bool operator&(const bundle_adjuster::optimization_inclusion lhs,
                      const bundle_adjuster::optimization_inclusion rhs)
{
    return static_cast<int>(lhs) & static_cast<int>(rhs);
}

inline bundle_adjuster::optimization_inclusion operator|(
  const bundle_adjuster::optimization_inclusion lhs,
  const bundle_adjuster::optimization_inclusion rhs)
{
    return static_cast<bundle_adjuster::optimization_inclusion>(
      static_cast<int>(lhs) | static_cast<int>(rhs));
}

} // namespace arba