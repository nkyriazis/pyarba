#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <string>
#include <thread>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>

#include "bundle_adjuster.hpp"
#include "camera_trajectory_smoother.hpp"
#include "pybind_helpers.hpp"

using namespace arba;

namespace py = pybind11;
using py::literals::operator""_a;

PYBIND11_MODULE(pyARBA, m)
{
    namespace logging = boost::log;

    // Clear all previous sinks
    logging::core::get()->remove_all_sinks();

    // Create a sink that writes to std::clog or std::cerr
    logging::add_console_log(std::cout);

    // Add common attributes like Time, Thread ID, etc.
    logging::add_common_attributes();

    // The solver's per-evaluation trace is diagnostic, so it is off unless
    // asked for: set ARBA_LOG_LEVEL to trace, debug, info, warning, error
    // or fatal (default: warning).
    {
        const char* env   = std::getenv("ARBA_LOG_LEVEL");
        const std::string level(env ? env : "warning");
        auto severity = logging::trivial::warning;
        if (level == "trace")
            severity = logging::trivial::trace;
        else if (level == "debug")
            severity = logging::trivial::debug;
        else if (level == "info")
            severity = logging::trivial::info;
        else if (level == "error")
            severity = logging::trivial::error;
        else if (level == "fatal")
            severity = logging::trivial::fatal;
        logging::core::get()->set_filter(logging::trivial::severity >=
                                         severity);
    }

    m.doc() = "pyARBA: triangulation-free bundle adjustment for camera "
              "pose refinement from coarse priors";

#ifdef NDEBUG
    m.attr("build_type") = "Release";
#else
    m.attr("build_type") = "Debug";
#endif

    py::enum_<bundle_adjuster::optimization_inclusion>(
      m, "optimization_inclusion", py::arithmetic())
      .value("none", bundle_adjuster::optimization_inclusion::none)
      .value("focal_length",
             bundle_adjuster::optimization_inclusion::focal_length)
      .value("principal_point",
             bundle_adjuster::optimization_inclusion::principal_point)
      .value("distortion_coefficients",
             bundle_adjuster::optimization_inclusion::distortion_coefficients)
      .value("camera_poses",
             bundle_adjuster::optimization_inclusion::camera_poses)
      .value("camera_positions",
             bundle_adjuster::optimization_inclusion::camera_positions)
      .value("camera_orientations",
             bundle_adjuster::optimization_inclusion::camera_orientations)
      .value("all", bundle_adjuster::optimization_inclusion::all)
      .export_values()
      .def(py::self | py::self)
      .def(py::self & py::self);

    py::class_<bundle_adjuster>(m, "bundle_adjuster")
      .def(py::init(
             [](const nd_vector<double, 2>& focal_length,
                const nd_vector<double, 2>& principal_point,
                const nd_vector<double, 4>& distortion_coefficients,
                const bundle_adjuster::camera_features_map& features,
                const bundle_adjuster::camera_matches_map& matches,
                const bundle_adjuster::camera_poses_map& camera_poses,
                const double arctan_loss              = 1e2,
                const double position_prior_lambda    = 1e-2,
                const double orientation_prior_lambda = 0.0,
                const bundle_adjuster::camera_tracks_map& feature_tracks =
                  {},
                const bundle_adjuster::camera_ray_times_map&
                  initial_ray_times   = {},
                const bool consensus       = false,
                const bool consensus_joint = false,
                const double lifted_weights_sigma = 0.0)
             {
                 py::scoped_ostream_redirect cout(
                   std::cout,                                // std::ostream&
                   py::module_::import("sys").attr("stdout") // Python output
                 );
                 py::scoped_ostream_redirect cerr(
                   std::cerr,                                // std::ostream&
                   py::module_::import("sys").attr("stderr") // Python output
                 );
                 return new bundle_adjuster(focal_length,
                                            principal_point,
                                            distortion_coefficients,
                                            features,
                                            matches,
                                            camera_poses,
                                            arctan_loss,
                                            position_prior_lambda,
                                            orientation_prior_lambda,
                                            feature_tracks,
                                            initial_ray_times,
                                            consensus,
                                            consensus_joint,
                                            lifted_weights_sigma);
             }),
           "focal_length"_a,
           "principal_point"_a,
           "distortion_coefficients"_a,
           "features"_a,
           "matches"_a,
           "camera_poses"_a,
           "arctan_loss"_a              = 1e2,
           "position_prior_lambda"_a    = 1e-2,
           "orientation_prior_lambda"_a = 0.0,
           "feature_tracks"_a =
             bundle_adjuster::camera_tracks_map{},
           "initial_ray_times"_a =
             bundle_adjuster::camera_ray_times_map{},
           "consensus"_a       = false,
           "consensus_joint"_a = false,
           "lifted_weights_sigma"_a = 0.0)
      .def_property_readonly("camera_poses", &bundle_adjuster::get_camera_poses)
      .def_property_readonly("focal_length", &bundle_adjuster::get_focal_length)
      .def_property_readonly("principal_point",
                             &bundle_adjuster::get_principal_point)
      .def_property_readonly("distortion_coefficients",
                             &bundle_adjuster::get_distortion_coefficients)
      .def_property_readonly("point_cloud", &bundle_adjuster::get_point_cloud)
      .def(
        "solve",
        [](bundle_adjuster& self,
           const std::size_t max_iterations,
           const bool verbose,
           const bundle_adjuster::optimization_inclusion inclusion,
           const double function_tolerance,
           const std::string& linear_solver)
        {
            py::scoped_ostream_redirect cout(
              std::cout,                                // std::ostream&
              py::module_::import("sys").attr("stdout") // Python output
            );
            py::scoped_ostream_redirect cerr(
              std::cerr,                                // std::ostream&
              py::module_::import("sys").attr("stderr") // Python output
            );
            return self.solve(max_iterations,
                              verbose,
                              inclusion,
                              function_tolerance,
                              linear_solver);
        },
        "max_iterations"_a     = 100,
        "verbose"_a            = false,
        "inclusion"_a          = bundle_adjuster::optimization_inclusion::all,
        "function_tolerance"_a = 1e-6,
        "linear_solver"_a      = "sparse_schur")
      .def("set_arctan_loss", &bundle_adjuster::set_arctan_loss, "a"_a)
      .def("set_position_prior_lambda",
           &bundle_adjuster::set_position_prior_lambda,
           "lambda"_a)
      .def("set_orientation_prior_lambda",
           &bundle_adjuster::set_orientation_prior_lambda,
           "lambda"_a)
      .def("set_consensus_weight",
           &bundle_adjuster::set_consensus_weight,
           "weight"_a)
      .def("update_consensus_targets",
           &bundle_adjuster::update_consensus_targets)
      .def("set_lifted_sigma", &bundle_adjuster::set_lifted_sigma,
           "sigma"_a);

    m.attr("bundle_adjuster").attr("optimization_inclusion") =
      m.attr("optimization_inclusion");
    m.attr("optimization_inclusion").attr("__module__") =
      "pyARBA.bundle_adjuster";

    m.def(
      "smooth_camera_trajectory",
      [](const nd_array<double, -1, 3>& camera_positions,
         const nd_array<double, -1, 4>& camera_orientations_xyzw,
         const nd_vector<int, -1>& fixed_knots,
         const bool loop = false,
         const std::size_t max_iterations,
         const double displacement_lambda,
         const double derotation_lambda,
         const double edge_lambda)
      {
          py::scoped_ostream_redirect cout(
            std::cout,                                // std::ostream&
            py::module_::import("sys").attr("stdout") // Python output
          );
          py::scoped_ostream_redirect cerr(
            std::cerr,                                // std::ostream&
            py::module_::import("sys").attr("stderr") // Python output
          );
          return smooth_camera_trajectory(camera_positions,
                                          camera_orientations_xyzw,
                                          fixed_knots,
                                          loop,
                                          max_iterations,
                                          displacement_lambda,
                                          derotation_lambda,
                                          edge_lambda);
      },
      "camera_positions"_a,
      "camera_orientations_xyzw"_a,
      "fixed_knots"_a,
      "loop"_a                = false,
      "max_iterations"_a      = 100,
      "displacement_lambda"_a = 1.0,
      "derotation_lambda"_a   = 1.0,
      "edge_lambda"_a         = 10.0);
}