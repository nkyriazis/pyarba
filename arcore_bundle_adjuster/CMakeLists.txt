cmake_minimum_required(VERSION 3.22.1)

project(arcore_bundle_adjuster)

set(CMAKE_CXX_STANDARD 20)

# None of the dependencies below can come from pip. When one is missing the
# default CMake failure is a wall of module-search output that buries the one
# thing the reader needs, which is the apt line. Each find below is quiet,
# and arba_missing() turns a failure into the list of things to install.
set(ARBA_APT_LINE
    "  sudo apt-get install -y build-essential cmake \\\n    libeigen3-dev libceres-dev libsuitesparse-dev libtbb-dev \\\n    libboost-log-dev libboost-system-dev libboost-filesystem-dev \\\n    libboost-test-dev python3-dev python3-pip\n  pip install 'pybind11>=2.12'"
)

function(arba_missing NAME WHAT)
  message(
    FATAL_ERROR
      "${NAME} was not found, and it cannot be installed from pip.\n"
      "${WHAT}\n"
      "On Ubuntu 24.04 the whole set comes from apt (24.04 is the oldest "
      "Ubuntu whose packaged Ceres has the ceres::Manifold API this code "
      "uses, so nothing has to be built from source):\n"
      "${ARBA_APT_LINE}\n"
      "See the Prerequisites section of the README for other platforms.")
endfunction()

find_package(Python3 COMPONENTS Interpreter Development)

# Prefer a pip-installed pybind11 over a distribution one: pybind11 < 2.12
# segfaults when it converts a NumPy >= 2.0 array, which is exactly what
# every entry point here does on its first argument. Ubuntu 24.04 ships
# pybind11 2.11.1 next to a NumPy 2 that pip will happily install, so
# without this the module builds and then crashes on first call.
if(NOT pybind11_DIR)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_PIP_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_PIP_RESULT
    ERROR_QUIET)
  if(PYBIND11_PIP_RESULT EQUAL 0 AND IS_DIRECTORY "${PYBIND11_PIP_CMAKEDIR}")
    set(pybind11_DIR "${PYBIND11_PIP_CMAKEDIR}")
  endif()
endif()

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  arba_missing("pybind11" "It provides the Python bindings for the solver.")
endif()

if(pybind11_VERSION VERSION_LESS 2.12)
  message(
    WARNING
      "pybind11 ${pybind11_VERSION} found: versions before 2.12 crash on "
      "NumPy >= 2 arrays. Either pip install 'pybind11>=2.12' (it will be "
      "picked up automatically) or pin numpy<2 in the runtime environment.")
endif()
find_package(Ceres 2.2 QUIET)
if(NOT Ceres_FOUND)
  arba_missing(
    "Ceres >= 2.2"
    "It is the solver, and 2.2 is the floor because this code uses the ceres::Manifold API that 2.1 does not have. Ceres in turn needs Eigen and SuiteSparse (for SPARSE_SCHUR)."
  )
endif()

find_package(Boost QUIET COMPONENTS log system filesystem unit_test_framework)
if(NOT Boost_FOUND)
  arba_missing(
    "Boost (log, system, filesystem, unit_test_framework)"
    "log/system/filesystem carry the solver's tracing; unit_test_framework builds the test binary."
  )
endif()

find_package(CUDAToolkit)

find_package(TBB QUIET)
if(NOT TBB_FOUND)
  arba_missing("TBB" "The residual evaluation is parallelised over it.")
endif()

# Eigen is header-only. /usr/include/eigen3 is where Debian and Ubuntu put
# it, which covers the documented build; anywhere else, ask CMake.
if(EXISTS "/usr/include/eigen3/signature_of_eigen3_matrix_library")
  set(EIGEN_INCLUDE_DIR /usr/include/eigen3)
else()
  find_package(Eigen3 3.4 QUIET NO_MODULE)
  if(NOT Eigen3_FOUND)
    arba_missing("Eigen 3.4" "Every matrix in the solver is an Eigen type.")
  endif()
  get_target_property(EIGEN_INCLUDE_DIR Eigen3::Eigen
                      INTERFACE_INCLUDE_DIRECTORIES)
endif()

include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

# The main lib
add_library(
  bundle_adjuster
  eigen.hpp
  bundle_adjuster_cross_projection_residuals.cpp
  bundle_adjuster.cpp
  bundle_adjuster.hpp
  camera_trajectory_smoother.hpp
  camera_trajectory_smoother.cpp
  cost_helpers.hpp
  util.hpp
  util.cpp
  math.hpp
  progress_iterator.hpp
  pybind_helpers.hpp)
target_link_libraries(bundle_adjuster PRIVATE Ceres::ceres TBB::tbb ${Boost_LIBRARIES})
set_property(TARGET bundle_adjuster PROPERTY POSITION_INDEPENDENT_CODE ON)

# The python bindings
pybind11_add_module(pyARBA pybind.cpp)
target_link_libraries(pyARBA PRIVATE bundle_adjuster ${Boost_LIBRARIES})

# unit tests in tests.cpp (link with boost unit test framework)
add_executable(tests tests.cpp)
target_link_libraries(tests PRIVATE bundle_adjuster Ceres::ceres
                                    ${Boost_LIBRARIES})

# to install we need to figure out the python site packages directory
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(
  STATUS
    "Python site packages: ${PYTHON_SITE_PACKAGES} (python=${Python3_EXECUTABLE})"
)

install(TARGETS pyARBA DESTINATION ${PYTHON_SITE_PACKAGES})
