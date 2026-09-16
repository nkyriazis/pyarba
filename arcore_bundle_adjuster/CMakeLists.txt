cmake_minimum_required(VERSION 3.22.1)

project(arcore_bundle_adjuster)

set(CMAKE_CXX_STANDARD 20)

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

find_package(pybind11 REQUIRED)

if(pybind11_VERSION VERSION_LESS 2.12)
  message(
    WARNING
      "pybind11 ${pybind11_VERSION} found: versions before 2.12 crash on "
      "NumPy >= 2 arrays. Either pip install 'pybind11>=2.12' (it will be "
      "picked up automatically) or pin numpy<2 in the runtime environment.")
endif()
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED log system filesystem unit_test_framework)
find_package(CUDAToolkit)
find_package(TBB REQUIRED)


set(EIGEN_INCLUDE_DIR /usr/include/eigen3)
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
