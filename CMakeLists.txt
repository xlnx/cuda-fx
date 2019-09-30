cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
project(vmutils)

option(CUDAFX_BUILD_TESTS "Set to ON to build tests" OFF)

# install build scripts
find_package(Git)
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)
# install external modules
include(scripts/external.cmake)
vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/VMUtils
  GIT_TAG        master
)
# export targets
include(export.cmake)

# build tests
if (CUDAFX_BUILD_TESTS)
  find_package(GTest REQUIRED)
  file(GLOB_RECURSE CUDAFX_TEST_SOURCES
    "tests/*.cc"
    "tests/*.cpp"
    "tests/*.cu"
  )
  find_package(CUDA REQUIRED)
  set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_NVCC_FLAGS ${CMAKE_NVCC_FLAGS}
    -std=c++14
    #  -v  
    --expt-extended-lambda
    --use_fast_math
    --keep-device-functions
    -keep
    -maxrregcount
    32
    # -w
    -lineinfo
    -Xcudafe --display_error_number
  )
  add_executable(cudafx_test_all ${CUDAFX_TEST_SOURCES})
  vm_target_dependency(cudafx_test_all cudafx PRIVATE)
  if (CMAKE_CXX_COMPILER MATCHES MSVC)
  else()
    target_link_libraries(cudafx_test_all pthread)
  endif()
  target_link_libraries(cudafx_test_all gtest gtest_main)
  # include(GoogleTest)
  # gtest_add_tests(cudafx_test_all "" AUTO)
endif()

