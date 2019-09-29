cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 14)
project(vmutils)

option(CUDAFX_BUILD_TESTS "Set to ON to build tests" OFF)

install(DIRECTORY "include" DESTINATION "include/cudafx")

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
  cuda_add_executable(test_all ${CUDAFX_TEST_SOURCES})
  if (CMAKE_CXX_COMPILER MATCHES MSVC)
  else()
    target_link_libraries(test_all pthread)
  endif()
  target_link_libraries(test_all gtest_main gtest)
  target_include_directories(test_all PRIVATE "include/cudafx")
  include(GoogleTest)
  gtest_add_tests(test_all "" AUTO)
endif()
