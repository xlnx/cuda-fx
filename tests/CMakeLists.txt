find_package(GTest REQUIRED)
file(GLOB_RECURSE CUDAFX_TEST_SOURCES
  *.cc
  *.cpp
  *.cu
)
#  find_package(CUDA REQUIRED)
#  set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
#  set(CUDA_SEPARABLE_COMPILATION ON)
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
target_link_libraries(cudafx_test_all gtest_main gtest)
include(GoogleTest)
gtest_add_tests(cudafx_test_all "" AUTO)
