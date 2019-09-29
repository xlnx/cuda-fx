add_library(cudafx INTERFACE)

target_include_directories(cudafx INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
vm_target_dependency(cudafx VMUtils INTERFACE)

