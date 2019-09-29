add_library(cudafx src/dummy.cc)

target_include_directories(cudafx INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}	
)
target_link_libraries(cudafx 
    ${CUDART_LIBRARY}
)
vm_target_dependency(cudafx VMUtils INTERFACE)

