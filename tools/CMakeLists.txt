add_executable(device_inspect 
  device_inspect.cc
)
vm_target_dependency(device_inspect cudafx PRIVATE)
