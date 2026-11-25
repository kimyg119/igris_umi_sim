# generated from rosidl_cmake/cmake/rosidl_cmake_aggregate_target-extras.cmake.in

# Create a convenience aggregate target igris_c_sdk::igris_c_sdk
# that links all generated interface targets, so downstream packages can use
# a single modern CMake target name instead of ${igris_c_sdk_TARGETS}.
if(igris_c_sdk_TARGETS AND NOT TARGET igris_c_sdk::igris_c_sdk)
  add_library(igris_c_sdk::igris_c_sdk INTERFACE IMPORTED)
  set_target_properties(igris_c_sdk::igris_c_sdk PROPERTIES
    INTERFACE_LINK_LIBRARIES "${igris_c_sdk_TARGETS}")
endif()
