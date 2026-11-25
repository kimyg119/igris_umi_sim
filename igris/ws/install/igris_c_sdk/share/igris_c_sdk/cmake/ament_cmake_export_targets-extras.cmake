# generated from ament_cmake_export_targets/cmake/ament_cmake_export_targets-extras.cmake.in

set(_exported_targets "export_igris_c_sdk__rosidl_generator_c;export_igris_c_sdk__rosidl_typesupport_fastrtps_c;igris_c_sdk__rosidl_typesupport_introspection_c;igris_c_sdk__rosidl_typesupport_c;export_igris_c_sdk__rosidl_generator_cpp;export_igris_c_sdk__rosidl_typesupport_fastrtps_cpp;igris_c_sdk__rosidl_typesupport_introspection_cpp;igris_c_sdk__rosidl_typesupport_cpp;export_igris_c_sdk__rosidl_generator_py")

# include all exported targets
if(NOT _exported_targets STREQUAL "")
  foreach(_target ${_exported_targets})
    set(_export_file "${igris_c_sdk_DIR}/${_target}Export.cmake")
    include("${_export_file}")

    # extract the target names associated with the export
    set(_regex "foreach\\((_cmake)?_expected_?[Tt]arget (IN ITEMS )?(.+)\\)")
    file(
      STRINGS "${_export_file}" _foreach_targets
      REGEX "${_regex}")
    list(LENGTH _foreach_targets _matches)
    if(NOT _matches EQUAL 1)
      message(FATAL_ERROR
        "Failed to find exported target names in '${_export_file}'")
    endif()
    string(REGEX REPLACE "${_regex}" "\\3" _targets "${_foreach_targets}")
    string(REPLACE " " ";" _targets "${_targets}")
    list(LENGTH _targets _length)

    list(APPEND igris_c_sdk_TARGETS ${_targets})
  endforeach()
endif()
