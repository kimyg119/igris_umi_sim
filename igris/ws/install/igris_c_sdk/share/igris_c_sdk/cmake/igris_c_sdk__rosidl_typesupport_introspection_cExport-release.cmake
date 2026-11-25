#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "igris_c_sdk::igris_c_sdk__rosidl_typesupport_introspection_c" for configuration "Release"
set_property(TARGET igris_c_sdk::igris_c_sdk__rosidl_typesupport_introspection_c APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(igris_c_sdk::igris_c_sdk__rosidl_typesupport_introspection_c PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libigris_c_sdk__rosidl_typesupport_introspection_c.so"
  IMPORTED_SONAME_RELEASE "libigris_c_sdk__rosidl_typesupport_introspection_c.so"
  )

list(APPEND _cmake_import_check_targets igris_c_sdk::igris_c_sdk__rosidl_typesupport_introspection_c )
list(APPEND _cmake_import_check_files_for_igris_c_sdk::igris_c_sdk__rosidl_typesupport_introspection_c "${_IMPORT_PREFIX}/lib/libigris_c_sdk__rosidl_typesupport_introspection_c.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
