#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "igris_c_bridge::igris_c_bridge_lib" for configuration "Release"
set_property(TARGET igris_c_bridge::igris_c_bridge_lib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(igris_c_bridge::igris_c_bridge_lib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libigris_c_bridge_lib.a"
  )

list(APPEND _cmake_import_check_targets igris_c_bridge::igris_c_bridge_lib )
list(APPEND _cmake_import_check_files_for_igris_c_bridge::igris_c_bridge_lib "${_IMPORT_PREFIX}/lib/libigris_c_bridge_lib.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
