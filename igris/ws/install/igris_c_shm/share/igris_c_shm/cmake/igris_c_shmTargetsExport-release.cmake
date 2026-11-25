#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "igris_c_shm::igris_c_shm_logger" for configuration "Release"
set_property(TARGET igris_c_shm::igris_c_shm_logger APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(igris_c_shm::igris_c_shm_logger PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libigris_c_shm_logger.a"
  )

list(APPEND _cmake_import_check_targets igris_c_shm::igris_c_shm_logger )
list(APPEND _cmake_import_check_files_for_igris_c_shm::igris_c_shm_logger "${_IMPORT_PREFIX}/lib/libigris_c_shm_logger.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
