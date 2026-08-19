cmake_minimum_required(VERSION 3.19)

project(deepsafe_fusion VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GLIB REQUIRED IMPORTED_TARGET glib-2.0)

set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream" CACHE PATH
    "DeepStream 9 SDK root")
set(DEEPSAFE_POSE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../models/pose/postprocess"
    CACHE PATH "Existing DeepSafe pose postprocess root")
set(DEEPSAFE_PPE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../models/ppe/postprocess"
    CACHE PATH "Existing DeepSafe PPE postprocess root")

if(NOT EXISTS "${DEEPSTREAM_ROOT}/sources/includes/gstnvdsinfer.h")
  message(FATAL_ERROR "DeepStream headers are missing below ${DEEPSTREAM_ROOT}")
endif()
if(NOT EXISTS "${DEEPSAFE_POSE_ROOT}/CMakeLists.txt" OR
   NOT EXISTS "${DEEPSAFE_PPE_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR "Existing pose/PPE postprocess cores are required")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(DEEPSAFE_POSE_WITH_DEEPSTREAM ON CACHE BOOL "" FORCE)
set(DEEPSTREAM_ROOT "${DEEPSTREAM_ROOT}" CACHE PATH "" FORCE)
set(DEEPSAFE_PPE_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

add_subdirectory("${DEEPSAFE_POSE_ROOT}" pose-core)
add_subdirectory("${DEEPSAFE_PPE_ROOT}" ppe-core)

find_library(NVDS_META_LIBRARY nvds_meta
  PATHS "${DEEPSTREAM_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
find_library(NVDS_GST_META_LIBRARY nvdsgst_meta
  PATHS "${DEEPSTREAM_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)

add_library(deepsafe_fusion_core STATIC
  src/core.cpp
)
target_include_directories(deepsafe_fusion_core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(deepsafe_fusion_core
  PUBLIC deepsafe_pose_postprocess
  PUBLIC deepsafe_ppe_postprocess)
target_compile_features(deepsafe_fusion_core PUBLIC cxx_std_17)
target_compile_options(deepsafe_fusion_core PRIVATE
  -Wall -Wextra -Wpedantic -Werror)

add_library(deepsafe_fusion SHARED
  src/config.cpp
  src/deepstream_adapter.cpp
)
target_include_directories(deepsafe_fusion
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  SYSTEM PRIVATE "${DEEPSTREAM_ROOT}/sources/includes")
target_link_libraries(deepsafe_fusion
  PRIVATE deepsafe_fusion_core
  PRIVATE deepsafe_pose_deepstream_adapter
  PRIVATE PkgConfig::GSTREAMER
  PRIVATE PkgConfig::GLIB
  PRIVATE "${NVDS_META_LIBRARY}"
  PRIVATE "${NVDS_GST_META_LIBRARY}")
target_compile_features(deepsafe_fusion PUBLIC cxx_std_17)
target_compile_options(deepsafe_fusion PRIVATE
  -Wall -Wextra -Wpedantic -Werror
  -fvisibility=hidden)
target_link_options(deepsafe_fusion PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_target_properties(deepsafe_fusion PROPERTIES
  OUTPUT_NAME deepsafe_fusion
  VERSION 1.0.0
  SOVERSION 1)

if(BUILD_TESTING)
  add_executable(fusion_core_tests tests/test_fusion_core.cpp)
  target_link_libraries(fusion_core_tests PRIVATE deepsafe_fusion_core)
  target_compile_features(fusion_core_tests PRIVATE cxx_std_17)
  target_compile_options(fusion_core_tests PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME fusion_core_tests COMMAND fusion_core_tests)

  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/default-runtime.conf"
       DEEPSAFE_FUSION_TEST_CONFIG_SHA256)
  add_executable(fusion_ds9_metadata_tests tests/test_ds9_metadata.cpp)
  target_include_directories(fusion_ds9_metadata_tests SYSTEM PRIVATE
    "${DEEPSTREAM_ROOT}/sources/includes")
  target_link_libraries(fusion_ds9_metadata_tests
    PRIVATE deepsafe_fusion
    PRIVATE deepsafe_pose_postprocess
    PRIVATE PkgConfig::GSTREAMER
    PRIVATE "${NVDS_META_LIBRARY}"
    PRIVATE "${NVDS_GST_META_LIBRARY}")
  target_compile_features(fusion_ds9_metadata_tests PRIVATE cxx_std_17)
  target_compile_options(fusion_ds9_metadata_tests PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME fusion_ds9_metadata_tests
    COMMAND fusion_ds9_metadata_tests
            "${CMAKE_CURRENT_SOURCE_DIR}/default-runtime.conf"
            "${DEEPSAFE_FUSION_TEST_CONFIG_SHA256}")
endif()

install(TARGETS deepsafe_fusion
  LIBRARY DESTINATION lib)
install(FILES default-runtime.conf DESTINATION share/deepsafe-fusion)
install(DIRECTORY include/ DESTINATION include)
