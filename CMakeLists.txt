cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)

include(cmake/Headless.cmake)
include(CTest)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 3.0 QUIET)
   if(NOT OpenCV_FOUND)
      find_package(OpenCV 2.4.3 QUIET)
      if(NOT OpenCV_FOUND)
         message(FATAL_ERROR "OpenCV >= 2.4.3 not found.")
      endif()
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)
if(ORB_SLAM2_BUILD_VIEWER)
   find_package(Pangolin REQUIRED)
endif()

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
${ZMQ_INCLUDE_DIRS}
)

if(ORB_SLAM2_BUILD_VIEWER)
   include_directories(${Pangolin_INCLUDE_DIRS})
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(ORB_SLAM2_SOURCES
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/semantic/FeatureMaskPolicy.cc
src/semantic/CacheMaskProvider.cc
src/semantic/Telemetry.cc
src/semantic/IpcMaskProvider.cc
src/semantic/OnlineDynamicState.cc
)

if(ORB_SLAM2_BUILD_VIEWER)
   list(APPEND ORB_SLAM2_SOURCES src/MapDrawer.cc src/Viewer.cc)
else()
   list(APPEND ORB_SLAM2_SOURCES src/headless/MapDrawer.cc src/headless/Viewer.cc)
endif()

add_library(${PROJECT_NAME} SHARED ${ORB_SLAM2_SOURCES})

if(ORB_SLAM2_BUILD_VIEWER)
   target_compile_definitions(${PROJECT_NAME} PUBLIC ORB_SLAM2_BUILD_VIEWER=1)
else()
   target_compile_definitions(${PROJECT_NAME} PUBLIC ORB_SLAM2_BUILD_VIEWER=0)
endif()

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
OpenSSL::Crypto
${ZMQ_LIBRARIES}
)

if(ORB_SLAM2_BUILD_VIEWER)
   target_link_libraries(${PROJECT_NAME} ${Pangolin_LIBRARIES})
endif()

if(BUILD_TESTING)
   find_package(GTest REQUIRED)
   add_executable(test_rgbd_api tests/cpp/test_rgbd_api.cc)
   target_link_libraries(test_rgbd_api ${PROJECT_NAME} GTest::gtest_main)
   add_test(NAME rgbd_api COMMAND test_rgbd_api)
   add_executable(test_run_telemetry tests/cpp/test_run_telemetry.cc)
   target_link_libraries(test_run_telemetry GTest::gtest_main)
   add_test(NAME run_telemetry COMMAND test_run_telemetry)
   add_executable(test_g2o_abi tests/cpp/test_g2o_abi.cc)
   target_link_libraries(test_g2o_abi
      ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
      GTest::gtest_main)
   add_test(NAME g2o_abi COMMAND test_g2o_abi)
   add_executable(test_feature_mask_policy tests/cpp/test_feature_mask_policy.cc)
   target_link_libraries(test_feature_mask_policy ${PROJECT_NAME} GTest::gtest_main)
   add_test(NAME feature_mask_policy COMMAND test_feature_mask_policy)
   add_executable(test_frame_mask_invariants tests/cpp/test_frame_mask_invariants.cc)
   target_link_libraries(test_frame_mask_invariants ${PROJECT_NAME} GTest::gtest_main)
   add_test(NAME frame_mask_invariants COMMAND test_frame_mask_invariants)
   add_executable(test_cache_mask_provider tests/cpp/test_cache_mask_provider.cc)
   target_link_libraries(test_cache_mask_provider ${PROJECT_NAME} GTest::gtest_main)
   add_test(NAME cache_mask_provider COMMAND test_cache_mask_provider)
   add_executable(test_ipc_mask_provider tests/cpp/test_ipc_mask_provider.cc)
   target_compile_definitions(test_ipc_mask_provider PRIVATE
      IPC_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures/ipc_packets")
   target_link_libraries(test_ipc_mask_provider ${PROJECT_NAME} GTest::gtest_main)
   add_test(NAME ipc_mask_provider COMMAND test_ipc_mask_provider)
   add_executable(test_online_dynamic_state tests/cpp/test_online_dynamic_state.cc)
   target_compile_definitions(test_online_dynamic_state PRIVATE
      IPC_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures/ipc_packets")
   target_link_libraries(test_online_dynamic_state ${PROJECT_NAME} GTest::gtest_main)
   add_test(NAME online_dynamic_state COMMAND test_online_dynamic_state)
endif()

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

add_executable(rgbd_tum_ov
Examples/RGB-D/rgbd_tum_ov.cc)
target_link_libraries(rgbd_tum_ov ${PROJECT_NAME})

add_executable(rgbd_tum_online
Examples/RGB-D/rgbd_tum_online.cc)
target_link_libraries(rgbd_tum_online ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})
