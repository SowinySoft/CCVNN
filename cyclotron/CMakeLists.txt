cmake_minimum_required(VERSION 3.18)
project(ccvnn_integration_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Dependencies
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBMODBUS REQUIRED libmodbus)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Target Executable
add_executable(test_pipeline
    test_pipeline.cpp
    v17_feature_extractor.cpp
    watcher_edge_daemon.cpp
)

target_include_directories(test_pipeline PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${LIBMODBUS_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(test_pipeline PRIVATE 
    ${OpenCV_LIBS}
    ${LIBMODBUS_LIBRARIES}
    OpenMP::OpenMP_CXX
    gtest_main
)

enable_testing()
add_test(NAME PipelineIntegrationTest COMMAND test_pipeline)

//
cmake_minimum_required(VERSION 3.20)
project(WatcherEdgeDaemon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build GoogleTest suite" OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
pkg_check_modules(MODBUS REQUIRED libmodbus)

add_executable(watcher_edge_daemon
    src/main.cpp
    src/watcher_edge_daemon.cpp
    src/mqtt_telemetry_exporter.cpp
)

target_include_directories(watcher_edge_daemon PRIVATE ${OPENCV_INCLUDE_DIRS} ${MODBUS_INCLUDE_DIRS} include)
target_link_libraries(watcher_edge_daemon PRIVATE ${OPENCV_LIBRARIES} ${MODBUS_LIBRARIES} OpenMP::OpenMP_CXX mosquitto)

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    
    add_executable(test_circuit_breaker tests/test_circuit_breaker.cpp)
    target_include_directories(test_circuit_breaker PRIVATE include ${MODBUS_INCLUDE_DIRS})
    target_link_libraries(test_circuit_breaker PRIVATE GTest::gtest GTest::gtest_main ${MODBUS_LIBRARIES})
    
    include(GoogleTest)
    gtest_discover_tests(test_circuit_breaker)
endif()