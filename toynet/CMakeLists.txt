cmake_minimum_required(VERSION 3.18)
project(toynet)
enable_testing()

find_package(Boost 1.73 REQUIRED program_options serialization unit_test_framework)

set(CMAKE_CXX_FLAGS "-g")

add_library(
    toynet
    math.cpp
    ublas/convert.cpp
    ublas/io.cpp
    w2v.cpp
    loss.cpp
)

add_library(
    toynet_diff
    examples/diff/diff.cpp
)

add_executable(
    unit_tests.tsk
    ublas/convert.t.cpp
    ublas/io.t.cpp
    w2v.t.cpp
    math.t.cpp
    examples/diff/diff.t.cpp
    loss.t.cpp
)

add_executable(
    diff.tsk
    examples/diff/diff.m.cpp
)

target_include_directories(toynet PRIVATE .. ${Boost_INCLUDE_DIRS})
target_include_directories(toynet_diff PRIVATE .. ${Boost_INCLUDE_DIRS})
target_include_directories(unit_tests.tsk PRIVATE .. ${Boost_INCLUDE_DIRS})
target_include_directories(diff.tsk PRIVATE .. ${Boost_INCLUDE_DIRS})

target_link_libraries(toynet PUBLIC)
target_link_libraries(toynet_diff PUBLIC)
target_link_libraries(unit_tests.tsk PRIVATE toynet_diff toynet ${Boost_LIBRARIES} rt)
target_link_libraries(diff.tsk PRIVATE toynet_diff toynet ${Boost_LIBRARIES} rt)
