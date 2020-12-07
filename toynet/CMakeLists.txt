cmake_minimum_required(VERSION 3.18)
project(toynet)
enable_testing()

find_package(Boost 1.73 REQUIRED program_options serialization unit_test_framework)

set(CMAKE_CXX_FLAGS "-g")

add_library(
    toynet
    loss.cpp
    math.cpp
    w2v.cpp
    ublas/convert.cpp
    ublas/io.cpp
)

add_library(
    toynet_diff
    examples/diff/diff.cpp
)

add_library(
    toynet_diff2
    examples/diff2/diff2.cpp
)

add_executable(
    unit_tests.tsk
    loss.t.cpp
    math.t.cpp
    w2v.t.cpp
    examples/diff/diff.t.cpp
    examples/diff2/diff2.t.cpp
    ublas/convert.t.cpp
    ublas/io.t.cpp
)

add_executable(
    diff.tsk
    examples/diff/diff.m.cpp
)

add_executable(
    diff2.tsk
    examples/diff2/diff2.m.cpp
)

target_include_directories(toynet PRIVATE .. ${Boost_INCLUDE_DIRS})
target_include_directories(toynet_diff PRIVATE .. ${Boost_INCLUDE_DIRS})
target_include_directories(toynet_diff2 PRIVATE .. ${Boost_INCLUDE_DIRS})
target_include_directories(unit_tests.tsk PRIVATE .. ${Boost_INCLUDE_DIRS})
target_include_directories(diff.tsk PRIVATE .. ${Boost_INCLUDE_DIRS})
target_include_directories(diff2.tsk PRIVATE .. ${Boost_INCLUDE_DIRS})

target_link_libraries(toynet PUBLIC)
target_link_libraries(toynet_diff PUBLIC)
target_link_libraries(toynet_diff2 PUBLIC)
target_link_libraries(unit_tests.tsk PRIVATE toynet_diff toynet_diff2 toynet ${Boost_LIBRARIES} rt)
target_link_libraries(diff.tsk PRIVATE toynet_diff toynet ${Boost_LIBRARIES} rt)
target_link_libraries(diff2.tsk PRIVATE toynet_diff2 toynet ${Boost_LIBRARIES} rt)
