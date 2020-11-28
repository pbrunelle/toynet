cmake_minimum_required(VERSION 3.18)
project(toynet)
enable_testing()

find_package(Boost 1.73 REQUIRED serialization unit_test_framework)

set(CMAKE_CXX_FLAGS "-g")

add_library(
    toynet
    w2v.cpp
    diff.cpp
)

add_executable(
    unit_tests.tsk
    w2v.t.cpp
    diff.t.cpp
)

target_include_directories(toynet PRIVATE . ${Boost_INCLUDE_DIRS})
target_include_directories(unit_tests.tsk PRIVATE . ${Boost_INCLUDE_DIRS})

target_link_libraries(toynet PUBLIC)
target_link_libraries(unit_tests.tsk PRIVATE toynet ${Boost_LIBRARIES} rt)
