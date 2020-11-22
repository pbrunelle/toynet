find_package(Boost 1.73 REQUIRED serialization unit_test_framework)

add_library(
    w2vlib
    w2v.cpp
)

add_executable(
    w2v_unit_tests.tsk
    w2v.t.cpp
)

target_include_directories(w2vlib PRIVATE . ${Boost_INCLUDE_DIRS})
target_include_directories(w2v_unit_tests.tsk PRIVATE . ${Boost_INCLUDE_DIRS})

target_link_libraries(w2vlib PUBLIC)
target_link_libraries(w2v_unit_tests.tsk PRIVATE w2vlib ${Boost_LIBRARIES} rt)
