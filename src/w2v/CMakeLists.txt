add_library(
    w2vlib
    w2v.cpp
)

add_executable(
    w2v_unit_tests.tsk
    w2v.t.cpp
)

target_include_directories(w2vlib PRIVATE .)
target_include_directories(w2v_unit_tests.tsk PRIVATE .)

target_link_libraries(w2vlib PUBLIC)
target_link_libraries(w2v_unit_tests.tsk PRIVATE w2vlib)
