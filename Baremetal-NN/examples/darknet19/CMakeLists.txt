cmake_minimum_required(VERSION 3.10)

project(darknet19 LANGUAGES C)

add_executable(darknet19 main.c crt0.S syscalls.c)

target_include_directories(darknet19 PUBLIC ./)


target_compile_features(darknet19 INTERFACE c_std_11)

if (X86)
    message("darknet19: building for x86")
    target_link_libraries(darknet19 PUBLIC target-x86)
elseif (RISCV)
    message("darknet19: building for RISC-V")
    target_link_libraries(darknet19 PUBLIC target-riscv)
    set_target_properties(darknet19 PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(darknet19 PRIVATE -O3 -Wall -Wextra)

target_link_libraries(darknet19 PUBLIC nn target-riscv)
target_link_libraries(darknet19 PUBLIC m)
