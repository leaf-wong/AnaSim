cmake_minimum_required(VERSION 3.10)

project(conv-simple LANGUAGES C)

add_executable(conv-simple main.c crt0.S syscalls.c)

target_include_directories(conv-simple PUBLIC ./)


target_compile_features(conv-simple INTERFACE c_std_11)

if (X86)
    message("conv-simple: building for x86")
    target_link_libraries(conv-simple PUBLIC target-x86)
elseif (RISCV)
    message("conv-simple: building for RISC-V")
    target_link_libraries(conv-simple PUBLIC target-riscv)
    set_target_properties(conv-simple PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(conv-simple PRIVATE -O3 -Wall -Wextra)

target_link_libraries(conv-simple PUBLIC nn target-riscv)
target_link_libraries(conv-simple PUBLIC m)
