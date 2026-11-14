cmake_minimum_required(VERSION 3.10)

project(resnet9-residual LANGUAGES C)

add_executable(resnet9-residual main.c crt0.S syscalls.c)

target_include_directories(resnet9-residual PUBLIC ./)


target_compile_features(resnet9-residual INTERFACE c_std_11)

if (X86)
    message("resnet9-residual: building for x86")
    target_link_libraries(resnet9-residual PUBLIC target-x86)
elseif (RISCV)
    message("resnet9-residual: building for RISC-V")
    target_link_libraries(resnet9-residual PUBLIC target-riscv)
    set_target_properties(resnet9-residual PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(resnet9-residual PRIVATE -O3 -Wall -Wextra)

target_link_libraries(resnet9-residual PUBLIC nn target-riscv)
target_link_libraries(resnet9-residual PUBLIC m)
