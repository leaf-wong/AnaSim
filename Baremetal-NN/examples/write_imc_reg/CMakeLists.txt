cmake_minimum_required(VERSION 3.10)

project(write_imc_reg LANGUAGES C)

add_executable(write_imc_reg crt0.S main.c syscalls.c)

target_include_directories(write_imc_reg PUBLIC ./)


target_compile_features(write_imc_reg INTERFACE c_std_11)

if (X86)
    message("write_imc_reg: building for x86")
    target_link_libraries(write_imc_reg PUBLIC target-x86)
elseif (RISCV)
    message("write_imc_reg: building for RISC-V")
    target_link_libraries(write_imc_reg PUBLIC target-riscv)
    set_target_properties(write_imc_reg PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(write_imc_reg PRIVATE -O3 -Wall -Wextra)

target_link_libraries(write_imc_reg PUBLIC nn target-riscv)
target_link_libraries(write_imc_reg PUBLIC m)
