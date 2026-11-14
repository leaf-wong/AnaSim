ifndef RISCV
$(error RISCV is unset)
else
$(info Running with RISCV=$(RISCV))
endif

PREFIX ?= $RISCV/
SRC_DIR := src

clean:
	rm -rf ./build/

compile:
	cmake -S ./ -B ./build/ -D CMAKE_BUILD_TYPE=Debug -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake
	cmake --build ./build/ --target darknet19 resnet9-residual
	#cmake --build ./build/ --target conv-simple resnet9-residual write_imc_reg
	# After create a new C program folder like examples/xxx/, update the CMakeLists.txt in examples/ and examples/xxx/

run:
	/opt/riscv_64_newlib/bin/spike build/examples/mlp-simple/mlp-simple

run_imc:
	#/opt/riscv/bin/spike --extlib=/opt/riscv/lib/libspikedevices.so --device=imc_device --device=sifive_uart --dtb=../riscv-isa-sim/spike-devices/imc_32.dtb build/examples/darknet19/darknet19
	/opt/riscv/bin/spike --extlib=/opt/riscv/lib/libspikedevices.so --device=imc_device --device=sifive_uart --dtb=../riscv-isa-sim/spike-devices/imc_32.dtb build/examples/resnet9-residual/resnet9-residual
	#/opt/riscv_64_newlib/bin/spike --extlib=/opt/riscv_64_newlib/lib/libspikedevices.so --device=imc_device --device=sifive_uart --dtb=../riscv-isa-sim/spike-devices/imc.dtb build/examples/conv-simple/conv-simple

riscvdumps:
	riscv32-unknown-elf-nm build/examples/conv-simple/conv-simple | grep model_weight_data
	riscv32-unknown-elf-objdump -d build/examples/conv-simple/conv-simple --start-address=0x80000000 --stop-address=0x800FFFFF > diasm.txt
	# run spike with log and # of instructions
	/opt/riscv/bin/spike --isa=rv32imac_zicsr --instructions=1386600 -l build/examples/llama2/llama2