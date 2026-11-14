ifndef RISCV
$(error RISCV is unset)
else
$(info Running with RISCV=$(RISCV))
endif

PREFIX ?= $RISCV/
SRC_DIR := src
SRCS=$(wildcard $(SRC_DIR)/*.cc)

default: libspikedevices.so

libspikedevices.so: $(SRCS)
	g++ -L $(RISCV)/lib -Wl,-rpath,$(RISCV)/lib -shared -o $@ -std=c++17 -I $(RISCV)/include -fPIC $^

.PHONY: install
install: libspikedevices.so
	sudo cp libspikedevices.so $(RISCV)/lib

clean:
	rm -rf *.o *.so

compile_dtb:
	dtc -I dts -O dtb imc.dts > imc.dtb

set_env:
	export PATH=$PATH:/opt/riscv/bin
	export RISCV=/opt/riscv

run_64:
	/opt/riscv_64_tick/bin/spike --extlib=$(RISCV)/lib/libspikedevices.so --device=imc_device --dtb=imc.dtb ../../riscv-tests/benchmarks/gemm_imc.riscv
	#spike --extlib=/opt/riscv_64/lib/libspikedevices.so --device=sifive_uart --dtb=test.dtb /opt/riscv_64/riscv64-unknown-linux-gnu/bin/pk ../test/hello_count/hello
