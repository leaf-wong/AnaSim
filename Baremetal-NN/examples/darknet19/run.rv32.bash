
riscv32-unknown-elf-objcopy -O verilog --only-section=.init --only-section=.text  ../../build/examples/resnet9-residual/resnet9-residual  vhx/resnet9.vhx8
#riscv32-unknown-elf-objcopy -O verilog --only-section=.rodata --only-section=.data --only-section=.bss --only-section=.stack  tensorflow/lite/micro/tools/make/gen/mcu_riscv_riscv32_mcu_release/bin/micro_speech_test  vhx/ResNet8_data.vhx8
riscv32-unknown-elf-objcopy -O verilog --only-section=.data_comb --only-section=.bss --only-section=.stack --only-section=.rodata  ../../build/examples/resnet9-residual/resnet9-residual  vhx/resnet9_data.vhx8
#riscv32-unknown-elf-objcopy -O verilog --only-section=.rodata --only-section=.lalign --only-section=.dalign --only-section=.data --only-section=.bss --only-section=.stack  tensorflow/lite/micro/tools/make/gen/mcu_riscv_riscv32_mcu_release/bin/micro_speech_test  vhx/ResNet8_data.vhx8

python3 ./hex8tohex32.py vhx/resnet9.vhx8 > vhx/resnet9.vhx32
python3 ./hex8tohex32.py vhx/resnet9_data.vhx8 > vhx/resnet9_data.vhx32

grep -v '^@' vhx/resnet9.vhx32 >  vhx/resnet9.vhx
grep -v '^@' vhx/resnet9_data.vhx32 >  vhx/resnet9_data.vhx


riscv32-unknown-elf-objcopy -O verilog --only-section=.rodata    ../../build/examples/resnet9-residual/resnet9-residual  vhx/_rodata.vhx8
riscv32-unknown-elf-objcopy -O verilog --only-section=.data    ../../build/examples/resnet9-residual/resnet9-residual  vhx/_data.vhx8
riscv32-unknown-elf-objcopy -O verilog --only-section=.bss    ../../build/examples/resnet9-residual/resnet9-residual  vhx/_bss.vhx8
riscv32-unknown-elf-objcopy -O verilog --only-section=.stack   ../../build/examples/resnet9-residual/resnet9-residual  vhx/_stack.vhx8

python3 ./hex8tohex32.py vhx/_rodata.vhx8 > vhx/_rodata.vhx32
python3 ./hex8tohex32.py vhx/_data.vhx8 > vhx/_data.vhx32
python3 ./hex8tohex32.py vhx/_bss.vhx8 > vhx/_bss.vhx32
python3 ./hex8tohex32.py vhx/_stack.vhx8 > vhx/_stack.vhx32

grep -v '^@' vhx/_rodata.vhx32 >  vhx/_rodata.vhx
grep -v '^@' vhx/_data.vhx32 >  vhx/_data.vhx
grep -v '^@' vhx/_bss.vhx32 >  vhx/_bss.vhx
grep -v '^@' vhx/_stack.vhx32 >  vhx/_stack.vhx
