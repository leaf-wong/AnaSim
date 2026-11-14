# AnaSim: Analog-Datapath Neural Network Processor Simulation Framework

AnaSim is a software-hardware simulation framework for exploring a RISC-V-based analog datapath digital control (ADDC) neural network processor. It combines a custom ADDC accelerator device model for the [Spike RISC-V ISA simulator](https://github.com/riscv-software-src/riscv-isa-sim) with [Baremetal-NN](https://github.com/ucb-bar/Baremetal-NN), which converts PyTorch models into C code for bare-metal targets.

## Project components

| Component | Purpose |
| --- | --- |
| [`spike-devices/`](spike-devices/) | Spike device plugin, including the ADDC accelerator model in `src/imc_device.cc`. |
| [`Baremetal-NN/`](Baremetal-NN/) | PyTorch-to-C conversion tools, generated-code examples, and neural-network kernels. |

## Simulation workflow

1. Train and quantize your DNN model using PyTorch.
2. Convert a supported PyTorch model to C and model data with Baremetal-NN.
3. Compile the generated code and the required bare-metal support code for RISC-V.
4. Build the Spike device plugin and run the RISC-V program in Spike with the `imc_device` and its device-tree description.

This repository contains the framework components; model conversion, target compilation, and simulation setup may require project-specific configuration. See the component documentation for details.

## Requirements

- `riscv-toolchains` for compiling bare-metal RISC-V programs. Set `RISCV` to the toolchain installation path.
- [riscv-isa-sim (Spike)](https://github.com/riscv-software-src/riscv-isa-sim), built with the libraries and headers needed by the device plugin.
- A host C++17 compiler and `make` to build the plugin.
- Python and the packages listed in [`Baremetal-NN/baremetal-nn/requirements.txt`](Baremetal-NN/baremetal-nn/requirements.txt) for model conversion.
- `dtc` (Device Tree Compiler) if you need to regenerate a DTB from a DTS file.

## Build the Spike device plugin

Set `RISCV` to the installation used for Spike, then build the plugin:

```bash
export RISCV=/path/to/riscv-installation
make -C spike-devices
```

To regenerate the IMC device-tree blob:

```bash
make -C spike-devices compile_dtb
```

The plugin build creates `spike-devices/libspikedevices.so`. To run a RISC-V ELF with the IMC device, use the Spike executable associated with the same installation:

```bash
spike \
  --extlib="$PWD/spike-devices/libspikedevices.so" \
  --device=imc_device \
  --dtb="$PWD/spike-devices/imc.dtb" \
  path/to/program.elf
```

Replace `path/to/program.elf` with the program built for your target. See [`spike-devices/README.md`](spike-devices/README.md) and its Makefile for additional plugin and simulator details.

## Convert a PyTorch model

Follow the setup and conversion instructions in [`Baremetal-NN/README.md`](Baremetal-NN/README.md). To install its Python dependencies and local converter package from the repository root:

```bash
python -m pip install -r Baremetal-NN/baremetal-nn/requirements.txt
python -m pip install -e Baremetal-NN/baremetal-nn
```

Baremetal-NN documents supported operations, conversion APIs, and target build examples in its README and `examples/` directory.

## Upstream projects

- [Spike device plugins](https://github.com/ucb-bar/spike-devices)
- [Baremetal-NN](https://github.com/ucb-bar/Baremetal-NN)
- [RISC-V ISA Simulator (Spike)](https://github.com/riscv-software-src/riscv-isa-sim)

## License

See the top-level [`LICENSE`](LICENSE) and the license files included with individual components.
