# Spike Devices

For spike device plugins.

## Dependencies

- `riscv-toolchains`
- If you are using this within Chipyard, the dependencies will be already installed in the `$RISCV` directory.

## Quick Start

Make sure `$RISCV` is set before running the following commands.

```bash
make
make install
```

Example usage:

```bash
spike --extlib libspikedevices.so --device sifive_uart ./hello.riscv
```


## Comment
1. Physical address can't be directly used in imc_device (host PC based simulator can't tell) --> using addr_to_mem in sim.cc
   spike uses page to manage memory, PGSIZE=4096. If memcpy function across two pages, need to divide  (see DRAM2SP_W_BURST in imc_device.cc)
