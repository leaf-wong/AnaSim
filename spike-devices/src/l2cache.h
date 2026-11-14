#ifndef _L2CACHE_H
#define _L2CACHE_H

#include <riscv/abstract_device.h>
#include <riscv/dts.h>
#include <riscv/sim.h>
#include <fdt/libfdt.h>
#include <fesvr/term.h>
#include <vector>




#define L2CACHE_BASE_ADDR 0x20000000

#define IMC_DIM 64


class l2cache_t : public abstract_device_t {
public:
  l2cache_t(const simif_t* sim,
      abstract_interrupt_controller_t *intctrl,
      reg_t base,
      reg_t int_id);
  ~l2cache_t();
  bool load(reg_t addr, size_t len, uint8_t* bytes) override;
  bool store(reg_t addr, size_t len, const uint8_t* bytes) override;
  void tick(reg_t rtc_ticks) override;
  reg_t size() override;
  

private:

  const simif_t* sim;
  abstract_interrupt_controller_t *intctrl;

  uint8_t memory[IMC_DIM * IMC_DIM + IMC_DIM];


};

#endif

