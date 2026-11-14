#ifndef _DMA_DEVICE_H
#define _DMA_DEVICE_H

#include <riscv/mmu.h>
#include <riscv/abstract_device.h>
#include <riscv/dts.h>
#include <riscv/sim.h>
#include <fdt/libfdt.h>
#include <fesvr/term.h>
#include <vector>


#define IMC_BASE_ADDR 0x10000000

#define IMC_REG_CONTROL (IMC_BASE_ADDR + 0x00)
#define IMC_REG_STATUS  (IMC_BASE_ADDR + 0x04)
#define IMC_REG_OUTPUT  (IMC_BASE_ADDR + 0x08)
#define IMC_AMEM_OFFSET  (IMC_BASE_ADDR + 0x0c)
#define IMC_IE     (IMC_BASE_ADDR + 0x10)
#define IMC_SP_INPUT  (IMC_BASE_ADDR + 0x1000)
#define IMC_SP_WEIGHT  (IMC_BASE_ADDR + 0x2000)
#define IMC_AMEM_0   (IMC_BASE_ADDR + 0x300000)
#define IMC_ARRAY_0  (IMC_BASE_ADDR + 0x400000)

#define IMC_CONTROL_MVM     (1)
#define IMC_CONTROL_RST     (0)

#define L2CACHE_BASE_ADDR 0x20000000

#define DMA_BASE_ADDR 0x30000000

#define SP_ROW_SIZE (64)
#define REPEATS (64)


#define DRAM2SP_IN 0x00
#define DRAM2SP_W 0x04
#define DRAM2SP_IN_BURST 0x08
#define DRAM2SP_W_BURST 0x0c
#define SP_IN2AMEM 0x10
#define SP_W2IMC 0x14
#define DMA_ROWIDX 0x18

class dma_t : public abstract_device_t {
public:
  dma_t(const simif_t* sim,
      abstract_interrupt_controller_t *intctrl,
      reg_t base,
      reg_t int_id);
  ~dma_t();
  bool load(reg_t addr, size_t len, uint8_t* bytes) override;
  bool store(reg_t addr, size_t len, const uint8_t* bytes) override;
  void tick(reg_t rtc_ticks) override;
  reg_t size() override;
  
private:
  int row_index;
  const simif_t* sim;
  abstract_interrupt_controller_t *intctrl;

  reg_t src_addr;
  reg_t dst_addr;
  reg_t length;
  reg_t start;


  // Latency simulation
  int dma_cycles;

  bool mvm_in_progress;
  int mvm_cycles_left;

  bool imc_call_reset;

  const int row_latency_cycles = 1;
  const int AMEM_write_latency_cycles = 5;
  const int IMC_write_latency_cycles = 5;

};

#endif

