#include "dma.h"
#include <cstring>
#include <cstdio>

dma_t::dma_t(
      const simif_t* sim,
      abstract_interrupt_controller_t *intctrl,
      reg_t base,
      reg_t int_id)
    : sim(sim), 
      intctrl(intctrl),
      src_addr(0),
      dst_addr(0),
      length(0),
      start(0), 
      row_index(0),
      dma_cycles(0) {
}

dma_t::~dma_t() {
  //free(blockdevice);
   printf(">>> DMA device destroyed\n");
}



bool dma_t::load(reg_t addr, size_t len, uint8_t* bytes) {
  //addr -= base_addr;
  //printf("IMC load addr = 0x%lx (offset 0x%lx), len = %lu\n", addr + base_addr, addr, len);

  if (len > 4) return false;
  uint32_t value = 0;

  switch (addr) {
    default:
      printf("DMA LOAD unknown addr 0x%lx\n", addr);
      return false;
  }

  memcpy(bytes, &value, len);
  return true;
}

bool dma_t::store(reg_t addr, size_t len, const uint8_t* bytes) {
  //addr -= base_addr;
  printf("DMA STORE addr = 0x%lx , len = %lu, size of bytes= %d, byte= %d\n", addr, len, sizeof(*bytes), *bytes);
  if (len > 4) return false;
  uint32_t value;
  memcpy(&value, bytes, len);

  switch (addr) {
    case DMA_ROWIDX:
      //printf("DMA_ROWIDX\n");
      row_index = value;
      printf("cell index = %d\n", row_index);
      return true;

    case DRAM2SP_IN:{
      //printf("DRAM2SP_IN\n");
      auto src_addr = L2CACHE_BASE_ADDR + value * SP_ROW_SIZE;
      auto dst_addr = IMC_SP_INPUT+row_index*SP_ROW_SIZE;

      for (int i = 0; i < SP_ROW_SIZE; ++i) {
        uint8_t byte = sim->debug_mmu->load<uint8_t>(src_addr + i);
        printf("DRAM2SP_IN: src=0x%lx dst=0x%lx data = %d\n", (uint64_t)(src_addr + i), (uint64_t)(dst_addr + i), byte);
        sim->debug_mmu->store<uint8_t>(dst_addr + i, byte);
      }
      //printf("DRAM2SP_IN: src=0x%lx dst=0x%lx len = %d last = %d\n", (uint64_t)(L2CACHE_BASE_ADDR + value * SP_ROW_SIZE), (uint64_t)(IMC_SP_INPUT + row_index * SP_ROW_SIZE), sizeof(uint8_t) * SP_ROW_SIZE, ((uint8_t*)(L2CACHE_BASE_ADDR+ value*SP_ROW_SIZE))[SP_ROW_SIZE-1]);
      //memcpy( (void*)(uintptr_t)(IMC_SP_INPUT+row_index*SP_ROW_SIZE), (void*)(uintptr_t)(L2CACHE_BASE_ADDR+value*SP_ROW_SIZE), sizeof(uint8_t) * SP_ROW_SIZE);
      //printf("DRAM2SP_IN: src=0x%lx dst=0x%lx\n", (uint64_t)(L2CACHE_BASE_ADDR + value * SP_ROW_SIZE), (uint64_t)(IMC_SP_INPUT + row_index * SP_ROW_SIZE));
      dma_cycles += row_latency_cycles;
      return true;
    }

    case DRAM2SP_W:
      memcpy( (void*)(IMC_SP_WEIGHT+row_index*SP_ROW_SIZE), (void*)(L2CACHE_BASE_ADDR+value*SP_ROW_SIZE), sizeof(uint8_t) * SP_ROW_SIZE);
      dma_cycles += row_latency_cycles;
      return true;

    case DRAM2SP_IN_BURST:
      memcpy( (void*)(IMC_SP_INPUT+row_index*SP_ROW_SIZE), (void*)(L2CACHE_BASE_ADDR+value*SP_ROW_SIZE), sizeof(uint8_t) * SP_ROW_SIZE * REPEATS);
      dma_cycles += row_latency_cycles * (int)REPEATS;
      return true;

    case DRAM2SP_W_BURST:
      memcpy( (void*)(IMC_SP_WEIGHT+row_index*SP_ROW_SIZE), (void*)(L2CACHE_BASE_ADDR+value*SP_ROW_SIZE), sizeof(uint8_t) * SP_ROW_SIZE* REPEATS) ;
      dma_cycles += row_latency_cycles * (int)REPEATS;
      return true;

    case SP_IN2AMEM:
      memcpy( (void*)(IMC_AMEM_0+row_index*SP_ROW_SIZE), (void*)(IMC_SP_INPUT+value*SP_ROW_SIZE), sizeof(uint8_t) * SP_ROW_SIZE);
      dma_cycles += AMEM_write_latency_cycles;
      printf("%d row of AMEM: first=%d last=%d\n", row_index, ((uint8_t*)(IMC_AMEM_0))[row_index], ((uint8_t*)(IMC_AMEM_0))[row_index + SP_ROW_SIZE-1]);
      return true;

    case SP_W2IMC:
      memcpy( (void*)(IMC_ARRAY_0+row_index*SP_ROW_SIZE), (void*)(IMC_SP_WEIGHT+value*SP_ROW_SIZE), sizeof(uint8_t) * SP_ROW_SIZE);
      dma_cycles += IMC_write_latency_cycles;
      printf("%d row of IMC ARRAY: first=%d last=%d\n", row_index, ((uint8_t*)(IMC_ARRAY_0))[row_index], ((uint8_t*)(IMC_ARRAY_0))[row_index + SP_ROW_SIZE-1]);
      return true;

    default:
      printf("DMA STORE unknown addr 0x%lx\n", addr);
      return false;
  }
}

void dma_t::tick(reg_t rtc_ticks) {
  //printf("IMC tick called, cycle =%lu\n", rtc_ticks);
}

reg_t dma_t::size() {
	//printf("IMC size() called, returning 0x1000\n");
	return 0x1000;
}

int fdt_parse_dma(const void* fdt, reg_t* addr, const char* compatible) {
  int rc;
  int nodeoffset = fdt_node_offset_by_compatible(fdt, -1, compatible);
  if (nodeoffset < 0)
    return nodeoffset;

  rc = fdt_get_node_addr_size(fdt, nodeoffset, addr, NULL, "reg");
  if (rc < 0 || !addr)
    return -ENODEV;

  return 0;
}


dma_t* dma_parse_from_fdt(
    const void* fdt,
    const sim_t* sim,
    reg_t* base,
    std::vector<std::string> sargs) {

  if (fdt_parse_dma(fdt, base, "elwood,dma0") == 0) {
    printf("DMA device found at 0x%lx\n", *base);
    abstract_interrupt_controller_t* intctrl = sim->get_intctrl();

    // Allocate using new, Spike will manage it
    return new dma_t(sim, intctrl, *base, 1);
  } else {
    return nullptr;
  }
}

std::string dma_generate_dts(const sim_t* sim, const std::vector<std::string>& args) {
  return std::string();  // Leave empty if not auto-generating
}

REGISTER_DEVICE(dma, dma_parse_from_fdt, dma_generate_dts);
