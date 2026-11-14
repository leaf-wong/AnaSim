#include "l2cache.h"
#include <cstring>
#include <cstdio>

l2cache_t::l2cache_t(
      const simif_t* sim,
      abstract_interrupt_controller_t *intctrl,
      reg_t base,
      reg_t int_id)
    : sim(sim), 
      intctrl(intctrl) {
}

l2cache_t::~l2cache_t() {
  //free(blockdevice);
   printf(">>> l2cache device destroyed\n");
}


bool l2cache_t::load(reg_t addr, size_t len, uint8_t* bytes) {
  //addr -= base_addr;
  //printf("IMC load addr = 0x%lx (offset 0x%lx), len = %lu\n", addr + base_addr, addr, len);

  if (len > 4) return false;
  uint32_t value = 0;

  //memcpy(bytes, &value, len);
  memcpy(bytes, &memory[addr], len);
  return true;
}

bool l2cache_t::store(reg_t addr, size_t len, const uint8_t* bytes) {
  //addr -= base_addr;
  
  if (len > 4) return false;
  uint32_t value;
  memcpy(&memory[addr], bytes, len);
  printf("l2cache STORE addr = 0x%lx , len = %lu, size of bytes= %d, byte= %d\n", addr, len, sizeof(*bytes), *bytes);
  return true;
}

void l2cache_t::tick(reg_t rtc_ticks) {
  //printf("IMC tick called, cycle =%lu\n", rtc_ticks);
}

reg_t l2cache_t::size() {
	//printf("IMC size() called, returning 0x1000\n");
	return 0x40000;
}

int fdt_parse_l2cache(const void* fdt, reg_t* addr, const char* compatible) {
  int rc;
  int nodeoffset = fdt_node_offset_by_compatible(fdt, -1, compatible);
  if (nodeoffset < 0)
    return nodeoffset;

  rc = fdt_get_node_addr_size(fdt, nodeoffset, addr, NULL, "reg");
  if (rc < 0 || !addr)
    return -ENODEV;

  return 0;
}


l2cache_t* l2cache_parse_from_fdt(
    const void* fdt,
    const sim_t* sim,
    reg_t* base,
    std::vector<std::string> sargs) {

  if (fdt_parse_l2cache(fdt, base, "elwood,l2cache0") == 0) {
    printf("l2cache device found at 0x%lx\n", *base);
    abstract_interrupt_controller_t* intctrl = sim->get_intctrl();

    // Allocate using new, Spike will manage it
    return new l2cache_t(sim, intctrl, *base, 1);
  } else {
    return nullptr;
  }
}

std::string l2cache_generate_dts(const sim_t* sim, const std::vector<std::string>& args) {
  return std::string();  // Leave empty if not auto-generating
}

REGISTER_DEVICE(l2cache, l2cache_parse_from_fdt, l2cache_generate_dts);
