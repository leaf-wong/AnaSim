#include "imc_device.h"
#include <cstring>
#include <cstdio>
#include <math.h>
#include <algorithm>

imc_device_t::imc_device_t(
      const simif_t* sim,
      abstract_interrupt_controller_t *intctrl,
      reg_t base,
      reg_t int_id)
    : sim(sim), 
      intctrl(intctrl), 
      offset(0),
      bias_offset(0),
      bias_existing(0),
      row_index(0),
      write_in_progress(false),
      amem_leakage_cycles(0),
      mvm_in_progress(false),
      mvm_cycles_left(0),
      imc_call_reset(false),
      pool_enable(false),
      conv_enable(false),
      layer_fuse_enable(0),
      load_first_weight(true),
      imc_cycles(0),
      dma_cycles(0),
      anu_cycles(0),
      amem_r_cycles(0),
      amem_w_cycles(0),
      refresh_cycles(0),
      ping_pong_computing_imc_cycles(0),
      ping_pong_computing_anu_cycles(0),
      ping_pong_computing_dma_cycles(0),
      ping_pong_computing_cycles(0),
      ping_pong_loading_cycles(0),
      total_layer_cycles(0),
      amem_read_energy(0),
      amem_write_energy(0),
      imc_leakage_energy(0),
      amem_leakage_energy(0),
      imc_computing_energy(0),
      anu_computing_energy(0),
      imc_loading_energy(0),
      refresh_energy(0),
      ADDA_energy(0),
      sram_buffer_leakage_energy(0),
      fsm_energy(0),
      total_energy(0),
      ie(0),
      interrupt_id(int_id) {
  //matrix.resize(IMC_DIM, std::vector<uint8_t>(IMC_DIM, 0));
  bias_enable.resize((IMC_DIM/MACRO_DIM), 0);
  output_vector.resize((IMC_DIM/MACRO_DIM), std::vector<int32_t>(IMC_DIM, 0));
}

imc_device_t::~imc_device_t() {
  //free(blockdevice);
   printf(">>> IMC device destroyed\n");
}

void imc_device_t::compute() {
  for (int i = 0; i < IMC_DIM; ++i) {
    for (int j = 0; j < (IMC_DIM/MACRO_DIM); ++j) {
      int32_t acc = 0;      
      for (int k = 0; k < MACRO_DIM; ++k) {
        if (imc0_computing){
          acc += (imc_array_0[(j*MACRO_DIM + k) * IMC_DIM+ i]-W_ZERO_POINT) * input_buffer[j*MACRO_DIM + k];
        }
        else{
          acc += (imc_array_1[(j*MACRO_DIM + k) * IMC_DIM+ i]-W_ZERO_POINT) * input_buffer[j*MACRO_DIM + k];
        }
      }
      output_vector[j][i] = acc+  bias_enable[j] * bias_scale*(bias_row_0[i]-BIAS_ZERO_POINT);
    }
    //output_vector[i] = acc + bias_enable * BIAS_SCALE*(bias_row_0[i]-BIAS_ZERO_POINT);
    /*if (bias_enable && i==IMC_DIM-1)
      if (imc0_computing)
        printf("%dth output result is %d, last-2 input=%d, w=%d, last-1 input=%d, w=%d, last input=%d, w=%d, bias=%d\n", i, output_vector[i], amem_0[offset* IMC_DIM +IMC_DIM-3], imc_array_0[(IMC_DIM-3) * IMC_DIM+ i], amem_0[offset* IMC_DIM +IMC_DIM-2], imc_array_0[(IMC_DIM-2) * IMC_DIM+ i], amem_0[offset* IMC_DIM +IMC_DIM-1], imc_array_0[(IMC_DIM-1) * IMC_DIM+ i],bias_row_0[i]);
      else
        printf("%dth output result is %d, last-2 input=%d, w=%d, last-1 input=%d, w=%d, last input=%d, w=%d, bias=%d\n", i, output_vector[i], amem_0[offset* IMC_DIM +IMC_DIM-3], imc_array_1[(IMC_DIM-3) * IMC_DIM+ i], amem_0[offset* IMC_DIM +IMC_DIM-2], imc_array_1[(IMC_DIM-2) * IMC_DIM+ i], amem_0[offset* IMC_DIM +IMC_DIM-1], imc_array_1[(IMC_DIM-1) * IMC_DIM+ i],bias_row_0[i]);
        */
  }
  //imc_cycles += mvm_latency_cycles;
}

void imc_device_t::dma(uint32_t read_addr) {
    //printf("DMA from DRAM addr 0x%x to SP weight buffer\n", read_addr);
    uint32_t row_addr = read_addr;
    uint32_t page_base = row_addr & ~(PGSIZE - 1);          // 4k aligned base, PGSIZE=4096 defined in mmu.h
    uint32_t page_off = row_addr & (PGSIZE - 1);           // offset into page
    uint32_t bytes_in_page = PGSIZE - page_off;
    //printf("byte in page = %d\n", bytes_in_page);
    
    if (bytes_in_page >= SP_ROW_SIZE) {
      char* ptr = ((sim_t*)sim)->get_mem_ptr(row_addr);
      memcpy( &imc_w_mux_in[0], ptr, SP_ROW_SIZE);
    }
    else {
      char* ptr1 = ((sim_t*)sim)->get_mem_ptr(row_addr);
      memcpy( &imc_w_mux_in[0], ptr1, bytes_in_page);

      uint32_t bytes_in_next_page = SP_ROW_SIZE - bytes_in_page;
      uint32_t row_addr2 = row_addr + bytes_in_page;
      char* ptr2 = ((sim_t*)sim)->get_mem_ptr(row_addr2);

      memcpy( &imc_w_mux_in[bytes_in_page], ptr2, bytes_in_next_page);
    }

    //printf("DRAM2SP_W_BURST: first=%d 11th=%d last=%d\n", imc_w_mux_in[0],imc_w_mux_in[10], imc_w_mux_in[SP_ROW_SIZE-1]);
}

// row-level counter for fine-grained AMEM leakage profile.
void imc_device_t::leakage(int cycle_inc) {
  refresh_counter += cycle_inc;
  for (int leak_j = 0; leak_j < AMEM_ROW_INDEX; ++leak_j) {
    amem_row_cycles_0[leak_j] += cycle_inc;

    /*if (amem_row_cycles_0[j] >= RETENTION_TIME){
      if (j<IMC_AMEM_TEMP_BASE){ 
        for (int i = 0; i < IMC_DIM; ++i) {
          if (amem_0[j*IMC_DIM +i] >AMEM_MIDDLE_VALUE)
            amem_0[j*IMC_DIM +i] = amem_0[j*IMC_DIM +i] -1; 
          else if (amem_0[j*IMC_DIM +i] < AMEM_MIDDLE_VALUE)
            amem_0[j*IMC_DIM +i] = amem_0[j*IMC_DIM +i] +1;
          else
            amem_0[j*IMC_DIM +i] = amem_0[j*IMC_DIM +i];       
        }
      }
      else {
        for (int i = 0; i < IMC_DIM; ++i) {
          if (amem_0[j*IMC_DIM +i] >AMEM_MIDDLE_VALUE*amem_scale)
            amem_0[j*IMC_DIM +i] = amem_0[j*IMC_DIM +i] - amem_scale; 
          else if (amem_0[j*IMC_DIM +i] < AMEM_MIDDLE_VALUE*amem_scale)
            amem_0[j*IMC_DIM +i] = amem_0[j*IMC_DIM +i] + amem_scale;
          else
            amem_0[j*IMC_DIM +i] = amem_0[j*IMC_DIM +i];       
        }
      }
      amem_row_cycles_0[j] = amem_row_cycles_0[j]-RETENTION_TIME;
      //leakage flag
      amem_row_leak_flag_0[j] += 1;
      //printf("AMEM leaks\n");
    }*/
  }
  for (int leak_j = 0; leak_j < (IMG_SIZE*64*3)/IMC_DIM; ++leak_j) {
    for (int leak_c = 0; leak_c < NUM_FUSED_LAYERS-1; leak_c++){
      amem_row_cycles[leak_c][leak_j] += cycle_inc;
    }
  }
}

void imc_device_t::reset_amem() {
  for (int j = 0; j < AMEM_ROW_INDEX; ++j) {
    refresh_counter = 0;
    amem_row_cycles_0[j] = 0;
    amem_row_leak_flag_0[j] = 0;
    /*for (int n = 0; n < SP_ROW_SIZE; ++n)
      amem_0[(j) * SP_ROW_SIZE+ n] = 0;*/
    if (j == AMEM_ROW_INDEX-1)
      printf("AMEM reset finish, %d rows\n", j+1);
  }
  for (int j = 0; j < (IMG_SIZE*64*3)/IMC_DIM; ++j) {
    for (int c = 0; c < NUM_FUSED_LAYERS-1; c++){
      amem_row_cycles[c][j] = 0;
    }
  }
}

// signle counter for bank-level AMEM refresh control.
void imc_device_t::refresh() {
  bool need_refresh = false;
  int active_macro_index = 0;
  int active_temp_macro_index = 0;

  // Row-level refresh counter is too complex, here we use a simple bank-level counter for refresh control.
  /*for (int j = 0; j < AMEM_ROW_INDEX; ++j) {
    if (amem_row_cycles_0[j] >= REFRESH_INTERVAL) 
      need_refresh = true;
  }*/

  if (refresh_counter >= REFRESH_INTERVAL) 
    need_refresh = true;

  // refresh only active macros
  //capacitor based AAIMC only has limited precision, only consider refresh energy for now (function later)
  if (need_refresh) {
    if (pool_enable){
      active_macro_index = (output_rem > 0) ? (amem_read_base+2*output_col*2*output_row*output_itr*output_rem/SP_ROW_SIZE) : (amem_read_base+2*output_col*2*output_row*output_itr);
      //printf("active_macro_index = %d, amem_read_base=%d, output_col=%d, output_row=%d, output_itr=%d, output_rem=%d\n", active_macro_index, amem_read_base, output_col, output_row, output_itr, output_rem);
    }
    else{
      active_macro_index = (row_index > amem_read_base) ? ((output_rem > 0) ? (row_index+output_col*output_row*output_itr*output_rem/SP_ROW_SIZE) : (row_index+output_col*output_row*output_itr)) : ((output_rem > 0) ? (amem_read_base+output_col*output_row*output_itr*output_rem/SP_ROW_SIZE) : (amem_read_base+output_col*output_row*output_itr));
      active_macro_index = (active_macro_index > IMC_AMEM_TEMP_BASE) ? IMC_AMEM_TEMP_BASE : active_macro_index;
    }

    if (conv_enable)
      active_temp_macro_index = (output_rem > 0) ? (IMC_AMEM_TEMP_BASE+output_col*output_row*output_itr*output_rem/SP_ROW_SIZE) : (IMC_AMEM_TEMP_BASE+output_col*output_row*output_itr);
    else
      active_temp_macro_index = IMC_AMEM_TEMP_BASE;

    /*for (int j = 0; j < AMEM_ROW_INDEX; ++j) {
      if (j<active_macro_index){ // for activations
      //if (j<IMC_AMEM_TEMP_BASE){ // for activations
        for (int i = 0; i < IMC_DIM; ++i) {
          if (amem_0[j*IMC_DIM +i] > (pow(2,AMEM_PRECISION) - 1)*1)
            amem_0[j*IMC_DIM +i] = (pow(2,AMEM_PRECISION) - 1)*1;
          else if (amem_0[j*IMC_DIM +i] < 0)
            amem_0[j*IMC_DIM +i] = 0;
          else
            amem_0[j*IMC_DIM +i] = (amem_0[j*IMC_DIM +i]/1) * 1; // activation no scale
        }
        amem_row_cycles_0[j] = 0;
        //printf("%d ", j);
      }
      else if (j>=IMC_AMEM_TEMP_BASE&&j<active_temp_macro_index){ // for pSUM
      //else {
        for (int i = 0; i < IMC_DIM; ++i) {
          if (amem_0[j*IMC_DIM +i] > (pow(2,AMEM_PRECISION-1) - 1)*amem_scale)
            amem_0[j*IMC_DIM +i] = (pow(2,AMEM_PRECISION-1) - 1)*amem_scale;
          else if (amem_0[j*IMC_DIM +i] < -pow(2,AMEM_PRECISION-1)*amem_scale)
            amem_0[j*IMC_DIM +i] = -pow(2,AMEM_PRECISION-1)*amem_scale;
          else
            amem_0[j*IMC_DIM +i] = (amem_0[j*IMC_DIM +i]/amem_scale) * amem_scale; // scale to AMEM_PRECISION
        }
        amem_row_cycles_0[j] = 0;
      }
    }*/
    printf("AMEM refresh, refresh_counter=%d\n", refresh_counter);
    refresh_counter = AMEM_refresh_latency_cycles*64; //macro-level refresh only takes 64 row latency cycles in AnaRefresh
    refresh_cycles += AMEM_refresh_latency_cycles*64;
    if (SRAM_IMC==1){
      refresh_energy += AMEM_refresh_latency_cycles*(AMEM_ARRAY_SIZE+FUSED_AMEM_ARRAY_SIZE)*AMEM_64_Refresh_POWER/64;
    }
    else {
      refresh_energy += AMEM_refresh_latency_cycles*(AMEM_ARRAY_SIZE+FUSED_AMEM_ARRAY_SIZE+IMC_ARRAY_SIZE+FUSED_IMC_ARRAY_SIZE)*AMEM_64_Refresh_POWER/64;
    }

    
  }
  //printf("AMEM refresh sampled value = %d and %d; %d and %d\n", (72/10)*10, (78/10)*10, (-75/10)*10, (-72/10)*10);

}


bool imc_device_t::load(reg_t addr, size_t len, uint8_t* bytes) {
  //addr -= base_addr;
  //printf("IMC load addr = 0x%lx, len = %lu\n", addr, len);

  if (len > 4) return false;
  uint32_t value = 0;

  switch (addr) {
    case IMC_REG_STATUS:
      //value = (!write_in_progress && !mvm_in_progress) ? 1 : 0;
      value = ie;
      break;

    case IMC_REG_OUTPUT:
      memcpy(&value, &output_vector[0], 4);  // Only read first word of result
      printf("Output0 result is %d\n", output_vector[0]);
      break;

    case L2CACHE_BASE_ADDR ... L2CACHE_BASE_ADDR + L2CACHE_SIZE - 1:
      memcpy(&value, &l2_cache[(addr-L2CACHE_BASE_ADDR)], len);
      break;

    case IMC_AMEM_0 ... IMC_AMEM_0 + AMEM_ARRAY_SIZE - 1:
      memcpy(&value, &amem_0[(addr-IMC_AMEM_0)], len);
      printf("Value = %d index = %d amem= %d int8 amem = %d len = %d\n", value, (addr-IMC_AMEM_0), amem_0[(addr-IMC_AMEM_0)], int8_t(amem_0[(addr-IMC_AMEM_0)]), len);
      
      total_layer_cycles += ADC_CYCLE;
      leakage((ADC_CYCLE));
      refresh();
      total_cycles = total_layer_cycles+refresh_cycles;
      printf("Total cycle= %lu\n", total_cycles);

      printf("Total area= %lu um^2\n", total_area*9*9/100);// 0.9 cale in 28nm

      ADDA_energy += ADC_CYCLE*ADC_POWER;
      amem_read_energy += ADC_CYCLE * 1 * AMEM_128_R_POWER/128;
      if (SRAM_IMC==1){
        imc_leakage_energy += (ADC_CYCLE * (1+PP_IMC_ENABLE)*IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024)
                              + (ADC_CYCLE) * FUSED_IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024));
      }
      else{
        imc_leakage_energy += (ADC_CYCLE * (1+PP_IMC_ENABLE)*IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024)
                              + (ADC_CYCLE) * FUSED_IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024));
      }
      sram_buffer_leakage_energy += (ADC_CYCLE) * ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_Leakage_POWER/(5152*1024);
      sram_leakage_energy_in_refresh = refresh_cycles * ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_Leakage_POWER/(5152*1024)
                                        + SRAM_IMC * ((1+PP_IMC_ENABLE)* IMC_ARRAY_SIZE + FUSED_IMC_ARRAY_SIZE)* AAIMC_PRECISION * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024);
      fsm_energy = total_cycles * CTRL_FSM_POWER;
      total_energy = amem_read_energy + amem_write_energy + amem_leakage_energy +  refresh_energy+
                    imc_computing_energy + anu_computing_energy + imc_loading_energy + sram_leakage_energy_in_refresh+
                    imc_leakage_energy + sram_buffer_leakage_energy + fsm_energy + ADDA_energy;
      printf("Total energy= %lu pJ\n", total_energy*PERIOD/1000);
      break;

    default:
      //printf("IMC LOAD unknown addr 0x%lx\n", addr);
      return false;
  }

  memcpy(bytes, &value, len);
  return true;
}

bool imc_device_t::store(reg_t addr, size_t len, const uint8_t* bytes) {
  //addr -= base_addr;
  //printf("IMC STORE addr = 0x%lx, len = %lu, byte= %d\n", addr, len, *bytes);
  //if (len > 4) return false;
  uint32_t value;
  memcpy(&value, bytes, len);

  switch (addr) {
    
    case IMC_REG_CONTROL:
      if (value & IMC_CONTROL_MVM && !mvm_in_progress && !imc_call_reset) {
        mvm_in_progress = true;
        mvm_cycles_left = mvm_latency_cycles;
        //compute();
      }
      else if (value & IMC_CONTROL_RST) {
        imc_call_reset = false;
      }
      return true;

    case AMEM_RD_OFFSET:
      offset = value;
      return true;

    case BIAS_ROW_RD_OFFSET:
      bias_offset = value;
      return true;

    case BIAS_EXIST:
      bias_existing = value;
      return true;

    case IMC_IE:     memcpy(&ie, bytes, len); update_interrupts(); printf("IMC_IE = %d\n", ie);return true;

    // L2 cache address
    case L2CACHE_BASE_ADDR ... L2CACHE_BASE_ADDR + L2CACHE_SIZE - 1:
      //memcpy(&l2_cache[(addr-L2CACHE_BASE_ADDR)], bytes, len);
      switch (len) {
        case 1:
          l2_cache[(addr-L2CACHE_BASE_ADDR)] = static_cast<uint8_t>(value & 0xFF);
          break;
        case 4:
          l2_cache[(addr-L2CACHE_BASE_ADDR)] = static_cast<uint8_t>(value & 0xFF);
          l2_cache[(addr-L2CACHE_BASE_ADDR+1)] = static_cast<uint8_t>((value >> 8) & 0xFF);
          l2_cache[(addr-L2CACHE_BASE_ADDR+2)] = static_cast<uint8_t>((value >> 16) & 0xFF);
          l2_cache[(addr-L2CACHE_BASE_ADDR+3)] = static_cast<uint8_t>((value >> 24) & 0xFF);
          break;
        case 8:
          // Implement double-word stores as a pair of word stores
          l2_cache[(addr-L2CACHE_BASE_ADDR)] = static_cast<uint8_t>(value & 0xFF);
          l2_cache[(addr-L2CACHE_BASE_ADDR+1)] = static_cast<uint8_t>((value >> 8) & 0xFF);
          l2_cache[(addr-L2CACHE_BASE_ADDR+2)] = static_cast<uint8_t>((value >> 16) & 0xFF);
          l2_cache[(addr-L2CACHE_BASE_ADDR+3)] = static_cast<uint8_t>((value >> 24) & 0xFF);
          l2_cache[(addr-L2CACHE_BASE_ADDR+5)] = static_cast<uint8_t>((value >> 32) & 0xFF);
          l2_cache[(addr-L2CACHE_BASE_ADDR+6)] = static_cast<uint8_t>((value >> 40) & 0xFF);
          l2_cache[(addr-L2CACHE_BASE_ADDR+7)] = static_cast<uint8_t>((value >> 48) & 0xFF);
          l2_cache[(addr-L2CACHE_BASE_ADDR+8)] = static_cast<uint8_t>((value >> 56) & 0xFF);
          break;
        default:
          // Other type stores are not supported
          printf("Unsupported L2 cache store: addr=0x%lx, len=%lu\n", addr, len);
          return false;
      }
      
      //printf(" value= %d len= %d \n", l2_cache[(addr-L2CACHE_BASE_ADDR)], len);
      //dma_cycles += row_latency_cycles;
      return true;

    // DMA related Register
    case DMA_ROWIDX:
      row_index = value;
      printf("cell index = %d\n", row_index);
      return true;

    case DRAM2SP_IN:{
      memcpy( &sp_input[row_index * SP_ROW_SIZE], &l2_cache[value * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
      printf("DRAM2SP_IN: first=%d last=%d\n", sp_input[row_index * SP_ROW_SIZE], sp_input[row_index * SP_ROW_SIZE+SP_ROW_SIZE-1]);
      dma_cycles += row_latency_cycles;
      //leakage(row_latency_cycles); SP initialization at begining, no need to leakage
      return true;
    }

    case DRAM2SP_W:{
      //memcpy( &sp_weight[row_index * SP_ROW_SIZE], &l2_cache[value * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
      uint32_t row_addr = value;
      uint32_t page_base = row_addr & ~(PGSIZE - 1);          // 4k aligned base, PGSIZE=4096 defined in mmu.h
      uint32_t page_off = row_addr & (PGSIZE - 1);           // offset into page
      uint32_t bytes_in_page = PGSIZE - page_off;
      //printf("byte in page = %d\n", bytes_in_page);
      
      if (bytes_in_page >= SP_ROW_SIZE) {
        char* ptr = ((sim_t*)sim)->get_mem_ptr(row_addr);
        if (!ptr) {
              fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr);
              return false;
          }
        memcpy( &sp_weight[row_index * SP_ROW_SIZE], ptr, SP_ROW_SIZE);
      }
      else {
        char* ptr1 = ((sim_t*)sim)->get_mem_ptr(row_addr);
        if (!ptr1) {
              fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr);
              return false;
          }
        memcpy( &sp_weight[row_index * SP_ROW_SIZE], ptr1, bytes_in_page);

        uint32_t bytes_in_next_page = SP_ROW_SIZE - bytes_in_page;
        uint32_t row_addr2 = row_addr + bytes_in_page;
        char* ptr2 = ((sim_t*)sim)->get_mem_ptr(row_addr2);
        if (!ptr2) {
              fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr2);
              return false;
          }
        memcpy( &sp_weight[row_index * SP_ROW_SIZE+bytes_in_page], ptr2, bytes_in_next_page);
      }

      //char* ptr = ((sim_t*)sim)->get_mem_ptr(value);
      //memcpy( &sp_weight[row_index * SP_ROW_SIZE], ptr, sizeof(uint8_t) * SP_ROW_SIZE);
      dma_cycles += row_latency_cycles;
      printf("DRAM2SP_W: first=%d 11th=%d last=%d\n", sp_weight[row_index * SP_ROW_SIZE], sp_weight[row_index * SP_ROW_SIZE+10], sp_weight[row_index * SP_ROW_SIZE+SP_ROW_SIZE-1]);
      //ie = 1;
      //update_interrupts();
      //leakage(row_latency_cycles); SP initialization at begining, no need to leakage
      return true;
    }

    case DRAM2SP_IN_BURST:{
      char* ptr = ((sim_t*)sim)->get_mem_ptr(value);
      memcpy( &sp_input[row_index * SP_ROW_SIZE], ptr, sizeof(uint8_t) * SP_ROW_SIZE* repeats);
      dma_cycles += row_latency_cycles * repeats;

      return true;
    }

    case DRAM2SP_W_BURST:{
      for (int i = 0; i < repeats; i++) {
        uint32_t row_addr = value + i * SP_ROW_SIZE;
        uint32_t page_base = row_addr & ~(PGSIZE - 1);          // 4k aligned base, PGSIZE=4096 defined in mmu.h
        uint32_t page_off = row_addr & (PGSIZE - 1);           // offset into page
        uint32_t bytes_in_page = PGSIZE - page_off;
        //printf("byte in page = %d\n", bytes_in_page);
        
        if (bytes_in_page >= SP_ROW_SIZE) {
          char* ptr = ((sim_t*)sim)->get_mem_ptr(row_addr);
          if (!ptr) {
                fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr);
                return false;
            }
          memcpy( &sp_weight[row_index * SP_ROW_SIZE+i*SP_ROW_SIZE], ptr, SP_ROW_SIZE);
        }
        else {
          char* ptr1 = ((sim_t*)sim)->get_mem_ptr(row_addr);
          if (!ptr1) {
                fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr);
                return false;
            }
          memcpy( &sp_weight[row_index * SP_ROW_SIZE+i*SP_ROW_SIZE], ptr1, bytes_in_page);

          uint32_t bytes_in_next_page = SP_ROW_SIZE - bytes_in_page;
          uint32_t row_addr2 = row_addr + bytes_in_page;
          char* ptr2 = ((sim_t*)sim)->get_mem_ptr(row_addr2);
          if (!ptr2) {
                fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr2);
                return false;
            }
          memcpy( &sp_weight[row_index * SP_ROW_SIZE+i*SP_ROW_SIZE+bytes_in_page], ptr2, bytes_in_next_page);
        }
      }
      dma_cycles += row_latency_cycles * repeats;
      char* ptr = ((sim_t*)sim)->get_mem_ptr(value);
      printf("repeats = %d\n", repeats);
      //printf("DEV: addr=%p ptr[0]=%u ptr[10]=%u ptr[11]=%u ptr[12]=%u ptr[last]=%u\n", value, *(((sim_t*)sim)->get_mem_ptr(value+ IMC_DIM-1)), *(((sim_t*)sim)->get_mem_ptr(value+10*SP_ROW_SIZE +IMC_DIM-1)),*(((sim_t*)sim)->get_mem_ptr(value+11*SP_ROW_SIZE +IMC_DIM-1)), *(((sim_t*)sim)->get_mem_ptr(value+12*SP_ROW_SIZE +IMC_DIM-1)), *(((sim_t*)sim)->get_mem_ptr(value+(SP_ROW_SIZE-1)*SP_ROW_SIZE +IMC_DIM-1)));
      //printf("DEV2: addr=%p ptr[0]=%u ptr[10]=%u ptr[11]=%u ptr[12]=%u ptr[last]=%u\n", value, (uint8_t)ptr[SP_ROW_SIZE-1], (uint8_t)ptr[10*SP_ROW_SIZE + SP_ROW_SIZE-1], (uint8_t)ptr[11*SP_ROW_SIZE + SP_ROW_SIZE-1], (uint8_t)ptr[12*SP_ROW_SIZE + SP_ROW_SIZE-1], (uint8_t)ptr[(SP_ROW_SIZE-1)*SP_ROW_SIZE + SP_ROW_SIZE-1]);
      //printf("DRAM2SP_W_IN: first=%d 3rd=%d last-1=%d last=%d\n", ptr[0], ptr[0+2*SP_ROW_SIZE],  ptr[0+(SP_ROW_SIZE-2)*SP_ROW_SIZE],  ptr[0+(SP_ROW_SIZE-1)*SP_ROW_SIZE]);
      printf("DRAM2SP_W_BURST: first=%d 11th=%d 12th=%d 13th=%d last=%d\n", sp_weight[row_index * SP_ROW_SIZE+0],sp_weight[row_index * SP_ROW_SIZE+10*SP_ROW_SIZE+0], sp_weight[row_index * SP_ROW_SIZE+11*SP_ROW_SIZE+0], sp_weight[row_index * SP_ROW_SIZE+12*SP_ROW_SIZE+0], sp_weight[row_index * SP_ROW_SIZE+(repeats-1)*SP_ROW_SIZE+SP_ROW_SIZE-1]);
      //ie = 1;
      //update_interrupts();
      return true;
    }

    case SP_IN2AMEM:
      if (value == 0)
        reset_amem();

      for (int i = 0; i < SP_ROW_SIZE; ++i) {
        amem_0[row_index * SP_ROW_SIZE + i] = static_cast<int32_t>(sp_input[value * SP_ROW_SIZE + i]);
      }
      //memcpy( &amem_0[row_index * SP_ROW_SIZE], &sp_input[value * SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
      dma_cycles += AMEM_write_latency_cycles;
      amem_row_cycles_0[row_index] = 0;
      leakage(AMEM_write_latency_cycles);
      printf("%d row of AMEM: first=%d last=%d\n", row_index, amem_0[row_index* SP_ROW_SIZE], amem_0[row_index* SP_ROW_SIZE + SP_ROW_SIZE-1]);
      return true;

    case SP_W2IMC:
      memcpy( &imc_array_0[row_index * SP_ROW_SIZE], &sp_weight[value * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
      dma_cycles += IMC_write_latency_cycles;
      leakage(IMC_write_latency_cycles);
      printf("%d row of IMC ARRAY: first=%d last=%d\n", row_index, imc_array_0[row_index* SP_ROW_SIZE], imc_array_0[row_index* SP_ROW_SIZE + SP_ROW_SIZE-1]);
      return true;

    case SP_W2BIAS_ROW:
      memcpy( &bias_row_0[row_index * SP_ROW_SIZE], &sp_weight[value * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
      dma_cycles += IMC_write_latency_cycles;
      leakage(IMC_write_latency_cycles);
      printf("%d row of BIAS_ROW: first=%d last=%d\n", row_index, bias_row_0[row_index* SP_ROW_SIZE], bias_row_0[row_index* SP_ROW_SIZE + SP_ROW_SIZE-1]);
      return true;

    // ANU related Register
    case ANU_ADD:
      printf("input1 of ANU: first=%d last=%d\n", amem_0[offset* SP_ROW_SIZE], amem_0[offset* SP_ROW_SIZE + SP_ROW_SIZE-1]);
      printf("input2 of ANU: first=%d last=%d\n", amem_0[value* SP_ROW_SIZE], amem_0[value* SP_ROW_SIZE + SP_ROW_SIZE-1]);
      for (int j = 0; j < SP_ROW_SIZE; ++j) {
        amem_0[row_index*SP_ROW_SIZE+j] = amem_0[offset*SP_ROW_SIZE+ j] + amem_0[value* IMC_DIM + j];
      }
      anu_cycles += anu_latency_cycles;
      dma_cycles += AMEM_write_latency_cycles;
      printf("output of ANU: first=%d last=%d\n", amem_0[row_index* SP_ROW_SIZE], amem_0[row_index* SP_ROW_SIZE + SP_ROW_SIZE-1]);
      return true;

    case ANU_ADD_RELU:{
      int32_t anu_result = 0;
      for (int j = 0; j < SP_ROW_SIZE; ++j) {
        anu_result = amem_0[offset*SP_ROW_SIZE+ j] + amem_0[value* IMC_DIM + j];
        if (anu_result > 0)
          amem_0[row_index*SP_ROW_SIZE+j] = anu_result;
        else
          amem_0[row_index*SP_ROW_SIZE+j] = 0;
      }
      anu_cycles += anu_latency_cycles;
      dma_cycles += AMEM_write_latency_cycles;
      printf("output of ANU: first=%d last=%d\n", amem_0[row_index* SP_ROW_SIZE], amem_0[row_index* SP_ROW_SIZE + SP_ROW_SIZE-1]);
      return true;
    }
    
    case AMEM_ROW_REFRESH:{
      int32_t anu_result = 0;
      for (int j = 0; j < SP_ROW_SIZE; ++j) {
        anu_result = amem_0[value*SP_ROW_SIZE+ j];
        if (anu_result > pow(2,AMEM_PRECISION) - 1)
          amem_0[value*SP_ROW_SIZE+ j] = pow(2,AMEM_PRECISION) - 1;
      }
      dma_cycles += AMEM_refresh_latency_cycles;
      printf("output after Refresh: first=%d last=%d\n", amem_0[value* SP_ROW_SIZE], amem_0[value* SP_ROW_SIZE + SP_ROW_SIZE-1]);
      return true;
    }

    // ML Kernel operation
    case DMA_BURST_NUM:
      repeats = value;
      printf("repeats=%d\n",repeats);
      return true;
    
    case DMA_READ_BASE:
      dma_read_base = value;
      return true;

    case DMA_READ_INPUT_BASE:
      dma_read_input_base = value;
      return true;
    
    case SP_W_WR_BASE:
      sp_weight_write_base = value;
      return true;

    case AMEM_RD_BASE:
      amem_read_base = value;
      return true;

    case AMEM_RD_BASE2:
      amem_read_base2 = value;
      return true;

    case INPUT_ITR:{
      if (layer_fuse_enable)
        input_itr_fused[layer_index] = value;
      else
        input_itr = value;
      return true;
    }

    case OUTPUT_ITR:{
      if (layer_fuse_enable)
        output_itr_fused[layer_index] = value;
      else
        output_itr = value;
      return true;
    }
    
    case OUTPUT_COL:{
      if (layer_fuse_enable)
        output_col_fused[layer_index] = value;
      else
        output_col = value;
      return true;
    }

    case OUTPUT_ROW:{
      if (layer_fuse_enable)
        output_row_fused[layer_index] = value;
      else
        output_row = value;
      return true;
    }

    case WEIGHT_COL:{
      if (layer_fuse_enable)
        weight_col_fused[layer_index] = value;
      else
        weight_col = value;
      return true;
    }

    case WEIGHT_ROW:{
      if (layer_fuse_enable)
        weight_row_fused[layer_index] = value;
      else
        weight_row = value;
      return true;
    }
    
    case INPUT_REMAINDER:{
      if (layer_fuse_enable)
        input_rem_fused[layer_index] = value;
      else
        input_rem = value;
      return true;
    }

    case OUTPUT_REMAINDER:{
      if (layer_fuse_enable)
        output_rem_fused[layer_index] = value;
      else
        output_rem = value;
      return true;
    }

    case NEXT_CONV_CHANNEL:
      next_conv_channel = value;
      return true;
    
    case PADDING:{
      if (layer_fuse_enable)
        padding_fused[layer_index] = value;
      else
        padding = value;
      return true;
    }
    
    case STRIDE:{
      if (layer_fuse_enable)
        stride_fused[layer_index] = value;
      else
        stride = value;
      return true;
    }

    case AMEM_SCALE_CFG:{
      if (layer_fuse_enable)
        amem_scale_fused[layer_index] = value;
      else
        amem_scale = value;
      return true;
    }

    case BIAS_SCALE_CFG:{
      if (layer_fuse_enable)
        bias_scale_fused[layer_index] = value;
      else
        bias_scale = value;
      return true;
    }

    //Layer fusion config
    case LAYER_INDEX:
      layer_index = value;
      return true;

    case LAYER_FUSE_ENABLE:
      layer_fuse_enable = value;
      return true;
    
    case LAYER_TYPE:
      layer_type[layer_index] = value;
      return true;

    case LINEAR_LAYER:{
      printf("Linear layer start input_itr= %d, output_itr= %d, output_rem= %d\n", input_itr, output_itr, output_rem);
      ping_pong_computing_cycles = 0;
      ping_pong_loading_cycles = 0;
      ping_pong_computing_imc_cycles = 0;
      ping_pong_computing_input_feeding_cycles = 0;
      ping_pong_computing_output_wb_cycles = 0;
      layer_cycles = 0;
      for (int j = 0; j < output_itr;j += 1) { // output_itr will be 1 for linear
        // Load bias from MEM -> IMC_row
        printf("input_itr= %d, output_itr= %d, output_rem= %d\n", input_itr, output_itr, output_rem);
        dma(dma_read_base+(input_itr*output_rem+j)* SP_ROW_SIZE);
        //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
        memcpy( &bias_row_0[0], &imc_w_mux_in[0], sizeof(uint8_t) * output_rem);
        printf("bias_row: first=%d 10th=%d last=%d\n", bias_row_0[0], bias_row_0[9], bias_row_0[63]);
        ping_pong_loading_cycles += IMC_write_latency_cycles;
        layer_cycles += IMC_write_latency_cycles;
        leakage(IMC_write_latency_cycles);
        refresh();

        if (SRAM_IMC==0){
          imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * output_rem * AAIMC_AMEM_32x4b_W_POWER/128;
        }
        else{
          imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * output_rem * AAIMC_SRAM_512b_W_POWER/512;
        }

        if (DRAM_ENABLE==1){
          imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * output_rem * DRAM_512b_DMA_POWER/512;
        }
        else{
          imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * output_rem * SRAM_512b_DMA_POWER/512;
        }
        
        //bias_scale = 21;

        imc0_computing = !imc0_computing;
        //linear layer compute << weight upload
        for (int k = 0; k < input_itr; k += 1) {
          refresh();

          //input index in AMEM
          offset = amem_read_base + k;
          memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
          memcpy( &input_buffer[0], &input_mux_in[0], sizeof(int32_t) * SP_ROW_SIZE);
          ping_pong_computing_input_feeding_cycles += AMEM_read_latency_cycles;
          if (PP_IMC_ENABLE==0){
            layer_cycles += AMEM_read_latency_cycles;
            leakage(AMEM_read_latency_cycles);
            refresh();
          }
          amem_read_energy += AMEM_read_latency_cycles * SP_ROW_SIZE * AMEM_128_R_POWER/128;
          imc_computing_energy += mvm_latency_cycles * SP_ROW_SIZE * AAIMC_SRAM_128IN_POWER /128;

          if (input_itr==1) {
            if (bias_existing)
              bias_enable[0] = 1;
            else 
              bias_enable[0] = 0;
            compute();
            ping_pong_computing_imc_cycles += mvm_latency_cycles;
            if (PP_IMC_ENABLE==0){
              layer_cycles += mvm_latency_cycles;
              leakage(mvm_latency_cycles);
              refresh();
            }
            
            for (int n = 0; n < SP_ROW_SIZE;n += 1){
              for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                if (mi==0)
                  output_buffer[n] = output_vector[mi][n];
                else
                  output_buffer[n] += output_vector[mi][n];
              }
            }

            memcpy( &amem_0[(row_index+j) * SP_ROW_SIZE], &output_buffer[0], sizeof(int32_t) * output_rem);
            amem_row_cycles_0[(row_index+j)] = 0;
            ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
            if (PP_IMC_ENABLE==0){
              layer_cycles += AMEM_write_latency_cycles;
              leakage(AMEM_write_latency_cycles);
              refresh();
            }
            if (SRAM_IMC==1){
              imc_computing_energy += mvm_latency_cycles * output_rem * AAIMC_SRAM_128O_POWER /128;
            }
            else{
              imc_computing_energy += mvm_latency_cycles * output_rem * AAIMC_AMEM_32O_POWER/32;
            }
            amem_write_energy += AMEM_write_latency_cycles * output_rem * AMEM_128_W_POWER/128;

          }
          else {
            if (k==0) {
              if (bias_existing)
                bias_enable[0] = 1;

              compute();
              ping_pong_computing_imc_cycles += mvm_latency_cycles;
              if (PP_IMC_ENABLE==0){
                layer_cycles += mvm_latency_cycles;
                leakage(mvm_latency_cycles);
                refresh();
              }

              for (int n = 0; n < MACRO_DIM;n += 1){
                for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                  if (mi==0)
                    output_buffer[n] = output_vector[mi][n];
                  else
                    output_buffer[n] += output_vector[mi][n];
                }
              }

              memcpy( &amem_0[(IMC_AMEM_TEMP_BASE+j) * SP_ROW_SIZE], &output_buffer[0], sizeof(int32_t) * output_rem);
              amem_row_cycles_0[(IMC_AMEM_TEMP_BASE+j)] = 0;
              ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
              if (PP_IMC_ENABLE==0){
                layer_cycles += AMEM_write_latency_cycles;
                leakage(AMEM_write_latency_cycles);
                refresh();
              }
              if (SRAM_IMC==1){
                imc_computing_energy += mvm_latency_cycles * output_rem * AAIMC_SRAM_128O_POWER /128;
              }
              else{
                imc_computing_energy += mvm_latency_cycles * output_rem * AAIMC_AMEM_32O_POWER/32;
              }
              amem_write_energy += AMEM_write_latency_cycles * output_rem * AMEM_128_W_POWER/128;
              

              for (int n = 0; n < SP_ROW_SIZE; n++) {
                dma(dma_read_base+((k+1)*output_rem+j)* SP_ROW_SIZE+n*output_rem);
                ping_pong_loading_cycles += IMC_write_latency_cycles;
                layer_cycles += IMC_write_latency_cycles;
                leakage((IMC_write_latency_cycles));
                refresh();
                
                if (imc0_computing) {
                  memcpy( &imc_array_1[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * output_rem);
                }
                else {
                  memcpy( &imc_array_0[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * output_rem);
                }
              }
              ping_pong_loading_cycles += DMA_LOADING_CYCLE;// pipeline DMA and IMC write, latency = cycle +1;
              layer_cycles += DMA_LOADING_CYCLE;
              leakage((DMA_LOADING_CYCLE));
              refresh();

              if (SRAM_IMC==0){
                imc_loading_energy += SP_ROW_SIZE*IMC_write_latency_cycles * AAIMC_PRECISION * output_rem * AAIMC_AMEM_32x4b_W_POWER/128;
              }
              else{
                imc_loading_energy += SP_ROW_SIZE*IMC_write_latency_cycles * AAIMC_PRECISION * output_rem * AAIMC_SRAM_512b_W_POWER/512;
              }

              if (DRAM_ENABLE==1){
                imc_loading_energy += SP_ROW_SIZE*DMA_LOADING_CYCLE * AAIMC_PRECISION * output_rem * DRAM_512b_DMA_POWER/512;
              }
              else{
                imc_loading_energy += SP_ROW_SIZE*DMA_LOADING_CYCLE * AAIMC_PRECISION * output_rem * SRAM_512b_DMA_POWER/512;
              }

            }
            else {
              bias_enable[0] = 0;
              compute();
              ping_pong_computing_imc_cycles += mvm_latency_cycles;
              if (PP_IMC_ENABLE==0){
                layer_cycles += mvm_latency_cycles;
                leakage(mvm_latency_cycles);
                refresh();
              }

              for (int n = k*MACRO_DIM; n < (k+1)*MACRO_DIM;n += 1){
                for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                  if (mi==0)
                    output_buffer[n] = output_vector[mi][n];
                  else
                    output_buffer[n] += output_vector[mi][n];
                }
              }
                //output_buffer[(n-k*MACRO_DIM)] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n]+output_vector[3][n];

              //printf("input1 of ANU: first=%d last=%d\n", amem_0[(IMC_AMEM_TEMP_BASE+1) * SP_ROW_SIZE], amem_0[(IMC_AMEM_TEMP_BASE+1) * SP_ROW_SIZE + 9]);
              //printf("input2 of ANU: first=%d last=%d\n", amem_0[(IMC_AMEM_TEMP_BASE+0) * SP_ROW_SIZE], amem_0[(IMC_AMEM_TEMP_BASE+0) * SP_ROW_SIZE + 9]);
              if (k==input_itr-1){
                for (int l = 0; l < output_rem; ++l) {
                  amem_0[(row_index+j)*SP_ROW_SIZE+ l] = amem_0[(IMC_AMEM_TEMP_BASE+j)*SP_ROW_SIZE+ l] + output_buffer[l];
                }
              }
              else {
                for (int l = 0; l < output_rem; ++l) {
                  amem_0[(IMC_AMEM_TEMP_BASE+j)*SP_ROW_SIZE+ l] = amem_0[(IMC_AMEM_TEMP_BASE+j)*SP_ROW_SIZE+ l] + output_buffer[l];
                }
              }
              ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
              if (PP_IMC_ENABLE==0){
                layer_cycles += AMEM_write_latency_cycles;
                leakage(AMEM_write_latency_cycles);
                refresh();
              }
              if (SRAM_IMC==1){
                imc_computing_energy += mvm_latency_cycles * output_rem * AAIMC_SRAM_128O_POWER /128;
              }
              else{
                imc_computing_energy += mvm_latency_cycles * output_rem * AAIMC_AMEM_32O_POWER/32;
              }
              amem_write_energy += AMEM_write_latency_cycles * output_rem * AMEM_128_W_POWER/128;
              amem_read_energy += AMEM_read_latency_cycles * output_rem * AMEM_128_R_POWER/128;
              

              if (k==input_itr-1&& j==output_itr-1){
                
              }
              else{
                for (int n = 0; n < SP_ROW_SIZE; n++) {
                  dma(dma_read_base+((k+1)*output_rem+j)* SP_ROW_SIZE+n*output_rem);
                  ping_pong_loading_cycles += IMC_write_latency_cycles;
                  layer_cycles += IMC_write_latency_cycles;
                  leakage((IMC_write_latency_cycles));
                  refresh();
                  
                  if (imc0_computing) {
                    memcpy( &imc_array_1[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * output_rem);
                  }
                  else {
                    memcpy( &imc_array_0[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * output_rem);
                  }
                }
                ping_pong_loading_cycles += DMA_LOADING_CYCLE;// pipeline DMA and IMC write, latency = cycle +1;
                layer_cycles += DMA_LOADING_CYCLE;
                leakage((DMA_LOADING_CYCLE));
                refresh();

                if (SRAM_IMC==0){
                  imc_loading_energy += SP_ROW_SIZE*IMC_write_latency_cycles * AAIMC_PRECISION * output_rem * AAIMC_AMEM_32x4b_W_POWER/128;
                }
                else{
                  imc_loading_energy += SP_ROW_SIZE*IMC_write_latency_cycles * AAIMC_PRECISION * output_rem * AAIMC_SRAM_512b_W_POWER/512;
                }

                if (DRAM_ENABLE==1){
                  imc_loading_energy += SP_ROW_SIZE*DMA_LOADING_CYCLE * AAIMC_PRECISION * output_rem * DRAM_512b_DMA_POWER/512;
                }
                else{
                  imc_loading_energy += SP_ROW_SIZE*DMA_LOADING_CYCLE * AAIMC_PRECISION * output_rem * SRAM_512b_DMA_POWER/512;
                }
              }
              //printf("output of ANU: first=%d last=%d\n", amem_0[(IMC_AMEM_TEMP_BASE+0) * SP_ROW_SIZE], amem_0[(IMC_AMEM_TEMP_BASE+0) * SP_ROW_SIZE + 9]);
            }
          }
        }
      }

      //output trace
      dma_cycles += ping_pong_loading_cycles;
      imc_cycles += ping_pong_computing_imc_cycles;
      amem_r_cycles += ping_pong_computing_input_feeding_cycles;
      amem_w_cycles += ping_pong_computing_output_wb_cycles;
      total_layer_cycles += layer_cycles;
      total_cycles = total_layer_cycles+refresh_cycles;
      //linear layer ony compute once, not even start pipeline
      ping_pong_computing_cycles = ping_pong_computing_imc_cycles + ping_pong_computing_input_feeding_cycles + ping_pong_computing_output_wb_cycles;
      
      printf("Data loading cycle= %lu\n", dma_cycles);
      printf("IMC cycle= %lu\n", imc_cycles);
      printf("AMEM read cycle= %lu\n", amem_r_cycles);
      printf("AMEM write cycle= %lu\n", amem_w_cycles);
      printf("refresh cycle= %lu\n", refresh_cycles);
      printf("Total cycle= %lu\n", total_cycles);
      //area
      if (SRAM_IMC==1){
        imc_area = ((1+PP_IMC_ENABLE) * IMC_ARRAY_SIZE + FUSED_IMC_ARRAY_SIZE) * AAIMC_PRECISION * AAIMC_4b_SRAM_128kb_AREA/(128*1024);
      }
      else{
        imc_area = ((1+PP_IMC_ENABLE) * IMC_ARRAY_SIZE + FUSED_IMC_ARRAY_SIZE) * AAIMC_PRECISION * AAIMC_4b_AMEM_2kc_AREA/(2*1024*4);
      }
      amem_area = (AMEM_ARRAY_SIZE + FUSED_AMEM_ARRAY_SIZE) * AMEM_PRECISION * AMEM_64k_AREA/(64*1024*6);
      anu_area = IMC_DIM * MAX_POOL_ANU_128O_AREA/128; 
      abus_area = IMC_DIM * ABUS_256_AREA/256; 
      sram_buffer_area = ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_AREA/(5152*1024);
      total_area = imc_area + amem_area + anu_area + abus_area + sram_buffer_area + ADC_AREA + DAC_AREA*MACRO_DIM  + CTRL_FSM_AREA + RISCV32_AREA * RISCV_ENABLE;
      printf("Total area= %lu um^2\n", total_area*9*9/100);// 0.9 cale in 28nm

      //energy
      amem_leakage_energy += ((layer_cycles) * AMEM_ARRAY_SIZE + layer_cycles*FUSED_AMEM_ARRAY_SIZE) * AMEM_64K_Leakage_POWER/(64*1024);

      if (SRAM_IMC==1){
        imc_leakage_energy += (PP_IMC_ENABLE*((layer_cycles-ping_pong_computing_imc_cycles) * IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024)
                              + (layer_cycles-ping_pong_loading_cycles) * IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024))
                              + layer_cycles * FUSED_IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024));
      }
      else{
        imc_leakage_energy += (PP_IMC_ENABLE*((layer_cycles-ping_pong_computing_imc_cycles) * IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024)
                              + (layer_cycles-ping_pong_loading_cycles) * IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024))
                              + layer_cycles * FUSED_IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024));
      }
      //anu_computing_energy = ping_pong_computing_anu_cycles * IMC_DIM * MAX_POOL_ANU_128O_POWER/128;
      sram_leakage_energy_in_refresh = refresh_cycles * ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_Leakage_POWER/(5152*1024)
                                        + SRAM_IMC * ((1+PP_IMC_ENABLE)* IMC_ARRAY_SIZE + FUSED_IMC_ARRAY_SIZE)* AAIMC_PRECISION * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024);

      sram_buffer_leakage_energy += ((layer_cycles) * ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_Leakage_POWER)/(5152*1024);

      fsm_energy = total_cycles * CTRL_FSM_POWER;
      //risc_energy = total_cycles * RISCV32_POWER;
      total_energy = amem_read_energy + amem_write_energy + amem_leakage_energy + refresh_energy+
                    imc_computing_energy + anu_computing_energy + imc_loading_energy + sram_leakage_energy_in_refresh+
                    imc_leakage_energy + sram_buffer_leakage_energy + fsm_energy + ADDA_energy;
      printf("Total energy= %lu pJ\n", total_energy*PERIOD/1000);
      printf("Average energy efficiency = %lu GOPS/W\n", OPs*1000000/(total_energy*PERIOD));


      ie = 1;
      update_interrupts();

      return true;
    }

    case CONV_LAYER:{
      printf("conv layer start\n");
      imc0_computing = true; 
      for (int h = 0; h < output_row;h += 1) {
        for (int i = 0; i < output_col;i += 1) {
          for (int j = 0; j < output_itr;j += 1) {
            // Load bias from L2CACHE -> SP; 
            //memcpy( &sp_weight[(sp_weight_write_base+output_itr*input_itr*IMC_DIM+j) * SP_ROW_SIZE], &l2_cache[(l2cache_read_base+output_itr*input_itr*IMC_DIM+j) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
            //dma_cycles += row_latency_cycles;

            // Load bias from SP -> IMC_row
            memcpy( &bias_row_0[0], &sp_weight[(sp_weight_write_base+output_itr*weight_col*weight_row*input_itr*IMC_DIM+j) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
            if (h==0 && i==0)
              printf("bias_row: first=%d 10th=%d last=%d\n", bias_row_0[0], bias_row_0[9], bias_row_0[SP_ROW_SIZE-1]);
            dma_cycles += IMC_write_latency_cycles;

            if (h==0){
              k_start = 1;
              k_end = weight_row;
            }
            else if (h==output_row-1){
              k_start = 0;
              k_end = weight_row-1;
            }
            else {
              k_start = 0;
              k_end = weight_row;
            }

            if (i==0){
              l_start = 1;
              l_end = weight_col;
            }
            else if (i==output_col-1){
              l_start = 0;
              l_end = weight_col-1;
            }
            else {
              l_start = 0;
              l_end = weight_col;
            }

            for (int k = k_start; k < k_end; k += 1) {
              for (int l = l_start; l < l_end; l += 1) {
                for (int m = 0; m < input_itr; m += 1) {

                  // Load weights from SP -> IMC_array
                  for (int n = 0; n < IMC_DIM; n++) { // Only store 3 rows(first layer) just for debug
                    memcpy( &imc_array_0[n * SP_ROW_SIZE], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*IMC_DIM+k*weight_row*input_itr*IMC_DIM+l*input_itr*IMC_DIM+m*IMC_DIM+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                    dma_cycles += IMC_write_latency_cycles;
                  }

                  if (h==1 && i==1)
                    printf("%d row %d col pixel of W: first=%d 2nd=%d 3rd=%d last=%d\n", k, l, imc_array_0[0 * SP_ROW_SIZE + SP_ROW_SIZE-1], imc_array_0[1 * SP_ROW_SIZE+SP_ROW_SIZE-1], imc_array_0[2 * SP_ROW_SIZE+SP_ROW_SIZE-1], imc_array_0[(SP_ROW_SIZE-1) * SP_ROW_SIZE+SP_ROW_SIZE-1]);

                  // selected input window from (i-1,h-1) to (i+1, h+1)
                  // TODO: fixed parameter of stride and kernel size for now --> Parameterize later
                  offset = amem_read_base + (h-1+k)*output_row*input_itr + (i-1+l)*input_itr + m;

                  if (k==k_start&&l==l_start&&m==0) {
                    bias_enable[0] = 0;
                    compute();
                    memcpy( &amem_0[(IMC_AMEM_TEMP_BASE) * SP_ROW_SIZE], output_vector.data(), sizeof(int32_t) * SP_ROW_SIZE);
                    dma_cycles += AMEM_write_latency_cycles;
                  }
                  else if (k==k_end-1&&l==l_end-1&&m==input_itr-1) {
                    if (bias_existing)
                      bias_enable[0] = 1;
                    compute();
                    memcpy( &amem_0[(IMC_AMEM_TEMP_BASE+1) * SP_ROW_SIZE], output_vector.data(), sizeof(int32_t) * SP_ROW_SIZE);
                    dma_cycles += AMEM_write_latency_cycles;
                    //printf("input1 of ANU: first=%d last=%d\n", amem_0[(IMC_AMEM_TEMP_BASE+1) * SP_ROW_SIZE], amem_0[(IMC_AMEM_TEMP_BASE+1) * SP_ROW_SIZE + 9]);
                    //printf("input2 of ANU: first=%d last=%d\n", amem_0[(IMC_AMEM_TEMP_BASE+0) * SP_ROW_SIZE], amem_0[(IMC_AMEM_TEMP_BASE+0) * SP_ROW_SIZE + 9]);
                    for (int x = 0; x < SP_ROW_SIZE; ++x) {
                      amem_0[(row_index+h*output_row*output_itr+i*output_itr+j)*SP_ROW_SIZE+x] = amem_0[(IMC_AMEM_TEMP_BASE)*SP_ROW_SIZE+ x] + amem_0[(IMC_AMEM_TEMP_BASE+1)* IMC_DIM + x];
                    }
                    anu_cycles += anu_latency_cycles;
                    dma_cycles += AMEM_write_latency_cycles;
                    if (h==output_col-1 && i==output_row-1 && j==output_itr-1) {
                      printf("Data loading cycle= %d\n", dma_cycles);
                      printf("IMC cycle= %d\n", imc_cycles);
                      printf("ANU cycle= %d\n", anu_cycles);
                      ie = 1;
                      update_interrupts();
                    }
                    //printf("output of ANU: first=%d last=%d\n", amem_0[(row_index+j)*SP_ROW_SIZE], amem_0[(row_index+j)*SP_ROW_SIZE + 9]);
                  }                  
                  else {
                    bias_enable[0] = 0;
                    compute();
                    memcpy( &amem_0[(IMC_AMEM_TEMP_BASE+1) * SP_ROW_SIZE], output_vector.data(), sizeof(int32_t) * SP_ROW_SIZE);
                    dma_cycles += AMEM_write_latency_cycles;
                    //printf("input1 of ANU: first=%d last=%d\n", amem_0[(IMC_AMEM_TEMP_BASE+1) * SP_ROW_SIZE], amem_0[(IMC_AMEM_TEMP_BASE+1) * SP_ROW_SIZE + 9]);
                    //printf("input2 of ANU: first=%d last=%d\n", amem_0[(IMC_AMEM_TEMP_BASE+0) * SP_ROW_SIZE], amem_0[(IMC_AMEM_TEMP_BASE+0) * SP_ROW_SIZE + 9]);
                    for (int x = 0; x < SP_ROW_SIZE; ++x) {
                      amem_0[(IMC_AMEM_TEMP_BASE)*SP_ROW_SIZE+ x] = amem_0[(IMC_AMEM_TEMP_BASE)*SP_ROW_SIZE+ x] + amem_0[(IMC_AMEM_TEMP_BASE+1)* IMC_DIM + x];
                    }
                    anu_cycles += anu_latency_cycles;
                    dma_cycles += AMEM_write_latency_cycles;
                    //printf("output of ANU: first=%d last=%d\n", amem_0[(IMC_AMEM_TEMP_BASE+0) * SP_ROW_SIZE], amem_0[(IMC_AMEM_TEMP_BASE+0) * SP_ROW_SIZE + 9]);
                  }
                }
              }
            }
          }
        }
      }
      return true;
    }

    case AVGPOOL_LAYER:{
      printf("avg pool layer start\n");
      pool_enable = true;
      for (int h = 0; h < output_row;h += 1) {
        for (int i = 0; i < output_col;i += 1) {
          for (int j = 0; j < output_itr;j += 1) {
            refresh();
            int row_length = (j==output_itr-1)&&output_rem ? output_rem : SP_ROW_SIZE;
            if (output_rem >0 ){
              for (int l = 0; l < row_length; ++l) {
                //amem_0[(row_index+h*output_row*output_itr + i*output_itr + j)*SP_ROW_SIZE+ l] = (amem_0[(amem_read_base + 2*h*output_row*output_itr + 2*i*output_itr + j)*SP_ROW_SIZE+ l] + amem_0[(amem_read_base + 2*h*output_row*output_itr + (2*i+1)*output_itr + j)* IMC_DIM + l] + amem_0[(amem_read_base + (2*h+1)*output_row*output_itr + 2*i*output_itr + j)*SP_ROW_SIZE+ l] + amem_0[(amem_read_base + (2*h+1)*output_row*output_itr + (2*i+1)*output_itr + j)* IMC_DIM + l])/4;
                amem_0[(IMC_AMEM_TEMP_BASE)*SP_ROW_SIZE+ l] = (amem_0[amem_read_base*SP_ROW_SIZE+(2*h*2*output_col + 2*i)*((output_itr-1)*SP_ROW_SIZE+output_rem)+j*SP_ROW_SIZE+ l] + amem_0[amem_read_base*SP_ROW_SIZE+(2*h*2*output_col + 2*i+1)*((output_itr-1)*SP_ROW_SIZE+output_rem)+j*SP_ROW_SIZE+ l])/2;
                amem_0[(IMC_AMEM_TEMP_BASE+1)*SP_ROW_SIZE+ l] = (amem_0[amem_read_base*SP_ROW_SIZE+((2*h+1)*2*output_col + 2*i)*((output_itr-1)*SP_ROW_SIZE+output_rem)+j*SP_ROW_SIZE+ l] + amem_0[amem_read_base*SP_ROW_SIZE+((2*h+1)*2*output_col + 2*i+1)*((output_itr-1)*SP_ROW_SIZE+output_rem)+j*SP_ROW_SIZE+ l])/2;
                amem_0[row_index*SP_ROW_SIZE+(h*output_col + i)*((output_itr-1)*SP_ROW_SIZE+output_rem)+j*SP_ROW_SIZE+l] = (amem_0[(IMC_AMEM_TEMP_BASE)*SP_ROW_SIZE+ l] + amem_0[(IMC_AMEM_TEMP_BASE+1)* SP_ROW_SIZE + l])/2;
              }
            }

            else{
              for (int l = 0; l < SP_ROW_SIZE; ++l) {
              //amem_0[(row_index+h*output_row*output_itr + i*output_itr + j)*SP_ROW_SIZE+ l] = (amem_0[(amem_read_base + 2*h*output_row*output_itr + 2*i*output_itr + j)*SP_ROW_SIZE+ l] + amem_0[(amem_read_base + 2*h*output_row*output_itr + (2*i+1)*output_itr + j)* IMC_DIM + l] + amem_0[(amem_read_base + (2*h+1)*output_row*output_itr + 2*i*output_itr + j)*SP_ROW_SIZE+ l] + amem_0[(amem_read_base + (2*h+1)*output_row*output_itr + (2*i+1)*output_itr + j)* IMC_DIM + l])/4;
                amem_0[(IMC_AMEM_TEMP_BASE)*SP_ROW_SIZE+ l] = (amem_0[amem_read_base*SP_ROW_SIZE+(2*h*2*output_col + 2*i)*output_itr*SP_ROW_SIZE+j*SP_ROW_SIZE+ l] + amem_0[amem_read_base*SP_ROW_SIZE+(2*h*2*output_col + 2*i+1)*output_itr*SP_ROW_SIZE+j*SP_ROW_SIZE+ l])/2;
                amem_0[(IMC_AMEM_TEMP_BASE+1)*SP_ROW_SIZE+ l] = (amem_0[amem_read_base*SP_ROW_SIZE+((2*h+1)*2*output_col + 2*i)*output_itr*SP_ROW_SIZE+j*SP_ROW_SIZE+ l] + amem_0[amem_read_base*SP_ROW_SIZE+((2*h+1)*2*output_col + 2*i+1)*output_itr*SP_ROW_SIZE+j*SP_ROW_SIZE+ l])/2;
                amem_0[row_index*SP_ROW_SIZE+(h*output_col + i)*output_itr*SP_ROW_SIZE+j*SP_ROW_SIZE+l] = (amem_0[(IMC_AMEM_TEMP_BASE)*SP_ROW_SIZE+ l] + amem_0[(IMC_AMEM_TEMP_BASE+1)* SP_ROW_SIZE + l])/2;
              }
            }

            amem_r_cycles += 3*AMEM_read_latency_cycles;
            leakage((3*AMEM_read_latency_cycles));
            refresh();
            //anu done after 2nd read, and pipeline with wb
            total_layer_cycles += 3*AMEM_read_latency_cycles;
            anu_cycles += 3*anu_latency_cycles;
            amem_w_cycles += 3*AMEM_write_latency_cycles;
            if (h==output_row-1 && i==output_col-1 && j==output_itr-1){
              /*total_cycles += 3*anu_latency_cycles;
              leakage(3*anu_latency_cycles);
              refresh();*/
              total_layer_cycles += 1*AMEM_write_latency_cycles;
              leakage(3*AMEM_write_latency_cycles);
              refresh();
            }

            total_cycles = total_layer_cycles+refresh_cycles;

            amem_read_energy +=  3*AMEM_read_latency_cycles * row_length * AMEM_128_R_POWER/128;
            amem_write_energy += 3*AMEM_write_latency_cycles * row_length * AMEM_128_W_POWER/128;
            anu_computing_energy += 3*anu_latency_cycles * row_length * ANU_128O_POWER/128;
          }
        }
      }
      
      for (int i=0; i<4*MACRO_DIM; i++)
        printf("%d ", amem_0[(row_index+(output_row-4)*output_col*output_itr + (output_col-6)*output_itr + 0)*SP_ROW_SIZE + i]);
      printf("\n");

      pool_enable = false;

      //printf("output of ANU: first=%d last=%d\n", amem_0[(row_index+13*output_col*output_itr + 13*output_itr + 0)*SP_ROW_SIZE], amem_0[(row_index+13*output_col*output_itr + 13*output_itr + 0)*SP_ROW_SIZE + 2*MACRO_DIM-1]);
      return true;
    }

    case MAXPOOL_LAYER:{
      printf("max pool layer start\n");
      printf("input_itr=%d, output_itr=%d, output_rem=%d, amem_read_base=%d, output_col=%d, output_row=%d\n", input_itr, output_itr, output_rem, amem_read_base, output_col, output_row);
      pool_enable = true;
      for (int h = 0; h < output_row;h += 1) {
        for (int i = 0; i < output_col;i += 1) {
          for (int j = 0; j < output_itr;j += 1) {
            refresh();
            int row_length = (j==output_itr-1)&&output_rem ? output_rem : SP_ROW_SIZE;
            if (output_rem >0 ){
              for (int l = 0; l < row_length; ++l) {
                amem_0[row_index*SP_ROW_SIZE+(h*output_col + i)*((output_itr-1)*SP_ROW_SIZE+output_rem)+j*SP_ROW_SIZE+l] = std::max({amem_0[amem_read_base*SP_ROW_SIZE+(2*h*2*output_col + 2*i)*((output_itr-1)*SP_ROW_SIZE+output_rem)+j*SP_ROW_SIZE+ l], amem_0[amem_read_base*SP_ROW_SIZE+(2*h*2*output_col + 2*i+1)*((output_itr-1)*SP_ROW_SIZE+output_rem)+j*SP_ROW_SIZE+ l], amem_0[amem_read_base*SP_ROW_SIZE+((2*h+1)*2*output_col + 2*i)*((output_itr-1)*SP_ROW_SIZE+output_rem)+j*SP_ROW_SIZE+ l], amem_0[amem_read_base*SP_ROW_SIZE+((2*h+1)*2*output_col + 2*i+1)*((output_itr-1)*SP_ROW_SIZE+output_rem)+j*SP_ROW_SIZE+ l]});
              }
              if (h==0&&i==0){
                printf("%d %d %d %d max=%d\n", amem_0[amem_read_base*SP_ROW_SIZE + (2*h*2*output_col*output_itr + 2*i*output_itr + j)*output_rem + 0], amem_0[amem_read_base*SP_ROW_SIZE + (2*h*2*output_col*output_itr + (2*i+1)*output_itr + j)*output_rem + 0], amem_0[amem_read_base*SP_ROW_SIZE + ((2*h+1)*2*output_col*output_itr + 2*i*output_itr + j)*output_rem + 0], amem_0[amem_read_base*SP_ROW_SIZE + ((2*h+1)*2*output_col*output_itr + (2*i+1)*output_itr + j)*output_rem + 0], std::max({amem_0[amem_read_base*SP_ROW_SIZE + (2*h*2*output_col*output_itr + 2*i*output_itr + j)*output_rem+ 0], amem_0[amem_read_base*SP_ROW_SIZE + (2*h*2*output_col*output_itr + (2*i+1)*output_itr + j)*output_rem+ 0], amem_0[amem_read_base*SP_ROW_SIZE + ((2*h+1)*2*output_col*output_itr + 2*i*output_itr + j)*output_rem+ 0], amem_0[amem_read_base*SP_ROW_SIZE + ((2*h+1)*2*output_col*output_itr + (2*i+1)*output_itr + j)*output_rem+ 0]}));
              }
            }

            else{
              for (int l = 0; l < SP_ROW_SIZE; ++l) {
                amem_0[row_index*SP_ROW_SIZE+(h*output_col + i)*output_itr*SP_ROW_SIZE+j*SP_ROW_SIZE+l] = std::max({amem_0[amem_read_base*SP_ROW_SIZE+(2*h*2*output_col + 2*i)*output_itr*SP_ROW_SIZE+j*SP_ROW_SIZE+ l], amem_0[amem_read_base*SP_ROW_SIZE+(2*h*2*output_col + 2*i+1)*output_itr*SP_ROW_SIZE+j*SP_ROW_SIZE+ l], amem_0[amem_read_base*SP_ROW_SIZE+((2*h+1)*2*output_col + 2*i)*output_itr*SP_ROW_SIZE+j*SP_ROW_SIZE+ l], amem_0[amem_read_base*SP_ROW_SIZE+((2*h+1)*2*output_col + 2*i+1)*output_itr*SP_ROW_SIZE+j*SP_ROW_SIZE+ l]});
              }
            }
            amem_r_cycles += 2*AMEM_read_latency_cycles;
            leakage((2*AMEM_read_latency_cycles));
            refresh();
            anu_cycles += 1*anu_latency_cycles;

            total_layer_cycles += (2*AMEM_read_latency_cycles);
            //pipeline wb and read/anu
            amem_w_cycles += 1*AMEM_write_latency_cycles;
            if (h==output_row-1 && i==output_col-1 && j==output_itr-1){
              total_layer_cycles += 1*AMEM_write_latency_cycles;
              leakage(1*AMEM_write_latency_cycles);
              refresh();
            }

            total_cycles = total_layer_cycles+refresh_cycles;

            amem_read_energy +=  2*AMEM_read_latency_cycles * row_length * AMEM_128_R_POWER/128;
            amem_write_energy += AMEM_write_latency_cycles * row_length * AMEM_128_W_POWER/128;
            anu_computing_energy += anu_latency_cycles * row_length * MAX_POOL_ANU_128O_POWER/128;
          }
        }
      }
      
      for (int i=0; i<4*MACRO_DIM; i++)
        printf("%d ", amem_0[(row_index+(output_row-4)*output_col*output_itr + (output_col-6)*output_itr + 0)*SP_ROW_SIZE + i]);
      printf("\n");

      pool_enable = false;

      //printf("output of ANU: first=%d last=%d\n", amem_0[(row_index+13*output_col*output_itr + 13*output_itr + 0)*SP_ROW_SIZE], amem_0[(row_index+13*output_col*output_itr + 13*output_itr + 0)*SP_ROW_SIZE + 2*MACRO_DIM-1]);
      return true;
    }

    case REORG_CAT:{
      printf("reorg&cat layer start\n");
      printf("input_itr=%d, output_itr=%d, output_rem=%d, amem_read_base=%d, amem_read_base2=%d, output_col=%d, output_row=%d\n", input_itr, output_itr, output_rem, amem_read_base, amem_read_base2, output_col, output_row);
      for (int h = 0; h < output_row;h += 1) {
        for (int i = 0; i < output_col;i += 1) {
          for (int j = 0; j < 4*input_itr + output_itr;j += 1) {
            //refresh();
            if (j<4*input_itr){
               for (int l = 0; l < SP_ROW_SIZE; ++l) {
                amem_0[IMC_AMEM_TEMP_BASE*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+l] = amem_0[amem_read_base*SP_ROW_SIZE+((2*h+(j/input_itr)/2)*2*output_col + 2*i+(j/input_itr)%2)*input_itr*SP_ROW_SIZE+(j%input_itr)*SP_ROW_SIZE+ l];
              }
              if(h==0&&i==0)
                printf("read row=%d, col=%d ", (2*h+(j/input_itr)/2), 2*i+(j/input_itr)%2);
            }

            else{
              for (int l = 0; l < SP_ROW_SIZE; ++l) {
                amem_0[IMC_AMEM_TEMP_BASE*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+l] = amem_0[amem_read_base2*SP_ROW_SIZE+(h*output_col + i)*output_itr*SP_ROW_SIZE+(j-4*input_itr)*SP_ROW_SIZE+l];
              }
              if(h==0&&i==0){
                printf("%d, %d, %d, %d ", amem_0[amem_read_base2*SP_ROW_SIZE+(h*output_col + i)*output_itr*SP_ROW_SIZE+(j-4*input_itr)*SP_ROW_SIZE+0], amem_0[amem_read_base2*SP_ROW_SIZE+(h*output_col + i)*output_itr*SP_ROW_SIZE+(j-4*input_itr)*SP_ROW_SIZE+1], amem_0[amem_read_base2*SP_ROW_SIZE+(h*output_col + i)*output_itr*SP_ROW_SIZE+(j-4*input_itr)*SP_ROW_SIZE+2], amem_0[amem_read_base2*SP_ROW_SIZE+(h*output_col + i)*output_itr*SP_ROW_SIZE+(j-4*input_itr)*SP_ROW_SIZE+3]);
                printf("%d, %d, %d, %d\n", amem_0[IMC_AMEM_TEMP_BASE*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+0], amem_0[IMC_AMEM_TEMP_BASE*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+1], amem_0[IMC_AMEM_TEMP_BASE*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+2], amem_0[IMC_AMEM_TEMP_BASE*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+3]);
              }
            }
            // REORG_CAT could be implemented by changing the index during conv input feeding, so no delay
            /*dma_cycles += 1*AMEM_read_latency_cycles;
            dma_cycles += 1*AMEM_write_latency_cycles;

            leakage(4*AMEM_read_latency_cycles);
            leakage(1*AMEM_write_latency_cycles);*/
          }
        }
      }

      for (int h = 0; h < output_row;h += 1) {
        for (int i = 0; i < output_col;i += 1) {
          for (int j = 0; j < 4*input_itr + output_itr;j += 1) {
            for (int l = 0; l < SP_ROW_SIZE; ++l) {
              amem_0[row_index*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+l] = amem_0[IMC_AMEM_TEMP_BASE*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+l];
            }
            if(h==0&&i==0)
              printf("%d, %d, %d, %d\n", amem_0[row_index*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+0], amem_0[row_index*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+1], amem_0[row_index*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+2], amem_0[row_index*SP_ROW_SIZE+(h*output_col + i)*(4*input_itr + output_itr)*SP_ROW_SIZE+j*SP_ROW_SIZE+3]);
          }
        }
      }

      return true;
    }

    case ADD_LAYER:{
      printf("add layer start\n");
      for (int h = 0; h < output_row;h += 1) {
        for (int i = 0; i < output_col;i += 1) {
          for (int j = 0; j < output_itr;j += 1) {
            refresh();
            int row_length = (j==output_itr-1)&&output_rem ? output_rem : SP_ROW_SIZE;
            for (int l = 0; l < row_length; ++l) {
              //amem_0[(row_index+h*output_row*output_itr + i*output_itr + j)*SP_ROW_SIZE+ l] = (amem_0[(amem_read_base + 2*h*output_row*output_itr + 2*i*output_itr + j)*SP_ROW_SIZE+ l] + amem_0[(amem_read_base + 2*h*output_row*output_itr + (2*i+1)*output_itr + j)* IMC_DIM + l] + amem_0[(amem_read_base + (2*h+1)*output_row*output_itr + 2*i*output_itr + j)*SP_ROW_SIZE+ l] + amem_0[(amem_read_base + (2*h+1)*output_row*output_itr + (2*i+1)*output_itr + j)* IMC_DIM + l])/4;
              amem_0[(IMC_AMEM_TEMP_BASE)*SP_ROW_SIZE+ l] = (amem_0[(amem_read_base + h*output_col*output_itr + i*output_itr + j)*SP_ROW_SIZE+ l] + amem_0[(row_index + h*output_col*output_itr + i*output_itr + j)* IMC_DIM + l]);
              amem_0[(row_index+h*output_col*output_itr + i*output_itr + j)*SP_ROW_SIZE+ l] = amem_0[(IMC_AMEM_TEMP_BASE)*SP_ROW_SIZE+ l];
            }
            amem_r_cycles += 1*AMEM_read_latency_cycles;
            leakage((1*AMEM_read_latency_cycles));
            refresh();
            //pipeline read/anu and wb
            total_layer_cycles += 1*AMEM_read_latency_cycles;
            anu_cycles += 1*anu_latency_cycles;
            amem_w_cycles += 1*AMEM_write_latency_cycles;
            if (h==output_row-1 && i==output_col-1 && j==output_itr-1){
              /*total_cycles += 1*anu_latency_cycles;
              leakage(1*anu_latency_cycles);
              refresh();*/
              total_layer_cycles += 1*AMEM_write_latency_cycles;
              leakage(1*AMEM_write_latency_cycles);
              refresh();
            }

            total_cycles = total_layer_cycles+refresh_cycles;

            amem_read_energy +=  1*AMEM_read_latency_cycles * row_length * AMEM_128_R_POWER/128;
            amem_write_energy += 1*AMEM_write_latency_cycles * row_length * AMEM_128_W_POWER/128;
            anu_computing_energy += 1*anu_latency_cycles * row_length * ANU_128O_POWER/128;
          }
        }
      }
      
      for (int i=2*MACRO_DIM; i<4*MACRO_DIM; i++)
        printf("%d ", amem_0[(row_index+(output_row-2)*output_col*output_itr + (output_col-2)*output_itr + 0)*SP_ROW_SIZE + i]);
      printf("\n");
      printf("output of ANU: first=%d last=%d\n", amem_0[row_index* SP_ROW_SIZE], amem_0[row_index* SP_ROW_SIZE + SP_ROW_SIZE-1]);
      return true;
    }


    case ReLU_LAYER:{
      printf("ReLU layer start\n");
      for (int h = 0; h < output_row;h += 1) {
        for (int i = 0; i < output_col;i += 1) {
          for (int j = 0; j < output_itr;j += 1) {
            refresh();
            for (int l = 0; l < SP_ROW_SIZE; ++l) {
              if (amem_0[(row_index+h*output_col*output_itr + i*output_itr + j)*SP_ROW_SIZE+ l] > (pow(2,AMEM_PRECISION) - 1))
                amem_0[(row_index+h*output_col*output_itr + i*output_itr + j)*SP_ROW_SIZE+ l] = pow(2,AMEM_PRECISION) - 1;
              else if (amem_0[(row_index+h*output_col*output_itr + i*output_itr + j)*SP_ROW_SIZE+ l] > 0)
                amem_0[(row_index+h*output_col*output_itr + i*output_itr + j)*SP_ROW_SIZE+ l] = amem_0[(row_index+h*output_col*output_itr + i*output_itr + j)*SP_ROW_SIZE+ l];
              else
                amem_0[(row_index+h*output_col*output_itr + i*output_itr + j)*SP_ROW_SIZE+ l] = 0;
            }
            // May not need if performed directly after conv layer
            //dma_cycles += 1*AMEM_read_latency_cycles;
            //anu_cycles += 1*anu_latency_cycles;
            //dma_cycles += 1*AMEM_write_latency_cycles;
            //leakage(AMEM_read_latency_cycles);
            //leakage(anu_latency_cycles);
            //leakage(AMEM_write_latency_cycles);
          }
        }
      }
      printf("output of ANU: first=%d last=%d output_col=%d output_row=%d output_itr=%d\n", amem_0[row_index* SP_ROW_SIZE], amem_0[row_index* SP_ROW_SIZE + SP_ROW_SIZE-1], output_col, output_row, output_itr);
      printf("offset=%d\n", (row_index+(output_row-4)*output_col*output_itr + (output_col-6)*output_itr + 0));
      for (int i=0*MACRO_DIM; i<4*MACRO_DIM; i++)
        printf("%d ", amem_0[(row_index+(0)*output_col*output_itr + (0)*output_itr + 0)*SP_ROW_SIZE + i]);
      printf("\n");
      //for (int i=3*MACRO_DIM; i<4*MACRO_DIM; i++)
      //  printf("%d ", amem_0[(row_index+(output_row-1)*output_col*output_itr + (output_col-1)*output_itr + 0)*SP_ROW_SIZE + i]);
      //printf("\n");
      return true;
    }

    // CASE A: 2*IMC_ARRAY for ping-pong weight updaing and computing, max resuing for weight. 
    // for concurrent weight updating and computing cycle modeling, to avoid record leakage twice, we precompute per array loading cycle and computing cycle to estimate which one is dominant.
    // Thanks to 3R1W AMEM design, we could pipeline input feeding, computing and output write back to reduce total latency.
    case CONV_LAYER_A:{
      printf("conv layer start\n");
      ping_pong_computing_cycles = 0;
      ping_pong_loading_cycles = 0;
      ping_pong_computing_imc_cycles = 0;
      ping_pong_computing_input_feeding_cycles = 0;
      ping_pong_computing_output_wb_cycles = 0;
      layer_cycles = 0;
      conv_enable=true;

      //Load weight if first chunk of Lary-by-layer, IMC can be preloaded, so not count cycle and energy.
      if (load_first_weight){
        int dma_weight_base = dma_read_base;
        int row_length = (output_itr==1&&output_rem>0) ? output_rem : SP_ROW_SIZE;
        int dma_row_index = (input_itr==1&&input_rem>0) ? input_rem : SP_ROW_SIZE; 
        for (int n = 0; n < dma_row_index; n++) {
          if (output_rem>0)
            dma(dma_weight_base + n * ((output_itr-1)*SP_ROW_SIZE+output_rem));
          else
            dma(dma_weight_base + n*output_itr*SP_ROW_SIZE);
          
          memcpy( &imc_array_0[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
          /*if (SRAM_IMC==0){
            layer_cycles += IMC_write_latency_cycles;
            dma_cycles += IMC_write_latency_cycles;
            leakage(IMC_write_latency_cycles);
            refresh();
          }*/
        }
        imc0_computing = false;
        /*if (SRAM_IMC==0){
          layer_cycles += DMA_LOADING_CYCLE;
          dma_cycles += DMA_LOADING_CYCLE;
          leakage(DMA_LOADING_CYCLE);
          refresh();
        }*/
        load_first_weight=false;
      }

      printf("input_itr=%d, output_itr=%d, input_rem=%d, output_rem=%d, weight_col=%d, weight_row=%d, output_col=%d, output_row=%d, amem_scale=%d, bias_scale=%d\n", input_itr, output_itr, input_rem, output_rem, weight_col, weight_row, output_col, output_row, amem_scale, bias_scale);
      for (int j = 0; j < output_itr;j += 1) {
        // Load bias from MEM -> IMC_row
        int dma_weight_base;
        if (input_rem>0&&output_rem>0)
          dma_weight_base = dma_read_base+ (weight_col*weight_row)*((input_itr-1)*SP_ROW_SIZE+input_rem)*((output_itr-1)*SP_ROW_SIZE+output_rem)+ j*SP_ROW_SIZE;
        else if (input_rem>0)
          dma_weight_base = dma_read_base+ (weight_col*weight_row)*((input_itr-1)*SP_ROW_SIZE+input_rem)*output_itr*SP_ROW_SIZE+ j*SP_ROW_SIZE;
        else if (output_rem>0)
          dma_weight_base = dma_read_base+ (weight_col*weight_row)*input_itr*SP_ROW_SIZE*((output_itr-1)*SP_ROW_SIZE+output_rem)+ j*SP_ROW_SIZE;
        else
          dma_weight_base = dma_read_base+ (weight_col*weight_row)*input_itr*SP_ROW_SIZE*output_itr*SP_ROW_SIZE+ j*SP_ROW_SIZE;
        
        int row_length = (j==output_itr-1&&output_rem>0) ? output_rem : SP_ROW_SIZE;
        dma(dma_weight_base);
        memcpy( &bias_row_0[0], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
        
        layer_cycles += (IMC_write_latency_cycles);
        dma_cycles += (IMC_write_latency_cycles);
        leakage(IMC_write_latency_cycles);
        refresh();
        if (SRAM_IMC==0){
          imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
        }
        else{
          imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_SRAM_512b_W_POWER/512;
        }

        if (DRAM_ENABLE==1){
          imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
        }
        else{
          imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
        }
        printf("bias_row: first=%d 11th=%d last=%d\n", bias_row_0[0], bias_row_0[10], bias_row_0[SP_ROW_SIZE-1]);

        for (int m = 0; m < input_itr; m += 1) {
          for (int k = 0; k < weight_row; k += 1) {
            for (int l = 0; l < weight_col; l += 1) {

              int dma_row_index; // how many dma repats in this chunk.
              if (k==weight_col-1&&l==weight_row-1&&m==input_itr-1&&j==output_itr-1){
                dma_row_index = (output_itr==1&&output_rem>0) ? output_rem : SP_ROW_SIZE; 
              }
              else {
                if (input_rem>0)
                  dma_row_index = ((k==weight_col-1&&l==weight_row-1&&m==input_itr-2)||m==input_itr-1) ? input_rem : SP_ROW_SIZE;
                else
                  dma_row_index = SP_ROW_SIZE;
              }

              if (k<padding){
                k_start = padding-k;
                k_end = output_row;
              }
              else if (k>weight_row-padding-1){
                k_start = 0;
                k_end = output_row-(k-weight_row+padding+1);
              }
              else {
                k_start = 0;
                k_end = output_row;
              }

              if (l<padding){
                l_start = padding-l;
                l_end = output_col;
              }
              else if (l>weight_col-padding-1){
                l_start = 0;
                l_end = output_col-(l-weight_col+padding+1);
              }
              else {
                l_start = 0;
                l_end = output_col;
              }

              // non ping-pong IMC should be implemented independently. For now we just change the cycle count. Will do if have time.
              if (PP_IMC_ENABLE==1){
                // compare dma cycle and computing cycle
                // 3 stage pipeline total latency = (output_pixels + 2) * mvm_latency_cycles
                if ((dma_row_index*IMC_write_latency_cycles+DMA_LOADING_CYCLE) > ((k_end-k_start)*(l_end-l_start)+2)*PIPELINE_STAGE_LATENCY){
                  dma_dominate = true;
                  layer_cycles += dma_row_index*IMC_write_latency_cycles+DMA_LOADING_CYCLE;
                }
                else{
                  dma_dominate = false;
                  layer_cycles += ((k_end-k_start)*(l_end-l_start)+2)*PIPELINE_STAGE_LATENCY;
                }
              }
              else{
                layer_cycles += (dma_row_index*IMC_write_latency_cycles+DMA_LOADING_CYCLE + ((k_end-k_start)*(l_end-l_start)+2)*PIPELINE_STAGE_LATENCY);
              }
              //dma_dominate = false;
              //layer_cycles += ((k_end-k_start)*(l_end-l_start)+2)*PIPELINE_STAGE_LATENCY;

              ping_pong_computing_cycles +=((k_end-k_start)*(l_end-l_start)+2)*PIPELINE_STAGE_LATENCY;

              // Load weights from SP -> IMC_array
              //imc0_computing = ((m+k+l) % 2 == 0) ? true : false;
              imc0_computing = !imc0_computing;

              for (int h = k_start; h < k_end;h += 1) {
                for (int i = l_start; i < l_end;i += 1) {

                  refresh();
                  
                  offset = (h*stride+k-(weight_row-1)/2)*output_col*stride + (i*stride+l-(weight_col-1)/2);
                  if (input_rem>0){
                    if (m==input_itr-1){
                      memcpy( &input_mux_in[0], &amem_0[amem_read_base* SP_ROW_SIZE + offset*((input_itr-1)*SP_ROW_SIZE+input_rem)+m*SP_ROW_SIZE], sizeof(int32_t) * input_rem);
                      amem_read_energy += AMEM_read_latency_cycles * input_rem * AMEM_128_R_POWER/128;
                      imc_computing_energy += mvm_latency_cycles * input_rem * AAIMC_SRAM_128IN_POWER /128;
                      for (int n = 0; n < SP_ROW_SIZE;n += 1)
                        if (n<input_rem)
                          input_buffer[n] = input_mux_in[n];
                        else
                          input_buffer[n] = 0;
                    }
                    else {
                      memcpy( &input_mux_in[0], &amem_0[amem_read_base* SP_ROW_SIZE + offset*((input_itr-1)*SP_ROW_SIZE+input_rem)+m*SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                      memcpy( &input_buffer[0], &input_mux_in[0], sizeof(int32_t) * SP_ROW_SIZE);
                      amem_read_energy += AMEM_read_latency_cycles * SP_ROW_SIZE * AMEM_128_R_POWER/128;
                      imc_computing_energy += mvm_latency_cycles * SP_ROW_SIZE * AAIMC_SRAM_128IN_POWER /128;
                    }
                  }
                  else {
                    memcpy( &input_mux_in[0], &amem_0[amem_read_base* SP_ROW_SIZE + offset*input_itr*SP_ROW_SIZE+m*SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                    memcpy( &input_buffer[0], &input_mux_in[0], sizeof(int32_t) * SP_ROW_SIZE);
                    amem_read_energy += AMEM_read_latency_cycles * SP_ROW_SIZE * AMEM_128_R_POWER/128;
                    imc_computing_energy += mvm_latency_cycles * SP_ROW_SIZE * AAIMC_SRAM_128IN_POWER /128;
                  }
                  
                  ping_pong_computing_input_feeding_cycles += AMEM_read_latency_cycles;
                  if (((!dma_dominate)&&i == l_start&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                    leakage((AMEM_read_latency_cycles));
                    refresh();
                  }


                  int row_length = j==output_itr-1&&output_rem ? output_rem : SP_ROW_SIZE;
                  if ((k==0&&l==0&&m==0)||(k==0&&l==padding&&m==0&&i==0)||(k==padding&&l==0&&m==0&&h==0)||(k==padding&&l==padding&&m==0&&h==0&&i==0)) { // only for padding=0/1
                    if (weight_col==1&&weight_row==1&&input_itr==1){
                      for (int n = 1; n < ((IMC_DIM/MACRO_DIM));n += 1)
                        bias_enable[n] = 0;

                      if (bias_existing)
                        bias_enable[0] = 1;
                      else 
                        bias_enable[0] = 0;

                      compute();
                      //imc_cycles += mvm_latency_cycles;//imc cycle is hidden by ping-pong
                      //ping_pong_computing_imc_cycles += mvm_latency_cycles;
                      ping_pong_computing_imc_cycles += mvm_latency_cycles;
                      if (((!dma_dominate)&&i == l_start&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                        leakage((mvm_latency_cycles));
                        refresh();
                      }

                      if (output_rem>0){
                        for (int n = 0; n < row_length;n += 1){
                          for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                            if (mi==0)
                              output_buffer[n] = output_vector[mi][n];
                            else
                              output_buffer[n] += output_vector[mi][n];
                          }
                          amem_0[row_index*IMC_DIM + (h*output_col+i)*((output_itr-1)*IMC_DIM+output_rem)+j*IMC_DIM+ n] = (output_buffer[n])/amem_scale;
                        }
                        if ((h*output_col+i)%(IMC_DIM/output_rem)==0)
                          amem_row_cycles_0[row_index + (h*output_col+i)*(output_itr-1)+j+ (h*output_col+i)*output_rem/IMC_DIM] = 0;
                      }
                      else {
                        for (int n = 0; n < IMC_DIM;n += 1){
                          for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                            if (mi==0)
                              output_buffer[n] = output_vector[mi][n];
                            else
                              output_buffer[n] += output_vector[mi][n];
                          }
                          amem_0[row_index*IMC_DIM + (h*output_col+i)*output_itr*IMC_DIM+j*IMC_DIM+ n] = (output_buffer[n])/amem_scale;
                        }
                        amem_row_cycles_0[row_index + (h*output_col+i)*output_itr+j] = 0;
                      }
                      
                      /*if (h==4&&i==4){
                        //printf("input buffer first pixel=%d second=%d last=%d\n", input_buffer[MACRO_DIM], input_buffer[MACRO_DIM+1], input_buffer[MACRO_DIM+MACRO_DIM-1]);
                        printf("output value in 1=%d 2=%d 3=%d buffer=%d\n",  output_vector[0][0], output_vector[1][0], output_vector[2][0], output_buffer[0]);
                      }*/
                      ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
                      if (((!dma_dominate)&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                        leakage((AMEM_write_latency_cycles));
                        refresh();
                      }
                    }

                    else{
                      for (int n = 0; n < ((IMC_DIM/MACRO_DIM));n += 1)
                        bias_enable[n] = 0;

                      compute();
                      //imc_cycles += mvm_latency_cycles;//imc cycle is hidden by ping-pong
                      ping_pong_computing_imc_cycles += mvm_latency_cycles;
                      if (((!dma_dominate)&&i == l_start&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                        leakage((mvm_latency_cycles));
                        refresh();
                      }

                      if (output_rem>0){
                        for (int n = 0; n < row_length;n += 1){
                          for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                            if (mi==0)
                              output_buffer[n] = output_vector[mi][n];
                            else
                              output_buffer[n] += output_vector[mi][n];
                          }
                        }
                        memcpy( &amem_0[IMC_AMEM_TEMP_BASE*IMC_DIM + (h*output_col+i)*((output_itr-1)*IMC_DIM+output_rem)+j*IMC_DIM ], &output_buffer[0], sizeof(int32_t) * row_length);
                        if ((h*output_col+i)%(IMC_DIM/output_rem)==0)
                          amem_row_cycles_0[IMC_AMEM_TEMP_BASE + (h*output_col+i)*(output_itr-1)+j+ (h*output_col+i)*output_rem/IMC_DIM] = 0;
                      }
                      else {
                        for (int n = 0; n < IMC_DIM;n += 1){
                          for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                            if (mi==0)
                              output_buffer[n] = output_vector[mi][n];
                            else
                              output_buffer[n] += output_vector[mi][n];
                          }
                        }
                        memcpy( &amem_0[IMC_AMEM_TEMP_BASE*IMC_DIM + (h*output_col+i)*output_itr*IMC_DIM+j*IMC_DIM ], &output_buffer[0], sizeof(int32_t) * IMC_DIM);
                        amem_row_cycles_0[IMC_AMEM_TEMP_BASE + (h*output_col+i)*output_itr+j] = 0;
                      }
                      
                      
                      ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
                      if (((!dma_dominate)&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                        leakage((AMEM_write_latency_cycles));
                        refresh();
                      }
                      //dma_cycles += AMEM_write_latency_cycles;//psum writing cycle is hidden by ping-pong
                      //leakage((AMEM_write_latency_cycles+mvm_latency_cycles));
                    }
                  }

                  else {
                    if ((k==weight_col-1&&l==weight_row-1&&m==input_itr-1)||(k==weight_col-1&&l==weight_row-1-padding&&m==input_itr-1&&i==output_row-1)||(k==weight_col-1-padding&&l==weight_row-1&&m==input_itr-1&&h==output_col-1)||(k==weight_col-1-padding&&l==weight_row-1-padding&&m==input_itr-1&&h==output_col-1&&i==output_row-1)) {
                      if (bias_existing)
                        bias_enable[0] = 1;
                      else 
                        bias_enable[0] = 0;

                      compute();
                      ping_pong_computing_imc_cycles += mvm_latency_cycles;
                      if (((!dma_dominate)&&i == l_start&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                        leakage((mvm_latency_cycles));
                        refresh();
                      }

                      if (output_rem>0){
                        for (int n = 0; n < row_length;n += 1){
                          for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                            if (mi==0)
                              output_buffer[n] = output_vector[mi][n];
                            else
                              output_buffer[n] += output_vector[mi][n];
                          }
                          amem_0[row_index*IMC_DIM + (h*output_col+i)*((output_itr-1)*IMC_DIM+output_rem)+j*IMC_DIM+ n] = (amem_0[IMC_AMEM_TEMP_BASE*IMC_DIM + (h*output_col+i)*((output_itr-1)*IMC_DIM+output_rem)+j*IMC_DIM+ n] + output_buffer[n])/amem_scale;
                        }
                        if ((h*output_col+i)%(IMC_DIM/output_rem)==0)
                          amem_row_cycles_0[row_index + (h*output_col+i)*(output_itr-1)+j+ (h*output_col+i)*output_rem/IMC_DIM] = 0;
                        
                        amem_read_energy += AMEM_read_latency_cycles * row_length * AMEM_128_R_POWER/128;
                      }
                      else {
                        for (int n = 0; n < IMC_DIM;n += 1){
                          for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                            if (mi==0)
                              output_buffer[n] = output_vector[mi][n];
                            else
                              output_buffer[n] += output_vector[mi][n];
                          }
                          amem_0[row_index*IMC_DIM + (h*output_col+i)*output_itr*IMC_DIM+j*IMC_DIM+ n] = (amem_0[IMC_AMEM_TEMP_BASE*IMC_DIM + (h*output_col+i)*output_itr*IMC_DIM+j*IMC_DIM+ n] + output_buffer[n])/amem_scale;
                        }
                        amem_row_cycles_0[row_index + (h*output_col+i)*output_itr+j] = 0;
                        amem_read_energy += AMEM_read_latency_cycles * IMC_DIM * AMEM_128_R_POWER/128;
                      }
                      ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
                      if (((!dma_dominate)&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                        leakage((AMEM_write_latency_cycles));
                        refresh();
                      }

                      
                    }
                    else{
                      bias_enable[0] = 0;
                      compute();
                      ping_pong_computing_imc_cycles += mvm_latency_cycles;
                      if (((!dma_dominate)&&i == l_start&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                        leakage((mvm_latency_cycles));
                        refresh();
                      }

                      if (output_rem>0){
                        for (int n = 0; n < row_length;n += 1){
                          for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                            if (mi==0)
                              output_buffer[n] = output_vector[mi][n];
                            else
                              output_buffer[n] += output_vector[mi][n];
                          }
                          amem_0[IMC_AMEM_TEMP_BASE*IMC_DIM + (h*output_col+i)*((output_itr-1)*IMC_DIM+output_rem)+j*IMC_DIM+ n] = (amem_0[IMC_AMEM_TEMP_BASE*IMC_DIM + (h*output_col+i)*((output_itr-1)*IMC_DIM+output_rem)+j*IMC_DIM+ n] + output_buffer[n]);
                        }
                        if ((h*output_col+i)%(IMC_DIM/output_rem)==0)
                          amem_row_cycles_0[IMC_AMEM_TEMP_BASE + (h*output_col+i)*(output_itr-1)+j+ (h*output_col+i)*output_rem/IMC_DIM] = 0;
                        
                        amem_read_energy += AMEM_read_latency_cycles * row_length * AMEM_128_R_POWER/128;
                      }
                      else {
                        for (int n = 0; n < IMC_DIM;n += 1){
                          for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                            if (mi==0)
                              output_buffer[n] = output_vector[mi][n];
                            else
                              output_buffer[n] += output_vector[mi][n];
                          }
                          amem_0[IMC_AMEM_TEMP_BASE*IMC_DIM + (h*output_col+i)*output_itr*IMC_DIM+j*IMC_DIM+ n] = (amem_0[IMC_AMEM_TEMP_BASE*IMC_DIM + (h*output_col+i)*output_itr*IMC_DIM+j*IMC_DIM+ n] + output_buffer[n]);
                        }
                        amem_row_cycles_0[IMC_AMEM_TEMP_BASE + (h*output_col+i)*output_itr+j] = 0;
                        amem_read_energy += AMEM_read_latency_cycles * IMC_DIM * AMEM_128_R_POWER/128;
                      }

                      ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
                      if (((!dma_dominate)&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                        leakage((AMEM_write_latency_cycles));
                        refresh();
                      }

                    }
                      
                  }

                  if (SRAM_IMC==1){
                    imc_computing_energy += mvm_latency_cycles * row_length * AAIMC_SRAM_128O_POWER /128;
                  }
                  else{
                    imc_computing_energy += mvm_latency_cycles * row_length * AAIMC_AMEM_32O_POWER/32;
                  }
                  amem_write_energy += AMEM_write_latency_cycles * row_length * AMEM_128_W_POWER/128;
                }
              }

              if (k==weight_col-1&&l==weight_row-1&&m==input_itr-1&&j==output_itr-1){
                int dma_weight_base;
                if (input_rem>0&&output_rem>0)
                  dma_weight_base = dma_read_base+ (weight_col*weight_row)*((input_itr-1)*SP_ROW_SIZE+input_rem)*((output_itr-1)*SP_ROW_SIZE+output_rem)+ (output_itr-1)*SP_ROW_SIZE+output_rem;
                else if (input_rem>0)
                  dma_weight_base = dma_read_base+ (weight_col*weight_row)*((input_itr-1)*SP_ROW_SIZE+input_rem)*output_itr*SP_ROW_SIZE+ output_itr*SP_ROW_SIZE;
                else if (output_rem>0)
                  dma_weight_base = dma_read_base+ (weight_col*weight_row)*input_itr*SP_ROW_SIZE*((output_itr-1)*SP_ROW_SIZE+output_rem)+ (output_itr-1)*SP_ROW_SIZE+output_rem;
                else
                  dma_weight_base = dma_read_base+ (weight_col*weight_row)*input_itr*SP_ROW_SIZE*output_itr*SP_ROW_SIZE+ output_itr*SP_ROW_SIZE;
                
                //int dma_row_index = (output_itr==1&&output_rem>0) ? output_rem : SP_ROW_SIZE;
                int row_length = (next_conv_channel<IMC_DIM) ? next_conv_channel : SP_ROW_SIZE;
                for (int n = 0; n < dma_row_index; n++) {
                  dma(dma_weight_base + n * next_conv_channel);
                  ping_pong_loading_cycles += IMC_write_latency_cycles;
                  if ((dma_dominate&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                    leakage((IMC_write_latency_cycles));
                    refresh();
                  }
                  
                  if (imc0_computing) {
                    memcpy( &imc_array_1[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
                  }
                  else {
                    memcpy( &imc_array_0[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
                  }

                  if (SRAM_IMC==0){
                    imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                  }
                  else{
                    imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_SRAM_512b_W_POWER/512;
                  }

                  if (DRAM_ENABLE==1){
                    imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
                  }
                  else{
                    imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
                  }
                }
                ping_pong_loading_cycles += DMA_LOADING_CYCLE;
                if ((dma_dominate&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                  leakage((DMA_LOADING_CYCLE));
                  refresh();
                }


                // output trace
                dma_cycles += ping_pong_loading_cycles;
                imc_cycles += ping_pong_computing_imc_cycles;
                amem_r_cycles += ping_pong_computing_input_feeding_cycles;
                amem_w_cycles += ping_pong_computing_output_wb_cycles;
                total_layer_cycles += layer_cycles;
                total_cycles = total_layer_cycles+refresh_cycles;

                printf("Data loading cycle= %lu\n", dma_cycles);
                printf("IMC cycle= %lu\n", imc_cycles);
                printf("AMEM read cycle= %lu\n", amem_r_cycles);
                printf("AMEM write cycle= %lu\n", amem_w_cycles);
                printf("refresh cycle= %lu\n", refresh_cycles);
                printf("Total cycle= %lu\n", total_cycles);

                //area
                if (SRAM_IMC==1){
                  imc_area = ((1+PP_IMC_ENABLE) * IMC_ARRAY_SIZE + FUSED_IMC_ARRAY_SIZE) * AAIMC_PRECISION * AAIMC_4b_SRAM_128kb_AREA/(128*1024);
                }
                else{
                  imc_area = ((1+PP_IMC_ENABLE) * IMC_ARRAY_SIZE + FUSED_IMC_ARRAY_SIZE) * AAIMC_PRECISION * AAIMC_4b_AMEM_2kc_AREA/(2*1024*4);
                }
                amem_area = (AMEM_ARRAY_SIZE + FUSED_AMEM_ARRAY_SIZE) * AMEM_PRECISION * AMEM_64k_AREA/(64*1024*6);
                anu_area = IMC_DIM * MAX_POOL_ANU_128O_AREA/128; 
                abus_area = IMC_DIM * ABUS_256_AREA/256; 
                sram_buffer_area = ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_AREA/(5152*1024);
                total_area = imc_area + amem_area + anu_area + abus_area + sram_buffer_area + ADC_AREA + DAC_AREA*MACRO_DIM  + CTRL_FSM_AREA + RISCV32_AREA * RISCV_ENABLE;
                printf("Total area= %lu um^2\n", total_area*9*9/100);// 0.9 cale in 28nm

                //energy
                //amem_leakage_energy += ((layer_cycles- ping_pong_computing_cycles) * AMEM_ARRAY_SIZE + layer_cycles*FUSED_AMEM_ARRAY_SIZE) * AMEM_64K_Leakage_POWER/(64*1024);
                amem_leakage_energy += ((layer_cycles) * AMEM_ARRAY_SIZE + layer_cycles*FUSED_AMEM_ARRAY_SIZE) * AMEM_64K_Leakage_POWER/(64*1024);

                if (SRAM_IMC==1){
                  imc_leakage_energy += (PP_IMC_ENABLE*((layer_cycles-ping_pong_computing_imc_cycles) * IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024)
                                        + (layer_cycles-ping_pong_loading_cycles) * IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024))
                                        + layer_cycles * FUSED_IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024));
                }
                else{
                  imc_leakage_energy += (PP_IMC_ENABLE*((layer_cycles-ping_pong_computing_imc_cycles) * IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024)
                                        + (layer_cycles-ping_pong_loading_cycles) * IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024))
                                        + layer_cycles * FUSED_IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024));
                }
                //anu_computing_energy = ping_pong_computing_anu_cycles * IMC_DIM * MAX_POOL_ANU_128O_POWER/128;
                
                //sram_buffer_leakage_energy += ((layer_cycles-ping_pong_loading_cycles) * ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_Leakage_POWER)/(5152*1024);
                sram_buffer_leakage_energy += ((layer_cycles) * ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_Leakage_POWER)/(5152*1024);

                sram_leakage_energy_in_refresh = refresh_cycles * ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_Leakage_POWER/(5152*1024)
                                        + SRAM_IMC * ((1+PP_IMC_ENABLE)* IMC_ARRAY_SIZE + FUSED_IMC_ARRAY_SIZE)* AAIMC_PRECISION * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024);

                fsm_energy = total_cycles * CTRL_FSM_POWER;
                //risc_energy = total_cycles * RISCV32_POWER;
                total_energy = amem_read_energy + amem_write_energy + amem_leakage_energy + refresh_energy+
                              imc_computing_energy + anu_computing_energy + imc_loading_energy + sram_leakage_energy_in_refresh+
                              //imc_computing_energy + anu_computing_energy + sram_leakage_energy_in_refresh+
                              imc_leakage_energy + sram_buffer_leakage_energy + fsm_energy + ADDA_energy;

                printf("Total energy= %lu pJ\n", total_energy*PERIOD/1000);
                printf("Average energy efficiency = %lu GOPS/W\n", (OPs*1000000)/(total_energy*PERIOD));

                
                printf("amem row cycles first = %d ofm = %d ofm2 = %d last = %d\n", amem_row_cycles_0[0], amem_row_cycles_0[row_index], amem_row_cycles_0[row_index+2], amem_row_cycles_0[AMEM_ROW_INDEX-1]);
                printf("amem fist channel 1st result= %d 2nd = %d last-1 = %d last = %d\n", amem_0[(row_index+30*16+13)*SP_ROW_SIZE], amem_0[(row_index+30*16+13)*SP_ROW_SIZE+1], amem_0[(row_index+30*16+13)*SP_ROW_SIZE+SP_ROW_SIZE-2], amem_0[(row_index+30*16+13)*SP_ROW_SIZE+SP_ROW_SIZE-1]);
                for (int i=0*MACRO_DIM; i<2*MACRO_DIM; i++)
                  printf("%d ", amem_0[(row_index +0)*SP_ROW_SIZE+i]);
                printf("\n");
                ie = 1;
                update_interrupts();
              }
              else {
                int dma_weight_base;
                if (input_rem>0&&output_rem>0){
                  if (k==weight_col-1&&l==weight_row-1&&m==input_itr-1)
                    dma_weight_base = dma_read_base+ (0)*((input_itr-1)*SP_ROW_SIZE+input_rem)*((output_itr-1)*SP_ROW_SIZE+output_rem)+ (0)*SP_ROW_SIZE*((output_itr-1)*SP_ROW_SIZE+output_rem)+ (j+1)*SP_ROW_SIZE;
                  else if (k==weight_col-1&&l==weight_row-1)
                    dma_weight_base = dma_read_base+ (0)*((input_itr-1)*SP_ROW_SIZE+input_rem)*((output_itr-1)*SP_ROW_SIZE+output_rem)+ (m+1)*SP_ROW_SIZE*((output_itr-1)*SP_ROW_SIZE+output_rem)+ j*SP_ROW_SIZE;
                  else
                    dma_weight_base = dma_read_base+ (k*weight_row+l+1)*((input_itr-1)*SP_ROW_SIZE+input_rem)*((output_itr-1)*SP_ROW_SIZE+output_rem)+ m*SP_ROW_SIZE*((output_itr-1)*SP_ROW_SIZE+output_rem)+ j*SP_ROW_SIZE;

                  //int dma_row_index = ((k==weight_col-1&&l==weight_row-1&&m==input_itr-2)||m==input_itr-1) ? input_rem : SP_ROW_SIZE;
                  int row_length = ((k==weight_col-1&&l==weight_row-1&&m==input_itr-1&&j==output_itr-2)||j==output_itr-1) ? output_rem : SP_ROW_SIZE;
                  for (int n = 0; n < dma_row_index; n++) {
                    dma(dma_weight_base + n * ((output_itr-1)*SP_ROW_SIZE+output_rem));
                    ping_pong_loading_cycles += IMC_write_latency_cycles;
                    if ((dma_dominate&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                      leakage((IMC_write_latency_cycles));
                      refresh();
                    }
                    
                    if (imc0_computing) {
                      memcpy( &imc_array_1[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
                    }
                    else {
                      memcpy( &imc_array_0[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
                    }
                  }
                  ping_pong_loading_cycles += DMA_LOADING_CYCLE; // pipeline DMA and IMC write, latency = n*IMC_write_latency_cycles +1*DMA;
                  if ((dma_dominate&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                    leakage((DMA_LOADING_CYCLE));
                    refresh();
                  }
                }
                else if (input_rem>0){
                  if (k==weight_col-1&&l==weight_row-1&&m==input_itr-1)
                    dma_weight_base = dma_read_base+ (0)*((input_itr-1)*SP_ROW_SIZE+input_rem)*output_itr*SP_ROW_SIZE+ (0)*SP_ROW_SIZE*output_itr*SP_ROW_SIZE+ (j+1)*SP_ROW_SIZE;
                  else if (k==weight_col-1&&l==weight_row-1)
                    dma_weight_base = dma_read_base+ (0)*((input_itr-1)*SP_ROW_SIZE+input_rem)*output_itr*SP_ROW_SIZE+ (m+1)*SP_ROW_SIZE*output_itr*SP_ROW_SIZE+ j*SP_ROW_SIZE;
                  else
                    dma_weight_base = dma_read_base+ (k*weight_row+l+1)*((input_itr-1)*SP_ROW_SIZE+input_rem)*output_itr*SP_ROW_SIZE+ m*SP_ROW_SIZE*output_itr*SP_ROW_SIZE+ j*SP_ROW_SIZE;

                  int row_length = SP_ROW_SIZE;
                  //int dma_row_index = ((k==weight_col-1&&l==weight_row-1&&m==input_itr-2)||m==input_itr-1) ? input_rem : SP_ROW_SIZE;
                  for (int n = 0; n < dma_row_index; n++) {
                    dma(dma_weight_base + n*output_itr*SP_ROW_SIZE);
                    ping_pong_loading_cycles += IMC_write_latency_cycles;
                    if ((dma_dominate&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                      leakage((IMC_write_latency_cycles));
                      refresh();
                    }
                    
                    if (imc0_computing) {
                      memcpy( &imc_array_1[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                    }
                    else {
                      memcpy( &imc_array_0[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                    }
                  }
                  ping_pong_loading_cycles += DMA_LOADING_CYCLE;// pipeline DMA and IMC write, latency = cycle +1;
                  if ((dma_dominate&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                    leakage((DMA_LOADING_CYCLE));
                    refresh();
                  }
                }
                else if (output_rem>0){
                  if (k==weight_col-1&&l==weight_row-1&&m==input_itr-1)
                    dma_weight_base = dma_read_base+ (0)*input_itr*SP_ROW_SIZE*((output_itr-1)*SP_ROW_SIZE+output_rem)+ (0)*SP_ROW_SIZE*((output_itr-1)*SP_ROW_SIZE+output_rem)+ (j+1)*SP_ROW_SIZE;
                  else if (k==weight_col-1&&l==weight_row-1)
                    dma_weight_base = dma_read_base+ (0)*input_itr*SP_ROW_SIZE*((output_itr-1)*SP_ROW_SIZE+output_rem)+ (m+1)*SP_ROW_SIZE*((output_itr-1)*SP_ROW_SIZE+output_rem)+ j*SP_ROW_SIZE;
                  else
                    dma_weight_base = dma_read_base+ (k*weight_row+l+1)*input_itr*SP_ROW_SIZE*((output_itr-1)*SP_ROW_SIZE+output_rem)+ m*SP_ROW_SIZE*((output_itr-1)*SP_ROW_SIZE+output_rem)+ j*SP_ROW_SIZE;

                  int row_length = ((k==weight_col-1&&l==weight_row-1&&m==input_itr-1&&j==output_itr-2)||j==output_itr-1) ? output_rem : SP_ROW_SIZE;
                  for (int n = 0; n < SP_ROW_SIZE; n++) {
                    dma(dma_weight_base + n * ((output_itr-1)*SP_ROW_SIZE+output_rem));
                    ping_pong_loading_cycles += IMC_write_latency_cycles;
                    if ((dma_dominate&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                      leakage((IMC_write_latency_cycles));
                      refresh();
                    }
                    
                    if (imc0_computing) {
                      memcpy( &imc_array_1[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
                    }
                    else {
                      memcpy( &imc_array_0[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
                    }
                  }
                  ping_pong_loading_cycles += DMA_LOADING_CYCLE;// pipeline DMA and IMC write, latency = cycle +1;
                  if ((dma_dominate&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                    leakage((DMA_LOADING_CYCLE));
                    refresh();
                  }
                }
                else {
                  if (k==weight_col-1&&l==weight_row-1&&m==input_itr-1)
                    dma_weight_base = dma_read_base+ (0)*input_itr*SP_ROW_SIZE*output_itr*SP_ROW_SIZE+ (0)*SP_ROW_SIZE*output_itr*SP_ROW_SIZE+ (j+1)*SP_ROW_SIZE;
                  else if (k==weight_col-1&&l==weight_row-1)
                    dma_weight_base = dma_read_base+ (0)*input_itr*SP_ROW_SIZE*output_itr*SP_ROW_SIZE+ (m+1)*SP_ROW_SIZE*output_itr*SP_ROW_SIZE+ j*SP_ROW_SIZE;
                  else
                    dma_weight_base = dma_read_base+ (k*weight_row+l+1)*input_itr*SP_ROW_SIZE*output_itr*SP_ROW_SIZE+ m*SP_ROW_SIZE*output_itr*SP_ROW_SIZE+ j*SP_ROW_SIZE;

                  int row_length = SP_ROW_SIZE;
                  for (int n = 0; n < SP_ROW_SIZE; n++) {
                    dma(dma_weight_base + n * output_itr * SP_ROW_SIZE);
                    ping_pong_loading_cycles += IMC_write_latency_cycles;
                    if ((dma_dominate&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                      leakage((IMC_write_latency_cycles));
                      refresh();
                    }
                    
                    if (imc0_computing) {
                      memcpy( &imc_array_1[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                    }
                    else {
                      memcpy( &imc_array_0[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                    }
                  }
                  ping_pong_loading_cycles += DMA_LOADING_CYCLE;// pipeline DMA and IMC write, latency = cycle +1;
                  if ((dma_dominate&&PP_IMC_ENABLE)||PP_IMC_ENABLE==0){
                    leakage((DMA_LOADING_CYCLE));
                    refresh();
                  }
                }

                if (SRAM_IMC==0){
                  imc_loading_energy += dma_row_index * IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                }
                else{
                  imc_loading_energy += dma_row_index * IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_SRAM_512b_W_POWER/512;
                }

                if (DRAM_ENABLE==1){
                  imc_loading_energy += dma_row_index * DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
                }
                else{
                  imc_loading_energy += dma_row_index * DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
                }
              }
            }
          }
        }
      }
      conv_enable=false;
      return true;
    }

    // CASE A_OTHER: Case study of optimized mapping for Resnet-9.
    // Each conv layer includes parallel imc_computing (input loading, imc, anu, and amem writing) and weight loading
    // For leakage calculation in ping-pong architecture, we consider the max cycle between computing and weight loading as the total cycle for one conv layer. (conv2-5: computing cycles > weight loading cycles; conv6-8: weight loading cycles > computing cycles)
    case CONV_LAYER_A_OTHER:{
      printf("conv layer start\n");
      ping_pong_computing_cycles = 0;
      ping_pong_loading_cycles = 0;
      ping_pong_computing_imc_cycles = 0;
      ping_pong_computing_anu_cycles = 0;
      ping_pong_computing_dma_cycles = 0;
      conv_enable=true;

      bias_scale = BIAS_SCALE;

      
      printf("input_itr=%d, output_itr=%d, input_rem=%d, output_rem=%d, weight_col=%d, weight_row=%d, output_col=%d, output_row=%d\n", input_itr, output_itr, input_rem, output_rem, weight_col, weight_row, output_col, output_row);
      for (int j = 0; j < output_itr;j += 1) {

        if (input_rem==MACRO_DIM)
          //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+(output_itr*input_itr*weight_col-1)*weight_row*input_rem*output_rem/SP_ROW_SIZE+j) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
          dma(dma_read_base+((output_itr*input_itr*weight_col)*weight_row*input_rem*output_rem/SP_ROW_SIZE+j)* SP_ROW_SIZE);
        else if (input_rem==2*MACRO_DIM){
          if (output_rem>0)
            //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-2)*input_rem*output_rem/SP_ROW_SIZE+j) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
            dma(dma_read_base+((output_itr*input_itr*weight_col)*weight_row*input_rem*output_rem/SP_ROW_SIZE+j)* SP_ROW_SIZE);
          else{
            //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-2)*input_rem+j) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
            dma(dma_read_base+((output_itr*input_itr*weight_col)*weight_row*input_rem+j)* SP_ROW_SIZE);
            printf("offset=%d\n", (sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-2)*input_rem+j));
            printf("bias_row: first=%d 11th=%d last=%d\n", imc_w_mux_in[0], imc_w_mux_in[10], imc_w_mux_in[SP_ROW_SIZE-1]);
          }
        }
        else
          //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-1)*IMC_DIM+j) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
          dma(dma_read_base+((output_itr*input_itr*weight_col)*weight_row*IMC_DIM+j)* SP_ROW_SIZE);
        memcpy( &bias_row_0[0], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
        printf("bias_row: first=%d 11th=%d last=%d\n", bias_row_0[0], bias_row_0[10], bias_row_0[SP_ROW_SIZE-1]);
        dma_cycles += IMC_write_latency_cycles;// 4 blocks to improve parallelism, each block has bia row
        leakage(IMC_write_latency_cycles);

        for (int n = 0; n < ((IMC_DIM/MACRO_DIM));n += 1)
          bias_enable[n] = 0;

        for (int m = 0; m < input_itr; m += 1) {

          // conv layer 2
          // 3x3 conv divided to 3 groups
          // Group0: W00 W01 W02
          // Group1: W10 W11 W12
          // Group2: W20 W21 W22
          
          if (input_rem == MACRO_DIM) { 
            for (int k = 0; k < weight_row; k += 1) {

              if (k==0){
                k_start = 1;
                k_end = output_row;
              }
              else if (k==weight_row-1){
                k_start = 0;
                k_end = output_row-1;
              }
              else {
                k_start = 0;
                k_end = output_row;
              }


              // Ping-pong control
              //imc0_computing = ((m+k+l) % 2 == 0) ? true : false;
              imc0_computing = !imc0_computing;

              for (int h = k_start; h < k_end;h += 1) {
                for (int i = 0; i < output_col;i += 1) {

                  refresh();
                
                  // selected input window from (i-1,h-1) to (i+1, h+1)
                  
                  int amem_readout_index = i/(IMC_DIM/input_rem);

                  switch (i%(IMC_DIM/input_rem)) {
                    case 0:{
                      if (i==0){ // the input is always aligned to amem initial index(0) when i==0
                        offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + amem_readout_index*input_itr + m;
                        memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                        /*if (k==2&&h==0){
                          for (int n = 0*MACRO_DIM; n < 2*MACRO_DIM;n += 1)
                            printf("%d ", input_mux_in[n]);
                          printf("\n");
                        }*/
                        for (int n = 0; n < (SP_ROW_SIZE);n += 1)
                          if (n<MACRO_DIM)
                            input_buffer[n] = 0;
                          else if (n<3*MACRO_DIM)
                            input_buffer[n] = input_mux_in[n-MACRO_DIM];
                          else
                            input_buffer[n] = 0;
                        ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                        leakage((AMEM_read_latency_cycles));
                      }

                      else{
                        offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index-1)*input_itr + m;
                        memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                        for (int n = 0; n < (SP_ROW_SIZE);n += 1)
                          if (n<MACRO_DIM)
                            input_buffer[n] = input_mux_in[n+3*MACRO_DIM];
                          else
                            input_buffer[n] = 0;
                        ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                        leakage((AMEM_read_latency_cycles));

                        offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index)*input_itr + m;
                        memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                        for (int n = 0; n < (SP_ROW_SIZE);n += 1)
                          if (n<MACRO_DIM)
                            input_buffer[n] = input_buffer[n];
                          else if (n<3*MACRO_DIM)
                            input_buffer[n] = input_mux_in[n-MACRO_DIM];
                          else
                            input_buffer[n] = 0;
                        ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                        leakage((AMEM_read_latency_cycles));
                      }

                      break;
                    }

                    case 1:{
                      offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index)*input_itr + m;
                      memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                      for (int n = 0; n < (SP_ROW_SIZE);n += 1)
                        if (n<3*MACRO_DIM)
                          input_buffer[n] = input_mux_in[n];
                        else
                          input_buffer[n] = 0;
                      ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                      leakage((AMEM_read_latency_cycles));

                      break;
                    }

                    case 2:{
                      offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index)*input_itr + m;
                      memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                      for (int n = 0; n < (SP_ROW_SIZE);n += 1)
                        if (n<3*MACRO_DIM)
                          input_buffer[n] = input_mux_in[n+MACRO_DIM];
                        else
                          input_buffer[n] = 0;
                      ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                      leakage((AMEM_read_latency_cycles));

                      break;
                    }
                    
                    case 3:{
                      if (i==output_col-1){ 
                        offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + amem_readout_index*input_itr + m;
                        memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                        for (int n = 0; n < (SP_ROW_SIZE);n += 1)
                          if (n<2*MACRO_DIM)
                            input_buffer[n] = input_mux_in[n+2*MACRO_DIM];
                          else
                            input_buffer[n] = 0;
                        ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                        leakage((AMEM_read_latency_cycles));
                      }

                      else{
                        offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index)*input_itr + m;
                        memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                        for (int n = 0; n < (SP_ROW_SIZE);n += 1)
                          if (n<2*MACRO_DIM)
                            input_buffer[n] = input_mux_in[n+2*MACRO_DIM];
                          else
                            input_buffer[n] = 0;
                        ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                        leakage((AMEM_read_latency_cycles));
                        
                        if (k==2&&h==30&&i==27){
                          printf("offset=%d\n", offset);
                          for (int n = 2*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                            printf("%d ", amem_0[offset* SP_ROW_SIZE+n]);
                          printf("\n");
                        }

                        offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index+1)*input_itr + m;
                        memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                        for (int n = 0; n < (SP_ROW_SIZE);n += 1)
                          if (n<2*MACRO_DIM)
                            input_buffer[n] = input_buffer[n];
                          else if (n<3*MACRO_DIM)
                            input_buffer[n] = input_mux_in[n-2*MACRO_DIM];
                          else
                            input_buffer[n] = 0;
                        ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                        leakage((AMEM_read_latency_cycles));
                      }

                      break;
                    }

                    default:
                      // Other type stores are not supported
                      printf("Unsupported conv dimension:\n");
                      return false;
                  }

                  if ((k==0&&m==0)||(k==1&&m==0&&h==0)) {
                    bias_enable[0] = 0;
                    
                    compute();
                    //imc_cycles += mvm_latency_cycles;//imc cycle is hidden by ping-pong
                    ping_pong_computing_imc_cycles += mvm_latency_cycles;
                    leakage((mvm_latency_cycles));

                    for (int n = 0; n < output_rem;n += 1)
                      output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n];
                    //ping_pong_computing_anu_cycles += anu_latency_cycles; //acc output_vector of multi macro does not need ANU
                    //leakage((anu_latency_cycles));
                    memcpy( &amem_0[(IMC_AMEM_TEMP_BASE + h*(output_col*output_rem/IMC_DIM)*output_itr+(i*output_rem/IMC_DIM)*output_itr+j) * IMC_DIM + (i%(IMC_DIM/output_rem))*output_rem], &output_buffer[0], sizeof(int32_t) * output_rem);
                    ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                    leakage((AMEM_write_latency_cycles));

                    if (i%(IMC_DIM/output_rem)==0){
                      amem_row_cycles_0[IMC_AMEM_TEMP_BASE + h*(output_col*output_rem/IMC_DIM)*output_itr+(i*output_rem/IMC_DIM)*output_itr+j] = 0;
                    }

                    if (h==16&&i==4){
                      //printf("input buffer first pixel=%d second=%d last=%d\n", input_buffer[MACRO_DIM], input_buffer[MACRO_DIM+1], input_buffer[MACRO_DIM+MACRO_DIM-1]);
                      printf("output value in 1=%d 2=%d 3=%d buffer=%d\n",  output_vector[0][0], output_vector[1][0], output_vector[2][0], output_buffer[0]);
                      for (int n = 1*MACRO_DIM; n < 2*MACRO_DIM;n += 1)
                        printf("%d ", input_buffer[n]);
                      printf("\n");
                      if (imc0_computing)
                        //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_0[0], imc_array_0[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_0[MACRO_DIM*SP_ROW_SIZE], imc_array_0[2*MACRO_DIM*SP_ROW_SIZE]);
                        for (int n = 1*MACRO_DIM; n < 2*MACRO_DIM;n += 1)
                          printf("%d ", imc_array_0[(n)*SP_ROW_SIZE]);
                      else
                        //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_1[0], imc_array_1[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_1[MACRO_DIM*SP_ROW_SIZE], imc_array_1[2*MACRO_DIM*SP_ROW_SIZE]);
                        for (int n = 1*MACRO_DIM; n < 2*MACRO_DIM;n += 1)
                          printf("%d ", imc_array_1[(n)*SP_ROW_SIZE]);
                    }
                  }

                  else {
                    if ((k==weight_row-1&&m==input_itr-1)||(k==weight_row-2&&m==input_itr-1&&h==output_row-1)) {
                      if (bias_existing)
                        bias_enable[0] = 1;
                      else 
                        bias_enable[0] = 0;
                    }
                    else{
                      bias_enable[0] = 0;
                    }
                    
                    compute();
                    ping_pong_computing_imc_cycles += mvm_latency_cycles;
                    leakage((mvm_latency_cycles));

                    for (int n = 0; n < output_rem;n += 1)
                      output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n];
                    //ping_pong_computing_anu_cycles += anu_latency_cycles; //acc output_vector of multi macro does not need ANU
                    //leakage((anu_latency_cycles));
                    if ((k==weight_row-1&&m==input_itr-1)){
                      for (int x = 0; x < output_rem; ++x) 
                        amem_0[(row_index +h*(output_col*output_rem/IMC_DIM)*output_itr+(i*output_rem/IMC_DIM)*output_itr+j)*SP_ROW_SIZE+ (i%(IMC_DIM/output_rem))*output_rem + x] = (amem_0[(IMC_AMEM_TEMP_BASE +h*(output_col*output_rem/IMC_DIM)*output_itr+(i*output_rem/IMC_DIM)*output_itr+j)*SP_ROW_SIZE+ (i%(IMC_DIM/output_rem))*output_rem + x] + output_buffer[x])/amem_scale;
                      if (i%(IMC_DIM/output_rem)==0){
                        amem_row_cycles_0[row_index + h*(output_col*output_rem/IMC_DIM)*output_itr+(i*output_rem/IMC_DIM)*output_itr+j] = 0;
                      }
                    } 
                    else {
                      for (int x = 0; x < output_rem; ++x)
                        amem_0[(IMC_AMEM_TEMP_BASE +h*(output_col*output_rem/IMC_DIM)*output_itr+(i*output_rem/IMC_DIM)*output_itr+j)*SP_ROW_SIZE+ (i%(IMC_DIM/output_rem))*output_rem + x] = amem_0[(IMC_AMEM_TEMP_BASE + +h*(output_col*output_rem/IMC_DIM)*output_itr+(i*output_rem/IMC_DIM)*output_itr+j)*SP_ROW_SIZE+ (i%(IMC_DIM/output_rem))*output_rem + x] + output_buffer[x];
                      if (i%(IMC_DIM/output_rem)==0){
                        amem_row_cycles_0[IMC_AMEM_TEMP_BASE + h*(output_col*output_rem/IMC_DIM)*output_itr+(i*output_rem/IMC_DIM)*output_itr+j] = 0;
                      }
                    } 
                    
                    ping_pong_computing_anu_cycles += AMEM_read_latency_cycles;
                    ping_pong_computing_anu_cycles += anu_latency_cycles;
                    ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                    leakage((anu_latency_cycles));
                    leakage((AMEM_write_latency_cycles));

                    /*
                    if (i%2==0){
                      for (int n = 0; n < 2*MACRO_DIM;n += 1)
                        output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n];
                      ping_pong_computing_anu_cycles += anu_latency_cycles;
                      leakage((anu_latency_cycles));

                      if ((k==weight_col-1&&m==input_itr-1)){
                        for (int x = 0; x < 2*MACRO_DIM; ++x) 
                          amem_0[(row_index +h*(output_col/2)*output_itr+(i/2)*output_itr+j)*SP_ROW_SIZE+ x] = (amem_0[(IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j) * SP_ROW_SIZE+ x] + output_buffer[x])/AMEM_SCALE;
                      } 
                      else {
                        for (int x = 0; x < 2*MACRO_DIM; ++x)
                          amem_0[(IMC_AMEM_TEMP_BASE +h*(output_col/2)*output_itr+(i/2)*output_itr+j)*SP_ROW_SIZE+ x] = amem_0[(IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j) * SP_ROW_SIZE+ x] + output_buffer[x];
                      } 
                      ping_pong_computing_anu_cycles += AMEM_read_latency_cycles;
                      ping_pong_computing_anu_cycles += anu_latency_cycles;
                      ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                      leakage((anu_latency_cycles));
                      leakage((AMEM_write_latency_cycles));
                    } 

                    else{
                      for (int n = 0; n < 2*MACRO_DIM;n += 1)
                        output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n];
                      ping_pong_computing_anu_cycles += anu_latency_cycles;
                      leakage((anu_latency_cycles));

                      if ((k==weight_col-1&&m==input_itr-1)){
                        for (int x = 0; x < 2*MACRO_DIM; ++x) 
                          amem_0[(row_index +h*(output_col/2)*output_itr+(i/2)*output_itr+j)*SP_ROW_SIZE+2*MACRO_DIM+ x] = (amem_0[(IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j) * SP_ROW_SIZE+2*MACRO_DIM+ x] + output_buffer[x])/AMEM_SCALE;
                        amem_row_cycles_0[row_index +h*(output_col/2)*output_itr+(i/2)*output_itr+j] = 0;
                        //printf("output of ANU: first=%d last=%d\n", amem_0[(row_index +h*(output_col/2)*output_itr+(i/2)*output_itr+j)* SP_ROW_SIZE], amem_0[(row_index +h*(output_col/2)*output_itr+(i/2)*output_itr+j)* SP_ROW_SIZE + SP_ROW_SIZE-1]);
                      } 
                      else {
                        for (int x = 0; x < 2*MACRO_DIM; ++x)
                          amem_0[(IMC_AMEM_TEMP_BASE +h*(output_col/2)*output_itr+(i/2)*output_itr+j)*SP_ROW_SIZE+2*MACRO_DIM+ x] = amem_0[(IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j) * SP_ROW_SIZE+2*MACRO_DIM+ x] + output_buffer[x];
                        amem_row_cycles_0[IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j] = 0;
                      } 
                      ping_pong_computing_anu_cycles += AMEM_read_latency_cycles;
                      ping_pong_computing_anu_cycles += anu_latency_cycles;
                      ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                      leakage((anu_latency_cycles));
                      leakage((AMEM_write_latency_cycles));
                    }
                    */

                    if (h==16&&i==4){
                      //printf("input buffer first pixel=%d second=%d last=%d\n", input_buffer[MACRO_DIM], input_buffer[MACRO_DIM+1], input_buffer[MACRO_DIM+MACRO_DIM-1]);
                      printf("output value in 1=%d 2=%d 3=%d buffer=%d\n",  output_vector[0][0], output_vector[1][0], output_vector[2][0], output_buffer[0]);
                      for (int n = 1*MACRO_DIM; n < 2*MACRO_DIM;n += 1)
                        printf("%d ", input_buffer[n]);
                      printf("\n");
                      
                      if (imc0_computing)
                        //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_0[0], imc_array_0[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_0[MACRO_DIM*SP_ROW_SIZE], imc_array_0[2*MACRO_DIM*SP_ROW_SIZE]);
                        for (int n = 1*MACRO_DIM; n < 2*MACRO_DIM;n += 1)
                          printf("%d ", imc_array_0[(n)*SP_ROW_SIZE]);
                      else
                        //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_1[0], imc_array_1[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_1[MACRO_DIM*SP_ROW_SIZE], imc_array_1[2*MACRO_DIM*SP_ROW_SIZE]);
                        for (int n = 1*MACRO_DIM; n < 2*MACRO_DIM;n += 1)
                          printf("%d ", imc_array_1[(n)*SP_ROW_SIZE]);
                      printf("\n");
                    }

                  }
                }
              }

              if ((k==weight_row-1)&&(m==input_itr-1)) {// avoid the overwriting of amem for last row before finish using it
                for (int i = 0; i < output_col*output_rem/IMC_DIM;i += 1) {
                  for (int x = 0; x < SP_ROW_SIZE; ++x) {
                    //amem_0[(row_index +(output_row-1)*(output_col/2)*output_itr+i*output_itr+j)*SP_ROW_SIZE+ x] = (amem_0[(IMC_AMEM_TEMP_BASE + (output_row-1)*(output_col/2)*output_itr+i*output_itr+j) * SP_ROW_SIZE+ x])/AMEM_SCALE;
                    //amem_row_cycles_0[row_index +(output_row-1)*(output_col/2)*output_itr+i*output_itr+j] = 0;
                    amem_0[(row_index +(output_row-1)*(output_col*output_rem/IMC_DIM)*output_itr+(i*output_rem/IMC_DIM)*output_itr+j)*SP_ROW_SIZE + x] = (amem_0[(row_index +(output_row-1)*(output_col*output_rem/IMC_DIM)*output_itr+(i*output_rem/IMC_DIM)*output_itr+j)*SP_ROW_SIZE + x])/amem_scale;
                    amem_row_cycles_0[row_index +(output_row-1)*(output_col*output_rem/IMC_DIM)*output_itr+(i*output_rem/IMC_DIM)*output_itr+j] = 0;
                  }
                  
                  ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                  leakage((AMEM_write_latency_cycles));
                }
              }
              

              //ping-pong weight loading
              if (k==weight_col-1&&m==input_itr-1&&j==output_itr-1){
                if (output_rem == MACRO_DIM){
                  for (int n = 0; n < (3*MACRO_DIM*next_conv_channel/IMC_DIM); n++) {
                    //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+(output_itr*input_itr*weight_row-1)*weight_col*input_rem*output_rem/SP_ROW_SIZE+output_itr+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                    dma(dma_read_base+(output_itr*input_itr*weight_col*weight_row*input_rem*output_rem)+output_itr*output_rem+(n)* SP_ROW_SIZE);
                    for (int x = 0; x < (IMC_DIM/next_conv_channel); x++) {
                      refresh();
                      if (imc0_computing){
                        memcpy( &imc_array_1[((IMC_DIM/next_conv_channel)*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*next_conv_channel], sizeof(uint8_t) * next_conv_channel);
                        //printf("%d row pixel in W imc_array1: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_1[0 * SP_ROW_SIZE], imc_array_1[(0+1) * SP_ROW_SIZE], imc_array_1[(0+2) * SP_ROW_SIZE], imc_array_1[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
                      }
                      else {
                        memcpy( &imc_array_0[((IMC_DIM/next_conv_channel)*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*next_conv_channel], sizeof(uint8_t) * next_conv_channel);
                        //printf("%d row pixel in W imc_array0: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_0[0 * SP_ROW_SIZE], imc_array_0[(0+1) * SP_ROW_SIZE], imc_array_0[(0+2) * SP_ROW_SIZE], imc_array_0[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
                      }
                      ping_pong_loading_cycles += IMC_write_latency_cycles;
                      //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong
                      
                    }
                  }
                }
                else {
                  for (int n = 0; n < (IMC_DIM*next_conv_channel/IMC_DIM); n++) {
                    //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+(output_itr*input_itr*weight_row-1)*weight_col*input_rem*output_rem/SP_ROW_SIZE+output_itr+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                    dma(dma_read_base+(output_itr*input_itr*weight_col*weight_row*input_rem*output_rem)+output_itr*output_rem+(n)* SP_ROW_SIZE);
                    for (int x = 0; x < (IMC_DIM/next_conv_channel); x++) {
                      refresh();
                      if (imc0_computing){
                        memcpy( &imc_array_1[((IMC_DIM/next_conv_channel)*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*next_conv_channel], sizeof(uint8_t) * next_conv_channel);
                        //printf("%d row pixel in W imc_array1: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_1[0 * SP_ROW_SIZE], imc_array_1[(0+1) * SP_ROW_SIZE], imc_array_1[(0+2) * SP_ROW_SIZE], imc_array_1[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
                      }
                      else {
                        memcpy( &imc_array_0[((IMC_DIM/next_conv_channel)*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*next_conv_channel], sizeof(uint8_t) * next_conv_channel);
                        //printf("%d row pixel in W imc_array0: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_0[0 * SP_ROW_SIZE], imc_array_0[(0+1) * SP_ROW_SIZE], imc_array_0[(0+2) * SP_ROW_SIZE], imc_array_0[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
                      }
                      ping_pong_loading_cycles += IMC_write_latency_cycles;
                      //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong
                      
                    }
                  }
                }

                // output trace
                ping_pong_computing_cycles = ping_pong_computing_imc_cycles + ping_pong_computing_anu_cycles + ping_pong_computing_dma_cycles;
                if (ping_pong_loading_cycles > ping_pong_computing_cycles){
                  dma_cycles += ping_pong_loading_cycles;
                }
                else {
                  imc_cycles += ping_pong_computing_imc_cycles;
                  anu_cycles += ping_pong_computing_anu_cycles;
                  dma_cycles += ping_pong_computing_dma_cycles;
                }
                printf("Data loading cycle= %d\n", dma_cycles);
                printf("IMC cycle= %d\n", imc_cycles);
                printf("ANU cycle= %d\n", anu_cycles);
                printf("amem row cycles first = %d ofm = %d ofm2 = %d last = %d\n", amem_row_cycles_0[0], amem_row_cycles_0[row_index], amem_row_cycles_0[row_index+2], amem_row_cycles_0[AMEM_ROW_INDEX-1]);
                printf("amem fist channel 1st result= %d 2nd = %d last-1 = %d last = %d\n", amem_0[(row_index+30*16+13)*SP_ROW_SIZE], amem_0[(row_index+30*16+13)*SP_ROW_SIZE+1], amem_0[(row_index+30*16+13)*SP_ROW_SIZE+SP_ROW_SIZE-2], amem_0[(row_index+30*16+13)*SP_ROW_SIZE+SP_ROW_SIZE-1]);
                for (int i=0*MACRO_DIM; i<2*MACRO_DIM; i++)
                  printf("%d ", amem_0[(row_index +0)*SP_ROW_SIZE+i]);
                printf("\n");
                for (int i=2*MACRO_DIM; i<4*MACRO_DIM; i++)
                  printf("%d ", amem_0[(row_index +(output_row-16)*(output_col/2)*output_itr+((output_col/2)-14)*output_itr+j)*SP_ROW_SIZE+i]);
                printf("\n");
                ie = 1;
                update_interrupts();

              }
              else {
                for (int n = 0; n < (3*MACRO_DIM*output_rem/IMC_DIM); n++) {
                  //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*(MACRO_DIM/2)+k*weight_col*input_itr*(MACRO_DIM/2)+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                  dma(dma_read_base+ (j*weight_col*weight_row*input_itr*MACRO_DIM*output_rem/IMC_DIM+(k+1)*weight_col*input_itr*MACRO_DIM*output_rem/IMC_DIM+n)* SP_ROW_SIZE);
                  for (int x = 0; x < (IMC_DIM/output_rem); x++) {
                    refresh();
                    if (n==0)
                      printf("wmux: first=%d 2nd=%d 3rd=%d\n", imc_w_mux_in[x*output_rem], imc_w_mux_in[x*output_rem+1], imc_w_mux_in[x*output_rem+2]);

                    if (imc0_computing){
                      memcpy( &imc_array_1[((IMC_DIM/output_rem)*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*output_rem], sizeof(uint8_t) * output_rem);
                      //printf("%d row pixel in W imc_array1: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_1[0 * SP_ROW_SIZE], imc_array_1[(0+1) * SP_ROW_SIZE], imc_array_1[(0+2) * SP_ROW_SIZE], imc_array_1[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
                    }
                    else {
                      memcpy( &imc_array_0[((IMC_DIM/output_rem)*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*output_rem], sizeof(uint8_t) * output_rem);
                      //printf("%d row pixel in W imc_array0: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_0[0 * SP_ROW_SIZE], imc_array_0[(0+1) * SP_ROW_SIZE], imc_array_0[(0+2) * SP_ROW_SIZE], imc_array_0[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
                    }
                    ping_pong_loading_cycles += IMC_write_latency_cycles;
                    //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong
                    
                  }
                }
                if (imc0_computing)
                  printf("%d group weight in W imc_array1: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_1[0 * SP_ROW_SIZE], imc_array_1[(0+1) * SP_ROW_SIZE], imc_array_1[(0+2) * SP_ROW_SIZE], imc_array_1[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
                else
                  printf("%d group weight in W imc_array0: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_0[0 * SP_ROW_SIZE], imc_array_0[(0+1) * SP_ROW_SIZE], imc_array_0[(0+2) * SP_ROW_SIZE], imc_array_0[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
              }

            }
          }

          // conv layer 3-5
          // 3x3 conv divided to 5 groups
          // Group0: W00 W01
          // Group1: W10 W11
          // Group2: W20 W21
          // Group3: W02 W12
          // Group4: W22
          else if (input_rem == 2*MACRO_DIM){
            for (int k = 0; k < 5; k += 1) { //(weight_row*weight_col)/(IMC_DIM/(2*MACRO_DIM) = 5
              if (k==0){
                k_start = 1;
                k_end = output_row;
                l_start = 0;
                l_end = output_col;
              }
              else if (k==1){
                k_start = 0;
                k_end = output_row;
                l_start = 0;
                l_end = output_col;
              }
              else if (k==2){
                k_start = 0;
                k_end = output_row-1;
                l_start = 0;
                l_end = output_col;
              }
              else if (k==3){
                k_start = 0;
                k_end = output_row;
                l_start = 0;
                l_end = output_col-1;
              }
              else {
                k_start = 0;
                k_end = output_row-1;
                l_start = 0;
                l_end = output_col-1;
              }


              // Ping-pong control
              //imc0_computing = ((m+k+l) % 2 == 0) ? true : false;
              imc0_computing = !imc0_computing;

              for (int h = k_start; h < k_end;h += 1) {
                for (int i = l_start; i < l_end;i += 1) {

                  refresh();
                
                  // selected input window from (i-1,h-1) to (i+1, h+1)
                  
                  int amem_readout_index = 0;
                  if (k<3){
                    amem_readout_index = i/(IMC_DIM/input_rem);
                    switch (i%(IMC_DIM/input_rem)) {
                      case 0:{
                        if (i==0){ // the input is always aligned to amem initial index(0) when i==0
                          offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + amem_readout_index*input_itr + m;
                          memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                          for (int n = 0; n < (SP_ROW_SIZE);n += 1)
                            if (n<2*MACRO_DIM)
                              input_buffer[n] = 0;
                            else
                              input_buffer[n] = input_mux_in[n-2*MACRO_DIM];
                          ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                          leakage((AMEM_read_latency_cycles));
                        }

                        else{
                          offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index-1)*input_itr + m;
                          memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                          for (int n = 0; n < (SP_ROW_SIZE);n += 1)
                            if (n<2*MACRO_DIM)
                              input_buffer[n] = input_mux_in[n+2*MACRO_DIM];
                            else
                              input_buffer[n] = input_buffer[n];
                          ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                          leakage((AMEM_read_latency_cycles));

                          offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index)*input_itr + m;
                          memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                          for (int n = 0; n < (SP_ROW_SIZE);n += 1)
                            if (n<2*MACRO_DIM)
                              input_buffer[n] = input_buffer[n];
                            else
                              input_buffer[n] = input_mux_in[n-2*MACRO_DIM];
                          ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                          leakage((AMEM_read_latency_cycles));
                        }

                        break;
                      }

                      case 1:{
                        offset = amem_read_base + (h-1+k)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index)*input_itr + m;
                        memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                        memcpy( &input_buffer[0], &input_mux_in[0], sizeof(int32_t) * SP_ROW_SIZE);
                        ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                        leakage((AMEM_read_latency_cycles));

                        break;
                      }

                      default:
                        // Other type stores are not supported
                        printf("Unsupported conv dimension:\n");
                        return false;
                    }
                  }

                  else if (k==3){
                    amem_readout_index = (i+1)/(IMC_DIM/input_rem);
                    if (h==0){
                      for (int n = 0; n < 2*MACRO_DIM;n += 1){
                        input_buffer[n] = 0;
                      }
                    }
                    else {
                      offset = amem_read_base + (h-1)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index)*input_itr + m;
                      memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                      for (int n = 0; n < 2*MACRO_DIM;n += 1){
                        if ((i+1)%(IMC_DIM/input_rem)==0)
                          input_buffer[n] = input_mux_in[n];
                        else
                          input_buffer[n] = input_mux_in[n+2*MACRO_DIM];
                      }
                      ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                      leakage((AMEM_read_latency_cycles));
                    }

                    offset = amem_read_base + (h)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index)*input_itr + m;
                    memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                    for (int n = 2*MACRO_DIM; n < (SP_ROW_SIZE);n += 1){
                      if ((i+1)%(IMC_DIM/input_rem)==0)
                        input_buffer[n] = input_mux_in[n-2*MACRO_DIM];
                      else
                        input_buffer[n] = input_mux_in[n];
                    }
                    ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                    leakage((AMEM_read_latency_cycles));
                    
                  }

                  else{
                    amem_readout_index = (i+1)/(IMC_DIM/input_rem);
                    offset = amem_read_base + (h+1)*(output_col/(IMC_DIM/input_rem))*input_itr + (amem_readout_index)*input_itr + m;
                    memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                    for (int n = 0; n < (SP_ROW_SIZE);n += 1){
                      if (n<2*MACRO_DIM){
                          if ((i+1)%(IMC_DIM/input_rem)==0)
                            input_buffer[n] = input_mux_in[n];
                          else
                            input_buffer[n] = input_mux_in[n+2*MACRO_DIM];
                        }
                        else
                          input_buffer[n] = 0;
                    }
                    ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                    leakage((AMEM_read_latency_cycles));
                    
                  }
                  
                  if ((k==0&&m==0)||(k==1&&m==0&&h==0)) {
                    bias_enable[0] = 0;
                    compute();
                    //imc_cycles += mvm_latency_cycles;//imc cycle is hidden by ping-pong
                    ping_pong_computing_imc_cycles += mvm_latency_cycles;
                    leakage((mvm_latency_cycles));
                    
                    if (output_rem == 2*MACRO_DIM){
                      if (i%2==0){
                        for (int n = 0; n < 2*MACRO_DIM;n += 1)
                          output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n]+output_vector[3][n];
                        ping_pong_computing_anu_cycles += anu_latency_cycles;
                        leakage((anu_latency_cycles));

                        memcpy( &amem_0[(IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j) * SP_ROW_SIZE], &output_buffer[0], sizeof(int32_t) * 2*MACRO_DIM);
                        ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                        leakage((AMEM_write_latency_cycles));
                      }
                      else{
                        for (int n = 0; n < 2*MACRO_DIM;n += 1)
                          output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n]+output_vector[3][n];
                        ping_pong_computing_anu_cycles += anu_latency_cycles;
                        leakage((anu_latency_cycles));

                        memcpy( &amem_0[(IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j) * SP_ROW_SIZE+2*MACRO_DIM], &output_buffer[0], sizeof(int32_t) * 2*MACRO_DIM);
                        amem_row_cycles_0[IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j] = 0;
                        ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                        leakage((AMEM_write_latency_cycles));
                      }
                    }

                    else {
                      for (int n = 0; n < IMC_DIM;n += 1)
                        output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n]+output_vector[3][n];
                      ping_pong_computing_anu_cycles += anu_latency_cycles;
                      leakage((anu_latency_cycles));

                      memcpy( &amem_0[(IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j) * SP_ROW_SIZE], &output_buffer[0], sizeof(int32_t) * SP_ROW_SIZE);
                      amem_row_cycles_0[IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j] = 0;
                      ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                      leakage((AMEM_write_latency_cycles));
                    }

                    if (h==8&&i==2){
                      for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                        printf("%d ", input_buffer[n]);
                      //printf("input buffer first pixel=%d second=%d last=%d\n", input_buffer[MACRO_DIM], input_buffer[MACRO_DIM+1], input_buffer[MACRO_DIM+MACRO_DIM-1]);
                      printf("output value in 1=%d 2=%d 3=%d 4=%d buffer=%d\n",  output_vector[0][0], output_vector[1][0], output_vector[2][0], output_vector[3][0], output_buffer[0]);
                      if (imc0_computing)
                        //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_0[0], imc_array_0[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_0[MACRO_DIM*SP_ROW_SIZE], imc_array_0[2*MACRO_DIM*SP_ROW_SIZE]);
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", imc_array_0[(n)*SP_ROW_SIZE+0]);
                      else
                        //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_1[0], imc_array_1[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_1[MACRO_DIM*SP_ROW_SIZE], imc_array_1[2*MACRO_DIM*SP_ROW_SIZE]);
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", imc_array_1[(n)*SP_ROW_SIZE+0]);
                      printf("\n");
                    }

                  }

                  else {
                    if ((k==4&&m==input_itr-1)||(k==3&&m==input_itr-1&&h==output_row-1)||(k==2&&m==input_itr-1&&i==output_col-1)||(k==1&&m==input_itr-1&&h==output_row-1&&i==output_col-1)) {
                      if (bias_existing)
                        bias_enable[0] = 1;
                      else 
                        bias_enable[0] = 0;
                    }
                    else{
                      bias_enable[0] = 0;
                    }
                    compute();
                    ping_pong_computing_imc_cycles += mvm_latency_cycles;
                    leakage((mvm_latency_cycles));

                    if (output_rem == 2*MACRO_DIM){
                      if (i%2==0){
                        for (int n = 0; n < 2*MACRO_DIM;n += 1)
                          output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n]+output_vector[3][n];
                        ping_pong_computing_anu_cycles += anu_latency_cycles;
                        leakage((anu_latency_cycles));

                        if ((k==4&&m==input_itr-1)||(k==3&&m==input_itr-1&&h==output_row-1)||(k==2&&m==input_itr-1&&i==output_col-1)||(k==1&&m==input_itr-1&&h==output_row-1&&i==output_col-1)){
                          for (int x = 0; x < 2*MACRO_DIM; ++x) 
                            amem_0[(row_index +h*(output_col/2)*output_itr+(i/2)*output_itr+j)*SP_ROW_SIZE+ x] = (amem_0[(IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j) * SP_ROW_SIZE+ x] + output_buffer[x])/amem_scale;
                        } 
                        else {
                          for (int x = 0; x < 2*MACRO_DIM; ++x)
                            amem_0[(IMC_AMEM_TEMP_BASE +h*(output_col/2)*output_itr+(i/2)*output_itr+j)*SP_ROW_SIZE+ x] = amem_0[(IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j) * SP_ROW_SIZE+ x] + output_buffer[x];
                        } 
                        ping_pong_computing_anu_cycles += AMEM_read_latency_cycles;
                        ping_pong_computing_anu_cycles += anu_latency_cycles;
                        ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                        leakage((anu_latency_cycles));
                        leakage((AMEM_write_latency_cycles));
                      }
                      else{
                        for (int n = 0; n < 2*MACRO_DIM;n += 1)
                          output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n]+output_vector[3][n];
                        ping_pong_computing_anu_cycles += anu_latency_cycles;
                        leakage((anu_latency_cycles));

                        if ((k==4&&m==input_itr-1)||(k==3&&m==input_itr-1&&h==output_row-1)||(k==2&&m==input_itr-1&&i==output_col-1)||(k==1&&m==input_itr-1&&h==output_row-1&&i==output_col-1)){
                          for (int x = 0; x < 2*MACRO_DIM; ++x) 
                            amem_0[(row_index +h*(output_col/2)*output_itr+(i/2)*output_itr+j)*SP_ROW_SIZE+2*MACRO_DIM+ x] = (amem_0[(IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j) * SP_ROW_SIZE+2*MACRO_DIM+ x] + output_buffer[x])/amem_scale;
                          amem_row_cycles_0[row_index +h*(output_col/2)*output_itr+(i/2)*output_itr+j] = 0;
                        } 
                        else {
                          for (int x = 0; x < 2*MACRO_DIM; ++x)
                            amem_0[(IMC_AMEM_TEMP_BASE +h*(output_col/2)*output_itr+(i/2)*output_itr+j)*SP_ROW_SIZE+2*MACRO_DIM+ x] = amem_0[(IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j) * SP_ROW_SIZE+2*MACRO_DIM+ x] + output_buffer[x];
                          amem_row_cycles_0[IMC_AMEM_TEMP_BASE + h*(output_col/2)*output_itr+(i/2)*output_itr+j] = 0;
                        } 
                        ping_pong_computing_anu_cycles += AMEM_read_latency_cycles;
                        ping_pong_computing_anu_cycles += anu_latency_cycles;
                        ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                        leakage((anu_latency_cycles));
                        leakage((AMEM_write_latency_cycles));
                      }
                    }

                    else{
                      for (int n = 0; n < IMC_DIM;n += 1)
                        output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n]+output_vector[3][n];
                      ping_pong_computing_anu_cycles += anu_latency_cycles;
                      leakage((anu_latency_cycles));

                      if ((k==4&&m==input_itr-1)||(k==3&&m==input_itr-1&&h==output_row-1)||(k==2&&m==input_itr-1&&i==output_col-1)||(k==1&&m==input_itr-1&&h==output_row-1&&i==output_col-1)){
                        for (int x = 0; x < SP_ROW_SIZE; ++x) 
                          amem_0[(row_index +h*output_col*output_itr+i*output_itr+j)*SP_ROW_SIZE+ x] = (amem_0[(IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j) * SP_ROW_SIZE+ x] + output_buffer[x])/amem_scale;
                        amem_row_cycles_0[row_index +h*output_col*output_itr+i*output_itr+j] = 0;
                      } 
                      else {
                        for (int x = 0; x < SP_ROW_SIZE; ++x)
                          amem_0[(IMC_AMEM_TEMP_BASE +h*output_col*output_itr+i*output_itr+j)*SP_ROW_SIZE+ x] = amem_0[(IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j) * SP_ROW_SIZE+ x] + output_buffer[x];
                        amem_row_cycles_0[IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j] = 0;
                      } 
                      ping_pong_computing_anu_cycles += AMEM_read_latency_cycles;
                      ping_pong_computing_anu_cycles += anu_latency_cycles;
                      ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                      leakage((anu_latency_cycles));
                      leakage((AMEM_write_latency_cycles));
                    }

                    
                    if (h==8&&i==2){
                      for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                        printf("%d ", input_buffer[n]);
                      //printf("input buffer first pixel=%d second=%d last=%d\n", input_buffer[MACRO_DIM], input_buffer[MACRO_DIM+1], input_buffer[MACRO_DIM+MACRO_DIM-1]);
                      printf("output value in 1=%d 2=%d 3=%d 4=%d buffer=%d\n",  output_vector[0][0], output_vector[1][0], output_vector[2][0], output_vector[3][0], output_buffer[0]);
                      if (imc0_computing)
                        //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_0[0], imc_array_0[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_0[MACRO_DIM*SP_ROW_SIZE], imc_array_0[2*MACRO_DIM*SP_ROW_SIZE]);
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", imc_array_0[(n)*SP_ROW_SIZE+0]);
                      else
                        //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_1[0], imc_array_1[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_1[MACRO_DIM*SP_ROW_SIZE], imc_array_1[2*MACRO_DIM*SP_ROW_SIZE]);
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", imc_array_1[(n)*SP_ROW_SIZE+0]);
                      printf("\n");
                    }

                  }
                }
              }

              //ping-pong weight loading
              // frist if-else: inter layer or intra layer
              //second if-else: output channel = 256 or 128
              // third if-else: group 0-3
              if (k==4&&m==input_itr-1&&j==output_itr-1){
                if (next_conv_channel == 2*MACRO_DIM){
                  for (int n = 0; n < (IMC_DIM/2); n++) {
                    //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-2)*input_rem*output_rem/SP_ROW_SIZE+output_itr+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                    dma(dma_read_base+ (output_itr*weight_col*weight_row*input_itr)*input_rem*output_rem+output_itr*output_rem+(n)* SP_ROW_SIZE);
                    for (int x = 0; x < 2; x++) {
                      refresh();
                      if (imc0_computing){
                        memcpy( &imc_array_1[(2*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        //printf("%d row pixel in W imc_array1: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_1[0 * SP_ROW_SIZE], imc_array_1[(0+1) * SP_ROW_SIZE], imc_array_1[(0+2) * SP_ROW_SIZE], imc_array_1[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
                      }
                      else {
                        memcpy( &imc_array_0[(2*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        //printf("%d row pixel in W imc_array0: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_0[0 * SP_ROW_SIZE], imc_array_0[(0+1) * SP_ROW_SIZE], imc_array_0[(0+2) * SP_ROW_SIZE], imc_array_0[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
                      }
                      ping_pong_loading_cycles += IMC_write_latency_cycles;
                      //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong
                      
                    }
                  }
                }

                else {
                  for (int n = 0; n < IMC_DIM; n++) {
                    if (output_rem>0)
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-2)*input_rem*output_rem/SP_ROW_SIZE+output_itr+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                      dma(dma_read_base+ (output_itr*weight_col*weight_row*input_itr)*input_rem*output_rem+output_itr*output_rem+(n)* SP_ROW_SIZE);
                    else
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-2)*input_rem+output_itr+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                      dma(dma_read_base+ ((output_itr*weight_col*weight_row*input_itr)*input_rem+output_itr+n)* SP_ROW_SIZE);
                    refresh();
                    if (imc0_computing){
                      memcpy( &imc_array_1[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      //printf("%d row pixel in W imc_array1: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_1[0 * SP_ROW_SIZE], imc_array_1[(0+1) * SP_ROW_SIZE], imc_array_1[(0+2) * SP_ROW_SIZE], imc_array_1[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
                    }
                    else {
                      memcpy( &imc_array_0[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      //printf("%d row pixel in W imc_array0: first=%d 2nd=%d 3rd=%d last=%d\n", k+1, imc_array_0[0 * SP_ROW_SIZE], imc_array_0[(0+1) * SP_ROW_SIZE], imc_array_0[(0+2) * SP_ROW_SIZE], imc_array_0[(3*MACRO_DIM-1) * SP_ROW_SIZE]);
                    }
                    ping_pong_loading_cycles += IMC_write_latency_cycles;
                    //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong
                  }

                  //printf("offset=%d\n", (sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-2)*input_rem*output_rem/SP_ROW_SIZE+output_itr));
                  printf("imc_w_mux0=%d 1=%d last=%d\n", sp_weight[(sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-2)*input_rem*output_rem/SP_ROW_SIZE+output_itr+0) * SP_ROW_SIZE], sp_weight[(sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-2)*input_rem*output_rem/SP_ROW_SIZE+output_itr+1) * SP_ROW_SIZE], sp_weight[(sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-2)*input_rem*output_rem/SP_ROW_SIZE+output_itr+IMC_DIM-1) * SP_ROW_SIZE]);
                  if (imc0_computing){
                    printf("fisrt row last channel pixel of next conv in W imc_array1: first=%d 2nd=%d 3rd=%d last=%d\n", imc_array_1[0 * SP_ROW_SIZE+0], imc_array_1[(0+1) * SP_ROW_SIZE+0], imc_array_1[(0+2) * SP_ROW_SIZE+0], imc_array_1[(IMC_DIM-1) * SP_ROW_SIZE+0]);
                  }
                  else{
                    printf("fisrt row last channel pixel of next conv in W imc_array0: first=%d 2nd=%d 3rd=%d last=%d\n", imc_array_0[0 * SP_ROW_SIZE+0], imc_array_0[(0+1) * SP_ROW_SIZE+0], imc_array_0[(0+2) * SP_ROW_SIZE+0], imc_array_0[(IMC_DIM-1) * SP_ROW_SIZE+0]);
                  }

                }
                
                // output trace
                ping_pong_computing_cycles = ping_pong_computing_imc_cycles + ping_pong_computing_anu_cycles + ping_pong_computing_dma_cycles;
                if (ping_pong_loading_cycles > ping_pong_computing_cycles){
                  dma_cycles += ping_pong_loading_cycles;
                }
                else {
                  imc_cycles += ping_pong_computing_imc_cycles;
                  anu_cycles += ping_pong_computing_anu_cycles;
                  dma_cycles += ping_pong_computing_dma_cycles;
                }
                printf("Data loading cycle= %d\n", dma_cycles);
                printf("IMC cycle= %d\n", imc_cycles);
                printf("ANU cycle= %d\n", anu_cycles);
                printf("amem row cycles first = %d ofm = %d ofm2 = %d last = %d\n", amem_row_cycles_0[0], amem_row_cycles_0[row_index], amem_row_cycles_0[row_index+2], amem_row_cycles_0[AMEM_ROW_INDEX-1]);
                printf("amem first pixel 1st result= %d 2nd = %d last-1 = %d last = %d\n", amem_0[row_index*SP_ROW_SIZE+255*2*MACRO_DIM], amem_0[row_index*SP_ROW_SIZE+255*2*MACRO_DIM+1], amem_0[row_index*SP_ROW_SIZE+255*2*MACRO_DIM+2*MACRO_DIM-2], amem_0[row_index*SP_ROW_SIZE+255*2*MACRO_DIM+2*MACRO_DIM-1]);
                ie = 1;
                update_interrupts();
              }
              else {
                if (output_rem == 2*MACRO_DIM){
                  if (k < 2){
                    for (int n = 0; n < (IMC_DIM/2); n++) {                      
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*(IMC_DIM/4)+(k)*2*input_itr*(IMC_DIM/4)+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);  
                      dma(dma_read_base+ (j*weight_col*weight_row*input_itr*(IMC_DIM/4)+(k+1)*weight_col*input_itr*(IMC_DIM/4)+n)* SP_ROW_SIZE);                    
                      for (int x = 0; x < 2; x++) {
                        refresh();
                        if (imc0_computing){
                          memcpy( &imc_array_1[(2*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        }
                        else {
                          memcpy( &imc_array_0[(2*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        }
                        ping_pong_loading_cycles += IMC_write_latency_cycles;
                        //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong
                      }                      
                    }
                  }

                  else if (k == 2){
                    for (int n = 0; n < (IMC_DIM/4); n++) {                      
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*(IMC_DIM/4)+(k)*2*input_itr*(IMC_DIM/4)+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);  
                      dma(dma_read_base+ (j*weight_col*weight_row*input_itr*(IMC_DIM/4)+(0)*weight_col*input_itr*(IMC_DIM/4)+2*(IMC_DIM/4)+n)* SP_ROW_SIZE);                    
                      for (int x = 0; x < 2; x++) {
                        refresh();
                        if (imc0_computing){
                          memcpy( &imc_array_1[(2*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        }
                        else {
                          memcpy( &imc_array_0[(2*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        }
                        ping_pong_loading_cycles += IMC_write_latency_cycles;
                        //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong
                      }                      
                    }

                    for (int n = 0; n < (IMC_DIM/4); n++) {                      
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*(IMC_DIM/4)+(k)*2*input_itr*(IMC_DIM/4)+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);  
                      dma(dma_read_base+ (j*weight_col*weight_row*input_itr*(IMC_DIM/4)+(1)*weight_col*input_itr*(IMC_DIM/4)+2*(IMC_DIM/4)+n)* SP_ROW_SIZE);                    
                      for (int x = 0; x < 2; x++) {
                        refresh();
                        if (imc0_computing){
                          memcpy( &imc_array_1[(2*(n+IMC_DIM/4)+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        }
                        else {
                          memcpy( &imc_array_0[(2*(n+IMC_DIM/4)+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        }
                        ping_pong_loading_cycles += IMC_write_latency_cycles;
                        //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong
                      }                      
                    }
                  }

                  else if (k == 3){
                    for (int n = 0; n < (IMC_DIM/4); n++) { 
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*(IMC_DIM/4)+(k)*2*input_itr*(IMC_DIM/4)+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE); 
                      dma(dma_read_base+ (j*weight_col*weight_row*input_itr*(IMC_DIM/4)+(2)*weight_col*input_itr*(IMC_DIM/4)+2*(IMC_DIM/4)+n)* SP_ROW_SIZE);                       
                      for (int x = 0; x < 2; x++) {
                        refresh();
                        if (imc0_computing){
                          memcpy( &imc_array_1[(2*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        }
                        else {
                          memcpy( &imc_array_0[(2*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        }
                        ping_pong_loading_cycles += IMC_write_latency_cycles;
                        //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong
                      }                      
                    }
                  }

                  else if (k == 4){
                    for (int n = 0; n < (IMC_DIM/2); n++) {
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+((j+1)*weight_col*weight_row*input_itr-2)*(IMC_DIM/4)+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE); 
                      dma(dma_read_base+ ((j+1)*weight_col*weight_row*input_itr*(IMC_DIM/4)+n)* SP_ROW_SIZE);
                      for (int x = 0; x < 2; x++) {
                        refresh();
                        if (imc0_computing){
                          memcpy( &imc_array_1[(2*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        }
                        else {
                          memcpy( &imc_array_0[(2*n+x) * SP_ROW_SIZE], &imc_w_mux_in[x*2*MACRO_DIM], sizeof(uint8_t) * 2*MACRO_DIM);
                        }
                        ping_pong_loading_cycles += IMC_write_latency_cycles;
                        //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong
                      }                      
                    }
                  }
                }

                else{
                  if (k < 2){
                    for (int n = 0; n < IMC_DIM; n++) {                      
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*(IMC_DIM/2)+(k)*2*input_itr*(IMC_DIM/2)+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE); 
                      dma(dma_read_base+ (j*weight_col*weight_row*input_itr*(IMC_DIM/2)+(k+1)*weight_col*input_itr*(IMC_DIM/2)+n)* SP_ROW_SIZE);                      
                      refresh();
                      if (imc0_computing){
                        memcpy( &imc_array_1[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      else {
                        memcpy( &imc_array_0[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      ping_pong_loading_cycles += IMC_write_latency_cycles;
                      //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong         
                    }
                  }

                  else if (k == 2){
                    for (int n = 0; n < (IMC_DIM/2); n++) { 
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*(IMC_DIM/2)+(k)*2*input_itr*(IMC_DIM/2)+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);  
                      dma(dma_read_base+ (j*weight_col*weight_row*input_itr*(IMC_DIM/2)+(0)*weight_col*input_itr*(IMC_DIM/2)+2*(IMC_DIM/2)+n)* SP_ROW_SIZE);                    
                      refresh();
                      if (imc0_computing){
                        memcpy( &imc_array_1[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      else {
                        memcpy( &imc_array_0[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      ping_pong_loading_cycles += IMC_write_latency_cycles;
                      //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong          
                    }                   

                    for (int n = 0; n < (IMC_DIM/2); n++) {                      
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*(IMC_DIM/4)+(k)*2*input_itr*(IMC_DIM/4)+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);  
                      dma(dma_read_base+ (j*weight_col*weight_row*input_itr*(IMC_DIM/2)+(1)*weight_col*input_itr*(IMC_DIM/2)+2*(IMC_DIM/2)+n)* SP_ROW_SIZE);                    
                      refresh();
                      if (imc0_computing){
                        memcpy( &imc_array_1[(n+IMC_DIM/2) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      else {
                        memcpy( &imc_array_0[(n+IMC_DIM/2) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      ping_pong_loading_cycles += IMC_write_latency_cycles;
                      //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong             
                    }
                  }

                  else if (k == 3){
                    for (int n = 0; n < (IMC_DIM/2); n++) { 
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*(IMC_DIM/2)+(k)*2*input_itr*(IMC_DIM/2)+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                      dma(dma_read_base+ (j*weight_col*weight_row*input_itr*(IMC_DIM/2)+(2)*weight_col*input_itr*(IMC_DIM/2)+2*(IMC_DIM/2)+n)* SP_ROW_SIZE);       
                      refresh();
                      if (imc0_computing){
                        memcpy( &imc_array_1[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      else {
                        memcpy( &imc_array_0[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      ping_pong_loading_cycles += IMC_write_latency_cycles;
                      //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong               
                    }
                  }

                  else if (k == 4){
                    for (int n = 0; n < IMC_DIM; n++) {                      
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+((j+1)*weight_col*weight_row*input_itr-2)*(IMC_DIM/2)+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);  
                      dma(dma_read_base+ ((j+1)*weight_col*weight_row*input_itr*(IMC_DIM/2)+n)* SP_ROW_SIZE);                    
                      refresh();
                      if (imc0_computing){
                        memcpy( &imc_array_1[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      else {
                        memcpy( &imc_array_0[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      ping_pong_loading_cycles += IMC_write_latency_cycles;
                      //leakage((AMEM_write_latency_cycles)); Hiden by ping-pong
                    }
                  }
                }
                
              }

            }
          }

          //conv layer 6,7,8
          // 3x3 conv divided to 9 groups --> channel-wise mapping
          else {
            for (int k = 0; k < weight_row; k += 1) {
              for (int l = 0; l < weight_col; l += 1) {

                if (k==0){
                  k_start = 1;
                  k_end = output_row;
                }
                else if (k==weight_row-1){
                  k_start = 0;
                  k_end = output_row-1;
                }
                else {
                  k_start = 0;
                  k_end = output_row;
                }

                if (l==0){
                  l_start = 1;
                  l_end = output_col;
                }
                else if (l==weight_col-1){
                  l_start = 0;
                  l_end = output_col-1;
                }
                else {
                  l_start = 0;
                  l_end = output_col;
                }

                // Load weights from SP -> IMC_array
                //imc0_computing = ((m+k+l) % 2 == 0) ? true : false;
                imc0_computing = !imc0_computing;

                for (int h = k_start; h < k_end;h += 1) {
                  for (int i = l_start; i < l_end;i += 1) {

                    refresh();
                  
                    // selected input window from (i-1,h-1) to (i+1, h+1)
                    // TODO: fixed parameter of stride and kernel size for now --> Parameterize later
                    offset = amem_read_base + (h-1+k)*output_col*input_itr + (i-1+l)*input_itr + m;

                    memcpy( &input_mux_in[0], &amem_0[offset* SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                    memcpy( &input_buffer[0], &input_mux_in[0], sizeof(int32_t) * SP_ROW_SIZE);
                    ping_pong_computing_imc_cycles += AMEM_read_latency_cycles;
                    //leakage((AMEM_read_latency_cycles)); Hidden by ping-pong

                    if ((k==0&&l==0&&m==0)||(k==0&&l==1&&m==0&&i==0)||(k==1&&l==0&&m==0&&h==0)||(k==1&&l==1&&m==0&&h==0&&i==0)) {
                      bias_enable[0] = 0;
                      compute();
                      ping_pong_computing_imc_cycles += mvm_latency_cycles;
                      //leakage((mvm_latency_cycles)); Hidden by ping-pong

                      for (int n = 0; n < IMC_DIM;n += 1)
                        output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n]+output_vector[3][n];
                      ping_pong_computing_anu_cycles += anu_latency_cycles;
                      //leakage((anu_latency_cycles)); Hidden by ping-pong

                      memcpy( &amem_0[(IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j) * SP_ROW_SIZE], &output_buffer[0], sizeof(int32_t) * SP_ROW_SIZE);
                      amem_row_cycles_0[IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j] = 0;

                      /*if ((h==output_col-2&&i==output_row-2)){
                        printf("read_addr: %d\n", offset);
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", (amem_0[10* SP_ROW_SIZE+n]));
                        printf("\n");
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", (output_vector[0][n]));
                        printf("\n"); 
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", (amem_0[(IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j) * SP_ROW_SIZE+ n]));
                        printf("\n");  
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", ( output_buffer[n]));
                        printf("\n");
                      }
                      
                      if (h==output_row-4 && i==output_col-6){
                        printf("output_vector[SP_ROW_SIZE-1]= %d\n", output_buffer[0]);
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", input_buffer[n]);
                        printf("\n");

                        if (imc0_computing)
                          //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_0[0], imc_array_0[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_0[MACRO_DIM*SP_ROW_SIZE], imc_array_0[2*MACRO_DIM*SP_ROW_SIZE]);
                          for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                            printf("%d ", imc_array_0[(n)*SP_ROW_SIZE+0]);
                        else
                          //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_1[0], imc_array_1[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_1[MACRO_DIM*SP_ROW_SIZE], imc_array_1[2*MACRO_DIM*SP_ROW_SIZE]);
                          for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                            printf("%d ", imc_array_1[(n)*SP_ROW_SIZE+0]);
                        printf("\n");
                      }*/
                      ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                      //leakage((AMEM_write_latency_cycles)); Hidden by ping-pong
                    }

                    else {
                      if ((k==weight_col-1&&l==weight_row-1&&m==input_itr-1)||(k==weight_col-1&&l==weight_row-2&&m==input_itr-1&&i==output_row-1)||(k==weight_col-2&&l==weight_row-1&&m==input_itr-1&&h==output_col-1)||(k==weight_col-2&&l==weight_row-2&&m==input_itr-1&&h==output_col-1&&i==output_row-1)) {
                        if (bias_existing)
                          bias_enable[0] = 1;
                        else 
                          bias_enable[0] = 0;
                      }
                      else{
                        bias_enable[0] = 0;
                      }
                      compute();
                      ping_pong_computing_imc_cycles += mvm_latency_cycles;
                     // leakage((mvm_latency_cycles)); Hidden by ping-pong

                      for (int n = 0; n < IMC_DIM;n += 1)
                        output_buffer[n] = output_vector[0][n]+output_vector[1][n]+output_vector[2][n]+output_vector[3][n];
                      ping_pong_computing_anu_cycles += anu_latency_cycles;
                      //leakage((anu_latency_cycles)); Hidden by ping-pong

                      /*if ((h==output_col-2&&i==output_row-2)){
                        printf("read_addr: %d\n", offset);
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", (amem_0[10* SP_ROW_SIZE+n]));
                        printf("\n");
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", (amem_0[offset* SP_ROW_SIZE+n]));
                        printf("\n");
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", (input_buffer[n]));
                        printf("\n");
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", (amem_0[(IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j) * SP_ROW_SIZE+ n]));
                        printf("\n");  
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", (amem_0[(IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j) * SP_ROW_SIZE+ n] + output_buffer[n]));
                        printf("\n");
                      }*/

                      if ((k==weight_col-1&&l==weight_row-1&&m==input_itr-1)||(k==weight_col-1&&l==weight_row-2&&m==input_itr-1&&i==output_row-1)||(k==weight_col-2&&l==weight_row-1&&m==input_itr-1&&h==output_col-1)||(k==weight_col-2&&l==weight_row-2&&m==input_itr-1&&h==output_col-1&&i==output_row-1)) {
                        for (int x = 0; x < SP_ROW_SIZE; ++x) 
                          amem_0[(row_index +h*output_col*output_itr+i*output_itr+j)*SP_ROW_SIZE+ x] = (amem_0[(IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j) * SP_ROW_SIZE+ x] + output_buffer[x])/amem_scale;
                        amem_row_cycles_0[row_index +h*output_col*output_itr+i*output_itr+j] = 0;
                      } 
                      else {
                        for (int x = 0; x < SP_ROW_SIZE; ++x)
                          amem_0[(IMC_AMEM_TEMP_BASE +h*output_col*output_itr+i*output_itr+j)*SP_ROW_SIZE+ x] = amem_0[(IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j) * SP_ROW_SIZE+ x] + output_buffer[x];
                        amem_row_cycles_0[IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j] = 0;
                      }
                      //memcpy( &amem_0[(IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j) * SP_ROW_SIZE], &output_buffer[0], sizeof(int32_t) * SP_ROW_SIZE);
                      //amem_row_cycles_0[IMC_AMEM_TEMP_BASE + h*output_col*output_itr+i*output_itr+j] = 0;
                      /*if (h==output_row-4 && i==output_col-6){
                        printf("output_vector[SP_ROW_SIZE-1]= %d\n", output_buffer[0]);
                        for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                          printf("%d ", input_buffer[n]);
                        printf("\n");

                        if (imc0_computing)
                          //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_0[0], imc_array_0[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_0[MACRO_DIM*SP_ROW_SIZE], imc_array_0[2*MACRO_DIM*SP_ROW_SIZE]);
                          for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                            printf("%d ", imc_array_0[(n)*SP_ROW_SIZE+0]);
                        else
                          //printf("weight first pixel=%d, last channel=%d, second=%d, third=%d\n", imc_array_1[0], imc_array_1[(MACRO_DIM-1)*SP_ROW_SIZE], imc_array_1[MACRO_DIM*SP_ROW_SIZE], imc_array_1[2*MACRO_DIM*SP_ROW_SIZE]);
                          for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                            printf("%d ", imc_array_1[(n)*SP_ROW_SIZE+0]);
                        printf("\n");
                      }*/

 
                      ping_pong_computing_anu_cycles += AMEM_read_latency_cycles;
                      ping_pong_computing_anu_cycles += anu_latency_cycles;
                      ping_pong_computing_dma_cycles += AMEM_write_latency_cycles;
                      //leakage((anu_latency_cycles)); Hidden by ping-pong
                      //leakage((AMEM_write_latency_cycles)); Hidden by ping-pong
                    }
                  }
                }

                //ping-pong weight loading
                if (k==weight_col-1&&l==weight_row-1&&m==input_itr-1&&j==output_itr-1){
                  if (next_conv_channel == 10){
                    int rest = 0;
                    int rest_new = 0;
                    int mux_start_addr = 0;
                    int mux_finish_addr = 0;
                    int imc_write_addr = 0;
                    
                    for (int n = 0; n < 40; n++) {
                      //special case for cifar-10: map 1024*10 weights to 256 rows * 40 cols in IMC arrat.
                      //memcpy( &imc_w_reg[0], &sp_weight[(sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-1)*IMC_DIM+output_itr+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                      dma(dma_read_base+ ((output_itr*weight_col*weight_row*input_itr)*IMC_DIM+output_itr+n)* SP_ROW_SIZE);

                      for (int x = 0; x < 8; x++){
                        if (x==0){
                          for (int y = mux_start_addr; y < SP_ROW_SIZE; y++) {
                            if (y%MACRO_DIM<10)
                              imc_w_reg[y] = imc_w_mux_in[x*40-rest+(y/MACRO_DIM)*10+(y%MACRO_DIM)];
                            else 
                              imc_w_reg[y] = W_ZERO_POINT;
                          }
                          refresh();
                          if (imc0_computing){
                            memcpy( &imc_array_1[(imc_write_addr) * SP_ROW_SIZE], &imc_w_reg[0], sizeof(uint8_t) * SP_ROW_SIZE);
                          }
                          else {
                            memcpy( &imc_array_0[(imc_write_addr) * SP_ROW_SIZE], &imc_w_reg[0], sizeof(uint8_t) * SP_ROW_SIZE);
                          }
                          imc_write_addr= imc_write_addr+1;
                          ping_pong_loading_cycles += IMC_write_latency_cycles;
                          leakage((AMEM_write_latency_cycles));
                        }

                        
                        else if ((SP_ROW_SIZE+rest-(x)*40)>=40){
                          for (int y = 0; y < SP_ROW_SIZE; y++) {
                            if (y%MACRO_DIM<10)
                              imc_w_reg[y] = imc_w_mux_in[x*40-rest+(y/MACRO_DIM)*10+(y%MACRO_DIM)];
                            else 
                              imc_w_reg[y] = W_ZERO_POINT;
                          }
                          refresh();
                          if (imc0_computing){
                            memcpy( &imc_array_1[(imc_write_addr) * SP_ROW_SIZE], &imc_w_reg[0], sizeof(uint8_t) * SP_ROW_SIZE);
                          }
                          else {
                            memcpy( &imc_array_0[(imc_write_addr) * SP_ROW_SIZE], &imc_w_reg[0], sizeof(uint8_t) * SP_ROW_SIZE);
                          }
                          imc_write_addr= imc_write_addr+1;
                          ping_pong_loading_cycles += IMC_write_latency_cycles;
                          leakage((AMEM_write_latency_cycles));
                        }

                        else{
                          rest_new = SP_ROW_SIZE+rest-(x)*40;
                          mux_finish_addr = (rest_new/10)*MACRO_DIM+rest_new%10;
                          for (int y = 0; y < mux_finish_addr; y++) {
                            if (y%MACRO_DIM<10)
                              imc_w_reg[y] = imc_w_mux_in[x*40-rest+(y/MACRO_DIM)*10+(y%MACRO_DIM)];
                            else 
                              imc_w_reg[y] = W_ZERO_POINT;
                          }
                          printf("rest=%d, rest_new=%d, mux_start_addr=%d, mux_finish_addr=%d, imc_w_reg0=%d\n", rest, rest_new, mux_start_addr, mux_finish_addr, imc_w_reg[0]);
                          mux_start_addr= mux_finish_addr;
                          rest = rest_new;
                          break;
                        }
                    
                      }
                    }
                    if (imc0_computing)
                      printf("fc W imc_array1: first=%d 2nd=%d 3rd=%d last=%d\n", imc_array_1[0 * SP_ROW_SIZE], imc_array_1[(0) * SP_ROW_SIZE+ 1*MACRO_DIM], imc_array_1[(0) * SP_ROW_SIZE+ 2*MACRO_DIM], imc_array_1[(0+IMC_DIM-1) * SP_ROW_SIZE+ (0) * SP_ROW_SIZE+ 3*MACRO_DIM]);
                    else
                      printf("fc W imc_array0: first=%d 2nd=%d 3rd=%d last=%d\n", imc_array_0[0 * SP_ROW_SIZE], imc_array_0[(IMC_DIM-1) * SP_ROW_SIZE+ 1*MACRO_DIM], imc_array_0[(IMC_DIM-1) * SP_ROW_SIZE+ 2*MACRO_DIM], imc_array_0[(0+IMC_DIM-1) * SP_ROW_SIZE+ (0) * SP_ROW_SIZE+ 3*MACRO_DIM]);
                      //for (int n = 0; n < MACRO_DIM;n += 1)
                      //  printf("%d ", imc_array_0[(n)*SP_ROW_SIZE+0]);
                    //printf("\n");                    
                  }

                  else {
                    for (int n = 0; n < IMC_DIM; n++) {
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+(output_itr*weight_col*weight_row*input_itr-1)*IMC_DIM+output_itr+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                      dma(dma_read_base+ ((output_itr*weight_col*weight_row*input_itr)*IMC_DIM+output_itr+n)* SP_ROW_SIZE);
                      refresh();
                      if (imc0_computing){
                        memcpy( &imc_array_1[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      else {
                        memcpy( &imc_array_0[(n) * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      }
                      ping_pong_loading_cycles += IMC_write_latency_cycles;
                      leakage((AMEM_write_latency_cycles));
                    }
                  }

                  // output trace
                  ping_pong_computing_cycles = ping_pong_computing_imc_cycles + ping_pong_computing_anu_cycles + ping_pong_computing_dma_cycles;
                  if (ping_pong_loading_cycles > ping_pong_computing_cycles){
                    dma_cycles += ping_pong_loading_cycles;
                  }
                  else {
                    imc_cycles += ping_pong_computing_imc_cycles;
                    anu_cycles += ping_pong_computing_anu_cycles;
                    dma_cycles += ping_pong_computing_dma_cycles;
                  }
                  printf("Data loading cycle= %d\n", dma_cycles);
                  printf("IMC cycle= %d\n", imc_cycles);
                  printf("ANU cycle= %d\n", anu_cycles);
                  printf("amem row cycles first = %d ofm = %d ofm2 = %d last = %d\n", amem_row_cycles_0[0], amem_row_cycles_0[row_index], amem_row_cycles_0[row_index+2], amem_row_cycles_0[AMEM_ROW_INDEX-1]);
                  printf("amem last channel 1st result= %d 2nd = %d last-1 = %d last = %d\n", amem_0[(row_index+15)*SP_ROW_SIZE], amem_0[(row_index+15)*SP_ROW_SIZE+1], amem_0[(row_index+15)*SP_ROW_SIZE+SP_ROW_SIZE-2], amem_0[(row_index+15)*SP_ROW_SIZE+SP_ROW_SIZE-1]);
                  for (int n = 0*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
                    printf("%d ", amem_0[(row_index +(output_row-2)*output_col*output_itr + (output_col-3)*output_itr + 0)*SP_ROW_SIZE+ n]);
                  printf("\n");
                  ie = 1;
                  update_interrupts();
                }
                else {
                  if (imc0_computing) {
                    //for (int n = 0; n < 3; n++) { 
                    for (int n = 0; n < IMC_DIM; n++) {
                      refresh();
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*IMC_DIM+k*weight_row*input_itr*IMC_DIM+l*input_itr*IMC_DIM+m*IMC_DIM+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                      dma(dma_read_base+ (j*weight_col*weight_row*input_itr*IMC_DIM+k*weight_row*input_itr*IMC_DIM+l*input_itr*IMC_DIM+(m+1)*IMC_DIM+n)* SP_ROW_SIZE);
                      memcpy( &imc_array_1[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      //dma_cycles += IMC_write_latency_cycles; //DMA cycle is hidden by ping-pong
                      ping_pong_loading_cycles += IMC_write_latency_cycles;
                      leakage((AMEM_write_latency_cycles));
                    }
                    printf("%d row %d col's next pixel of W imc_array1: first=%d 2nd=%d 3rd=%d last=%d\n", k, l, imc_array_1[0 * SP_ROW_SIZE+ SP_ROW_SIZE-1], imc_array_1[(0+1) * SP_ROW_SIZE+ SP_ROW_SIZE-1], imc_array_1[(0+2) * SP_ROW_SIZE+ SP_ROW_SIZE-1], imc_array_1[(0+IMC_DIM-1) * SP_ROW_SIZE+ SP_ROW_SIZE-1]);
                  }
                  else {
                    //for (int n = 0; n < 3; n++) { 
                    for (int n = 0; n < IMC_DIM; n++) {
                      refresh();
                      //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+j*weight_col*weight_row*input_itr*IMC_DIM+k*weight_row*input_itr*IMC_DIM+l*input_itr*IMC_DIM+m*IMC_DIM+n) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                      dma(dma_read_base+ (j*weight_col*weight_row*input_itr*IMC_DIM+k*weight_row*input_itr*IMC_DIM+l*input_itr*IMC_DIM+(m+1)*IMC_DIM+n)* SP_ROW_SIZE);
                      memcpy( &imc_array_0[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * SP_ROW_SIZE);
                      //dma_cycles += IMC_write_latency_cycles; //DMA cycle is hidden by ping-pong
                      ping_pong_loading_cycles += IMC_write_latency_cycles;
                      leakage((AMEM_write_latency_cycles));
                    }
                    printf("%d row %d col's next pixel of W imc_array0: first=%d 2nd=%d 3rd=%d last=%d\n", k, l, imc_array_0[0 * SP_ROW_SIZE+ SP_ROW_SIZE-1], imc_array_0[(0+1) * SP_ROW_SIZE+ SP_ROW_SIZE-1], imc_array_0[(0+2) * SP_ROW_SIZE+ SP_ROW_SIZE-1], imc_array_0[(0+IMC_DIM-1) * SP_ROW_SIZE+ SP_ROW_SIZE-1]);
                  }
                }

              }
            }
          }

        }
      }
      conv_enable=false;
      return true;
    }

    // CASE A: conv1 unroll 3*3*3 inputs to do conv at once
    case CONV_LAYER_A_CONV1:{
      printf("conv1 layer start\n");
      ping_pong_computing_cycles = 0;
      ping_pong_loading_cycles = 0;
      ping_pong_computing_imc_cycles = 0;
      ping_pong_computing_input_feeding_cycles = 0;
      ping_pong_computing_output_wb_cycles = 0;
      layer_cycles = 0;
      conv_enable=true;

      imc0_computing = false;

      //bias_scale = BIAS_SCALE_CONV1;

      // first layer IMC array can be preloaded, not count cycle and energy.
      int row_length = (output_itr==1&&output_rem>0) ? output_rem : SP_ROW_SIZE;
      if (output_rem>0){
        dma(dma_read_base+input_rem*weight_row*weight_col*((output_itr-1)*SP_ROW_SIZE+output_rem));
        for (int n = 0; n < row_length; n++) {
          bias_row_0[n] = imc_w_mux_in[n];
        }
        for (int y = 0; y < input_rem*weight_row*weight_col; y++) { 
          dma(dma_read_base+y*((output_itr-1)*SP_ROW_SIZE+output_rem));
          memcpy( &imc_array_0[y * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
        }
      }
      else{
        dma(dma_read_base+input_rem*weight_row*weight_col*(output_itr*SP_ROW_SIZE));
        for (int n = 0; n < row_length; n++) {
          bias_row_0[n] = imc_w_mux_in[n];
        }
        for (int y = 0; y < input_rem*weight_row*weight_col; y++) { 
          dma(dma_read_base+y*(output_itr*SP_ROW_SIZE));
          memcpy( &imc_array_0[y * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
        }
      }
      load_first_weight = false;
      
      printf("input_itr=%d, output_itr=%d, input_rem=%d, output_rem=%d, weight_col=%d, weight_row=%d, output_col=%d, output_row=%d\n", input_itr, output_itr, input_rem, output_rem, weight_col, weight_row, output_col, output_row);
      for (int j = 0; j < output_itr;j += 1) {
        
        printf("First channel of W imc_array0: first=%d 2nd=%d 3rd=%d last=%d\n", imc_array_0[0 * SP_ROW_SIZE], imc_array_0[(0+1) * SP_ROW_SIZE], imc_array_0[(0+2) * SP_ROW_SIZE], imc_array_0[(0+26) * SP_ROW_SIZE]);
        printf("Last channel of W imc_array0: first=%d 2nd=%d 3rd=%d last=%d\n", imc_array_0[0 * SP_ROW_SIZE+MACRO_DIM-1], imc_array_0[(0+1) * SP_ROW_SIZE+MACRO_DIM-1], imc_array_0[(0+2) * SP_ROW_SIZE+MACRO_DIM-1], imc_array_0[(0+26) * SP_ROW_SIZE+MACRO_DIM-1]);

        //for (int n = 0; n < 4;n += 1)
        //  bias_enable[n] = 1;
        bias_enable[0] = 1;

        for (int m = 0; m < input_itr; m += 1) {

          // Load weights from SP -> IMC_array
          imc0_computing = !imc0_computing;

          for (int h = 0; h < output_row;h += 1) {
            //for (int i = 0; i < (output_col/4);i += 1) {
            for (int i = 0; i < output_col;i += 1) {
              //ping_pong_computing_cycles =(output_row*output_col+1)*PIPELINE_STAGE_LATENCY;
              refresh();
                          
              // selected input window from (i-1,h-1) to (i+1, h+1)
              uint32_t row_addr = dma_read_input_base + (h*output_col+i)*(m*SP_ROW_SIZE + MACRO_DIM);
              uint32_t page_base = row_addr & ~(PGSIZE - 1);          // 4k aligned base, PGSIZE=4096 defined in mmu.h
              uint32_t page_off = row_addr & (PGSIZE - 1);           // offset into page
              uint32_t bytes_in_page = PGSIZE - page_off;
              //printf("byte in page = %d\n", bytes_in_page);
              
              if (bytes_in_page >= MACRO_DIM) {
                char* ptr = ((sim_t*)sim)->get_mem_ptr(row_addr);
                if (!ptr) {
                      fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr);
                      return false;
                  }
                memcpy( &imc_dac_input[0], ptr, MACRO_DIM);
              }
              else {
                char* ptr1 = ((sim_t*)sim)->get_mem_ptr(row_addr);
                if (!ptr1) {
                      fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr);
                      return false;
                  }
                memcpy( &imc_dac_input[0], ptr1, bytes_in_page);

                uint32_t bytes_in_next_page = MACRO_DIM - bytes_in_page;
                uint32_t row_addr2 = row_addr + bytes_in_page;
                char* ptr2 = ((sim_t*)sim)->get_mem_ptr(row_addr2);
                if (!ptr2) {
                      fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr2);
                      return false;
                  }
                memcpy( &imc_dac_input[bytes_in_page], ptr2, bytes_in_next_page);
              }
              //memcpy( &input_mux_in[0], &amem_0[amem_read_base* SP_ROW_SIZE + (output_row_index_fused[c]*output_col_fused[c]+i)*(m*SP_ROW_SIZE + MACRO_DIM)], sizeof(int32_t) * MACRO_DIM);
              for (int n = 0; n < SP_ROW_SIZE;n += 1){
                if (n<MACRO_DIM){
                  input_buffer[n] = imc_dac_input[n];
                  //printf("%d ", imc_dac_input[n]);
                }
                else
                  input_buffer[n] = 0;
              }

              //only count 1 time due to pipeline
              ping_pong_computing_input_feeding_cycles += DAC_CYCLE;
              //amem_read_energy += DAC_CYCLE * input_rem*weight_row*weight_col * AMEM_128_R_POWER/128;
              ADDA_energy += DAC_CYCLE * input_rem*weight_row*weight_col * DAC_POWER;
              if (h==0&&i == 0){
                layer_cycles +=DAC_CYCLE;
                leakage((DAC_CYCLE));
                refresh();
              }

              compute();
              ping_pong_computing_imc_cycles += mvm_latency_cycles;
              imc_computing_energy += mvm_latency_cycles * input_rem*weight_row*weight_col * AAIMC_SRAM_128IN_POWER /128;
              if (h==0&&i == 0){
                layer_cycles +=mvm_latency_cycles;
                leakage((mvm_latency_cycles));
                refresh();
              }

              /*for (int x = 0; x < 4;x += 1){
                memcpy( &output_buffer[x * MACRO_DIM], &output_vector[x][x * MACRO_DIM], sizeof(int32_t) * MACRO_DIM);
              }

              for (int x = 0; x < SP_ROW_SIZE; ++x) 
                amem_0[(row_index + h*(output_col/4)*output_itr+i*output_itr) * SP_ROW_SIZE+ x] = ( output_buffer[x])/amem_scale;
              */
              memcpy( &output_buffer[0], &output_vector[0][0], sizeof(int32_t) * row_length);
              for (int x = 0; x < row_length; ++x){
                if (output_rem>0){
                  amem_0[row_index*IMC_DIM + (h*output_col+i)*((output_itr-1)*IMC_DIM+output_rem)+j*IMC_DIM+ x] = ( output_buffer[x])/amem_scale;
                }
                else{
                  amem_0[row_index*IMC_DIM + (h*output_col+i)*output_itr*IMC_DIM+j*IMC_DIM+ x] = ( output_buffer[x])/amem_scale;
                }
              }
              amem_row_cycles_0[row_index + (h*output_col+i)*output_itr+j] = 0;
              ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
              layer_cycles +=AMEM_write_latency_cycles;
              leakage((AMEM_write_latency_cycles));
              refresh(); 

              if (SRAM_IMC==1){
                imc_computing_energy += mvm_latency_cycles * row_length * AAIMC_SRAM_128O_POWER /128;
              }
              else{
                imc_computing_energy += mvm_latency_cycles * row_length * AAIMC_AMEM_32O_POWER/32;
              }
              amem_write_energy += AMEM_write_latency_cycles * row_length * AMEM_128_W_POWER/128;
            }
          }
        }

        // In conv1, computing time is larger than loading
        if (j==output_itr-1){
          // Ping-Pong imc array1 loading for next conv layer
          row_length = (next_conv_channel<IMC_DIM) ? next_conv_channel : SP_ROW_SIZE;
          int dma_row_index = (output_itr==1&&output_rem>0) ? output_rem : SP_ROW_SIZE;
          
          if (output_rem>0){
            for (int n = 0; n < dma_row_index; n++) { 
              dma(dma_read_base+input_rem*weight_row*weight_col*((output_itr-1)*SP_ROW_SIZE+output_rem)+((output_itr-1)*SP_ROW_SIZE+output_rem)+n*next_conv_channel);
              if (imc0_computing) {
                memcpy( &imc_array_1[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
              }
              else {
                memcpy( &imc_array_0[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
              }
              ping_pong_loading_cycles += IMC_write_latency_cycles;
              if (PP_IMC_ENABLE==0){
                layer_cycles +=IMC_write_latency_cycles;
                leakage((IMC_write_latency_cycles));
                refresh();
              }
            }
          }
          else{
            for (int n = 0; n < dma_row_index; n++) { 
              dma(dma_read_base+input_rem*weight_row*weight_col*(output_itr*SP_ROW_SIZE)+output_itr*SP_ROW_SIZE+n*next_conv_channel);
              if (imc0_computing) {
                memcpy( &imc_array_1[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
              }
              else {
                memcpy( &imc_array_0[n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
              }
              ping_pong_loading_cycles += IMC_write_latency_cycles;
              if (PP_IMC_ENABLE==0){
                layer_cycles +=IMC_write_latency_cycles;
                leakage((IMC_write_latency_cycles));
                refresh();
              }
            }
          }
          ping_pong_loading_cycles += DMA_LOADING_CYCLE;// pipeline DMA and IMC write, latency = cycle +1;
          if (PP_IMC_ENABLE==0){
            layer_cycles +=DMA_LOADING_CYCLE;
            leakage((DMA_LOADING_CYCLE));
            refresh();
          }
          if (SRAM_IMC==0){
            imc_loading_energy += dma_row_index * IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
          }
          else{
            imc_loading_energy += dma_row_index * IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_SRAM_512b_W_POWER/512;
          }

          if (DRAM_ENABLE==1){
            imc_loading_energy += dma_row_index * DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
          }
          else{
            imc_loading_energy += dma_row_index * DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
          }
        }
        else{
          // Load bias from SP -> IMC_row
          row_length = (j==output_itr-2) ? output_rem : SP_ROW_SIZE;
          if (output_rem>0){
            dma(dma_read_base+input_rem*weight_row*weight_col*((output_itr-1)*SP_ROW_SIZE+output_rem)+(j+1)*SP_ROW_SIZE);
            for (int n = 0; n < row_length; n++) {
              bias_row_0[n] = imc_w_mux_in[n];
            }
            ping_pong_loading_cycles += IMC_write_latency_cycles;
            if (PP_IMC_ENABLE==0){
              layer_cycles +=IMC_write_latency_cycles;
              leakage((IMC_write_latency_cycles));
              refresh();
            }
            for (int y = 0; y < input_rem*weight_row*weight_col; y++) { 
              dma(dma_read_base+y*((output_itr-1)*SP_ROW_SIZE+output_rem)+(j+1)*SP_ROW_SIZE);
              memcpy( &imc_array_0[y * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
              ping_pong_loading_cycles += IMC_write_latency_cycles;
              if (PP_IMC_ENABLE==0){
                layer_cycles +=IMC_write_latency_cycles;
                leakage((IMC_write_latency_cycles));
                refresh();
              }
            }
          }
          else{
            dma(dma_read_base+input_rem*weight_row*weight_col*(output_itr*SP_ROW_SIZE)+(j+1)*SP_ROW_SIZE);
            for (int n = 0; n < row_length; n++) {
              bias_row_0[n] = imc_w_mux_in[n];
            }
            ping_pong_loading_cycles += IMC_write_latency_cycles;
            if (PP_IMC_ENABLE==0){
              layer_cycles +=IMC_write_latency_cycles;
              leakage((IMC_write_latency_cycles));
              refresh();
            }
            for (int y = 0; y < input_rem*weight_row*weight_col; y++) { 
              dma(dma_read_base+y*(output_itr*SP_ROW_SIZE)+(j+1)*SP_ROW_SIZE);
              memcpy( &imc_array_0[y * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
              ping_pong_loading_cycles += IMC_write_latency_cycles;
              if (PP_IMC_ENABLE==0){
                layer_cycles +=IMC_write_latency_cycles;
                leakage((IMC_write_latency_cycles));
                refresh();
              }
            }
          }
          ping_pong_loading_cycles += DMA_LOADING_CYCLE;// pipeline DMA and IMC write, latency = cycle +1;
          if (PP_IMC_ENABLE==0){
            layer_cycles +=DMA_LOADING_CYCLE;
            leakage((DMA_LOADING_CYCLE));
            refresh();
          }
          if (SRAM_IMC==0){
            imc_loading_energy += input_rem*weight_row*weight_col * IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
          }
          else{
            imc_loading_energy += input_rem*weight_row*weight_col * IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_SRAM_512b_W_POWER/512;
          }

          if (DRAM_ENABLE==1){
            imc_loading_energy += input_rem*weight_row*weight_col * DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
          }
          else{
            imc_loading_energy += input_rem*weight_row*weight_col * DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
          }
        }
        

        /*
        for (int y = 0; y < 7; y++) { 
          //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+y) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
          dma(dma_read_base+(y)* SP_ROW_SIZE);
          for (int x = 0; x < 4; x++) {
            refresh(); 
            if (y==6){
              for (int n = 0; n < 3; n++) {
                memcpy( &imc_array_0[x*MACRO_DIM*SP_ROW_SIZE + (y*4+n)*SP_ROW_SIZE + x*MACRO_DIM], &imc_w_mux_in[n * MACRO_DIM], sizeof(uint8_t) * MACRO_DIM);
                dma_cycles += IMC_write_latency_cycles;
                leakage(IMC_write_latency_cycles);
              }
            }
            else {
              for (int n = 0; n < 4; n++) {
                //memcpy( &imc_w_mux_in[0], &sp_weight[(sp_weight_write_base+y) * SP_ROW_SIZE], sizeof(uint8_t) * SP_ROW_SIZE);
                memcpy( &imc_array_0[x*MACRO_DIM*SP_ROW_SIZE + (y*4+n)*SP_ROW_SIZE + x*MACRO_DIM], &imc_w_mux_in[n * MACRO_DIM], sizeof(uint8_t) * MACRO_DIM);
                dma_cycles += IMC_write_latency_cycles;
                leakage(IMC_write_latency_cycles);
              }
            }
          }
        }*/
      }

      
      //output trace
      dma_cycles += ping_pong_loading_cycles;
      imc_cycles += ping_pong_computing_imc_cycles;
      amem_r_cycles += ping_pong_computing_input_feeding_cycles;
      amem_w_cycles += ping_pong_computing_output_wb_cycles;
      total_layer_cycles += layer_cycles;
      total_cycles = total_layer_cycles+refresh_cycles;

      printf("Data loading cycle= %lu\n", dma_cycles);
      printf("IMC cycle= %lu\n", imc_cycles);
      printf("ANU cycle= %lu\n", anu_cycles);
      printf("AMEM read cycle= %lu\n", amem_r_cycles);
      printf("AMEM write cycle= %lu\n", amem_w_cycles);
      printf("refresh cycle= %lu\n", refresh_cycles);
      printf("Total cycle= %lu\n", total_cycles);

      //energy
      amem_leakage_energy += ((layer_cycles) * AMEM_ARRAY_SIZE + layer_cycles*FUSED_AMEM_ARRAY_SIZE) * AMEM_64K_Leakage_POWER/(64*1024);

      if (SRAM_IMC==1){
        imc_leakage_energy += (PP_IMC_ENABLE*((layer_cycles-ping_pong_computing_imc_cycles) * IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024)
                              + (layer_cycles-ping_pong_loading_cycles) * IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024))
                              + layer_cycles * FUSED_IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024));
      }
      else{
        imc_leakage_energy += (PP_IMC_ENABLE*((layer_cycles-ping_pong_computing_imc_cycles) * IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024)
                              + (layer_cycles-ping_pong_loading_cycles) * IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024))
                              + layer_cycles * FUSED_IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024));
      }
      //anu_computing_energy = ping_pong_computing_anu_cycles * IMC_DIM * MAX_POOL_ANU_128O_POWER/128;
      

      sram_buffer_leakage_energy += ((layer_cycles) * ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_Leakage_POWER)/(5152*1024);
      sram_leakage_energy_in_refresh = refresh_cycles * ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_Leakage_POWER/(5152*1024)
                                        + SRAM_IMC * ((1+PP_IMC_ENABLE)* IMC_ARRAY_SIZE + FUSED_IMC_ARRAY_SIZE)* AAIMC_PRECISION * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024);
                                        
      fsm_energy = total_cycles * CTRL_FSM_POWER;
      //risc_energy = total_cycles * RISCV32_POWER;
      total_energy = amem_read_energy + amem_write_energy + amem_leakage_energy +  refresh_energy+
                    imc_computing_energy + anu_computing_energy + imc_loading_energy + sram_leakage_energy_in_refresh+
                    imc_leakage_energy + sram_buffer_leakage_energy + fsm_energy + ADDA_energy;

      printf("Total energy= %lu\n", total_energy);
      printf("AMEM read energy= %lu\n", amem_read_energy);
      printf("AMEM write energy= %lu\n", amem_write_energy);
      printf("AMEM leakage energy= %lu\n", amem_leakage_energy);
      printf("IMC computing energy= %lu\n", imc_computing_energy);
      printf("ANU computing energy= %lu\n", anu_computing_energy);
      printf("IMC loading energy= %lu\n", imc_loading_energy);
      printf("IMC leakage energy= %lu\n", imc_leakage_energy);
      printf("SRAM buffer leakage energy= %lu\n", sram_buffer_leakage_energy);
      printf("FSM energy= %lu\n", fsm_energy);
      conv_enable=false;
      printf("conv1 finish\n");
      printf("amem row cycles first = %d ofm = %d ofm2 = %d last = %d\n", amem_row_cycles_0[0], amem_row_cycles_0[row_index], amem_row_cycles_0[row_index+2], amem_row_cycles_0[AMEM_ROW_INDEX-1]);
      printf("amem first channel 1st result= %d 2nd = %d last-1 = %d last = %d\n", amem_0[(row_index+0)*SP_ROW_SIZE], amem_0[(row_index+0)*SP_ROW_SIZE+1], amem_0[(row_index+0)*SP_ROW_SIZE+MACRO_DIM-2], amem_0[(row_index+0)*SP_ROW_SIZE+MACRO_DIM-1]);
      for (int n = 3*MACRO_DIM; n < 4*MACRO_DIM;n += 1)
        printf("%d ", amem_0[(row_index+255)*SP_ROW_SIZE+n]);
      ie = 1;
      update_interrupts();
      return true;
    }



    // CASE B: Storge all weights on IMC_ARRAY and use streaming dataflow (layer fusion) to reduced AMEM usage
    // Row by row processing for: convrelu1(64, 3, 3, 3), maxpool1(2,2), convrelu2(64, 64, 3, 3), maxpool2(2,2), convrelu3(128, 64, 3, 3), convrelu3(64, 128, 1, 1), convrelu4(128, 64, 3, 3), maxpool3(2,2)
    case LAYER_FUSED:{
      printf("layer fusion start\n");
      ping_pong_computing_cycles = 0;
      ping_pong_loading_cycles = 0;
      ping_pong_computing_imc_cycles = 0;
      ping_pong_computing_input_feeding_cycles = 0;
      ping_pong_computing_output_wb_cycles = 0;
      layer_cycles = 0;
      conv_enable=true;
      
      int run_layer[NUM_FUSED_LAYERS];
      run_layer[0] = 1;
      for (int c = 1; c < NUM_FUSED_LAYERS; c++){
        run_layer[c] = 0;
      }

      //weight&bias storge. IMC can be preloeaded so no need to count cycle and energy
      int dma_weight_base;
      int dma_row_index;
      int row_length;
      for (int c = 0; c < NUM_FUSED_LAYERS; c++){
        printf("layer %d: input_itr=%d, output_itr=%d, input_rem=%d, output_rem=%d, weight_col=%d, weight_row=%d, output_col=%d, output_row=%d, stride=%d, padding=%d\n", c, input_itr_fused[c], output_itr_fused[c], input_rem_fused[c], output_rem_fused[c], weight_col_fused[c], weight_row_fused[c], output_col_fused[c], output_row_fused[c], stride_fused[c], padding_fused[c]);
        output_row_index_fused[c] = 0; // init
        amem_row_index_fused[c] = 0;

        if (layer_type[c]>0){
        }
        else{
          if (input_rem_fused[c]>0&&output_rem_fused[c]>0){
            for (int j = 0; j < output_itr_fused[c];j += 1) {
              for (int m = 0; m < input_itr_fused[c]; m += 1) {
                row_length = (j==output_itr_fused[c]-1) ? output_rem_fused[c] : SP_ROW_SIZE;
                if(c==0){
                  dma_weight_base = dma_read_base;
                  for (int n = 0; n < input_rem_fused[c]*weight_row_fused[c]*weight_col_fused[c]; n++) {
                    dma(dma_weight_base + n * ((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c]));
                    /*if (SRAM_IMC==0){
                      dma_cycles += IMC_write_latency_cycles;
                      leakage((IMC_write_latency_cycles));// assume AMEM based IMC
                      refresh();
                      imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                      if (DRAM_ENABLE==1){
                        imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
                      }
                      else{
                        imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
                      }
                    }*/

                    memcpy( &imc_array[c][m*output_itr_fused[c]+j][n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * output_rem_fused[c]);
                    
                  }
                  for (int i=0*MACRO_DIM; i<MACRO_DIM; i++)
                    printf("%d ", imc_array[c][m*output_itr_fused[c]+j][i]);
                  printf("\n");
                }
                else{
                  for (int k = 0; k < weight_row_fused[c]; k += 1) {
                    for (int l = 0; l < weight_col_fused[c]; l += 1) {
                      dma_weight_base = dma_read_base + (k*weight_row_fused[c]+l)*((input_itr_fused[c]-1)*SP_ROW_SIZE+input_rem_fused[c])*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+ m*SP_ROW_SIZE*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+ j*SP_ROW_SIZE;
                      dma_row_index = (m==input_itr_fused[c]-1) ? input_rem_fused[c] : SP_ROW_SIZE;
                      row_length = (j==output_itr_fused[c]-1) ? output_rem_fused[c] : SP_ROW_SIZE;
                      for (int n = 0; n < dma_row_index; n++) {
                        dma(dma_weight_base + n * ((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c]));
                        /*if (SRAM_IMC==0){
                          dma_cycles += IMC_write_latency_cycles;
                          leakage((IMC_write_latency_cycles));// assume AMEM based IMC
                          refresh();
                          imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                          if (DRAM_ENABLE==1){
                            imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
                          }
                          else{
                            imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
                          }
                        }*/


                        memcpy( &imc_array[c][(k*weight_col_fused[c]+l)*input_itr_fused[c]*output_itr_fused[c]+m*output_itr_fused[c]+j][n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
                        //for (int i=0*MACRO_DIM; i<MACRO_DIM; i++)
                        //  printf("%d ", imc_array[c][(k*weight_col_fused[c]+l)*input_itr_fused[c]*output_itr_fused[c]+m*output_itr_fused[c]+j][n * SP_ROW_SIZE+i]);
                        //printf("\n");
                      }
                    }
                  }
                }
              }
              dma_weight_base = dma_read_base+ (weight_col_fused[c]*weight_row_fused[c])*((input_itr_fused[c]-1)*SP_ROW_SIZE+input_rem_fused[c])*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+ j*SP_ROW_SIZE;
              dma(dma_weight_base);
              /*if (SRAM_IMC==0){
                dma_cycles += IMC_write_latency_cycles+DMA_LOADING_CYCLE; // pipeline DMA and IMC write, latency = cycle +1;
                leakage((IMC_write_latency_cycles+DMA_LOADING_CYCLE));// assume AMEM based IMC
                refresh();
                imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                if (DRAM_ENABLE==1){
                  imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
                }
                else{
                  imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
                }
              }*/
              memcpy( &bias_row[c][j][0], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
              printf("bias_row: first=%d 11th=%d last=%d\n", bias_row[c][j][0], bias_row[c][j][10], bias_row[c][j][SP_ROW_SIZE-1]);
            }
            dma_read_base = dma_read_base + (weight_col_fused[c]*weight_row_fused[c])*((input_itr_fused[c]-1)*SP_ROW_SIZE+input_rem_fused[c])*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c]) + ((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c]);
          }
          else if (input_rem_fused[c]>0){
            for (int j = 0; j < output_itr_fused[c];j += 1) {
              for (int m = 0; m < input_itr_fused[c]; m += 1) {
                for (int k = 0; k < weight_row_fused[c]; k += 1) {
                  for (int l = 0; l < weight_col_fused[c]; l += 1) {
                    
                    dma_weight_base = dma_read_base + (k*weight_row_fused[c]+l)*((input_itr_fused[c]-1)*SP_ROW_SIZE+input_rem_fused[c])*(output_itr_fused[c]*SP_ROW_SIZE)+ m*SP_ROW_SIZE*(output_itr_fused[c]*SP_ROW_SIZE)+ j*SP_ROW_SIZE;
                    dma_row_index = (m==input_itr_fused[c]-1) ? input_rem_fused[c] : SP_ROW_SIZE;
                    row_length = SP_ROW_SIZE;
                    for (int n = 0; n < dma_row_index; n++) {
                      dma(dma_weight_base + n * output_itr_fused[c]*SP_ROW_SIZE);
                      /*if (SRAM_IMC==0){
                        dma_cycles += IMC_write_latency_cycles;
                        leakage((IMC_write_latency_cycles));// assume AMEM based IMC
                        refresh();
                        imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                        if (DRAM_ENABLE==1){
                          imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
                        }
                        else{
                          imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
                        }
                      }*/

                      memcpy( &imc_array[c][(k*weight_col_fused[c]+l)*input_itr_fused[c]*output_itr_fused[c]+m*output_itr_fused[c]+j][n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
                    }
                  }
                }
              }
              dma_weight_base = dma_read_base+ (weight_col_fused[c]*weight_row_fused[c])*((input_itr_fused[c]-1)*SP_ROW_SIZE+input_rem_fused[c])*(output_itr_fused[c]*SP_ROW_SIZE)+ j*SP_ROW_SIZE;
              dma(dma_weight_base);
              /*if (SRAM_IMC==0){
                dma_cycles += IMC_write_latency_cycles+DMA_LOADING_CYCLE; // pipeline DMA and IMC write, latency = cycle +1;
                leakage((IMC_write_latency_cycles+DMA_LOADING_CYCLE));// assume AMEM based IMC
                refresh();
                imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                if (DRAM_ENABLE==1){
                  imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
                }
                else{
                  imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
                }
              }*/
              memcpy( &bias_row[c][j][0], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
              printf("bias_row: first=%d 11th=%d last=%d\n", bias_row[c][j][0], bias_row[c][j][10], bias_row[c][j][SP_ROW_SIZE-1]);
            }
            dma_read_base = dma_read_base + (weight_col_fused[c]*weight_row_fused[c])*((input_itr_fused[c]-1)*SP_ROW_SIZE+input_rem_fused[c])*(output_itr_fused[c]*SP_ROW_SIZE) + (output_itr_fused[c]*SP_ROW_SIZE);
          }
          else if (output_rem_fused[c]>0){
            for (int j = 0; j < output_itr_fused[c];j += 1) {
              for (int m = 0; m < input_itr_fused[c]; m += 1) {
                for (int k = 0; k < weight_row_fused[c]; k += 1) {
                  for (int l = 0; l < weight_col_fused[c]; l += 1) {
                    
                    dma_weight_base = dma_read_base + (k*weight_row_fused[c]+l)*(input_itr_fused[c]*SP_ROW_SIZE)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+ m*SP_ROW_SIZE*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+ j*SP_ROW_SIZE;
                    dma_row_index = SP_ROW_SIZE;
                    row_length = (j==output_itr_fused[c]-1) ? output_rem_fused[c] : SP_ROW_SIZE;
                    for (int n = 0; n < dma_row_index; n++) {
                      dma(dma_weight_base + n * ((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c]));
                      /*if (SRAM_IMC==0){
                        dma_cycles += IMC_write_latency_cycles;
                        leakage((IMC_write_latency_cycles));// assume AMEM based IMC
                        refresh();
                        imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                        if (DRAM_ENABLE==1){
                          imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
                        }
                        else{
                          imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
                        }
                      }*/

                      memcpy( &imc_array[c][(k*weight_col_fused[c]+l)*input_itr_fused[c]*output_itr_fused[c]+m*output_itr_fused[c]+j][n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
                    }
                  }
                }
              }
              dma_weight_base = dma_read_base+ (weight_col_fused[c]*weight_row_fused[c])*(input_itr_fused[c]*SP_ROW_SIZE)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+ j*SP_ROW_SIZE;
              dma(dma_weight_base);
              /*if (SRAM_IMC==0){
                dma_cycles += IMC_write_latency_cycles+DMA_LOADING_CYCLE; // pipeline DMA and IMC write, latency = cycle +1;
                leakage((IMC_write_latency_cycles+DMA_LOADING_CYCLE));// assume AMEM based IMC
                refresh();
                imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                if (DRAM_ENABLE==1){
                  imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
                }
                else{
                  imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
                }
              }*/
              memcpy( &bias_row[c][j][0], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
              printf("bias_row: first=%d 11th=%d last=%d\n", bias_row[c][j][0], bias_row[c][j][10], bias_row[c][j][SP_ROW_SIZE-1]);
            }
            dma_read_base = dma_read_base + (weight_col_fused[c]*weight_row_fused[c])*(input_itr_fused[c]*SP_ROW_SIZE)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c]) + ((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c]);
          }
          else {
            for (int j = 0; j < output_itr_fused[c];j += 1) {
              for (int m = 0; m < input_itr_fused[c]; m += 1) {
                for (int k = 0; k < weight_row_fused[c]; k += 1) {
                  for (int l = 0; l < weight_col_fused[c]; l += 1) {
                    
                    dma_weight_base = dma_read_base + (k*weight_row_fused[c]+l)*(input_itr_fused[c]*SP_ROW_SIZE)*(output_itr_fused[c]*SP_ROW_SIZE)+ m*SP_ROW_SIZE*(output_itr_fused[c]*SP_ROW_SIZE)+ j*SP_ROW_SIZE;
                    /*if (SRAM_IMC==0&&DRAM_ENABLE==1){ // count_cycle only for AMEM_IMC
                      imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                    }*/
                    dma_row_index = SP_ROW_SIZE;
                    row_length = SP_ROW_SIZE;
                    for (int n = 0; n < dma_row_index; n++) {
                      dma(dma_weight_base + n * output_itr_fused[c]*SP_ROW_SIZE);
                      
                      /*if (SRAM_IMC==0){
                        dma_cycles += IMC_write_latency_cycles;
                        leakage((IMC_write_latency_cycles));// assume AMEM based IMC
                        refresh();
                        imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                        if (DRAM_ENABLE==1){
                          imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
                        }
                        else{
                          imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
                        }
                      }*/
                      memcpy( &imc_array[c][(k*weight_col_fused[c]+l)*input_itr_fused[c]*output_itr_fused[c]+m*output_itr_fused[c]+j][n * SP_ROW_SIZE], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);

                    }
                    //printf("weight_store finish, output_itr=%d, input_itr=%d, weight_row=%d, weight_col=%d\n", j, m, k, l);
                  }
                }
              }
              dma_weight_base = dma_read_base+ (weight_col_fused[c]*weight_row_fused[c])*(input_itr_fused[c]*SP_ROW_SIZE)*(output_itr_fused[c]*SP_ROW_SIZE)+ j*SP_ROW_SIZE;
              dma(dma_weight_base);
              
              /*if (SRAM_IMC==0){
                dma_cycles += IMC_write_latency_cycles+DMA_LOADING_CYCLE; // pipeline DMA and IMC write, latency = cycle +1;
                leakage((IMC_write_latency_cycles+DMA_LOADING_CYCLE));// assume AMEM based IMC
                refresh();
                imc_loading_energy += IMC_write_latency_cycles * AAIMC_PRECISION * row_length * AAIMC_AMEM_32x4b_W_POWER/128;
                if (DRAM_ENABLE==1){
                  imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * DRAM_512b_DMA_POWER/512;
                }
                else{
                  imc_loading_energy += DMA_LOADING_CYCLE * AAIMC_PRECISION * row_length * SRAM_512b_DMA_POWER/512;
                }
              }*/
              memcpy( &bias_row[c][j][0], &imc_w_mux_in[0], sizeof(uint8_t) * row_length);
              printf("bias_row: first=%d 11th=%d last=%d\n", bias_row[c][j][0], bias_row[c][j][10], bias_row[c][j][SP_ROW_SIZE-1]);
            }
            dma_read_base = dma_read_base + (weight_col_fused[c]*weight_row_fused[c])*(input_itr_fused[c]*SP_ROW_SIZE)*(output_itr_fused[c]*SP_ROW_SIZE) + (output_itr_fused[c]*SP_ROW_SIZE);
          }
        }
      }
      

      if (SRAM_IMC==0){ // count_cycle only for AMEM_IMC
        layer_cycles += dma_cycles; 
      }

      //run layer in fused_layers
      while (output_row_index_fused[NUM_FUSED_LAYERS-1] < output_row_fused[NUM_FUSED_LAYERS-1]) {
        printf("last output_row_index_fused=%d last output_row_fused=%d\n", output_row_index_fused[NUM_FUSED_LAYERS-1], output_row_fused[NUM_FUSED_LAYERS-1]);
        for (int c = 0; c < NUM_FUSED_LAYERS; c++){

          if (run_layer[c]){
            printf("generating %dth output row of layer %d\n", output_row_index_fused[c], c);
            for (int j = 0; j < output_itr_fused[c];j += 1) {
              //maxpool
              if (layer_type[c]==1){
                for (int i = 0; i < output_col_fused[c]; i += 1) {
                  refresh();
                  
                  if (c==NUM_FUSED_LAYERS-1){
                    if (output_rem_fused[c] >0 ){
                      row_length = j==output_itr_fused[c]-1 ? output_rem_fused[c] : SP_ROW_SIZE;
                      for (int l = 0; l < row_length; ++l) {
                        amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ l] = std::max({amem_fused[c-1][(2*0*2*output_col_fused[c] + 2*i)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+j*SP_ROW_SIZE+ l], amem_fused[c-1][(2*0*2*output_col_fused[c] + 2*i+1)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+j*SP_ROW_SIZE+ l], amem_fused[c-1][((2*0+1)*2*output_col_fused[c] + 2*i)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+j*SP_ROW_SIZE+ l], amem_fused[c-1][((2*0+1)*2*output_col_fused[c] + 2*i+1)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+j*SP_ROW_SIZE+ l]});
                      }
                      if ((output_row_index_fused[c]*output_col_fused[c]+i)%(IMC_DIM/output_rem_fused[c])==0)
                        amem_row_cycles_0[row_index + (output_row_index_fused[c]*output_col_fused[c]+i)*(output_itr_fused[c]-1+output_rem_fused[c]/IMC_DIM)+j] = 0;
                    }

                    else{
                      row_length = SP_ROW_SIZE;
                      for (int l = 0; l < SP_ROW_SIZE; ++l) {
                        amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*(output_itr_fused[c]*IMC_DIM)+j*IMC_DIM+ l] = std::max({amem_fused[c-1][(2*0*2*output_col_fused[c] + 2*i)*output_itr_fused[c]*SP_ROW_SIZE+j*SP_ROW_SIZE+ l], amem_fused[c-1][(2*0*2*output_col_fused[c] + 2*i+1)*output_itr_fused[c]*SP_ROW_SIZE+j*SP_ROW_SIZE+ l], amem_fused[c-1][((2*0+1)*2*output_col_fused[c] + 2*i)*output_itr_fused[c]*SP_ROW_SIZE+j*SP_ROW_SIZE+ l], amem_fused[c-1][((2*0+1)*2*output_col_fused[c] + 2*i+1)*output_itr_fused[c]*SP_ROW_SIZE+j*SP_ROW_SIZE+ l]});
                      }
                      amem_row_cycles_0[row_index + (output_row_index_fused[c]*output_col_fused[c]+i)*output_itr_fused[c]+j] = 0;
                    }
                  }
                  else {
                    if (output_rem_fused[c] >0 ){
                      row_length = j==output_itr_fused[c]-1 ? output_rem_fused[c] : SP_ROW_SIZE;
                      for (int l = 0; l < row_length; ++l) {
                        amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ l] = std::max({amem_fused[c-1][(2*0*2*output_col_fused[c] + 2*i)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+j*SP_ROW_SIZE+ l], amem_fused[c-1][(2*0*2*output_col_fused[c] + 2*i+1)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+j*SP_ROW_SIZE+ l], amem_fused[c-1][((2*0+1)*2*output_col_fused[c] + 2*i)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+j*SP_ROW_SIZE+ l], amem_fused[c-1][((2*0+1)*2*output_col_fused[c] + 2*i+1)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+j*SP_ROW_SIZE+ l]});
                      }
                      if (((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)%(IMC_DIM/output_rem_fused[c])==0)
                        amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*(output_itr_fused[c]-1+output_rem_fused[c]/IMC_DIM)+j] = 0;
                    }

                    else{
                      row_length = SP_ROW_SIZE;
                      for (int l = 0; l < SP_ROW_SIZE; ++l) {
                        amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*(output_itr_fused[c]*IMC_DIM)+j*IMC_DIM+ l] = std::max({amem_fused[c-1][(2*0*2*output_col_fused[c] + 2*i)*output_itr_fused[c]*SP_ROW_SIZE+j*SP_ROW_SIZE+ l], amem_fused[c-1][(2*0*2*output_col_fused[c] + 2*i+1)*output_itr_fused[c]*SP_ROW_SIZE+j*SP_ROW_SIZE+ l], amem_fused[c-1][((2*0+1)*2*output_col_fused[c] + 2*i)*output_itr_fused[c]*SP_ROW_SIZE+j*SP_ROW_SIZE+ l], amem_fused[c-1][((2*0+1)*2*output_col_fused[c] + 2*i+1)*output_itr_fused[c]*SP_ROW_SIZE+j*SP_ROW_SIZE+ l]});
                      }
                      amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]+j] = 0;
                    }
                    if (output_row_index_fused[c]==0&i==0){
                      for (int n=0*MACRO_DIM; n<2*MACRO_DIM; n++)
                        printf("%d ", amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n]);
                      printf("\n");
                      for (int n=0*MACRO_DIM; n<2*MACRO_DIM; n++)
                        printf("%d ", amem_fused[c-1][((2*0+1)*2*output_col_fused[c] + 2*i)*((output_itr_fused[c]-1)*SP_ROW_SIZE+output_rem_fused[c])+j*SP_ROW_SIZE+ n]);
                      printf("\n");
                    }
                  }
                  
                  ping_pong_computing_input_feeding_cycles += 2*AMEM_read_latency_cycles;
                  leakage((2*AMEM_read_latency_cycles));
                  refresh();
                  anu_cycles += 1*anu_latency_cycles;

                  layer_cycles += (2*AMEM_read_latency_cycles);
                  //pipeline wb and read
                  ping_pong_computing_output_wb_cycles += 1*AMEM_write_latency_cycles;
                  if (i == output_col_fused[c]-1){
                    layer_cycles += AMEM_write_latency_cycles;
                    leakage(1*AMEM_write_latency_cycles);
                    refresh();
                  }

                  amem_read_energy +=  2*AMEM_read_latency_cycles * row_length * AMEM_128_R_POWER/128;
                  amem_write_energy += AMEM_write_latency_cycles * row_length * AMEM_128_W_POWER/128;
                  anu_computing_energy += anu_latency_cycles * row_length * MAX_POOL_ANU_128O_POWER/128;
                  //amem_leakage_energy += dma_cycles * (AMEM_ARRAY_SIZE + FUSED_AMEM_ARRAY_SIZE) * AMEM_64K_Leakage_POWER/(64*1024);


                }
              }
              //add
              else if (layer_type[c]==2){
                for (int i = 0; i < output_col_fused[c]; i += 1) {
                  refresh();

                  if (output_rem_fused[c] >0 ){
                    row_length = j==output_itr_fused[c]-1 ? output_rem_fused[c] : SP_ROW_SIZE;
                    for (int l = 0; l < row_length; ++l) {
                      amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ l] = (amem_fused[c-1][(0*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ l] + amem_fused[c-3][(((output_row_index_fused[c-3]-1)%weight_row_fused[c-2])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ l]);
                    }
                    if (((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)%(IMC_DIM/output_rem_fused[c])==0)
                      amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*(output_itr_fused[c]-1+output_rem_fused[c]/IMC_DIM)+j] = 0;
                  }

                  else{
                    row_length = SP_ROW_SIZE;
                    for (int l = 0; l < SP_ROW_SIZE; ++l) {
                      amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*(output_itr_fused[c]*IMC_DIM)+j*IMC_DIM+ l] = (amem_fused[c-1][(0*output_col_fused[c]+i)*(output_itr_fused[c]*IMC_DIM)+j*IMC_DIM+ l] + amem_fused[c-3][(((output_row_index_fused[c-3]-1)%weight_row_fused[c-2])*output_col_fused[c]+i)*(output_itr_fused[c]*IMC_DIM)+j*IMC_DIM+ l]);
                    }
                    amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]+j] = 0;
                  }
                  
                  
                  ping_pong_computing_input_feeding_cycles += 1*AMEM_read_latency_cycles;
                  leakage((1*AMEM_read_latency_cycles));
                  refresh();
                  anu_cycles += 1*anu_latency_cycles;

                  layer_cycles += (1*AMEM_read_latency_cycles);
                  //pipeline wb and read
                  ping_pong_computing_output_wb_cycles += 1*AMEM_write_latency_cycles;
                  if (i == output_col_fused[c]-1){
                    layer_cycles += AMEM_write_latency_cycles;
                    leakage(1*AMEM_write_latency_cycles);
                    refresh();
                  }

                  amem_read_energy +=  1*AMEM_read_latency_cycles * row_length * AMEM_128_R_POWER/128;
                  amem_write_energy += AMEM_write_latency_cycles * row_length * AMEM_128_W_POWER/128;
                  anu_computing_energy += anu_latency_cycles * row_length * ANU_128O_POWER/128;
                  //amem_leakage_energy += dma_cycles * (AMEM_ARRAY_SIZE + FUSED_AMEM_ARRAY_SIZE) * AMEM_64K_Leakage_POWER/(64*1024);


                }
              }
              //conv
              else{
                for (int m = 0; m < input_itr_fused[c]; m += 1) {

                  //first conv
                  if (c==0) {
                    for (int i = 0; i < output_col_fused[c]; i += 1) {
                      uint32_t row_addr = dma_read_input_base + (output_row_index_fused[c]*output_col_fused[c]+i)*(m*SP_ROW_SIZE + MACRO_DIM);
                      uint32_t page_base = row_addr & ~(PGSIZE - 1);          // 4k aligned base, PGSIZE=4096 defined in mmu.h
                      uint32_t page_off = row_addr & (PGSIZE - 1);           // offset into page
                      uint32_t bytes_in_page = PGSIZE - page_off;
                      //printf("byte in page = %d\n", bytes_in_page);
                      
                      if (bytes_in_page >= MACRO_DIM) {
                        char* ptr = ((sim_t*)sim)->get_mem_ptr(row_addr);
                        if (!ptr) {
                              fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr);
                              return false;
                          }
                        memcpy( &imc_dac_input[0], ptr, MACRO_DIM);
                      }
                      else {
                        char* ptr1 = ((sim_t*)sim)->get_mem_ptr(row_addr);
                        if (!ptr1) {
                              fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr);
                              return false;
                          }
                        memcpy( &imc_dac_input[0], ptr1, bytes_in_page);

                        uint32_t bytes_in_next_page = MACRO_DIM - bytes_in_page;
                        uint32_t row_addr2 = row_addr + bytes_in_page;
                        char* ptr2 = ((sim_t*)sim)->get_mem_ptr(row_addr2);
                        if (!ptr2) {
                              fprintf(stderr, "Invalid memory at 0x%lx\n", (long)row_addr2);
                              return false;
                          }
                        memcpy( &imc_dac_input[bytes_in_page], ptr2, bytes_in_next_page);
                      }
                      //memcpy( &input_mux_in[0], &amem_0[amem_read_base* SP_ROW_SIZE + (output_row_index_fused[c]*output_col_fused[c]+i)*(m*SP_ROW_SIZE + MACRO_DIM)], sizeof(int32_t) * MACRO_DIM);
                      for (int n = 0; n < SP_ROW_SIZE;n += 1){
                        if (n<MACRO_DIM){
                          input_buffer[n] = imc_dac_input[n];
                          //printf("%d ", imc_dac_input[n]);
                        }
                        else
                          input_buffer[n] = 0;
                      }
                      //printf("\n");

                      //only count 1 time due to pipeline
                      ping_pong_computing_input_feeding_cycles += DAC_CYCLE;
                      //amem_read_energy += AMEM_read_latency_cycles * input_rem_fused[c]*weight_row_fused[c]*weight_col_fused[c] * AMEM_128_R_POWER/128;
                      ADDA_energy += DAC_CYCLE * input_rem_fused[c]*weight_row_fused[c]*weight_col_fused[c] * DAC_POWER;
                      if (i == 0){
                        layer_cycles +=DAC_CYCLE;
                        leakage((DAC_CYCLE));
                        refresh();
                      }

                      for (int n = 1; n < ((IMC_DIM/MACRO_DIM));n += 1)
                        bias_enable[n] = 0;

                      if (bias_existing)
                        bias_enable[0] = 1;
                      else 
                        bias_enable[0] = 0;

                      for (int ii = 0; ii < IMC_DIM; ++ii) {
                        for (int jj = 0; jj < (IMC_DIM/MACRO_DIM); ++jj) {
                          int32_t acc = 0;      
                          for (int kk = 0; kk < MACRO_DIM; ++kk) {
                            acc += (imc_array[0][0][(jj*MACRO_DIM + kk) * IMC_DIM+ ii]-W_ZERO_POINT) * input_buffer[jj*MACRO_DIM + kk];
                          }
                          output_vector[jj][ii] = acc+  bias_enable[jj] * bias_scale_fused[c]*(bias_row[c][j][ii]-BIAS_ZERO_POINT);
                        }
                      }
                      //imc_cycles += mvm_latency_cycles;//imc cycle is hidden by ping-pong
                      ping_pong_computing_imc_cycles += mvm_latency_cycles;
                      if (i == 0){
                        layer_cycles +=mvm_latency_cycles;
                        leakage((mvm_latency_cycles));
                        refresh();
                      }

                      imc_computing_energy += mvm_latency_cycles *  input_rem_fused[c]*weight_row_fused[c]*weight_col_fused[c] * AAIMC_SRAM_128IN_POWER /128;
                      if (SRAM_IMC==1){
                        imc_computing_energy += mvm_latency_cycles * output_rem_fused[c] * AAIMC_SRAM_128O_POWER /128;
                      }
                      else{
                        imc_computing_energy += mvm_latency_cycles * output_rem_fused[c] * AAIMC_AMEM_32O_POWER/32;
                      }

                      row_length = (j==output_itr_fused[c]-1)&&output_rem_fused[c] ? output_rem_fused[c] : SP_ROW_SIZE;
                      if (output_rem_fused[c]>0){
                        for (int n = 0; n < row_length;n += 1){
                          for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                            if (mi==0)
                              output_buffer[n] = output_vector[mi][n];
                            else
                              output_buffer[n] += output_vector[mi][n];
                          }
                          amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] = (output_buffer[n]>0) ? (output_buffer[n])/amem_scale_fused[c] : 0;
                        }
                        if (((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)%(IMC_DIM/output_rem_fused[c])==0)
                          amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*(output_itr_fused[c]-1+output_rem_fused[c]/IMC_DIM)+j] = 0;
                      }
                      else {
                        for (int n = 0; n < IMC_DIM;n += 1){
                          for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                            if (mi==0)
                              output_buffer[n] = output_vector[mi][n];
                            else
                              output_buffer[n] += output_vector[mi][n];
                          }
                          amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] = (output_buffer[n]>0) ? (output_buffer[n])/amem_scale_fused[c] : 0;
                        }
                        amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]+j] = 0;
                      }

                      ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
                      layer_cycles +=AMEM_write_latency_cycles;
                      leakage((AMEM_write_latency_cycles));
                      refresh();

                      amem_write_energy += AMEM_write_latency_cycles * row_length * AMEM_128_W_POWER/128;

                      if (output_row_index_fused[c]==1&i==1){
                        
                        for (int n=0*MACRO_DIM; n<MACRO_DIM; n++)
                          printf("%d ", imc_array[0][0][n]);
                        printf("\n");
                        for (int n=0*MACRO_DIM; n<MACRO_DIM; n++)
                          printf("%d ", input_buffer[n]);
                        printf("\n");
                        for (int n=0*MACRO_DIM; n<2*MACRO_DIM; n++)
                          printf("%d ", amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n]);
                        printf("\n");
                      }
                    }
                    
                  }
                  
                  //other
                  else {
                    if (output_row_index_fused[c]<padding_fused[c]){
                      k_start = padding_fused[c]-output_row_index_fused[c];
                      k_end = weight_row_fused[c];
                    }
                    else if (output_row_index_fused[c]>output_row_fused[c]-padding_fused[c]-1){
                      k_start = 0;
                      k_end = weight_row_fused[c]-(output_row_index_fused[c]-output_row_fused[c]+padding_fused[c]+1);
                    }
                    else{
                      k_start = 0;
                      k_end = weight_row_fused[c];
                    }

                    for (int k = k_start; k < k_end; k += 1) {
                      for (int l = 0; l < weight_col_fused[c]; l += 1) {
                        if (l<padding_fused[c]){
                          l_start = padding_fused[c]-l;
                          l_end = output_col_fused[c];
                        }
                        else if (l>weight_col_fused[c]-padding_fused[c]-1){
                          l_start = 0;
                          l_end = output_col_fused[c]-(l-weight_col_fused[c]+padding_fused[c]+1);
                        }
                        else {
                          l_start = 0;
                          l_end = output_col_fused[c];
                        }


                        for (int i = l_start; i < l_end;i += 1) {
                          refresh();
                          
                          // selected input window from (i-1,h-1) to (i+1, h+1)
                          offset = i*stride_fused[c]+l-padding_fused[c];
                          int amem_read_base_fused = (output_row_index_fused[c]*stride_fused[c]+k+weight_row_fused[c]-padding_fused[c])%weight_row_fused[c];
                          if (input_rem_fused[c]>0){
                            if (m==input_itr_fused[c]-1){
                              memcpy( &input_mux_in[0], &amem_fused[c-1][(amem_read_base_fused*output_col_fused[c]*stride_fused[c] + offset)*((input_itr_fused[c]-1)*SP_ROW_SIZE+input_rem_fused[c])+m*SP_ROW_SIZE], sizeof(int32_t) * input_rem_fused[c]);
                              for (int n = 0; n < SP_ROW_SIZE;n += 1){
                                if (n<input_rem_fused[c])
                                  input_buffer[n] = input_mux_in[n];
                                else
                                  input_buffer[n] = 0;
                              }

                              amem_read_energy += AMEM_read_latency_cycles * input_rem_fused[c] * AMEM_128_R_POWER/128;
                              imc_computing_energy += mvm_latency_cycles * input_rem_fused[c] * AAIMC_SRAM_128IN_POWER /128;
                            }
                            else {
                              memcpy( &input_mux_in[0], &amem_fused[c-1][(amem_read_base_fused*output_col_fused[c]*stride_fused[c] + offset)*((input_itr_fused[c]-1)*SP_ROW_SIZE+input_rem_fused[c])+m*SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                              memcpy( &input_buffer[0], &input_mux_in[0], sizeof(int32_t) * SP_ROW_SIZE);
                              amem_read_energy += AMEM_read_latency_cycles * SP_ROW_SIZE * AMEM_128_R_POWER/128;
                              imc_computing_energy += mvm_latency_cycles * SP_ROW_SIZE * AAIMC_SRAM_128IN_POWER /128;
                            }
                          }
                          else {
                            memcpy( &input_mux_in[0], &amem_fused[c-1][(amem_read_base_fused*output_col_fused[c]*stride_fused[c] + offset)*input_itr_fused[c]*SP_ROW_SIZE+m*SP_ROW_SIZE], sizeof(int32_t) * SP_ROW_SIZE);
                            memcpy( &input_buffer[0], &input_mux_in[0], sizeof(int32_t) * SP_ROW_SIZE);
                            amem_read_energy += AMEM_read_latency_cycles * SP_ROW_SIZE * AMEM_128_R_POWER/128;
                            imc_computing_energy += mvm_latency_cycles * SP_ROW_SIZE * AAIMC_SRAM_128IN_POWER /128;
                          }

                          ping_pong_computing_input_feeding_cycles += AMEM_read_latency_cycles;
                          if (i == l_start){
                            layer_cycles +=AMEM_read_latency_cycles;
                            leakage((AMEM_read_latency_cycles));
                            refresh();
                          }

                          //printf("input feeding finish, output_col=%d weight_row=%d weight_col=%d\n", i, k, l );

                          row_length = (j==output_itr_fused[c]-1)&&output_rem_fused[c] ? output_rem_fused[c] : SP_ROW_SIZE;
                          if ((k==0&&l==0&&m==0)||(k==0&&l==padding_fused[c]&&m==0&&i==0)||(k==padding_fused[c]&&l==0&&m==0&&output_row_index_fused[c]==0)||(k==padding_fused[c]&&l==padding_fused[c]&&m==0&&output_row_index_fused[c]==0&&i==0)) { // only for padding=0/1
                            if (weight_col_fused[c]==1&&weight_row_fused[c]==1&&input_itr_fused[c]==1){
                              for (int n = 1; n < ((IMC_DIM/MACRO_DIM));n += 1)
                                bias_enable[n] = 0;

                              if (bias_existing)
                                bias_enable[0] = 1;
                              else 
                                bias_enable[0] = 0;

                              for (int ii = 0; ii < IMC_DIM; ++ii) {
                                for (int jj = 0; jj < (IMC_DIM/MACRO_DIM); ++jj) {
                                  int32_t acc = 0;      
                                  for (int kk = 0; kk < MACRO_DIM; ++kk) {
                                    acc += (imc_array[c][(k*weight_col_fused[c]+l)*input_itr_fused[c]*output_itr_fused[c]+m*output_itr_fused[c]+j][(jj*MACRO_DIM + kk) * IMC_DIM+ ii]-W_ZERO_POINT) * input_buffer[jj*MACRO_DIM + kk];
                                  }
                                  output_vector[jj][ii] = acc+  bias_enable[jj] * bias_scale_fused[c]*(bias_row[c][j][ii]-BIAS_ZERO_POINT);
                                }
                              }
                              //imc_cycles += mvm_latency_cycles;//imc cycle is hidden by ping-pong
                              ping_pong_computing_imc_cycles += mvm_latency_cycles;
                              if (i == l_start){
                                layer_cycles +=mvm_latency_cycles;
                                leakage((mvm_latency_cycles));
                                refresh();
                              }

                              if (output_rem_fused[c]>0){
                                for (int n = 0; n < row_length;n += 1){
                                  for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                    if (mi==0)
                                      output_buffer[n] = output_vector[mi][n];
                                    else
                                      output_buffer[n] += output_vector[mi][n];
                                  }
                                  amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] = (output_buffer[n]>0) ? (output_buffer[n])/amem_scale_fused[c] : 0;
                                }
                                if (((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)%(IMC_DIM/output_rem_fused[c])==0)
                                  amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*(output_itr_fused[c]-1+output_rem_fused[c]/IMC_DIM)+j] = 0;
                              }
                              else {
                                for (int n = 0; n < IMC_DIM;n += 1){
                                  for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                    if (mi==0)
                                      output_buffer[n] = output_vector[mi][n];
                                    else
                                      output_buffer[n] += output_vector[mi][n];
                                  }
                                  amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] = (output_buffer[n]>0) ? (output_buffer[n])/amem_scale_fused[c] : 0;
                                }
                                amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]+j] = 0;
                              }
                              
                              ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
                              layer_cycles +=AMEM_write_latency_cycles;
                              leakage((AMEM_write_latency_cycles));
                              refresh();

                              if (output_row_index_fused[c]==1&i==1){
                                for (int n=0*MACRO_DIM; n<2*MACRO_DIM; n++)
                                  printf("%d ", amem_fused[c][n]);
                                printf("\n");
                              }
                            }

                            else{
                              for (int n = 0; n < ((IMC_DIM/MACRO_DIM));n += 1)
                                bias_enable[n] = 0;

                              for (int ii = 0; ii < IMC_DIM; ++ii) {
                                for (int jj = 0; jj < (IMC_DIM/MACRO_DIM); ++jj) {
                                  int32_t acc = 0;      
                                  for (int kk = 0; kk < MACRO_DIM; ++kk) {
                                    acc += (imc_array[c][(k*weight_col_fused[c]+l)*input_itr_fused[c]*output_itr_fused[c]+m*output_itr_fused[c]+j][(jj*MACRO_DIM + kk) * IMC_DIM+ ii]-W_ZERO_POINT) * input_buffer[jj*MACRO_DIM + kk];
                                  }
                                  output_vector[jj][ii] = acc+  bias_enable[jj] * bias_scale_fused[c]*(bias_row[c][j][ii]-BIAS_ZERO_POINT);
                                }
                              }
                              //imc_cycles += mvm_latency_cycles;//imc cycle is hidden by ping-pong
                              ping_pong_computing_imc_cycles += mvm_latency_cycles;
                              if (i == l_start){
                                layer_cycles +=mvm_latency_cycles;
                                leakage((mvm_latency_cycles));
                                refresh();
                              }

                              //printf("computing finish, output_col=%d weight_row=%d weight_col=%d\n", i, k, l );

                              if (c==NUM_FUSED_LAYERS-1){
                                if (output_rem_fused[c]>0){
                                  for (int n = 0; n < row_length;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] = output_buffer[n];
                                  }
                                  if ((output_row_index_fused[c]*output_col_fused[c]+i)%(IMC_DIM/output_rem_fused[c])==0)
                                    amem_row_cycles_0[row_index + (output_row_index_fused[c]*output_col_fused[c]+i)*(output_itr_fused[c]-1+output_rem_fused[c]/IMC_DIM)+j] = 0;
                                }
                                else {
                                  for (int n = 0; n < IMC_DIM;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] = output_buffer[n];
                                  }
                                  amem_row_cycles_0[row_index + (output_row_index_fused[c]*output_col_fused[c]+i)*output_itr_fused[c]+j] = 0;
                                }
                              }
                              else{
                                if (output_rem_fused[c]>0){
                                  for (int n = 0; n < row_length;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] = output_buffer[n];
                                  }
                                  if (((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)%(IMC_DIM/output_rem_fused[c])==0)
                                    amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*(output_itr_fused[c]-1+output_rem_fused[c]/IMC_DIM)+j] = 0;
                                  //printf("write back finish, output_col=%d weight_row=%d weight_col=%d\n", i, k, l );
                                }
                                else {
                                  for (int n = 0; n < IMC_DIM;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] = output_buffer[n];
                                  }
                                  amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]+j] = 0;
                                }
                              }
                              
                              ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
                              layer_cycles +=AMEM_write_latency_cycles;
                              leakage((AMEM_write_latency_cycles));
                              refresh();
                            }
                          }

                          else {
                            if ((k==weight_col_fused[c]-1&&l==weight_row_fused[c]-1&&m==input_itr_fused[c]-1)||(k==weight_col_fused[c]-1&&l==weight_row_fused[c]-1-padding_fused[c]&&m==input_itr_fused[c]-1&&i==output_row_fused[c]-1)||(k==weight_col_fused[c]-1-padding_fused[c]&&l==weight_row_fused[c]-1&&m==input_itr_fused[c]-1&&output_row_index_fused[c]==output_col_fused[c]-1)||(k==weight_col_fused[c]-1-padding_fused[c]&&l==weight_row_fused[c]-1-padding_fused[c]&&m==input_itr_fused[c]-1&&output_row_index_fused[c]==output_col_fused[c]-1&&i==output_row_fused[c]-1)) {
                              if (bias_existing)
                                bias_enable[0] = 1;
                              else 
                                bias_enable[0] = 0;

                              for (int ii = 0; ii < IMC_DIM; ++ii) {
                                for (int jj = 0; jj < (IMC_DIM/MACRO_DIM); ++jj) {
                                  int32_t acc = 0;      
                                  for (int kk = 0; kk < MACRO_DIM; ++kk) {
                                    acc += (imc_array[c][(k*weight_col_fused[c]+l)*input_itr_fused[c]*output_itr_fused[c]+m*output_itr_fused[c]+j][(jj*MACRO_DIM + kk) * IMC_DIM+ ii]-W_ZERO_POINT) * input_buffer[jj*MACRO_DIM + kk];
                                  }
                                  output_vector[jj][ii] = acc+  bias_enable[jj] * bias_scale_fused[c]*(bias_row[c][j][ii]-BIAS_ZERO_POINT);
                                }
                              }

                              ping_pong_computing_imc_cycles += mvm_latency_cycles;
                              if (i == l_start){
                                layer_cycles +=mvm_latency_cycles;
                                leakage((mvm_latency_cycles));
                                refresh();
                              }
                              
                              if (c==NUM_FUSED_LAYERS-1){
                                if (output_rem_fused[c]>0){
                                  for (int n = 0; n < row_length;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] = (amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] + output_buffer[n] >0) ? (amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] + output_buffer[n])/amem_scale_fused[c] : 0;
                                  }
                                  if ((output_row_index_fused[c]*output_col_fused[c]+i)%(IMC_DIM/output_rem_fused[c])==0)
                                    amem_row_cycles_0[row_index + (output_row_index_fused[c]*output_col_fused[c]+i)*(output_itr_fused[c]-1+output_rem_fused[c]/IMC_DIM)+j] = 0;

                                  amem_read_energy += AMEM_read_latency_cycles * row_length * AMEM_128_R_POWER/128;
                                }
                                else {
                                  for (int n = 0; n < IMC_DIM;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] = (amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] + output_buffer[n] >0) ? (amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] + output_buffer[n])/amem_scale_fused[c] : 0;
                                  }
                                
                                  amem_row_cycles_0[row_index + (output_row_index_fused[c]*output_col_fused[c]+i)*output_itr_fused[c]+j] = 0;
                                  amem_read_energy += AMEM_read_latency_cycles * IMC_DIM * AMEM_128_R_POWER/128;
                                }
                                if (output_row_index_fused[c]==output_row_fused[c]-1&&i==output_col_fused[c]-1){
                                  for (int n=0*MACRO_DIM; n<2*MACRO_DIM; n++)
                                    printf("%d ", amem_0[row_index*IMC_DIM + n]);
                                  printf("\n");
                                }
                              }
                              else{
                                if (output_rem_fused[c]>0){
                                  for (int n = 0; n < row_length;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] = (amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] + output_buffer[n] > 0) ? (amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] + output_buffer[n])/amem_scale_fused[c] : 0;
                                  }
                                  if (((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)%(IMC_DIM/output_rem_fused[c])==0)
                                    amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*(output_itr_fused[c]-1+output_rem_fused[c]/IMC_DIM)+j] = 0;

                                  amem_read_energy += AMEM_read_latency_cycles * row_length * AMEM_128_R_POWER/128;
                                }
                                else {
                                  for (int n = 0; n < IMC_DIM;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] = (amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] + output_buffer[n]>0) ? (amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] + output_buffer[n])/amem_scale_fused[c] : 0;
                                  }
                                  amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]+j] = 0;
                                  amem_read_energy += AMEM_read_latency_cycles * IMC_DIM * AMEM_128_R_POWER/128;
                                }

                                //if (output_row_index_fused[c]==output_row_fused[c]-1&&i==output_col_fused[c]-1){
                                if (output_row_index_fused[c]==1&&i==1){
                                  for (int n=0*MACRO_DIM; n<MACRO_DIM; n++)
                                    printf("%d ", imc_array[c][(k*weight_col_fused[c]+l)*input_itr_fused[c]*output_itr_fused[c]+m*output_itr_fused[c]+j][n]);
                                  printf("\n");
                                  for (int n=0*MACRO_DIM; n<MACRO_DIM; n++)
                                    printf("%d ", input_buffer[n]);
                                  printf("\n");
                                  for (int n=0*MACRO_DIM; n<2*MACRO_DIM; n++)
                                    printf("%d ", amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+n]);
                                  printf("\n");
                                }
                              }
                              ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
                              layer_cycles +=AMEM_write_latency_cycles;
                              leakage((AMEM_write_latency_cycles));
                              refresh();
                            }
                            else{
                              bias_enable[0] = 0;
                              for (int ii = 0; ii < IMC_DIM; ++ii) {
                                for (int jj = 0; jj < (IMC_DIM/MACRO_DIM); ++jj) {
                                  int32_t acc = 0;      
                                  for (int kk = 0; kk < MACRO_DIM; ++kk) {
                                    acc += (imc_array[c][(k*weight_col_fused[c]+l)*input_itr_fused[c]*output_itr_fused[c]+m*output_itr_fused[c]+j][(jj*MACRO_DIM + kk) * IMC_DIM+ ii]-W_ZERO_POINT) * input_buffer[jj*MACRO_DIM + kk];
                                  }
                                  output_vector[jj][ii] = acc+  bias_enable[jj] * bias_scale_fused[c]*(bias_row[c][j][ii]-BIAS_ZERO_POINT);
                                }
                              }

                              ping_pong_computing_imc_cycles += mvm_latency_cycles;
                              if (i == l_start){
                                layer_cycles +=mvm_latency_cycles;
                                leakage((mvm_latency_cycles));
                                refresh();
                              }

                              //printf("computing finish, output_col=%d weight_row=%d weight_col=%d\n", i, k, l );

                              if (c==NUM_FUSED_LAYERS-1){
                                if (output_rem_fused[c]>0){
                                  for (int n = 0; n < row_length;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] = (amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] + output_buffer[n]);
                                  }
                                  if ((output_row_index_fused[c]*output_col_fused[c]+i)%(IMC_DIM/output_rem_fused[c])==0)
                                    amem_row_cycles_0[row_index + (output_row_index_fused[c]*output_col_fused[c]+i)*(output_itr_fused[c]-1+output_rem_fused[c]/IMC_DIM)+j] = 0;

                                  amem_read_energy += AMEM_read_latency_cycles * row_length * AMEM_128_R_POWER/128;
                                }
                                else {
                                  for (int n = 0; n < IMC_DIM;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] = (amem_0[row_index*IMC_DIM + (output_row_index_fused[c]*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] + output_buffer[n]);
                                  }
                                
                                  amem_row_cycles_0[row_index + (output_row_index_fused[c]*output_col_fused[c]+i)*output_itr_fused[c]+j] = 0;
                                  amem_read_energy += AMEM_read_latency_cycles * IMC_DIM * AMEM_128_R_POWER/128;
                                }
                              }
                              else {
                                if (output_rem_fused[c]>0){
                                  for (int n = 0; n < row_length;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] = (amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*((output_itr_fused[c]-1)*IMC_DIM+output_rem_fused[c])+j*IMC_DIM+ n] + output_buffer[n]);
                                  }
                                  if (((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)%(IMC_DIM/output_rem_fused[c])==0)
                                    amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*(output_itr_fused[c]-1+output_rem_fused[c]/IMC_DIM)+j] = 0;

                                  amem_read_energy += AMEM_read_latency_cycles * row_length * AMEM_128_R_POWER/128;
                                }
                                else {
                                  for (int n = 0; n < IMC_DIM;n += 1){
                                    for (int mi = 0; mi < (IMC_DIM/MACRO_DIM); mi += 1){
                                      if (mi==0)
                                        output_buffer[n] = output_vector[mi][n];
                                      else
                                        output_buffer[n] += output_vector[mi][n];
                                    }
                                    amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] = (amem_fused[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]*IMC_DIM+j*IMC_DIM+ n] + output_buffer[n]);
                                  }
                                  amem_row_cycles[c][((output_row_index_fused[c]%weight_row_fused[c+1])*output_col_fused[c]+i)*output_itr_fused[c]+j] = 0;
                                  amem_read_energy += AMEM_read_latency_cycles * IMC_DIM * AMEM_128_R_POWER/128;
                                }
                              }

                              ping_pong_computing_output_wb_cycles += AMEM_write_latency_cycles;
                              layer_cycles +=AMEM_write_latency_cycles;
                              leakage((AMEM_write_latency_cycles));
                              refresh();
                            }
                              
                          }

                          if (SRAM_IMC==1){
                            imc_computing_energy += mvm_latency_cycles * row_length * AAIMC_SRAM_128O_POWER /128;
                          }
                          else{
                            imc_computing_energy += mvm_latency_cycles * row_length * AAIMC_AMEM_32O_POWER/32;
                          }
                          amem_write_energy += AMEM_write_latency_cycles * row_length * AMEM_128_W_POWER/128;
                  
                        }
                      }
                    }
                  }
                }
              }
            }
            if (c>0)
              amem_row_index_fused[c-1] = amem_row_index_fused[c-1] - stride_fused[c];

            output_row_index_fused[c] += 1;
            amem_row_index_fused[c] += 1;

          }
          if (c<NUM_FUSED_LAYERS-1){
            run_layer[c + 1] = ((amem_row_index_fused[c] == weight_row_fused[c + 1] - padding_fused[c + 1]) && (output_row_index_fused[c+1] < output_row_fused[c+1]))||
                                ((output_row_index_fused[c+1] >= output_row_fused[c+1]-padding_fused[c+1])&&(output_row_index_fused[c+1] < output_row_fused[c+1]));
            //printf("current amem_row_index=%d, next run_layer=%d, output_row_index=%d\n", amem_row_index_fused[c], run_layer[c + 1], output_row_index_fused[c+1]);
          }
          else{
            run_layer[0] = output_row_index_fused[0] < output_row_fused[0] ? 1 : 0;
          }
          
        }
      }

      // output trace
      //ping_pong_computing_cycles = ping_pong_computing_imc_cycles + ping_pong_computing_anu_cycles + ping_pong_computing_dma_cycles;

      imc_cycles += ping_pong_computing_imc_cycles;
      anu_cycles += ping_pong_computing_anu_cycles;
      amem_r_cycles += ping_pong_computing_input_feeding_cycles;
      amem_w_cycles += ping_pong_computing_output_wb_cycles; 
      total_layer_cycles += layer_cycles;
      total_cycles = total_layer_cycles+refresh_cycles;

      printf("Data loading cycle= %lu\n", dma_cycles);
      printf("IMC cycle= %lu\n", imc_cycles);
      printf("ANU cycle= %lu\n", anu_cycles);
      printf("AMEM read cycle= %lu\n", amem_r_cycles);
      printf("AMEM write cycle= %lu\n", amem_w_cycles);
      printf("refresh cycle= %lu\n", refresh_cycles);
      printf("Total cycle= %lu\n", total_cycles);
      
      //area
      if (SRAM_IMC==1){
        imc_area = ((1+PP_IMC_ENABLE) * IMC_ARRAY_SIZE + FUSED_IMC_ARRAY_SIZE) * AAIMC_PRECISION * AAIMC_4b_SRAM_128kb_AREA/(128*1024);
      }
      else{
        imc_area = ((1+PP_IMC_ENABLE) * IMC_ARRAY_SIZE + FUSED_IMC_ARRAY_SIZE) * AAIMC_PRECISION * AAIMC_4b_AMEM_2kc_AREA/(2*1024*4);
      }
      amem_area = (AMEM_ARRAY_SIZE + FUSED_AMEM_ARRAY_SIZE) * AMEM_PRECISION * AMEM_64k_AREA/(64*1024*6);
      anu_area = IMC_DIM * MAX_POOL_ANU_128O_AREA/128; 
      abus_area = IMC_DIM * ABUS_256_AREA/256; 
      sram_buffer_area = ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_AREA/(5152*1024);
      total_area = imc_area + amem_area + anu_area + abus_area + sram_buffer_area + ADC_AREA + DAC_AREA*MACRO_DIM + CTRL_FSM_AREA + RISCV32_AREA * RISCV_ENABLE;
      printf("Total area= %lu um^2\n", total_area*9*9/100);// 0.9 cale in 28nm

      //energy
      amem_leakage_energy += ((layer_cycles- ping_pong_computing_output_wb_cycles) * FUSED_AMEM_ARRAY_SIZE + layer_cycles*AMEM_ARRAY_SIZE) * AMEM_64K_Leakage_POWER/(64*1024);

      if (SRAM_IMC==1){
        imc_leakage_energy += (layer_cycles * (1+PP_IMC_ENABLE)*IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024)
                              //+ (layer_cycles-ping_pong_computing_imc_cycles) * FUSED_IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024));
                              + (layer_cycles) * FUSED_IMC_ARRAY_SIZE * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024));
      }
      else{
        imc_leakage_energy += (layer_cycles * (1+PP_IMC_ENABLE)*IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024)
                              //+ (layer_cycles-ping_pong_computing_imc_cycles) * FUSED_IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024));
                              + (layer_cycles) * FUSED_IMC_ARRAY_SIZE * AAIMC_AMEM_2kc_Leakage_POWER/(2*1024));
      }
      //anu_computing_energy = ping_pong_computing_anu_cycles * IMC_DIM * MAX_POOL_ANU_128O_POWER/128;
      
      sram_buffer_leakage_energy += (layer_cycles-dma_cycles) * ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_Leakage_POWER/(5152*1024);
      sram_leakage_energy_in_refresh = refresh_cycles * ON_CHIP_SRAM_BUFFER_SIZE * SRAM_5152kb_Leakage_POWER/(5152*1024)
                                        + SRAM_IMC * ((1+PP_IMC_ENABLE)* IMC_ARRAY_SIZE + FUSED_IMC_ARRAY_SIZE)* AAIMC_PRECISION * AAIMC_SRAM_128kb_Leakage_POWER/(128*1024);
      fsm_energy = total_cycles * CTRL_FSM_POWER;
      //risc_energy = total_cycles * RISCV32_POWER;
      total_energy = amem_read_energy + amem_write_energy + amem_leakage_energy +  refresh_energy+
                    imc_computing_energy + anu_computing_energy + imc_loading_energy + sram_leakage_energy_in_refresh+
                    imc_leakage_energy + sram_buffer_leakage_energy + fsm_energy + ADDA_energy;

      printf("AMEM read energy= %lu\n", amem_read_energy);
      printf("AMEM write energy= %lu\n", amem_write_energy);
      printf("AMEM leakage energy= %lu\n", amem_leakage_energy);
      printf("IMC computing energy= %lu\n", imc_computing_energy);
      printf("ANU computing energy= %lu\n", anu_computing_energy);
      printf("IMC loading energy= %lu\n", imc_loading_energy);
      printf("IMC leakage energy= %lu\n", imc_leakage_energy);
      printf("SRAM buffer leakage energy= %lu\n", sram_buffer_leakage_energy);
      printf("FSM energy= %lu\n", fsm_energy);
      printf("Total energy= %lu pJ\n", total_energy*PERIOD/1000);

      printf("amem row cycles ofm = %d ofm1 = %d ofm2 = %d\n", amem_row_cycles[NUM_FUSED_LAYERS-2][0], amem_row_cycles[NUM_FUSED_LAYERS-2][1], amem_row_cycles[NUM_FUSED_LAYERS-2][2]);
      //printf("amem fist channel 1st result= %d 2nd = %d last-1 = %d last = %d\n", amem_0[(row_index+30*16+13)*SP_ROW_SIZE], amem_0[(row_index+30*16+13)*SP_ROW_SIZE+1], amem_0[(row_index+30*16+13)*SP_ROW_SIZE+SP_ROW_SIZE-2], amem_0[(row_index+30*16+13)*SP_ROW_SIZE+SP_ROW_SIZE-1]);
      for (int i=0*MACRO_DIM; i<2*MACRO_DIM; i++)
        printf("%d ", amem_fused[NUM_FUSED_LAYERS-2][i]);
      printf("\n");
      ie = 1;
      update_interrupts();
              
      conv_enable=false;
      return true;
    }

    default:
      printf("IMC STORE unknown addr 0x%lx\n", addr);
      return false;
  }
}

void imc_device_t::tick(reg_t rtc_ticks) {
  //printf("IMC tick called, cycle =%lu\n", rtc_ticks);

  /*if (mvm_in_progress) {
    printf("mvm_cycles_left =%d\n", mvm_cycles_left);
    if (--mvm_cycles_left <= 0) {
      compute();
      printf("DMA cycle= %d\n", dma_cycles);
      printf("IMC cycle= %d\n", imc_cycles);
      mvm_in_progress = false;
      imc_call_reset = true;
      ie = 1;
      update_interrupts();
    }
  }*/

}

reg_t imc_device_t::size() {
	//printf("IMC size() called, returning 0x1000\n");
	return 0x20000000;
}

int fdt_parse_imc_device(const void* fdt, reg_t* addr, const char* compatible) {
  int rc;
  int nodeoffset = fdt_node_offset_by_compatible(fdt, -1, compatible);
  if (nodeoffset < 0)
    return nodeoffset;

  rc = fdt_get_node_addr_size(fdt, nodeoffset, addr, NULL, "reg");
  if (rc < 0 || !addr)
    return -ENODEV;

  return 0;
}

//imc_device_t* imc_parse_from_fdt(const void* fdt, const sim_t* sim, reg_t* base, std::vector<std::string> sargs) {
//  if (fdt_parse_imc_device(fdt, base, "elwood,imc_device0") == 0) {
//    printf("IMC device found at 0x%lx\n", *base);
//    abstract_interrupt_controller_t* intctrl = sim->get_intctrl();
//
//    static std::vector<std::shared_ptr<imc_device_t>> devices;
//    auto dev = std::make_shared<imc_device_t>(sim, intctrl, *base);
//    devices.push_back(dev); // Store it so it doesnt get deleted
//    return dev.get();       // REGISTER_DEVICE will register it
    //return dev.get();  // return raw pointer as REGISTER_DEVICE expects
    //((sim_t*)sim)->add_device(*base, dev); 
    //return dev.get();
    //return new imc_device_t(sim, intctrl, *base);
//  } else {
//    return nullptr;
//  }
//}

imc_device_t* imc_parse_from_fdt(
    const void* fdt,
    const sim_t* sim,
    reg_t* base,
    std::vector<std::string> sargs) {

  if (fdt_parse_imc_device(fdt, base, "elwood,imc_device0") == 0) {
    printf("IMC device found at 0x%lx\n", *base);
    abstract_interrupt_controller_t* intctrl = sim->get_intctrl();

    // Allocate using new, Spike will manage it
    return new imc_device_t(sim, intctrl, *base, 1);
  } else {
    return nullptr;
  }
}

std::string imc_generate_dts(const sim_t* sim, const std::vector<std::string>& args) {
  return std::string();  // Leave empty if not auto-generating
}

REGISTER_DEVICE(imc_device, imc_parse_from_fdt, imc_generate_dts);
