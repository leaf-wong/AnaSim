#ifndef _IMC_DEVICE_H
#define _IMC_DEVICE_H

#include <riscv/abstract_device.h>
#include <riscv/dts.h>
#include <riscv/sim.h>
#include <fdt/libfdt.h>
#include <fesvr/term.h>
#include <vector>
#include "imc_hardware_PPA.h"

#define SRAM_IMC (1) // 1: SRAM_IMC enabled, 0: AMEM_IMC enabled.
#define DRAM_ENABLE (1) // 1: DRAM enabled, 0: On-chip SRAM buffer enabled.
#define RISCV_ENABLE (1) // 1: RISCV enabled, 0: Only FSM control enabled.
#define PP_IMC_ENABLE (0) // 1: two set of IMC arrays work in ping-pong enabled, 0: Only single IMC array.

#define IMC_DIM (1024)
#define SP_ROW_SIZE (IMC_DIM)
#define MACRO_DIM (64)

#define PGSIZE (4096) //defined in mmu.h of spike

#define AMEM_PRECISION (6)
#define AAIMC_PRECISION (6)

//#define W_ZERO_POINT     (pow(2,AAIMC_PRECISION-1))
//#define BIAS_ZERO_POINT     (pow(2,AAIMC_PRECISION-1))
#define W_ZERO_POINT     (128)
#define BIAS_ZERO_POINT     (128)
#define BIAS_SCALE_CONV1     (1)
#define BIAS_SCALE     (55)

#define AMEM_SCALE     (315) //630
#define AMEM_SCALE_CONV1     (3)

#define IMC_BASE_ADDR 0x10000000

#define IMC_REG_CONTROL (0x00)
#define IMC_REG_STATUS  (0x04)
#define IMC_REG_OUTPUT  (0x08)
#define AMEM_RD_OFFSET  (0x0c)

#define IMC_IE     (0x10)

#define DRAM2SP_IN (0x14)
#define DRAM2SP_W (0x18)
#define DRAM2SP_IN_BURST (0x1c)
#define DRAM2SP_W_BURST (0x20)
#define SP_IN2AMEM (0x24)
#define SP_W2IMC (0x28)
#define DMA_ROWIDX (0x2c)

#define ANU_ADD (0x30)
#define ANU_ADD_RELU (0x34)

#define AMEM_ROW_REFRESH (0x38)

#define BIAS_ROW_RD_OFFSET (0x3c)
#define SP_W2BIAS_ROW (0x40)
#define BIAS_EXIST (0x44)

#define DMA_BURST_NUM (0x48)
#define SP_W_WR_BASE (0x4c)
#define AMEM_RD_BASE (0x50)
#define INPUT_ITR (0x54)
#define OUTPUT_ITR (0x58)
#define OUTPUT_COL (0x5c)
#define OUTPUT_ROW (0x60)
#define WEIGHT_COL (0x64)
#define WEIGHT_ROW (0x68)
#define INPUT_REMAINDER (0x6c)
#define OUTPUT_REMAINDER (0x70)
#define NEXT_CONV_CHANNEL (0x74)
#define PADDING (0x78)
#define STRIDE (0x7c)
#define DMA_READ_BASE (0x80)
#define AMEM_SCALE_CFG (0x84)
#define BIAS_SCALE_CFG (0x88)
#define AMEM_RD_BASE2 (0x8c)
#define REORG_CAT (0x90)
#define DMA_READ_INPUT_BASE (0x94)

#define LINEAR_LAYER (0x100)
#define CONV_LAYER (0x104)
#define AVGPOOL_LAYER (0x108)
#define MAXPOOL_LAYER (0x120)
#define ReLU_LAYER (0x10c)
#define ADD_LAYER (0x110)


#define CONV_LAYER_A (0x114)
#define CONV_LAYER_A_CONV1 (0x118)
#define CONV_LAYER_A_OTHER (0x11c)

#define LAYER_FUSED (0x124)
#define LAYER_FUSE_ENABLE (0x128)
#define LAYER_INDEX (0x12c)
#define LAYER_TYPE (0x130)

#define NUM_FUSED_LAYERS (8) //YOLOv2: lf 17 MD 8 resnet: lf 14 MD 8

#define IMC_SP_INPUT   (0x1000)
#define IMC_SP_WEIGHT  (0x2000)
//#define IMC_AMEM_0   (0x300000)
//#define IMC_AMEM_0_TEMP_BASE   (0x380000)
//#define IMC_ARRAY_0  (0x400000)
//#define IMC_ARRAY_1  (0x500000)
#define IMC_ARRAY_0  (0x300000)
#define IMC_ARRAY_1  (0x400000)
#define IMC_AMEM_0  (0x500000)

#define L2CACHE_BASE_ADDR (0x10000000)


#define IMC_CONTROL_MVM     (1)
#define IMC_CONTROL_RST     (0)


#define L2CACHE_SIZE (12845056)
#define SP_IN_SIZE (4096)
#define SP_W_SIZE (3137536)
//#define AMEM_ARRAY_SIZE (65536) // 64 AMEM BANK (Each 4k)
#define AMEM_ARRAY_SIZE (1384448) // for YOLOV2, MD 1384448 lbl 5537792 lf 21125
//#define AMEM_ARRAY_SIZE (196608) // for Resnet9, MD 16384 lbl 196608 lf 1024
#define IMC_ARRAY_SIZE (IMC_DIM*IMC_DIM) // IMC_DIM*IMC_DIM or 0
#define BIAS_ROW_SIZE (IMC_DIM)

// define layer fused hardware
#define IMG_SIZE (416) // for YOLOV2 416 for Resnet9 32
#define OPs (36792136212) // for YOLOV2 36792136212; ResNet-9: 569780224UL (167823360 modified)
// IMC: 32*32, 3*3*32*64, 3*3*64*128, 1*1*128*64, 3*3*64*128
#define FUSED_IMC_ARRAY_SIZE (175104) // YOLO: MD 175104, lbl 0, 1f 4598784 67146496
//#define FUSED_IMC_ARRAY_SIZE (0) // resnet: MD 665280, lbl 0, 1f 2444992
// ANEM: 32*2*IMG_SIZE, 32*3*IMG_SIZE/2, 64*2*IMG_SIZE/2, 64*3*IMG_SIZE/4, 128*1*IMG_SIZE/4, 64*3*IMG_SIZE/4, 128*2*IMG_SIZE/4
#define FUSED_AMEM_ARRAY_SIZE ((32*2+32*3/2+64*2/2+64*3/4+128*1/4+64*3/4+128*2/4)*IMG_SIZE)
//#define FUSED_AMEM_ARRAY_SIZE (0) // resnet: lbl 0 lf 62464 MD 43008
//#define FUSED_AMEM_ARRAY_SIZE ((32*2+32*3/2+64*2/2+64*3/4+128*1/4+64*3/4+128*2/4+128*3/8+256*1/8+128*3/8+256*2/8+256*3/16+512*1/16+256*3/16+512*1/16+256*3/16 +512*2/16+512*3/32+1024*1/32+512*3/32+1024*1/32+512*3/32+1024*3/32+1024*3/32)*IMG_SIZE)
#define ON_CHIP_SRAM_BUFFER_SIZE (IMC_DIM*AAIMC_PRECISION) //for DRAM_en: IMC_DIM*AAIMC_PRECISION; for YOLOV2: 67146496*AAIMC_PRECISION for Resnet9: 2444992*AAIMC_PRECISION  MD 1779712*AAIMC_PRECISION

//#define AMEM_ROW_INDEX (AMEM_ARRAY_SIZE/IMC_DIM) 
#define AMEM_ROW_INDEX (AMEM_ARRAY_SIZE*2/IMC_DIM) //#define AMEM_ROW_INDEX (327680/IMC_DIM) 

//#define IMC_AMEM_TEMP_BASE (AMEM_ARRAY_SIZE/(2*IMC_DIM))
#define IMC_AMEM_TEMP_BASE (AMEM_ARRAY_SIZE/(IMC_DIM))

class imc_device_t : public abstract_device_t {
public:
  imc_device_t(const simif_t* sim,
      abstract_interrupt_controller_t *intctrl,
      reg_t base,
      reg_t int_id);
  ~imc_device_t();
  bool load(reg_t addr, size_t len, uint8_t* bytes) override;
  bool store(reg_t addr, size_t len, const uint8_t* bytes) override;
  void tick(reg_t rtc_ticks) override;
  reg_t size() override;
  

private:
  //std::vector<std::vector<uint8_t>> matrix;
  std::vector<bool> bias_enable;
  std::vector<std::vector<int32_t>> output_vector;
  int offset;
  int bias_offset;
  int row_index;
  int repeats;

  int l2cache_read_base;
  int sp_weight_write_base;
  uint32_t dma_read_base;
  uint32_t dma_read_input_base;
  int amem_read_base;
  int amem_read_base2;
  int input_itr;
  int output_itr;
  int output_col;
  int output_row;
  int weight_col;
  int weight_row;
  int input_rem;
  int output_rem;
  int next_conv_channel;
  int stride;
  int padding;

  int k_start;
  int k_end;
  int l_start;
  int l_end;

  bool imc0_computing;


  bool bias_existing;

  int bias_scale;
  int amem_scale;

  const simif_t* sim;
  uint32_t ie;
  reg_t interrupt_id;
  abstract_interrupt_controller_t *intctrl;

  bool imc_call_reset;

  // Latency simulation
  bool write_in_progress;
  int amem_leakage_cycles;

  bool mvm_in_progress;
  int mvm_cycles_left;

  unsigned long  imc_cycles;
  unsigned long  dma_cycles;
  unsigned long  anu_cycles;
  unsigned long  amem_r_cycles;
  unsigned long  amem_w_cycles;
  unsigned long  refresh_cycles;
  unsigned long  ping_pong_computing_imc_cycles;
  unsigned long  ping_pong_computing_anu_cycles;
  unsigned long  ping_pong_computing_dma_cycles;
  unsigned long  ping_pong_computing_input_feeding_cycles;
  unsigned long  ping_pong_computing_output_wb_cycles;
  unsigned long  ping_pong_computing_cycles;
  unsigned long  ping_pong_loading_cycles;
  unsigned long  layer_cycles;
  unsigned long  total_layer_cycles;
  unsigned long  total_cycles;

  const unsigned long  anu_latency_cycles = ANU_LATENCY;
  const unsigned long  mvm_latency_cycles = IMC_COMPUTING_LATENCY;
  const unsigned long  row_latency_cycles = SRAM_ACCESS_LATENCY;
  const unsigned long  AMEM_write_latency_cycles = AMEM_WRITE_LATENCY;
  const unsigned long  AMEM_read_latency_cycles = AMEM_READ_LATENCY;
  const unsigned long  IMC_write_latency_cycles = (SRAM_IMC) ? AAIMC_SRAM_WRITE_LATENCY : AAIMC_AMEM_WRITE_LATENCY;
  const unsigned long  AMEM_refresh_latency_cycles = AMEM_REFRESH_LATENCY;
  
  unsigned long imc_area;
  unsigned long amem_area;
  unsigned long anu_area;
  unsigned long abus_area;
  unsigned long sram_buffer_area;
  unsigned long total_area;

  unsigned long amem_read_energy;
  unsigned long amem_write_energy;
  unsigned long refresh_energy;
  unsigned long amem_leakage_energy;
  unsigned long imc_computing_energy;
  unsigned long anu_computing_energy;
  unsigned long imc_loading_energy;
  unsigned long imc_leakage_energy;
  unsigned long sram_buffer_leakage_energy;
  unsigned long sram_leakage_energy_in_refresh;
  unsigned long fsm_energy;
  unsigned long ADDA_energy;
  unsigned long risc_energy;
  unsigned long total_energy;


  uint8_t l2_cache[L2CACHE_SIZE];
  uint8_t sp_input[SP_IN_SIZE];
  uint8_t sp_weight[SP_W_SIZE];
  int32_t amem_0[AMEM_ARRAY_SIZE*2];
  uint8_t imc_array_0[IMC_DIM*IMC_DIM];
  uint8_t imc_array_1[IMC_DIM*IMC_DIM];
  uint8_t bias_row_0[BIAS_ROW_SIZE];
  //uint8_t bias_row_1[BIAS_ROW_SIZE];
  uint8_t imc_w_mux_in[SP_ROW_SIZE];
  uint8_t imc_w_reg[SP_ROW_SIZE];
  uint8_t imc_dac_input[SP_ROW_SIZE];
  int32_t input_mux_in[IMC_DIM];
  int32_t input_buffer[IMC_DIM];
  int32_t output_buffer[BIAS_ROW_SIZE];
//  int32_t amem_temp[AMEM_TEMP_SIZE];
  int amem_row_cycles_0[AMEM_ROW_INDEX];
  int amem_row_leak_flag_0[AMEM_ROW_INDEX];
  int refresh_counter;
  bool pool_enable;
  bool conv_enable;
  bool dma_dominate;
  bool load_first_weight;

  //layer_fusion
  int layer_fuse_enable;
  int layer_index;
  int layer_type[NUM_FUSED_LAYERS];
  int output_row_index_fused[NUM_FUSED_LAYERS];
  int amem_row_index_fused[NUM_FUSED_LAYERS];

  int input_itr_fused[NUM_FUSED_LAYERS];
  int output_itr_fused[NUM_FUSED_LAYERS];
  int output_col_fused[NUM_FUSED_LAYERS];
  int output_row_fused[NUM_FUSED_LAYERS];
  int weight_col_fused[NUM_FUSED_LAYERS];
  int weight_row_fused[NUM_FUSED_LAYERS];
  int input_rem_fused[NUM_FUSED_LAYERS];
  int output_rem_fused[NUM_FUSED_LAYERS];
  int stride_fused[NUM_FUSED_LAYERS];
  int padding_fused[NUM_FUSED_LAYERS];
  int bias_scale_fused[NUM_FUSED_LAYERS];
  int amem_scale_fused[NUM_FUSED_LAYERS];

  //Be careful!!
  //The addr of these vars are continous, if amem_row_cycles overflows, the value of imc_array will be changed!!
  int32_t amem_fused[NUM_FUSED_LAYERS-1][IMG_SIZE*128*3];
  int amem_row_cycles[NUM_FUSED_LAYERS-1][(IMG_SIZE*128*3)/IMC_DIM];
  uint8_t imc_array[NUM_FUSED_LAYERS][9*(1024/IMC_DIM)*(1024/IMC_DIM)][IMC_DIM*IMC_DIM];
  uint8_t bias_row[NUM_FUSED_LAYERS][(1024/IMC_DIM)][BIAS_ROW_SIZE];



  void compute();  // MVM operation 

  void dma(uint32_t read_addr);  // DMA leakage operation 

  void leakage(int cycle_inc);  // AMEM leakage operation 

  void reset_amem();

  void refresh();

  void update_interrupts() {
    int cond = 0;
    if (!mvm_in_progress && ie) {
      cond = 1;
    }
    if (intctrl) {
      intctrl->set_interrupt_level(interrupt_id, cond ? 1 : 0);
      printf("IMC device %d: external interrupt level set to %d\n", interrupt_id, cond);
    }
  }
};

#endif

