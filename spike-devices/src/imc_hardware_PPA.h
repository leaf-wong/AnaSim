
////////unit////////
// frequency: MHz
// period: ns
// delay: cycle (If write delay=20ns, clck frequency=100Mhz, write delay=2)
// area:  um^2
// power: uW
////////////////////

//latency
#define FREQUENCY (100)
#define PERIOD (1000/FREQUENCY)

#define SRAM_ACCESS_LATENCY (1UL)
#define AMEM_WRITE_LATENCY (4UL)
#define AMEM_READ_LATENCY (4UL)
#define AMEM_REFRESH_LATENCY (3UL)
#define AAIMC_SRAM_WRITE_LATENCY (1UL)
#define AAIMC_AMEM_WRITE_LATENCY (2UL)
#define ANU_LATENCY (4UL)
#define IMC_COMPUTING_LATENCY (4UL) //include AMEM read latency
#define PIPELINE_STAGE_LATENCY (4UL) 
#define DMA_LOADING_CYCLE (1UL) 
#define ADC_CYCLE (3UL) 
#define DAC_CYCLE (1UL) 

#define RETENTION_TIME (43500) 
#define AMEM_MIDDLE_VALUE (32) 
#define REFRESH_INTERVAL (40000) 
#define REFRESH_SCALE (630) 

//power
#define AMEM_128_W_POWER 1660UL
#define AMEM_128_R_POWER 920UL
#define AMEM_64K_Leakage_POWER 200UL
#define AMEM_64_Refresh_POWER 820UL
#define AAIMC_SRAM_128IN_POWER 333UL
#define AAIMC_SRAM_128O_POWER 932UL // power together with AAU
#define AAIMC_SRAM_128kb_Leakage_POWER 64UL
#define AAIMC_SRAM_512b_W_POWER 2000UL
//#define AAIMC_AMEM_64IN_POWER 333 
// from ESSCIRC 2024 AACIM, doesn't include input driver
#define AAIMC_AMEM_32O_POWER 251UL
#define AAIMC_AMEM_2kc_Leakage_POWER 6UL
#define AAIMC_AMEM_32x4b_W_POWER 539UL
#define ANU_128O_POWER 932UL
#define CTRL_FSM_POWER 854UL
#define RISCV32_POWER 2263UL // from ESSCIRC 2025 EPIC
#define SRAM_512b_DMA_POWER 3405UL
#define SRAM_5152kb_Leakage_POWER 343UL
#define DRAM_512b_DMA_POWER 189440UL  // 3.7pJ/bit

// from TVLSI 2024 <<A High-Precision and High-Dynamic-Range Current-Mode WTA Circuit for Low-Supply-Voltage Applications>>
#define MAX_POOL_ANU_128O_POWER 585UL

#define ADC_POWER 21UL
#define DAC_POWER 6UL


//area
#define AMEM_64k_AREA 320600ULL
#define AAIMC_4b_SRAM_128kb_AREA 155800ULL
//#define AAIMC_6b_SRAM_36kb_AREA 90800ULL
#define AAIMC_4b_AMEM_2kc_AREA 4800ULL
#define ABUS_256_AREA 60800ULL
#define SRAM_5152kb_AREA 1898770ULL
#define CTRL_FSM_AREA 40000ULL // from ESSCIRC 2025 EPIC
#define RISCV32_AREA 132000ULL
#define MAX_POOL_ANU_128O_AREA 6643ULL
#define ADC_AREA 178UL
#define DAC_AREA 60UL


