OUTPUT_ARCH( "riscv" )

ENTRY( _start )

MEMORY
{
  

/*  ram :  	ORIGIN = 0x00100000, LENGTH = 0x18000      	*/	/* 96 kB */
/*  data_ram :  	ORIGIN = 0x00180000, LENGTH = 0x1E000   */  	/* 120 kB */
/*  stack :  	ORIGIN = 0x0019E000, LENGTH = 0x2000       	*/	/*  8 kB */

/*  ram :  	ORIGIN = 0x00100000, LENGTH = 0x18000     	*/	/* 96 kB */
/*  data_ram :  	ORIGIN = 0x00180000, LENGTH = 0x20000   */  	/* 128 kB */
/*  stack :  	ORIGIN = 0x001A0000, LENGTH = 0x20000     	*/	/* 128 kB */


/*  ram :         ORIGIN = 0x00100000, LENGTH = 0x18000  */   /* 96 KB */
  ram :         ORIGIN = 0x00100000, LENGTH = 0x11600     /* 69 KB */


/*  data_ram :    ORIGIN = 0x00180000, LENGTH = 0x5F000  */   /* 380 KB */  /* 0x0018_0000 ~ 0x001D_F000  */
/*  data_ram :    ORIGIN = 0x00180000, LENGTH = 0x5E0F0  */   /*  KB */
  data_ram :    ORIGIN = 0x00180000, LENGTH = 0x20000     /*  KB */


/* stack :       ORIGIN = 0x00600000, LENGTH = 0x60000  */   /* 384 KB */
/*  stack :       ORIGIN = 0x00600000, LENGTH = 0x40000 */    /* 256 KB */
/*  stack :       ORIGIN = 0x00600000, LENGTH = 0x33000 */    /* 204 KB */
/*  stack :       ORIGIN = 0x00600000, LENGTH = 0x32F00 */    /* 204 KB */
  stack :       ORIGIN = 0x00600000, LENGTH = 0x20000     /*  KB */




}

/* _min_stack      = 0x2000; */  /* 8K - minimum stack space to reserve */

_stack_start   = ORIGIN(stack) + LENGTH(stack);
_sp = _stack_start;

/*
PHDRS
{
  flash PT_LOAD;
  ram_init PT_LOAD;
  ram PT_NULL;
}
*/



SECTIONS
{
/*  __stack_size = DEFINED(__stack_size) ? __stack_size : 8K;  */
  __stack_size = DEFINED(__stack_size) ? __stack_size : 128K;


/*
  .vectors :
  {
   . = ALIGN(4);
   	_vectors_start = .;
   KEEP(*(.vectors))
   	_vectors_end = .;
  } > ram
*/

  .vectors :
  {
   . = ALIGN(4);
   	_vectors_start = .;
   KEEP(*(.vectors))
   	_vectors_end = .;
  } > ram









  .init           :
  {
    KEEP (*(SORT_NONE(.init)))
  } >ram  

  .text           :
  {
    *(.text.unlikely .text.unlikely.*)
    *(.text.startup .text.startup.*)
    *(.text .text.*)
    *(.gnu.linkonce.t.*)
  } >ram  

  .fini           :
  {
    KEEP (*(SORT_NONE(.fini)))
  } >ram  

  PROVIDE (__etext = .);
  PROVIDE (_etext = .);
  PROVIDE (etext = .);
/*
  .rodata         :
  {
    *(.rdata)
    *(.rodata .rodata.*)
    *(.gnu.linkonce.r.*)
  } >data_ram AT>data_ram 

  . = ALIGN(4);
*/
  /*
  .preinit_array  :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } >data_ram AT>data_ram 

  .init_array     :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
    KEEP (*(.init_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .ctors))
    PROVIDE_HIDDEN (__init_array_end = .);
  } >data_ram AT>data_ram 

  .fini_array     :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.dtors.*)))
    KEEP (*(.fini_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .dtors))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } >data_ram AT>data_ram 

  .ctors          :
  {
    KEEP (*crtbegin.o(.ctors))
    KEEP (*crtbegin?.o(.ctors))
    KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o ) .ctors))
    KEEP (*(SORT(.ctors.*)))
    KEEP (*(.ctors))
  } >data_ram AT>data_ram 

  .dtors          :
  {
    KEEP (*crtbegin.o(.dtors))
    KEEP (*crtbegin?.o(.dtors))
    KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o ) .dtors))
    KEEP (*(SORT(.dtors.*)))
    KEEP (*(.dtors))
  } >data_ram AT>data_ram 

  .lalign         :
  {
    . = ALIGN(4);
    PROVIDE( _data_lma = . );
  } >data_ram AT>data_ram 

  .dalign         :
  {
    . = ALIGN(4);
    PROVIDE( _data = . );
  } >data_ram AT>data_ram 
*/


  .data_comb       :
  {
    *(.rdata)
    *(.rodata .rodata.*)
    *(.gnu.linkonce.r.*)
    *(.data .data.*)
    *(.gnu.linkonce.d.*)
    PROVIDE( __global_pointer$ = . + 0x800 );
    *(.sdata .sdata.*)
    *(.gnu.linkonce.s.*)
    *(.srodata.cst16)
    *(.srodata.cst8)
    *(.srodata.cst4)
    *(.srodata.cst2)
    *(.srodata .srodata.*)
  } >data_ram 

/*
  .data          :
  {
    *(.data .data.*)
    *(.gnu.linkonce.d.*)
    . = ALIGN(4);
    PROVIDE( __global_pointer$ = . + 0x800 );
    *(.sdata .sdata.*)
    *(.gnu.linkonce.s.*)
    . = ALIGN(4);
    *(.srodata.cst16)
    *(.srodata.cst8)
    *(.srodata.cst4)
    *(.srodata.cst2)
    *(.srodata .srodata.*)
  } >data_ram AT>data_ram 
*/
  . = ALIGN(4);
  PROVIDE( _edata = . );
  PROVIDE( edata = . );

  PROVIDE( _fbss = . );
  PROVIDE( __bss_start = . );
  .bss            :
  {
    *(.sbss*)
    *(.gnu.linkonce.sb.*)
    *(.bss .bss.*)
    *(.gnu.linkonce.b.*)
    *(COMMON)
    . = ALIGN(4);
  } >data_ram 

  /* Define section boundary symbols */
  PROVIDE(__bss_end = .);
  __bss_end = .;

  /* Heap immediately follows BSS in data_ram */
  PROVIDE(__heap_start = .);
  __heap_start = .;

  . = ORIGIN(data_ram) + LENGTH(data_ram);
  PROVIDE(__heap_end = .);
  __heap_end = .;

  /* Define stack start for crt0 */
  PROVIDE(__stack_start = ORIGIN(stack) + LENGTH(stack));
  __stack_start = ORIGIN(stack) + LENGTH(stack);

  /*. = ALIGN(8); */
  . = ALIGN(4);
  PROVIDE( _end = . );
  PROVIDE( end = . );

  /*.stack ORIGIN(stack) + LENGTH(stack) - __stack_size : */
  .stack (NOLOAD) :
  {
    PROVIDE( _heap_end = . );
    /*. = __stack_size; */
    . = . + __stack_size;
    PROVIDE( _sp = . );
    /*_sp = . ; */
  } >stack  
}
