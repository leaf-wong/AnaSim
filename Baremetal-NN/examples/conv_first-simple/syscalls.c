#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

#define UART_BASE 0x40000000
#define UART_TXFIFO ((volatile uint8_t *)(UART_BASE))

extern char __heap_start;
extern char __heap_end;
static char *heap_ptr = &__heap_start;

void * _sbrk(ptrdiff_t incr) {
    char *prev_heap = heap_ptr;
    if (heap_ptr + incr > &__heap_end)
        return (void *)-1;
    heap_ptr += incr;
    return prev_heap;
}

int _write(int fd, const void *buf, size_t count) {
    const char *c = (const char *)buf;
    if (fd != STDOUT_FILENO && fd != STDERR_FILENO)
        return -1;

    for (size_t i = 0; i < count; i++) {
        *UART_TXFIFO = c[i];
    }

    return count;
}

// Minimal stubs for newlib compatibility
int _close(int fd) { return -1; }
int _fstat(int fd, struct stat *st) { st->st_mode = S_IFCHR; return 0; }
int _isatty(int fd) { return 1; }
int _lseek(int fd, int ptr, int dir) { return 0; }
int _read(int fd, void *buf, size_t count) { return 0; }
void _exit(int status) {
    const char *msg = "Exiting...\n";
    _write(STDOUT_FILENO, msg, strlen(msg));
    // Optional: print status or write to a debug register
    volatile uint64_t *tohost = (uint64_t *)0x80001000;
    *tohost = (uint64_t)((status << 1) | 1);  // Convention: status left-shifted + 1
    //while (1) {
    //    asm volatile ("wfi");
    //}
}
int _kill(int pid, int sig) { return -1; }
int _getpid(void) { return 1; }