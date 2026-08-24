#include "embedded_core.h"
#include <stdio.h>
#include <time.h>

int main(void) {
    uint8_t bits[128] = {0};
    uint8_t mapping[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t ram[32] = {0};
    const size_t iterations = 1000000;
    clock_t start = clock();
    for (size_t index = 0; index < iterations; ++index) {
        uint32_t address = pi_wnn_tuple_address(bits, mapping, 8);
        pi_wnn_set_bit(ram, address);
        (void)pi_wnn_get_bit(ram, address);
    }
    double elapsed = (double)(clock() - start) / (double)CLOCKS_PER_SEC;
    printf("iterations=%zu seconds=%.9f ns_per_iteration=%.3f hit=%u\n", iterations, elapsed, elapsed * 1e9 / iterations, pi_wnn_get_bit(ram, 0));
    return 0;
}