#include "embedded_core.h"

uint32_t pi_wnn_tuple_address(const uint8_t *bits, const uint8_t *mapping, size_t tuple_size) {
    uint32_t address = 0;
    for (size_t index = 0; index < tuple_size; ++index) {
        address = (address << 1) | (uint32_t)(bits[mapping[index]] != 0);
    }
    return address;
}

void pi_wnn_set_bit(uint8_t *ram, uint32_t address) {
    ram[address >> 3] |= (uint8_t)(1u << (address & 7u));
}

uint8_t pi_wnn_get_bit(const uint8_t *ram, uint32_t address) {
    return (uint8_t)((ram[address >> 3] >> (address & 7u)) & 1u);
}