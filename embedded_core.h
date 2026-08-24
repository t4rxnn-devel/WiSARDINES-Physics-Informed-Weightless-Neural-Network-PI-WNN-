#ifndef PI_WNN_EMBEDDED_CORE_H
#define PI_WNN_EMBEDDED_CORE_H

#include <stddef.h>
#include <stdint.h>

uint32_t pi_wnn_tuple_address(const uint8_t *bits, const uint8_t *mapping, size_t tuple_size);
void pi_wnn_set_bit(uint8_t *ram, uint32_t address);
uint8_t pi_wnn_get_bit(const uint8_t *ram, uint32_t address);

#endif