"""Dependency-free deterministic cuckoo table for exact logical addresses."""


class CuckooAddressTable:
    def __init__(self, capacity: int = 4096, max_kicks: int = 64) -> None:
        if capacity <= 1 or max_kicks <= 0:
            raise ValueError("capacity and max_kicks must be positive")
        self.capacity = capacity
        self.max_kicks = max_kicks
        self._slots: list[tuple[int, int] | None] = [None] * capacity
        self.size = 0

    def _positions(self, key: int) -> tuple[int, int]:
        key = int(key)
        first = (key * 0x9E3779B1) % self.capacity
        second = ((key ^ (key >> 16)) * 0x85EBCA77) % self.capacity
        return first, second

    def contains(self, key: int) -> bool:
        return any(slot is not None and slot[0] == key for slot in (self._slots[index] for index in self._positions(key)))

    def insert(self, key: int, value: int = 1) -> None:
        if self.contains(key):
            return
        if self.size >= self.capacity:
            raise OverflowError("cuckoo table is full; increase capacity")
        entry = (int(key), int(value))
        position = self._positions(entry[0])[0]
        for _ in range(self.max_kicks):
            if self._slots[position] is None:
                self._slots[position] = entry
                self.size += 1
                return
            self._slots[position], entry = entry, self._slots[position]
            alternatives = self._positions(entry[0])
            position = alternatives[1] if position == alternatives[0] else alternatives[0]
        raise OverflowError("cuckoo insertion exceeded max_kicks; increase capacity")

    def get(self, key: int, default: int = 0) -> int:
        for position in self._positions(key):
            slot = self._slots[position]
            if slot is not None and slot[0] == key:
                return slot[1]
        return default