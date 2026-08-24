"""Analytical hardware-alignment estimates for the bit-packed WiSARD core."""
from dataclasses import asdict, dataclass
import time

import numpy as np

from config import WiSARDPhysicsConfig
from pipeline import NaturalTrainingPipeline


@dataclass(frozen=True)
class HardwareEstimate:
    storage_mode: str
    tuple_size: int
    total_input_bits: int
    rams_per_discriminator: int
    model_bytes: int
    address_logic_gates_per_sample: int
    memory_growth_factor: float
    hash_buckets: int


def estimate_hardware(config: WiSARDPhysicsConfig, baseline_tuple_size: int = 8) -> HardwareEstimate:
    slots = 3 * (2 ** config.TUPLE_SIZE)
    if config.STORAGE_MODE == "sparse":
        model_bytes = 0
    else:
        stored_slots = slots if config.STORAGE_MODE == "dense" else config.HASH_BUCKETS
        model_bytes = config.NUM_DISCRIMINATORS * config.NUM_RAMS_PER_DISCRIMINATOR * ((stored_slots + 7) // 8)
    # Each tuple bit needs one input route and one address contribution; the OR tree
    # is approximated by tuple_size - 1 two-input gates.
    gates_per_ram = config.TUPLE_SIZE + max(config.TUPLE_SIZE - 1, 0)
    if config.STORAGE_MODE == "hashed":
        gates_per_ram += max(config.HASH_BUCKETS.bit_length() - 1, 1)
    baseline_ram_bytes = config.NUM_DISCRIMINATORS * (config.TOTAL_INPUT_BITS // baseline_tuple_size) * ((3 * (2 ** baseline_tuple_size) + 7) // 8)
    return HardwareEstimate(
        storage_mode=config.STORAGE_MODE,
        tuple_size=config.TUPLE_SIZE,
        total_input_bits=config.TOTAL_INPUT_BITS,
        rams_per_discriminator=config.NUM_RAMS_PER_DISCRIMINATOR,
        model_bytes=model_bytes,
        address_logic_gates_per_sample=config.NUM_RAMS_PER_DISCRIMINATOR * gates_per_ram,
        memory_growth_factor=model_bytes / baseline_ram_bytes if baseline_ram_bytes else 0.0,
        hash_buckets=config.HASH_BUCKETS,
    )


def profile_runtime(config: WiSARDPhysicsConfig, batch_size: int = 512, seed: int = 7) -> dict[str, float | int]:
    np.random.seed(seed)
    pipeline = NaturalTrainingPipeline(config)
    bits, raw, labels = pipeline.generate_streaming_batch(batch_size)
    start = time.perf_counter()
    pipeline.engine.memorize(bits, raw, labels)
    elapsed = time.perf_counter() - start
    estimate = asdict(estimate_hardware(config))
    estimate["memorize_samples_per_second"] = batch_size / elapsed
    estimate["sparse_counter_entries"] = len(pipeline.engine.ram_counts)
    estimate["allocated_hard_memory_bytes"] = pipeline.engine.memory_bytes
    if config.STORAGE_MODE == "hashed":
        logical_addresses = len(pipeline.engine.ram_counts)
        stored_addresses = len({
            (discriminator, ram, pipeline.engine._stored_address(discriminator, ram, address))
            for discriminator, ram, address in pipeline.engine.ram_counts
        })
        estimate["hash_collision_count"] = logical_addresses - stored_addresses
    return estimate
