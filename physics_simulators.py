"""Small physical simulators used to evaluate physics-aware weightless models."""
from dataclasses import dataclass
import math

import numpy as np


class IsingTransitionSimulator:
    """2D Ising model with 3x3 neighborhood transition memory."""

    def __init__(self, size: int = 16, temperature: float = 2.269, seed: int = 7) -> None:
        if size < 3 or temperature <= 0:
            raise ValueError("size must be >= 3 and temperature must be positive")
        self.size = size
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(size, size))
        self.transition_counts: dict[tuple[int, int], np.ndarray] = {}

    def neighborhood_address(self, row: int, column: int) -> int:
        address = 0
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                address = (address << 1) | int(
                    self.spins[(row + row_offset) % self.size, (column + column_offset) % self.size] > 0
                )
        return address

    def local_energy_delta(self, row: int, column: int) -> float:
        neighbor_sum = sum(
            self.spins[(row + row_offset) % self.size, (column + column_offset) % self.size]
            for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        return float(2 * self.spins[row, column] * neighbor_sum)

    def metropolis_step(self) -> bool:
        row = int(self.rng.integers(self.size))
        column = int(self.rng.integers(self.size))
        before = self.neighborhood_address(row, column)
        delta = self.local_energy_delta(row, column)
        accepted = delta <= 0 or self.rng.random() < math.exp(-delta / self.temperature)
        after = before
        if accepted:
            self.spins[row, column] *= -1
            after = self.neighborhood_address(row, column)
        key = (before, after)
        if key not in self.transition_counts:
            self.transition_counts[key] = np.zeros(2, dtype=np.int64)
        self.transition_counts[key][int(accepted)] += 1
        return accepted

    def run(self, steps: int) -> float:
        if steps <= 0:
            raise ValueError("steps must be positive")
        accepted = sum(self.metropolis_step() for _ in range(steps))
        return accepted / steps

    def transition_probability(self, before: int, after: int) -> float:
        counts = self.transition_counts.get((before, after))
        if counts is None:
            return 0.5
        return float((counts[1] + 1) / (counts.sum() + 2))


@dataclass
class PendulumState:
    theta1: float
    theta2: float
    omega1: float
    omega2: float


class EnergyConstrainedDoublePendulum:
    """Chaotic double pendulum with an explicit energy-preserving projection."""

    def __init__(self, state: PendulumState | None = None, *, dt: float = 0.01, energy_budget: float | None = None) -> None:
        self.state = state or PendulumState(1.2, 1.0, 0.0, 0.0)
        self.dt = float(dt)
        self.g = 9.81
        self.m1 = self.m2 = 1.0
        self.l1 = self.l2 = 1.0
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        self.energy_budget = self.energy() if energy_budget is None else float(energy_budget)
        if self.energy_budget <= 0:
            raise ValueError("energy_budget must be positive")

    def energy(self, state: PendulumState | None = None) -> float:
        state = self.state if state is None else state
        delta = state.theta1 - state.theta2
        kinetic = 0.5 * self.m1 * self.l1**2 * state.omega1**2
        kinetic += 0.5 * self.m2 * (
            self.l1**2 * state.omega1**2 + self.l2**2 * state.omega2**2
            + 2 * self.l1 * self.l2 * state.omega1 * state.omega2 * math.cos(delta)
        )
        potential = -self.g * ((self.m1 + self.m2) * self.l1 * math.cos(state.theta1) + self.m2 * self.l2 * math.cos(state.theta2))
        return kinetic + potential + 2 * self.g * self.l1

    def _derivative(self, state: PendulumState) -> PendulumState:
        delta = state.theta1 - state.theta2
        denominator = 2 * self.m1 + self.m2 - self.m2 * math.cos(2 * delta)
        acceleration1 = (-self.g * (2 * self.m1 + self.m2) * math.sin(state.theta1)
                        - self.m2 * self.g * math.sin(state.theta1 - 2 * state.theta2)
                        - 2 * math.sin(delta) * self.m2 * (state.omega2**2 * self.l2 + state.omega1**2 * self.l1 * math.cos(delta))) / (self.l1 * denominator)
        acceleration2 = (2 * math.sin(delta) * (state.omega1**2 * self.l1 * (self.m1 + self.m2)
                        + self.g * (self.m1 + self.m2) * math.cos(state.theta1)
                        + state.omega2**2 * self.l2 * self.m2 * math.cos(delta))) / (self.l2 * denominator)
        return PendulumState(state.omega1, state.omega2, acceleration1, acceleration2)

    @staticmethod
    def _add(state: PendulumState, derivative: PendulumState, scale: float) -> PendulumState:
        return PendulumState(
            state.theta1 + derivative.theta1 * scale,
            state.theta2 + derivative.theta2 * scale,
            state.omega1 + derivative.omega1 * scale,
            state.omega2 + derivative.omega2 * scale,
        )

    def _project_energy(self, state: PendulumState) -> tuple[PendulumState, bool]:
        potential_state = PendulumState(state.theta1, state.theta2, 0.0, 0.0)
        potential = self.energy(potential_state)
        target_kinetic = self.energy_budget - potential
        if target_kinetic < 0:
            return PendulumState(state.theta1, state.theta2, 0.0, 0.0), False
        current_kinetic = self.energy(state) - potential
        scale = 0.0 if current_kinetic <= 0 else math.sqrt(target_kinetic / current_kinetic)
        projected = PendulumState(state.theta1, state.theta2, state.omega1 * scale, state.omega2 * scale)
        return projected, True

    def step(self) -> bool:
        initial = self.state
        k1 = self._derivative(initial)
        k2 = self._derivative(self._add(initial, k1, self.dt / 2))
        k3 = self._derivative(self._add(initial, k2, self.dt / 2))
        k4 = self._derivative(self._add(initial, k3, self.dt))
        candidate = PendulumState(
            initial.theta1 + self.dt * (k1.theta1 + 2 * k2.theta1 + 2 * k3.theta1 + k4.theta1) / 6,
            initial.theta2 + self.dt * (k1.theta2 + 2 * k2.theta2 + 2 * k3.theta2 + k4.theta2) / 6,
            initial.omega1 + self.dt * (k1.omega1 + 2 * k2.omega1 + 2 * k3.omega1 + k4.omega1) / 6,
            initial.omega2 + self.dt * (k1.omega2 + 2 * k2.omega2 + 2 * k3.omega2 + k4.omega2) / 6,
        )
        self.state, valid = self._project_energy(candidate)
        return valid

    def trajectory(self, steps: int) -> tuple[np.ndarray, np.ndarray]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        states = []
        valid = []
        for _ in range(steps):
            valid.append(self.step())
            states.append((self.state.theta1, self.state.theta2, self.state.omega1, self.state.omega2))
        return np.asarray(states), np.asarray(valid, dtype=bool)
