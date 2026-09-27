"""Reproduce the numerical results in Chapter 5 of the heat equation thesis.

Run with Python 3.10 or newer; only the standard library is required.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import time


T = 0.1
PI2 = math.pi**2


def initial_mode(n: int, k: int = 1) -> list[float]:
    return [math.sin(k * math.pi * i / n) for i in range(1, n)]


def factor(n: int, steps: int, method: str, final_time: float = T,
           mode: int = 1) -> float:
    dt = final_time / steps
    lam = 4 * n * n * math.sin(mode * math.pi / (2 * n)) ** 2
    z = lam * dt
    if method == "E":
        return 1 - z
    if method == "B":
        return 1 / (1 + z)
    return (1 - z / 2) / (1 + z / 2)


def solve(method: str, n: int, steps: int, final_time: float = T,
          mode: int = 1, trace: bool = False,
          initial_values: list[float] | None = None,
          snapshots: list[list[float]] | None = None):
    """Advance the full grid using explicit update or prefactored Thomas solve."""
    dt = final_time / steps
    r = dt * n * n
    values = initial_mode(n, mode) if initial_values is None else initial_values.copy()
    m = n - 1
    assert len(values) == m
    if snapshots is not None:
        snapshots.append(values.copy())
    energy = [sum(x * x for x in values) / (2 * n)] if trace else None

    if method in ("B", "CN"):
        off = -r if method == "B" else -r / 2
        diagonal = 1 + 2 * r if method == "B" else 1 + r
        inverse = [0.0] * m
        upper = [0.0] * m
        inverse[0] = 1 / diagonal
        upper[0] = off * inverse[0]
        for i in range(1, m):
            inverse[i] = 1 / (diagonal - off * upper[i - 1])
            upper[i] = off * inverse[i]

    for _ in range(steps):
        if method == "E":
            new = [0.0] * m
            for i in range(m):
                left = values[i - 1] if i else 0.0
                right = values[i + 1] if i < m - 1 else 0.0
                new[i] = r * left + (1 - 2 * r) * values[i] + r * right
            values = new
        else:
            if method == "CN":
                rhs = [0.0] * m
                for i in range(m):
                    left = values[i - 1] if i else 0.0
                    right = values[i + 1] if i < m - 1 else 0.0
                    rhs[i] = (1 - r) * values[i] + (r / 2) * (left + right)
            else:
                rhs = values
            forward = [0.0] * m
            forward[0] = rhs[0] * inverse[0]
            for i in range(1, m):
                forward[i] = (rhs[i] - off * forward[i - 1]) * inverse[i]
            new = [0.0] * m
            new[-1] = forward[-1]
            for i in range(m - 2, -1, -1):
                new[i] = forward[i] - upper[i] * new[i + 1]
            values = new
        if trace:
            energy.append(sum(x * x for x in values) / (2 * n))
        if snapshots is not None:
            snapshots.append(values.copy())
    return (values, energy) if trace else values


def dst_coefficient(values: list[float], n: int, mode: int) -> float:
    """DST-I coefficient with the discrete-sine orthogonality normalization."""
    return 2 * math.fsum(value * math.sin(mode * math.pi * i / n)
                        for i, value in enumerate(values, 1)) / n


def multimode_experiment() -> dict:
    """Run both step sizes on the full grid and verify resolvable DST values."""
    n, final_time = 40, 0.2
    low, high = initial_mode(n), initial_mode(n, n - 1)
    initial = [a + 0.5 * b for a, b in zip(low, high)]
    assert min(initial) > 0
    exact = [math.exp(-PI2 * final_time) * a
             + 0.5 * math.exp(-(n - 1) ** 2 * PI2 * final_time) * b
             for a, b in zip(low, high)]
    result = {"N": n, "T": final_time, "initial_min": min(initial),
              "cases": {}}
    for steps in (800, 8):
        r = n * n * final_time / steps
        group = {}
        for method in ("E", "B", "CN"):
            snapshots: list[list[float]] = []
            final, energies = solve(method, n, steps, final_time,
                                    trace=True, initial_values=initial,
                                    snapshots=snapshots)
            g1 = factor(n, steps, method, final_time, 1)
            g39 = factor(n, steps, method, final_time, n - 1)
            error = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(final, exact)) / n)
            # The endpoint may underflow. The logarithm retains the formula's scale.
            log10_high_ratio = steps * math.log10(abs(g39))
            predicted_low = g1 ** steps
            predicted_high = 0.5 * g39 ** steps
            predicted_energy = (predicted_low ** 2 + predicted_high ** 2) / 4
            predicted_error = math.sqrt(((predicted_low - math.exp(-PI2 * final_time)) ** 2
                                         + (predicted_high - 0.5 * math.exp(-(n - 1) ** 2 * PI2 * final_time)) ** 2) / 2)
            assert abs(energies[-1] - predicted_energy) <= 1e-10 * max(1, predicted_energy)
            assert abs(error - predicted_error) <= 1e-10 * max(1, predicted_error)
            checked = []
            for j, values in enumerate(snapshots):
                projected_low = dst_coefficient(values, n, 1)
                projected_high = dst_coefficient(values, n, n - 1)
                expected_low = g1 ** j
                expected_high = 0.5 * g39 ** j
                if abs(expected_high) > 1e-12 * max(1, abs(expected_low)):
                    assert abs(projected_high - expected_high) <= 2e-10 * max(1, abs(expected_high)), (steps, method, j)
                    checked.append(j)
                if j in (0, 1, steps) and abs(expected_high) < 1e4:
                    assert abs(projected_low - expected_low) <= 2e-10 * max(1, abs(expected_low))
            group[method] = {
                "g1": g1, "g39": g39,
                "predicted_high_ratio_log10": log10_high_ratio,
                "L2_continuous": error, "energy_final": energies[-1],
                "grid_min_final": min(final),
                "first_negative_step": next((j for j, values in enumerate(snapshots)
                                             if min(values) < -1e-12), None),
                "dst_verified_through_step": max(checked),
                "high_curve": [[j, j * math.log10(abs(g39)),
                                1 if g39 > 0 or j % 2 == 0 else -1]
                               for j in range(steps + 1)] if steps == 8 else None,
                "spatial_final": [0.0, *final, 0.0] if steps == 8 and method != "E" else None,
            }
        group["r"] = r
        result["cases"][str(steps)] = group
    assert result["cases"]["8"]["CN"]["first_negative_step"] == 4
    return result


def error(n: int, steps: int, method: str, semidiscrete: bool = False) -> tuple[float, float]:
    values = solve(method, n, steps)
    lam = 4 * n * n * math.sin(math.pi / (2 * n)) ** 2
    amplitude = math.exp(-lam * T) if semidiscrete else math.exp(-PI2 * T)
    exact = [amplitude * x for x in initial_mode(n)]
    err = [a - b for a, b in zip(values, exact)]
    l2 = math.sqrt(sum(x * x for x in err) / n)
    linf = max(abs(x) for x in err)
    # Independent modal oracle catches errors in the grid implementation.
    predicted = abs(factor(n, steps, method) ** steps - amplitude) / math.sqrt(2)
    assert abs(l2 - predicted) < 1e-11, (method, n, steps, l2, predicted)
    return l2, linf


def rate(current: float, previous: float) -> float:
    return math.log(previous / current, 2)


def bench(method: str, n: int, steps: int, repetitions: int = 7) -> dict:
    solve(method, n, steps)
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        solve(method, n, steps)
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples),
            "samples_ms": samples}


def main() -> None:
    result = {"time": {}, "spatial": {}, "energy": {}, "instability": {},
              "multimode": {}, "cost": {}}

    for method in ("E", "B", "CN"):
        rows = []
        prior = None
        for steps in (256, 512, 1024, 2048):
            e2, einf = error(32, steps, method, semidiscrete=True)
            rows.append({"M": steps, "dt": T / steps, "L2": e2,
                         "Linf": einf, "rate": None if prior is None else rate(e2, prior)})
            prior = e2
        result["time"][method] = rows

        rows = []
        prior = None
        for n in (20, 40, 80, 160):
            steps = n * n // 4
            e2, einf = error(n, steps, method)
            rows.append({"N": n, "M": steps, "r": T / steps * n * n,
                         "L2": e2, "Linf": einf,
                         "rate": None if prior is None else rate(e2, prior)})
            prior = e2
        result["spatial"][method] = rows

    for method in ("E", "B", "CN"):
        _, energies = solve(method, 20, 200, final_time=0.2, trace=True)
        result["energy"][method] = {
            "final": energies[-1],
            "max_increment": max(b - a for a, b in zip(energies, energies[1:])),
            "curve": [[i / 1000, energies[i] / energies[0]] for i in range(0, 201, 20)],
        }
    result["energy"]["exact"] = [[i / 1000, math.exp(-2 * PI2 * i / 1000)]
                                     for i in range(0, 201, 20)]

    n = 40
    threshold = 1 / (2 * math.cos(math.pi / (2 * n)) ** 2)
    result["instability"]["threshold"] = threshold
    for r in (0.49, 0.5, 0.55):
        g = 1 - 4 * r * math.cos(math.pi / (2 * n)) ** 2
        result["instability"][str(r)] = {
            "g": g, "energy_ratio_40": abs(g) ** 80,
            "curve": [[j, 2 * j * math.log10(abs(g))] for j in range(0, 41, 5)],
        }
        # Direct grid update agrees with the modal energy prediction.
        _, energies = solve("E", n, 40, final_time=40 * r / n**2, mode=n - 1, trace=True)
        assert abs(energies[-1] / energies[0] / abs(g) ** 80 - 1) < 1e-10

    result["multimode"] = multimode_experiment()

    n = 80
    tolerance = 2e-4
    result["cost"]["tolerance"] = tolerance
    result["cost"]["environment"] = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "processor": platform.processor(),
    }
    for method in ("E", "B", "CN"):
        candidates = []
        for steps in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            r = T / steps * n * n
            if method == "E" and r > 0.5:
                continue
            e2, _ = error(n, steps, method)
            if e2 <= tolerance:
                candidates.append((steps, r, e2))
        steps, r, e2 = candidates[0]
        result["cost"][method] = {"N": n, "M": steps, "r": r,
                                   "L2": e2, **bench(method, n, steps)}

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
