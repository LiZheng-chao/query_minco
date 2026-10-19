"""Independent FP64 equality-constrained minimum-jerk QP, not the MINCO matrix.

Uses normalized local u in [0,1], C2 and waypoint constraints only. The KKT
optimality conditions determine the free interior velocities/accelerations.
No import of YOPO, torch solver or adapter is used in reference construction.
"""
import math
import numpy as np


def basis(t, order):
    return np.array([0.0 if k < order else math.factorial(k) / math.factorial(k-order) * t**(k-order)
                     for k in range(6)], dtype=np.float64)


def reference_solve(head, tail, inner, durations):
    head, tail, inner, durations = [np.asarray(x, dtype=np.float64) for x in (head, tail, inner, durations)]
    H = np.zeros((18, 18), dtype=np.float64)
    for segment, T in enumerate(durations):
        for i in range(3, 6):
            for j in range(3, 6):
                H[segment*6+i, segment*6+j] = (
                    math.factorial(i)/math.factorial(i-3) * math.factorial(j)/math.factorial(j-3)
                    / (i+j-5) / T**5)
    rows, rhs = [], []
    def row(segment, u, order):
        result = np.zeros(18)
        result[segment*6:(segment+1)*6] = basis(u, order) / durations[segment]**order
        return result
    for order in range(3):
        rows.extend([row(0, 0, order), row(2, 1, order)])
        rhs.extend([head[order], tail[order]])
    for joint in range(2):
        rows.extend([row(joint, 1, 0), row(joint+1, 0, 0)])
        rhs.extend([inner[joint], inner[joint]])
        for order in (1, 2):
            rows.append(row(joint, 1, order)-row(joint+1, 0, order))
            rhs.append(np.zeros(3))
    A = np.stack(rows)
    kkt = np.block([[H, A.T], [A, np.zeros((len(rows), len(rows)))]])
    solution = np.linalg.solve(kkt, np.concatenate((np.zeros((18, 3)), np.stack(rhs))))
    normalized = solution[:18].reshape(3, 6, 3)
    physical = normalized / durations[:, None, None]**np.arange(6)[None, :, None]
    return physical.transpose(0, 2, 1)


def reference_evaluate(coeffs, durations, times, order=0):
    values = []
    starts = np.r_[0., np.cumsum(durations)[:-1]]
    for t in times:
        segment = min(int(np.searchsorted(np.cumsum(durations), t, side="right")), 2)
        values.append(coeffs[segment] @ basis(t-starts[segment], order))
    return np.stack(values)


def reference_energy(coeffs, durations):
    energy = 0.
    for c, T in zip(coeffs, durations):
        for i in range(3, 6):
            for j in range(3, 6):
                energy += np.dot(c[:, i], c[:, j]) * (
                    math.factorial(i)/math.factorial(i-3)*math.factorial(j)/math.factorial(j-3)
                    * T**(i+j-5)/(i+j-5))
    return float(energy)
