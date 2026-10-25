import json
import pennylane as qp
import pennylane.numpy as np

# The unitary sizes your circuit is graded on, and the wire budget.
TEST_SIZES = (2, 3, 4, 5)
MAX_TOTAL_WIRES = 8
# The grader has to finish inside the challenge platform's time limit, so a decomposed
# circuit may hold at most this many gates, which is more than a good solution should need.
MAX_GATES = 100_000

ATOL = 1e-8
GRADES = {"Bronze": 5.0, "Silver": 20.0, "Gold": 60.0}  # minimum saving at TEST_SIZES[-1]

# ---------------------------------------------------------------------------------------
# The cost model. You do not need to change any of this, but read it: it is the scoring.
#
# 1. Your circuit is decomposed with qp.decompose into the gate set below.
#
# 2. Every surviving gate must be one of these exact PennyLane classes. Anything else, a
#    subclass, a look-alike with a borrowed name, or an operator with no decomposition,
#    is rejected rather than charged.
#
# 3. Each gate is charged in T-gate equivalents: a rotation 44, a T gate 1, the rest 0.
#
# 4. Qubits are charged too. Every wire beyond the n data wires multiplies the whole cost
#    by AUX_WIRE_FACTOR. A circuit on more than MAX_TOTAL_WIRES wires is rejected without
#    being simulated.
# ---------------------------------------------------------------------------------------
GATE_COSTS = {
    qp.RX: 44, qp.RY: 44, qp.RZ: 44,  
    qp.T: 1,
    qp.S: 0, qp.Hadamard: 0, qp.PauliX: 0, qp.PauliY: 0, qp.PauliZ: 0,
    qp.CNOT: 0, qp.GlobalPhase: 0,
}
TARGET_GATE_SET = {"RX", "RY", "RZ", "T", "S", "H", "X", "Y", "Z", "CNOT", "GlobalPhase"}
ROTATION_COST = 44
AUX_WIRE_FACTOR = 2


class IllegalGateError(ValueError):
    """Raised when a decomposed circuit contains a gate outside the cost model."""


def compile_circuit(circuit, u_mat):
    """Decomposes `circuit(u_mat)` into the target gate set and checks every gate.

    Returns:
        qp.tape.QuantumScript: the decomposed circuit, which is what is charged and
            what is simulated.
    """
    tape = qp.tape.make_qscript(circuit)(u_mat)
    (decomposed,), _ = qp.decompose(tape, gate_set=TARGET_GATE_SET)
    for op in decomposed.operations:
        if type(op) not in GATE_COSTS:
            raise IllegalGateError(
                f"{type(op).__name__} on wires {list(op.wires)} is not in the gate set. Only "
                "the exact PennyLane classes RX, RY, RZ, T, S, H, X, Y, Z, CNOT and "
                "GlobalPhase are accepted after decomposition; subclasses and custom "
                "operators are rejected, whatever they are called."
            )
    return decomposed


def wire_usage(tape, num_qubits):
    """(total wires, auxiliary wires) used by a tape whose data wires are 0 to num_qubits-1."""
    data = set(range(num_qubits))
    used = set(tape.wires.tolist())
    return len(used | data), len(used - data)


def tape_cost(tape, num_qubits):
    """The T-gate-equivalent cost of an already compiled tape, (44 N_rot + N_T) * 2**N_aux.
    A tape on more than MAX_TOTAL_WIRES wires is not charged: the grader rejects it."""
    gates = sum(GATE_COSTS[type(op)] for op in tape.operations)
    _, aux_wires = wire_usage(tape, num_qubits)
    return int(gates * AUX_WIRE_FACTOR ** aux_wires)


def cost_of(circuit, u_mat):
    """The T-gate-equivalent cost of `circuit(u_mat)`. Call this on your own circuit."""
    num_qubits = int(round(np.log2(len(u_mat))))
    return tape_cost(compile_circuit(circuit, u_mat), num_qubits)


def _plain(array):
    """A plain NumPy array, without pennylane.numpy's autodiff bookkeeping (faster)."""
    array = np.array(array, dtype=complex, requires_grad=False)
    return array.unwrap() if hasattr(array, "unwrap") else array


_FIXED_GATES = {
    qp.T: _plain([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
    qp.S: _plain([[1, 0], [0, 1j]]),
    qp.Hadamard: _plain([[2 ** -0.5, 2 ** -0.5], [2 ** -0.5, -(2 ** -0.5)]]),
    qp.PauliX: _plain([[0, 1], [1, 0]]),
    qp.PauliY: _plain([[0, -1j], [1j, 0]]),
    qp.PauliZ: _plain([[1, 0], [0, -1]]),
    qp.CNOT: _plain([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
}


def _gate_matrix(op):
    """The grader's own matrix for each gate. Gates are never asked for their matrices."""
    kind = type(op)
    if kind in (qp.RX, qp.RY, qp.RZ):
        half = float(op.data[0]) / 2
        c, s = float(np.cos(half)), float(np.sin(half))
        if kind is qp.RX:
            return _plain([[c, -1j * s], [-1j * s, c]])
        if kind is qp.RY:
            return _plain([[c, -s], [s, c]])
        return _plain([[complex(c, -s), 0], [0, complex(c, s)]])
    return _FIXED_GATES[kind]


def tape_matrix(tape, num_qubits):
    """The matrix a compiled tape implements on wires 0 to num_qubits-1, every other
    wire starting in |0>. Simulated with the grader's own gate matrices.

    Runs of single-qubit gates on a wire are multiplied together before they are applied,
    and a CNOT is applied as a swap of two slices, which keeps grading fast."""
    order = list(range(num_qubits)) + sorted(
        (w for w in tape.wires.tolist() if w not in range(num_qubits)), key=str
    )
    position = {wire: k for k, wire in enumerate(order)}
    dim = 2 ** num_qubits
    states = _plain(np.zeros((dim,) + (2,) * len(order)))
    for column in range(dim):
        bits = [(column >> (num_qubits - 1 - k)) & 1 for k in range(num_qubits)]
        states[(column, *bits) + (0,) * (len(order) - num_qubits)] = 1.0
    every = [slice(None)] * states.ndim

    def axis_slice(fixed):
        index = list(every)
        for axis, value in fixed.items():
            index[axis] = value
        return tuple(index)

    def apply_single(gate, axis):
        zero, one = states[axis_slice({axis: 0})], states[axis_slice({axis: 1})]
        new_zero = gate[0, 0] * zero + gate[0, 1] * one
        one *= gate[1, 1]
        one += gate[1, 0] * zero
        zero[...] = new_zero

    pending = {}  # wire -> product of the single-qubit gates not applied yet

    def flush(wire):
        gate = pending.pop(wire, None)
        if gate is not None:
            apply_single(gate, 1 + position[wire])

    phase = 1.0 + 0j
    for op in tape.operations:
        kind = type(op)
        if kind is qp.GlobalPhase:
            phase *= complex(np.exp(-1j * float(op.data[0])))
            continue
        wires = op.wires.tolist()
        if len(wires) == 1:
            gate = _gate_matrix(op)
            pending[wires[0]] = gate @ pending[wires[0]] if wires[0] in pending else gate
            continue
        for wire in wires:
            flush(wire)
        axes = [1 + position[w] for w in wires]
        if kind is qp.CNOT:
            low = axis_slice({axes[0]: 1, axes[1]: 0})
            high = axis_slice({axes[0]: 1, axes[1]: 1})
            swapped = states[low].copy()
            states[low] = states[high]
            states[high] = swapped
        else:
            gate = _gate_matrix(op).reshape((2,) * (2 * len(axes)))
            states = np.tensordot(gate, states, axes=(list(range(len(axes), 2 * len(axes))), axes))
            states = np.moveaxis(states, list(range(len(axes))), axes)
    for wire in list(pending):
        flush(wire)
    kept = states.reshape(dim, dim, -1)[:, :, 0]
    return np.array(phase * kept.T, requires_grad=False)


def circuit_matrix(circuit, u_mat):
    """The matrix of `circuit(u_mat)` on wires 0 to n-1, every other wire starting in |0>."""
    num_qubits = int(round(np.log2(len(u_mat))))
    return tape_matrix(compile_circuit(circuit, u_mat), num_qubits)


def baseline_cost(num_qubits):
    """The cost of the 2006 Shannon-decomposition baseline, 44 * (21/16 * 4**n - 3 * 2**(n-1))."""
    return ROTATION_COST * ((21 * 4 ** num_qubits) // 16 - 3 * 2 ** (num_qubits - 1))


def haar_unitary(num_qubits, seed):
    """A Haar-random unitary of dimension 2**num_qubits. You do not need to change this."""
    rng = np.random.default_rng(seed)
    dim = 2 ** num_qubits
    ginibre = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    q_mat, r_mat = np.linalg.qr(ginibre)
    diagonal = np.diag(r_mat)
    return np.array(q_mat * (diagonal / np.abs(diagonal)), requires_grad=False)


# Two pieces of linear algebra you are given. You do not need to change these.
_TOL = 1e-8
_MIXERS = (0.6180339887498949, 1.3027756377319946, 0.2360679774997896, 2.7320508075688772)


def diagonalize_unitary(d_mat):
    """Eigendecomposes a unitary in an orthonormal basis. Returns (v_mat, phases) with

        d_mat == v_mat @ diag(exp(1j * phases)) @ v_mat.conj().T
    """
    herm = d_mat + d_mat.conj().T
    skew = -1j * (d_mat - d_mat.conj().T)
    # herm and skew are Hermitian and commute with d_mat, so an eigenbasis of a generic
    # real combination of the two is an orthonormal eigenbasis of d_mat itself.
    best = None
    for mixer in _MIXERS:
        _, v_mat = np.linalg.eigh(herm + mixer * skew)
        rotated = v_mat.conj().T @ d_mat @ v_mat
        residual = np.abs(rotated - np.diag(np.diag(rotated))).max()
        if best is None or residual < best[0]:
            best = (residual, v_mat, np.angle(np.diag(rotated)))
        if residual < 1e-13:
            break
    return best[1], best[2]


def cs_decomposition(u_mat):
    """Cosine-sine decomposition of a unitary of even dimension 2m. Returns
    (l_0, l_1, theta, r_0, r_1) such that, in m x m blocks,

        u_mat == [[l_0, 0], [0, l_1]] @ [[C, -S], [S, C]] @ [[r_0, 0], [0, r_1]]

    with C = diag(cos(theta)) and S = diag(sin(theta)).
    """
    half = len(u_mat) // 2
    u_00, u_01 = u_mat[:half, :half], u_mat[:half, half:]
    u_10, u_11 = u_mat[half:, :half], u_mat[half:, half:]
    l_0, cosines, r_0 = np.linalg.svd(u_00)
    lower = u_10 @ r_0.conj().T
    sines = np.linalg.norm(lower, axis=0)
    theta = np.arctan2(sines, cosines)
    big_s, big_c = sines > _TOL, cosines > _TOL
    l_1 = _complete_basis(lower / np.where(big_s, sines, 1.0), big_s)
    r_1 = np.where(
        big_s[:, None],
        -(l_0.conj().T @ u_01) / np.where(big_s, sines, 1.0)[:, None],
        (l_1.conj().T @ u_11) / np.where(big_c, cosines, 1.0)[:, None],
    )
    # l_1 and r_1 come from dividing by small sines, so they drift from unitarity; snap
    # each to the nearest unitary so the drift is not amplified by a deep recursion.
    return l_0, _nearest_unitary(l_1), theta, r_0, _nearest_unitary(r_1)


def _nearest_unitary(mat):
    """The unitary closest to `mat` in Frobenius norm (its polar factor)."""
    left, _, right = np.linalg.svd(mat)
    return left @ right


def _complete_basis(partial, mask):
    """Fills the columns of `partial` where `mask` is False with an orthonormal completion."""
    missing = int((~mask).sum())
    if missing == 0:
        return partial
    good = partial[:, mask]
    extra, _, _ = np.linalg.svd(np.eye(len(partial)) - good @ good.conj().T)
    out = np.array(partial, dtype=complex)
    out[:, ~mask] = extra[:, :missing]
    return out


def submission_circ(u_mat):
    """Queues a circuit equal to u_mat to within ATOL, global phase included, at low cost.

    Args:
        u_mat (np.ndarray): A 2**n x 2**n unitary matrix, with n in TEST_SIZES. It is
            Haar-random, so you may assume no special structure whatsoever.

    Returns:
        None. The gates you queue on wires 0 to n-1 are the answer. Auxiliary wires are
        allowed, up to MAX_TOTAL_WIRES in total, but each one doubles your cost.
    """


    # Put your code here #
    
    # paper method: https://arxiv.org/abs/2603.14202
    # original code reference: https://github.com/Cass490/Yamazaki-akebui
    # Used claude opus 5.5 to help optimize the runtime around 4 minute per testcase with single thread
    import sys
    import types

    # The whole synthesis pipeline (pure Python + NumPy) is bundled below as module sources and
    # loaded once per process; later calls reuse the modules and their caches (prefix sets etc.).
    state = submission_circ.__dict__
    if "modules" not in state:
        sources = {
            'zomega': r'''"""Exact arithmetic for single-qubit Clifford+T synthesis (pure Python).

Z[omega], omega = e^{i pi/4}: an element a0 + a1 w + a2 w^2 + a3 w^3 is a 4-tuple of ints
(w^4 = -1). An exact unitary is (k, (u, w, v, z)) meaning 2^{-k/2} [[u, w], [v, z]].
"""

import cmath
import math

OMEGA = cmath.exp(1j * math.pi / 4)
ZERO = (0, 0, 0, 0)
ONE = (1, 0, 0, 0)
W1 = (0, 1, 0, 0)            # omega
SQRT2 = (0, 1, 0, -1)        # omega - omega^3 = sqrt(2)


def zmul(a, b):
    c0 = a[0] * b[0] - a[1] * b[3] - a[2] * b[2] - a[3] * b[1]
    c1 = a[0] * b[1] + a[1] * b[0] - a[2] * b[3] - a[3] * b[2]
    c2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0] - a[3] * b[3]
    c3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0]
    return (c0, c1, c2, c3)


def zadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def zneg(a):
    return (-a[0], -a[1], -a[2], -a[3])


def zconj(a):
    """Complex conjugate (omega -> omega^-1)."""
    return (a[0], -a[3], -a[2], -a[1])


def zpow_omega(m):
    m %= 8
    return zneg(zpow_omega(m - 4)) if m >= 4 else tuple(1 if i == m else 0 for i in range(4))


def sigma1(a):
    return complex(a[0] + a[1] * OMEGA + a[2] * 1j + a[3] * OMEGA ** 3)


def sigma2(a):
    """The sqrt(2)-conjugate embedding (omega -> -omega)."""
    return complex(a[0] - a[1] * OMEGA + a[2] * 1j - a[3] * OMEGA ** 3)


def divisible_by_sqrt2(a):
    b = zmul(a, SQRT2)          # a/sqrt2 = a*sqrt2/2
    return all(x % 2 == 0 for x in b)


def div_sqrt2(a):
    b = zmul(a, SQRT2)
    return (b[0] // 2, b[1] // 2, b[2] // 2, b[3] // 2)


# ---- exact 2x2 unitaries ------------------------------------------------------------------

def umul(A, B):
    ka, (a00, a01, a10, a11) = A
    kb, (b00, b01, b10, b11) = B
    m = (zadd(zmul(a00, b00), zmul(a01, b10)), zadd(zmul(a00, b01), zmul(a01, b11)),
         zadd(zmul(a10, b00), zmul(a11, b10)), zadd(zmul(a10, b01), zmul(a11, b11)))
    return ureduce((ka + kb, m))


def ureduce(U):
    k, m = U
    while k > 0 and all(divisible_by_sqrt2(e) for e in m):
        m = tuple(div_sqrt2(e) for e in m)
        k -= 1
    return (k, m)


def udagger(U):
    k, (a, b, c, d) = U
    return (k, (zconj(a), zconj(c), zconj(b), zconj(d)))


def ufloat(U):
    k, m = U
    s = 2.0 ** (-k / 2)
    return [[sigma1(m[0]) * s, sigma1(m[1]) * s], [sigma1(m[2]) * s, sigma1(m[3]) * s]]


GATES = {
    "H": (1, (ONE, ONE, ONE, zneg(ONE))),
    "T": (0, (ONE, ZERO, ZERO, W1)),
    "S": (0, (ONE, ZERO, ZERO, (0, 0, 1, 0))),
    "X": (0, (ZERO, ONE, ONE, ZERO)),
    "Z": (0, (ONE, ZERO, ZERO, zneg(ONE))),
    "I": (0, (ONE, ZERO, ZERO, ONE)),
}


def uword(word):
    """Exact unitary of a gate string (leftmost = leftmost matrix factor)."""
    U = GATES["I"]
    for g in word:
        if g == "t":
            U = umul(U, GATES["T"]); U = umul(U, GATES["S"]); U = umul(U, GATES["Z"])
        elif g == "s":
            U = umul(U, GATES["S"]); U = umul(U, GATES["Z"])
        elif g == "W":
            U = (U[0], tuple(zmul(e, W1) for e in U[1]))
        else:
            U = umul(U, GATES[g])
    return U


# ---- Matsumoto-Amano normal-form prefixes ---------------------------------------------------

def ma_prefixes(t_prime):
    """All normal-form prefixes (T|e)(HT|SHT)^* containing exactly t_prime T gates, as
    (word, exact unitary). Count 3 * 2^(t_prime - 1) for t_prime >= 1."""
    if t_prime == 0:
        return [("", GATES["I"])]
    out = []
    HT = umul(GATES["H"], GATES["T"])
    SHT = umul(GATES["S"], HT)

    def grow(word, U, remaining):
        if remaining == 0:
            out.append((word, U))
            return
        grow(word + "HT", umul(U, HT), remaining - 1)
        grow(word + "SHT", umul(U, SHT), remaining - 1)

    grow("T", GATES["T"], t_prime - 1)
    grow("", GATES["I"], t_prime)
    return out
''',
            'normal_form': r'''"""Matsumoto-Amano normal form of a single-qubit Clifford+T word, in plain integers.

Same rewriting system as pygridsynth.normal_form (syllables T | HT | SHT followed by a Clifford
E^a X^b S^c w^d, with pygridsynth's tables), applied to the word directly in matrix order,
without building gate objects or exact matrices. Output string format matches
gold_local.ma_normal_form (pygridsynth's to_simple_str: letters H, T, S, X, W)."""

CONJ2 = [(0, 0), (0, 0), (1, 0), (3, 2), (2, 0), (2, 4), (3, 0), (1, 6)]
CONJ3 = [(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (0, 1, 0, 0), (0, 1, 1, 0),
         (0, 1, 2, 0), (0, 1, 3, 0), (1, 0, 0, 0), (2, 0, 3, 6), (1, 1, 2, 2), (2, 1, 3, 6),
         (1, 0, 2, 0), (2, 1, 1, 0), (1, 1, 0, 6), (2, 0, 1, 4), (2, 0, 0, 0), (1, 1, 3, 4),
         (2, 1, 0, 0), (1, 0, 1, 2), (2, 1, 2, 2), (1, 1, 1, 0), (2, 0, 2, 6), (1, 0, 3, 2)]
CINV = [(0, 0, 0, 0), (0, 0, 3, 0), (0, 0, 2, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 1, 1, 6),
        (0, 1, 2, 4), (0, 1, 3, 2), (2, 0, 0, 0), (1, 0, 1, 2), (2, 1, 0, 0), (1, 1, 3, 4),
        (2, 1, 1, 2), (1, 1, 1, 6), (2, 0, 2, 2), (1, 0, 3, 4), (1, 0, 0, 0), (2, 1, 3, 6),
        (1, 1, 2, 2), (2, 0, 3, 6), (1, 0, 2, 0), (2, 1, 1, 6), (1, 1, 0, 2), (2, 0, 1, 6)]
TCONJ = [(0, 0, 0), (0, 1, 7), (1, 3, 3), (1, 2, 0), (2, 0, 5), (2, 1, 4)]   # (axis, c, d)


def _norm(a, b, c, d):
    return (a % 3, b & 1, c & 3, d & 7)


def cmul(x, y):
    a0, b0, c0, d0 = x
    a1, b1, c1, d1 = CONJ3[y[0] << 3 | b0 << 2 | c0]
    c2, d2 = CONJ2[c1 << 1 | y[1]]
    return _norm(a0 + a1, b1 + y[1], c2 + y[2], d2 + d1 + d0 + y[3])


def cinv(x):
    a1, b1, c1, d1 = CINV[x[0] << 3 | x[1] << 2 | x[2]]
    return _norm(a1, b1, c1, d1 - x[3])


C_I = (0, 0, 0, 0)
C_X = (0, 1, 0, 0)
C_H = (1, 0, 1, 5)
C_S = (0, 0, 1, 0)
C_W = (0, 0, 0, 1)
C_SH = cmul(C_S, C_H)
C_HS = cmul(C_H, C_S)
C_SHS = cmul(C_SH, C_S)
LETTER = {"H": C_H, "S": C_S, "X": C_X, "W": C_W}
# letters outside pygridsynth's alphabet, as products of H, S, X, W, T (see gold_local.to_basic)
EXPAND = {"s": "SSS", "t": "TTTTTTT", "Z": "SS", "Y": "WWXSS", "I": ""}


def normal_form(word):
    """(syllables, clifford) for the word read in matrix order (leftmost letter = leftmost factor)."""
    syl = []                     # 1 = T, 2 = HT, 3 = SHT
    c = C_I
    for g in word:
        if g in EXPAND:
            for h in EXPAND[g]:
                syl, c = _push(syl, c, h)
        else:
            syl, c = _push(syl, c, g)
    return syl, c


def _push(syl, c, g):
    if g != "T":
        return syl, cmul(c, LETTER[g])
    axis, c1, d1 = TCONJ[c[0] << 1 | c[1]]
    nc = _norm(0, c[1], c1 + c[2], d1 + c[3])
    if axis == 0:
        if not syl:
            syl.append(1)
            c = nc
        else:
            last = syl.pop()
            c = cmul({1: C_S, 2: C_HS, 3: C_SHS}[last], nc)
    elif axis == 1:
        syl.append(2)
        c = nc
    else:
        syl.append(3)
        c = nc
    return syl, c


def _clifford_str(c):
    if c[0] == 0:
        axis, r = "", c
    elif c[0] == 1:
        axis, r = "H", cmul(cinv(C_H), c)
    else:
        axis, r = "SH", cmul(cinv(C_SH), c)
    return axis + "X" * r[1] + "S" * r[2] + "W" * r[3]


def ma_normal_form(word):
    syl, c = normal_form(word)
    return "".join({1: "T", 2: "HT", 3: "SHT"}[s] for s in syl) + _clifford_str(c)
''',
            'lattice2': r'''"""Faster aligned lattice-point search: integer fixed point (no mpmath), warm-started LLL.

Same contract as lattice.aligned_points. Differences:
  * every high-precision quantity is a Python int at scale 2^PB (sqrt2 handled as a fixed-point
    constant, so the thin cap direction stays exactly inside the sigma1 subspace);
  * Q = Q_base(k, eps) + c * s s^T: Q_base is LLL-reduced once per (k, eps); each target only
    adds the rank-1 term in the warm basis and re-reduces;
  * the inverse basis is tracked through the LLL, so the centre needs no linear solve.
"""

import math
from fractions import Fraction

def norm_sq(x):
    return sum(v * v for v in x)


def bform(x):
    a0, a1, a2, a3, b0, b1, b2, b3 = x
    return a0 * a1 + a1 * a2 + a2 * a3 - a0 * a3 + b0 * b1 + b1 * b2 + b2 * b3 - b0 * b3

PB = 220                                   # fixed-point bits
ONE_FP = 1 << PB
SQ2_FP = math.isqrt(2 << (2 * PB))         # sqrt(2) * 2^PB
R2_FP = SQ2_FP >> 1                        # 1/sqrt(2) * 2^PB
GRAM_BITS = 180                            # target magnitude of the largest Gram entry


def _fp_mul(a, b):
    return (a * b) >> PB


# Exact sigma1 / sigma2 rows over Q(sqrt2): each entry is (rational, coefficient of 1/sqrt2).
def _rows():
    s1 = [[(1, 0), (0, 1), (0, 0), (0, -1)], [(0, 0), (0, 1), (1, 0), (0, 1)]]
    s2 = [[(1, 0), (0, -1), (0, 0), (0, 1)], [(0, 0), (0, -1), (1, 0), (0, -1)]]
    z = [(0, 0)] * 4
    S1 = [s1[0] + z, s1[1] + z, z + s1[0], z + s1[1]]
    S2 = [s2[0] + z, s2[1] + z, z + s2[0], z + s2[1]]
    return S1, S2


def _gram_fp(S):
    """S^T S as fixed-point ints (entries of the form a + b/sqrt2 + c/2)."""
    out = [[0] * 8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            acc = Fraction(0)
            irr = Fraction(0)          # coefficient of 1/sqrt2
            for row in S:
                (a1, b1), (a2, b2) = row[i], row[j]
                acc += a1 * a2 + Fraction(b1 * b2, 2)
                irr += a1 * b2 + a2 * b1
            out[i][j] = (acc.numerator * ONE_FP) // acc.denominator + (irr.numerator * R2_FP) // irr.denominator
    return out


_S1, _S2 = _rows()
_P1 = _gram_fp(_S1)
_P2 = _gram_fp(_S2)


def _isqrt_fp(x_fp):
    """sqrt of a fixed-point number."""
    return math.isqrt(x_fp << PB)


# Search ellipsoid alpha*a + beta*b + gamma*c <= 1 (times bound 3), with a = slab term along the
# target about a centre shifted by delta slab half-widths, b = transverse sigma1 term, c = sigma2
# term. Exact solutions lie on a = ((1 - 2b) - delta)^2, c = 1, b in [0, 1], so any weights with
# alpha*(1 - delta)^2 + gamma <= 1 and alpha*(1 + delta)^2 + beta + gamma <= 1 contain them all.
# "equal" is the original (1/3, 1/3, 1/3, 0); "fitted" minimises the volume (0.348 of "equal").
SHAPES = {"equal": (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3), Fraction(0)),
          "fitted": (Fraction(25, 128), Fraction(15, 32), Fraction(1, 2), Fraction(-3, 5))}
SHAPE = "fitted"


def _scalars(k, eps):
    """Fixed-point R, R_c, and the metric weights wp, wr, cy = wy - wp as exact Fractions."""
    e = Fraction(eps)
    e2 = e * e
    one_m = 1 - e2
    sq_fp = _isqrt_fp((one_m.numerator << PB) // one_m.denominator)       # sqrt(1-eps^2)
    sq = Fraction(sq_fp, ONE_FP)
    R2 = Fraction(2) ** k                                                # R^2
    if k % 2 == 0:
        R = Fraction(2) ** (k // 2)
    else:
        R = Fraction(2) ** (k // 2) * Fraction(SQ2_FP, ONE_FP)
    dy = R * e2 / (2 * (1 + sq))
    alpha, beta, gamma, delta = SHAPES[SHAPE]
    infl = Fraction(_inflation(float(eps), *map(float, SHAPES[SHAPE])))
    wy = 3 * alpha / (dy * dy * infl)
    wp = 3 * beta / (e2 * R2 * infl)
    wr = 3 * gamma / (R2 * infl)
    return R, R - (1 - delta) * dy, wp, wr, wy - wp


def _inflation(eps, alpha, beta, gamma, delta, grid=4000):
    """max over the exact solution curve of alpha*(f(u)-delta)^2 + beta*u + gamma (>= 1 needed).

    On the cap sin(theta) <= eps, a solution with transverse fraction s (u = s^2) has slab
    coordinate f(u) = 1 - 2 g(u)/g(1), g(u) = 1 - sqrt(1 - eps^2 u) (concave, = 1 - 2u + O(eps^2))."""
    def g(u):
        return eps * eps * u / (1 + math.sqrt(1 - eps * eps * u))
    g1 = g(1.0)
    worst = 0.0
    for i in range(grid + 1):
        u = i / grid
        f = 1 - 2 * g(u) / g1
        worst = max(worst, alpha * (f - delta) ** 2 + beta * u + gamma)
    # the grid misses at most the curvature between nodes; pad generously
    return max(1.0, worst) * (1 + 1e-9) + 4.0 / grid ** 2


_BASE_CACHE = {}


def _lll(G, B, Binv, delta=0.99, eta=0.51):
    """L^2-style LLL on exact integer Gram G (rows/cols of the current basis); updates B (rows) and
    Binv (columns) alongside. All in place; returns nothing."""
    d = 8
    r = [[0.0] * d for _ in range(d)]
    mu = [[0.0] * d for _ in range(d)]
    r[0][0] = float(G[0][0])
    k = 1
    guard = 0
    while k < d:
        guard += 1
        if guard > 200000:
            raise RuntimeError("LLL did not converge")
        Gk = G[k]
        for _ in range(400):
            rk = r[k]
            muk = mu[k]
            for j in range(k):
                acc = float(Gk[j])
                muj = mu[j]
                for i in range(j):
                    acc -= muj[i] * rk[i]
                rk[j] = acc
                muk[j] = acc / r[j][j]
            done = True
            for j in range(k):
                if abs(muk[j]) > eta:
                    done = False
                    break
            if done:
                break
            for j in range(k - 1, -1, -1):
                X = round(muk[j])
                if X == 0:
                    continue
                Gj = G[j]
                gkk = Gk[k] - 2 * X * Gk[j] + X * X * Gj[j]
                for i in range(d):
                    if i != k:
                        v = Gk[i] - X * Gj[i]
                        Gk[i] = v
                        G[i][k] = v
                Gk[k] = gkk
                bk, bj = B[k], B[j]
                for i in range(d):
                    bk[i] -= X * bj[i]
                for row in Binv:                 # column j += X * column k
                    row[j] += X * row[k]
                muj = mu[j]
                for i in range(j):
                    muk[i] -= X * muj[i]
                muk[j] -= X
        rkk = float(Gk[k])
        for j in range(k):
            rkk -= mu[k][j] * r[k][j]
        if delta * r[k - 1][k - 1] <= rkk + mu[k][k - 1] ** 2 * r[k - 1][k - 1]:
            r[k][k] = rkk
            k += 1
        else:
            B[k - 1], B[k] = B[k], B[k - 1]
            G[k - 1], G[k] = G[k], G[k - 1]
            for row in G:
                row[k - 1], row[k] = row[k], row[k - 1]
            for row in Binv:
                row[k - 1], row[k] = row[k], row[k - 1]
            k = max(k - 1, 1)
            if k == 1:
                r[0][0] = float(G[0][0])


def _base(k, eps):
    key = (k, eps, SHAPE)
    if key in _BASE_CACHE:
        return _BASE_CACHE[key]
    R, Rc, wp, wr, cy = _scalars(k, eps)
    # choose the Gram scale from the dominant rank-1 weight (|s1e| <= ~1.5)
    s = GRAM_BITS - max(0, int(math.log2(float(cy) * 2.25) if cy > 0 else 0))
    scale = Fraction(2) ** s
    # Q_base * 2^s as ints
    wpS, wrS = wp * scale, wr * scale
    Gb = [[(wpS.numerator * _P1[i][j]) // (wpS.denominator * ONE_FP)
           + (wrS.numerator * _P2[i][j]) // (wrS.denominator * ONE_FP) for j in range(8)] for i in range(8)]
    for i in range(8):                      # symmetrise rounding
        for j in range(i):
            Gb[i][j] = Gb[j][i]
    B0 = [[1 if i == j else 0 for j in range(8)] for i in range(8)]
    B0inv = [[1 if i == j else 0 for j in range(8)] for i in range(8)]
    _lll(Gb, B0, B0inv)
    cyS = cy * scale
    Rc_fp = (Rc.numerator * ONE_FP) // Rc.denominator
    entry = {"B0": B0, "B0inv": B0inv, "G0": Gb, "cy_num": cyS.numerator, "cy_den": cyS.denominator,
             "Rc_fp": Rc_fp, "scale": float(scale), "s": s}
    _BASE_CACHE[key] = entry
    return entry


def _s1e_fp(p):
    """S1^T e0 as fixed-point ints, e0 = p/|p|.

    e0 must be a unit vector to PB bits, not to float64: the slab centre is R_c |e0|^2 along e0,
    and the slab half-width is only ~eps^2 R / 4 (2.5e-19 R at eps 1e-9), so a float64-normalised
    e0 (|e0|^2 = 1 +- 1e-16) misplaces the slab by hundreds of widths at eps 1e-9 and loses
    solutions. The direction itself may carry float64 error (it only tilts the cap by ~1e-16)."""
    v = (p[0].real, p[0].imag, p[1].real, p[1].imag)
    n = math.sqrt(sum(t * t for t in v))
    e = [int((t / n) * ONE_FP) for t in v]
    n_fp = _isqrt_fp(sum(t * t for t in e) >> PB)          # |e| in fixed point, exact to PB bits
    e = [(t << PB) // n_fp for t in e]
    e0r, e0i, e1r, e1i = e
    return [e0r, ((e0r + e0i) * R2_FP) >> PB, e0i, ((e0i - e0r) * R2_FP) >> PB,
            e1r, ((e1r + e1i) * R2_FP) >> PB, e1i, ((e1i - e1r) * R2_FP) >> PB]


def aligned_points_shells(p, k, eps, max_solutions=1, node_budget=200000, bound=3.0):
    """Search shells k and k+1 for target direction p with ONE lattice reduction.

    All metric weights scale as 1/R^2, so Q(k+1) = Q(k)/2: the reduced basis is shared and only
    the centre (times sqrt2) and the norm target change. Returns [(shell, solutions, nodes)]."""
    base = _base(k, eps)
    B0, B0inv, G0 = base["B0"], base["B0inv"], base["G0"]
    s1e = _s1e_fp(p)
    w = [sum(B0[i][c] * s1e[c] for c in range(8)) for i in range(8)]
    cn, cd = base["cy_num"], base["cy_den"]
    den = cd << (2 * PB)
    G = [[G0[i][j] + (cn * w[i] * w[j]) // den for j in range(8)] for i in range(8)]
    for i in range(8):
        for j in range(i):
            G[i][j] = G[j][i]
    Bred = [[1 if i == j else 0 for j in range(8)] for i in range(8)]
    Bredinv = [[1 if i == j else 0 for j in range(8)] for i in range(8)]
    _lll(G, Bred, Bredinv)
    B = [[sum(Bred[i][m] * B0[m][c] for m in range(8)) for c in range(8)] for i in range(8)]
    gscale = base["scale"]
    Gf = [[float(G[i][j]) / gscale for j in range(8)] for i in range(8)]
    out = []
    for shell, gfac, rc_fp in ((k, 1.0, base["Rc_fp"]), (k + 1, 0.5, (base["Rc_fp"] * SQ2_FP) >> PB)):
        xc = [(rc_fp * v) >> (PB + 1) for v in s1e]
        y0 = [sum(B0inv[c][i] * xc[c] for c in range(8)) for i in range(8)]
        zc = [sum(Bredinv[c][i] * y0[c] for c in range(8)) for i in range(8)]
        zint = [(v + (ONE_FP >> 1)) >> PB for v in zc]
        frac = [(zc[i] - (zint[i] << PB)) / ONE_FP for i in range(8)]
        Gs = Gf if gfac == 1.0 else [[v * gfac for v in row] for row in Gf]
        sols, nodes, _ = _se_walk(Gs, B, zint, frac, shell, bound, max_solutions, node_budget)
        out.append((shell, sols, nodes))
        if sols:
            break
    return out


def aligned_points(p, k, eps, max_solutions=1, node_budget=200000, bound=3.0):
    base = _base(k, eps)
    B0, B0inv, G0 = base["B0"], base["B0inv"], base["G0"]
    s1e = _s1e_fp(p)
    # w = B0 s1e (fixed point), rank-1 Gram term in the warm basis
    w = [sum(B0[i][c] * s1e[c] for c in range(8)) for i in range(8)]
    cn, cd = base["cy_num"], base["cy_den"]
    den = cd << (2 * PB)
    G = [[G0[i][j] + (cn * w[i] * w[j]) // den for j in range(8)] for i in range(8)]
    for i in range(8):
        for j in range(i):
            G[i][j] = G[j][i]
    Bred = [[1 if i == j else 0 for j in range(8)] for i in range(8)]
    Bredinv = [[1 if i == j else 0 for j in range(8)] for i in range(8)]
    _lll(G, Bred, Bredinv)
    # full basis B = Bred * B0 (rows), inverse B^{-1} = B0inv * Bredinv
    B = [[sum(Bred[i][m] * B0[m][c] for m in range(8)) for c in range(8)] for i in range(8)]
    # centre x_c = Rc * s1e / 2 (fixed point); z_c = (B^{-1})^T x_c = Bredinv^T (B0inv^T x_c)
    Rc_fp = base["Rc_fp"]
    xc = [(Rc_fp * v) >> (PB + 1) for v in s1e]                 # scale 2^PB
    y0 = [sum(B0inv[c][i] * xc[c] for c in range(8)) for i in range(8)]
    zc = [sum(Bredinv[c][i] * y0[c] for c in range(8)) for i in range(8)]
    zint = [(v + (ONE_FP >> 1)) >> PB for v in zc]
    frac = [(zc[i] - (zint[i] << PB)) / ONE_FP for i in range(8)]
    gscale = base["scale"]
    Gf = [[float(G[i][j]) / gscale for j in range(8)] for i in range(8)]
    return _se_walk(Gf, B, zint, frac, k, bound, max_solutions, node_budget)


def _se_walk(Gf, B, zint, frac, k, bound, max_solutions, node_budget):
    """Schnorr-Euchner with incremental centres (Q(o - f) <= bound), exact leaf checks."""
    d = 8
    # Cholesky G = L L^T; q(y) = sum_i r_ii (y_i + sum_{j>i} m_ji y_j)^2 with r_ii = L_ii^2, m_ji = L_ji / L_ii
    L = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            s = Gf[i][j]
            for m in range(j):
                s -= L[i][m] * L[j][m]
            if i == j:
                if s <= 0:
                    raise ValueError("reduced Gram not positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    rii = [L[i][i] ** 2 for i in range(d)]
    m = [[L[j][i] / L[i][i] if j > i else 0.0 for j in range(d)] for i in range(d)]  # m[i][j], j>i
    target = 1 << k
    sols = []
    nodes = 0
    o = [0] * d
    c = [0.0] * d
    dist = [0.0] * (d + 1)
    step = [0] * d
    base_o = [0] * d
    dirn = [1] * d

    def centre(i):
        s = 0.0
        mi = m[i]
        for j in range(i + 1, d):
            s += mi[j] * (o[j] - frac[j])
        return frac[i] - s

    # xpart[i] = sum_{t > i} B[t] * (zint[t] + o[t]), kept incrementally (exact ints)
    xpart = [[0] * d for _ in range(d + 1)]
    i = d - 1
    c[i] = centre(i)
    base_o[i] = o[i] = round(c[i])
    step[i] = 0
    dirn[i] = 1 if c[i] >= o[i] else -1
    budget_hit = False
    while True:
        nodes += 1
        if nodes > node_budget:
            budget_hit = True
            break
        diff = o[i] - c[i]
        di = dist[i + 1] + rii[i] * diff * diff
        if di <= bound:
            zi = zint[i] + o[i]
            Bi = B[i]
            xp = xpart[i + 1]
            if i == 0:
                x = [xp[col] + Bi[col] * zi for col in range(d)]
                if sum(v * v for v in x) == target and bform(x) == 0:
                    sols.append(tuple(x))
                    if len(sols) >= max_solutions:
                        break
            else:
                xpart[i] = [xp[col] + Bi[col] * zi for col in range(d)]
                dist[i] = di
                i -= 1
                c[i] = centre(i)
                base_o[i] = o[i] = round(c[i])
                step[i] = 0
                dirn[i] = 1 if c[i] >= o[i] else -1
                continue
        else:
            i += 1
            if i >= d:
                break
        step[i] += 1
        st = step[i]
        off = (st + 1) // 2 if st % 2 else -(st // 2)
        o[i] = base_o[i] + dirn[i] * off
    return sols, nodes, budget_hit
''',
            'batch': r'''"""Batched lattice search: lattice2.aligned_points_shells for many targets at once, in numpy.

Same lattice problem, region and exact checks as lattice2; only the evaluation order changes.
  * Reduction: all targets share the base basis B0 (Gram G0 = Lc Lc^T) and differ in the rank-1
    term cy (w^T z)^2. Each basis vector z is embedded explicitly as (z Lc, sqrt(cy) w^T z) in R^9
    and reduced with a lockstep LLL on batched QR (no Gram squaring). w^T z is recomputed
    exactly every round from w held as three balanced 30-bit int64 limbs (90 bits), so the
    float error never accumulates in the one direction that needs ~70 bits.
  * Enumeration: breadth-first over the 8 levels for every (target, shell) at once; only nodes
    inside the ellipsoid are generated.
  * Leaves: x = x_centre + B_full^T o in int64; exact checks sum x^2 == 2^shell and B(x) == 0.
The basis and Gram used by the enumeration only need float accuracy in reduced coordinates,
where everything is O(1); the slab offset sits in the exact centre (Python ints)."""

import math

import numpy as np

import _qhg_lattice2 as L2

PB = L2.PB
LIMB = 30
W_BITS = 90                      # fixed-point bits kept for w (>= ~70 needed at eps 1e-9)
_BASEF = {}


def _base_float(k, eps):
    key = (k, eps, L2.SHAPE)
    if key not in _BASEF:
        b = L2._base(k, eps)
        G0f = np.array(b["G0"], dtype=float) / b["scale"]
        Lc = np.linalg.cholesky(G0f)
        cy = b["cy_num"] / b["cy_den"] / b["scale"]
        B0 = np.array(b["B0"], dtype=np.int64)
        _BASEF[key] = {"b": b, "Lc": Lc, "sqrt_cy": math.sqrt(cy), "B0": B0,
                       "B0inv_obj": np.array(b["B0inv"], dtype=object)}
    return _BASEF[key]


def _balanced_limbs(v):
    """Python int -> three int64 limbs (l0, l1, l2), v = l0 + l1 2^30 + l2 2^60, |l0|,|l1| <= 2^29."""
    out = []
    for _ in range(2):
        lo = ((v + (1 << (LIMB - 1))) & ((1 << LIMB) - 1)) - (1 << (LIMB - 1))
        out.append(lo)
        v = (v - lo) >> LIMB
    out.append(v)
    return out


def _normalise_limbs(l0, l1, l2):
    c = (l0 + (1 << (LIMB - 1))) >> LIMB
    l0 = l0 - (c << LIMB)
    l1 = l1 + c
    c = (l1 + (1 << (LIMB - 1))) >> LIMB
    l1 = l1 - (c << LIMB)
    l2 = l2 + c
    return l0, l1, l2


def _wz(B, WL):
    """w^T z for every basis row, exact in limbs, returned as float (true units)."""
    l0 = np.einsum("nij,nj->ni", B, WL[:, :, 0])
    l1 = np.einsum("nij,nj->ni", B, WL[:, :, 1])
    l2 = np.einsum("nij,nj->ni", B, WL[:, :, 2])
    l0, l1, l2 = _normalise_limbs(l0, l1, l2)
    s = 2.0 ** -W_BITS
    return (l2.astype(float) * 2.0 ** (2 * LIMB) + l1.astype(float) * 2.0 ** LIMB + l0.astype(float)) * s


def _embed(B, WL, Lc, sqrt_cy):
    V = np.empty(B.shape[:2] + (9,))
    V[:, :, :8] = B.astype(float) @ Lc
    V[:, :, 8] = sqrt_cy * _wz(B, WL)
    return V


def _size_reduce(B, R):
    """Size reduction along subdiagonals d = 1..7: for every i at once, subtract round(mu_{i,i-d})
    times vector i-d. GS vectors are invariant and mu_{i,j} is final after step i-j, so this
    gives the same guarantee (|mu| <= 1/2) as the usual sequential order."""
    for d in range(1, 8):
        X = np.rint(np.diagonal(R, d, 1, 2) / np.diagonal(R, 0, 1, 2)[:, :8 - d])   # (n, 8-d)
        if not X.any():
            continue
        B[:, d:, :] -= X.astype(np.int64)[:, :, None] * B[:, :-d, :]
        R[:, :, d:] -= R[:, :, :-d] * X[:, None, :]


def _size_reduce_adjacent(B, R):
    """The d = 1 step only: makes |mu_{k,k-1}| <= 1/2 so the next Lovasz test is valid."""
    X = np.rint(np.diagonal(R, 1, 1, 2) / np.diagonal(R, 0, 1, 2)[:, :7])
    if X.any():
        B[:, 1:, :] -= X.astype(np.int64)[:, :, None] * B[:, :-1, :]
        R[:, :, 1:] -= R[:, :, :-1] * X[:, None, :]


def _swap_pass(B, R, parity, delta):
    """Lovasz test on pairs (k-1, k) of one parity; swap and restore R by a Givens rotation."""
    swapped = np.zeros(B.shape[0], dtype=bool)
    for kk in range(1 + parity, 8, 2):
        d0 = R[:, kk - 1, kk - 1] ** 2
        d1 = R[:, kk, kk] ** 2
        sw = delta * d0 > d1 + R[:, kk - 1, kk] ** 2
        if not sw.any():
            continue
        swapped |= sw
        idx = np.nonzero(sw)[0]
        t = B[idx, kk - 1, :].copy()
        B[idx, kk - 1, :] = B[idx, kk, :]
        B[idx, kk, :] = t
        Rs = R[idx]
        t = Rs[:, :, kk - 1].copy()
        Rs[:, :, kk - 1] = Rs[:, :, kk]
        Rs[:, :, kk] = t
        a, b = Rs[:, kk - 1, kk - 1], Rs[:, kk, kk - 1]
        r = np.hypot(a, b)
        c, s_ = a / r, b / r
        r0, r1 = Rs[:, kk - 1, :].copy(), Rs[:, kk, :].copy()
        Rs[:, kk - 1, :] = c[:, None] * r0 + s_[:, None] * r1
        Rs[:, kk, :] = -s_[:, None] * r0 + c[:, None] * r1
        Rs[:, kk, kk - 1] = 0.0
        R[idx] = Rs
    return swapped


def _lll_stage(B, WL, Lc, sqrt_cy, todo, delta, max_rounds):
    """Lockstep LLL on the members in todo. R is updated by Givens rotations between exact
    refreshes; only members whose R changed are size-reduced again; a member leaves after two
    rounds without swaps on a freshly recomputed R."""
    active = todo.copy()
    Ba = B[active]
    R = np.linalg.qr(_embed(Ba, WL[active], Lc, sqrt_cy).transpose(0, 2, 1), mode="r")
    calm = np.zeros(active.size, dtype=np.int64)
    fresh = np.ones(active.size, dtype=bool)        # R recomputed exactly since the last swap
    dirty = np.ones(active.size, dtype=bool)        # needs size reduction
    rounds = 0
    while active.size and rounds < max_rounds:
        rounds += 1
        di = np.nonzero(dirty)[0]
        if di.size == active.size:
            _size_reduce(Ba, R)
        elif di.size:
            Bs, Rs = Ba[di], R[di]
            _size_reduce(Bs, Rs)
            Ba[di], R[di] = Bs, Rs
        sw = _swap_pass(Ba, R, 0, delta)
        si = np.nonzero(sw)[0]
        if si.size:
            Bs, Rs = Ba[si], R[si]
            _size_reduce_adjacent(Bs, Rs)
            Ba[si], R[si] = Bs, Rs
        sw |= _swap_pass(Ba, R, 1, delta)
        dirty = sw.copy()
        calm = np.where(sw, 0, calm + 1)
        fresh &= ~sw
        blown = np.abs(Ba).reshape(active.size, -1).max(axis=1) > BLOWUP
        done = (calm >= 2) & fresh
        refresh = (calm >= 2) & ~fresh & ~blown
        if refresh.any():
            ri = np.nonzero(refresh)[0]
            R[ri] = np.linalg.qr(_embed(Ba[ri], WL[active[ri]], Lc, sqrt_cy).transpose(0, 2, 1), mode="r")
            fresh[ri] = True
            calm[ri] = 0
            dirty[ri] = True
        leave = done | blown
        if leave.any():
            B[active[leave]] = Ba[leave]
            stay = ~leave
            active, Ba, R = active[stay], Ba[stay], R[stay]
            calm, fresh, dirty = calm[stay], fresh[stay], dirty[stay]
    if active.size:
        B[active] = Ba
    return rounds, active


def _inverse_mod_2_64(B):
    """Exact inverse of unimodular int64 matrices by Gauss-Jordan elimination modulo 2^64.

    det = +-1 is odd, so B is invertible mod 2 and every column has an odd pivot; odd numbers are
    units mod 2^64. All arithmetic wraps in int64. The true inverse has entries bounded by the
    7x7 minors of B (Hadamard), far below 2^63 here, so the residue is the exact inverse; callers
    still check B @ Binv == I."""
    N = B.shape[0]
    A = B.copy()
    X = np.tile(np.eye(8, dtype=np.int64), (N, 1, 1))
    n_idx = np.arange(N)
    with np.errstate(over="ignore"):
        for col in range(8):
            odd = (A[:, col:, col] & 1).astype(bool)
            piv = col + np.argmax(odd, axis=1)
            for M in (A, X):
                t = M[n_idx, piv, :].copy()
                M[n_idx, piv, :] = M[:, col, :]
                M[:, col, :] = t
            p = A[:, col, col]
            inv = p.copy()
            for _ in range(6):                      # Newton: inv <- inv (2 - p inv), mod 2^64
                inv = inv * (2 - p * inv)
            A[:, col, :] *= inv[:, None]
            X[:, col, :] *= inv[:, None]
            F = A[:, :, col].copy()
            F[:, col] = 0
            A -= F[:, :, None] * A[:, col, None, :]
            X -= F[:, :, None] * X[:, col, None, :]
    return X


BLOWUP = 1 << 40
CHUNK = 1 << 16
STAGE_BITS = 20


def batched_lll(WL, Lc, sqrt_cy, delta=0.99, max_rounds=400):
    """Precision ladder: reduce with the rank-1 weight scaled down so that the target direction is
    at most 2^STAGE_BITS larger than the base directions, then raise it 2^STAGE_BITS at a time.
    Each stage starts nearly reduced and stays inside float64 accuracy.

    Returns B, Binv (int64), total rounds, and the members that did not converge, blew up or
    lack an exact inverse (callers fall back to lattice2 for them)."""
    N = WL.shape[0]
    B = np.tile(np.eye(8, dtype=np.int64), (N, 1, 1))
    wmax = float(np.abs(_wz(B, WL)).max()) if N else 0.0
    base = float(np.abs(np.diag(Lc)).min())
    ratio = sqrt_cy * max(wmax, 1e-300) / base
    S = max(0, int(math.ceil(math.log2(ratio))) - STAGE_BITS) if ratio > 1 else 0
    total, bad = 0, np.zeros(N, dtype=bool)
    shifts = list(range(S, -1, -STAGE_BITS))
    if shifts[-1] != 0:
        shifts.append(0)
    for sh in shifts:
        todo = np.nonzero(~bad)[0]
        r, left = _lll_stage(B, WL, Lc, sqrt_cy * 2.0 ** -sh, todo, delta, max_rounds)
        total += r
        bad[left] = True
    bad |= np.abs(B).reshape(N, -1).max(axis=1) > BLOWUP
    Binv = _inverse_mod_2_64(B)
    with np.errstate(over="ignore"):
        ok = ~(np.eye(8, dtype=np.int64) - B @ Binv).reshape(N, -1).any(axis=1)
    bad |= ~ok
    bad |= np.abs(Binv).reshape(N, -1).max(axis=1) > (1 << 40)
    return B, Binv, total, np.nonzero(bad)[0]


def _enumerate(Gs, frac, bound):
    """All integer o with (o - frac)^T G (o - frac) <= bound, for every problem p (BFS).

    Gs: (P, 8, 8) Gram matrices, frac: (P, 8). Returns (problem index, o) of every point.
    Each level keeps only what the levels below still need (the centre accumulators for those
    levels) plus a parent pointer; the offsets are rebuilt for the surviving leaves at the end."""
    P = Gs.shape[0]
    Lch = np.linalg.cholesky(Gs)                         # G = L L^T
    diag = np.einsum("pii->pi", Lch)
    rii = diag ** 2
    M = Lch / diag[:, None, :]                           # M[p, j, i] = L_ji / L_ii (j > i used)
    pid = np.arange(P)
    dist = np.zeros(P)
    acc = np.zeros((P, 8))
    parents, values = [], []
    for i in range(7, -1, -1):
        c = frac[pid, i] - acc[:, i]
        rem = bound - dist
        h = np.sqrt(np.maximum(rem, 0.0) / rii[pid, i])
        lo = np.ceil(c - h).astype(np.int64)
        cnt = np.floor(c + h).astype(np.int64) - lo + 1
        np.maximum(cnt, 0, out=cnt)
        cnt[rem < 0] = 0
        tot = int(cnt.sum())
        if tot == 0:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 8), dtype=np.int64)
        rep = np.repeat(np.arange(pid.size), cnt)
        o_i = lo[rep] + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        diff = o_i - c[rep]
        pid_r = pid[rep]
        dist_r = dist[rep] + rii[pid_r, i] * diff * diff
        keep = dist_r <= bound
        rep, o_i, pid, dist = rep[keep], o_i[keep], pid_r[keep], dist_r[keep]
        parents.append(rep)
        values.append(o_i)
        if i:
            acc = acc[rep, :i] + M[pid, i, :i] * (o_i - frac[pid, i])[:, None]
    # rebuild offsets from the leaves upwards
    off = np.empty((pid.size, 8), dtype=np.int64)
    idx = np.arange(pid.size)
    for lvl, i in enumerate(range(0, 8)):
        k = 7 - i                                        # parents/values were appended for i = 7..0
        off[:, i] = values[k][idx]
        idx = parents[k][idx]
    return pid, off


def _wrap64(a):
    """Object array of Python ints -> int64 modulo 2^64. Centre coordinates in a skewed reduced
    basis can exceed 2^63; every later use (parity, offsets, x = Bfull^T z) is exact modulo 2^64
    and the final points are small, so wrapping keeps the results exact."""
    return ((a + (1 << 63)) % (1 << 64) - (1 << 63)).astype(np.int64)


def _bform_rows(X):
    a0, a1, a2, a3, b0, b1, b2, b3 = (X[:, t] for t in range(8))
    return a0 * a1 + a1 * a2 + a2 * a3 - a0 * a3 + b0 * b1 + b1 * b2 + b2 * b3 - b0 * b3


def search(cols, k, eps, bound=3.0):
    """For each target column (complex pair, any scale), the exact solutions on shells k, k+1.

    Returns (sols, stats): sols[t] = [(shell, x), ...] in the same sense as
    lattice2.aligned_points_shells (all solutions inside the region, no max_solutions cut)."""
    bf = _base_float(k, eps)
    b, Lc, sqrt_cy = bf["b"], bf["Lc"], bf["sqrt_cy"]
    B0, B0inv = b["B0"], b["B0inv"]
    N = len(cols)
    B0a = bf["B0"]
    # per target, exact (object arrays of Python ints): s1e (PB bits, exactly unit) and
    # u = B0inv^T s1e (centre direction); w = B0 s1e is formed in int64 limbs from s1e's limbs
    S = np.array([L2._s1e_fp(p) for p in cols], dtype=object)              # (N, 8)
    U = S @ bf["B0inv_obj"]                                                  # (N, 8), = B0inv^T s1e
    v = S >> (PB - W_BITS)
    half = 1 << (LIMB - 1)
    mask = (1 << LIMB) - 1
    l0 = ((v + half) & mask) - half
    v = (v - l0) >> LIMB
    l1 = ((v + half) & mask) - half
    l2 = (v - l1) >> LIMB
    SL = np.stack([l0.astype(np.int64), l1.astype(np.int64), l2.astype(np.int64)], axis=2)
    WL = np.einsum("ij,njl->nil", B0a, SL)
    l0, l1, l2 = _normalise_limbs(WL[:, :, 0], WL[:, :, 1], WL[:, :, 2])
    WL = np.stack([l0, l1, l2], axis=2)
    rc = (b["Rc_fp"], (b["Rc_fp"] * L2.SQ2_FP) >> PB)
    Bred, Bredinv, rounds, stuck = batched_lll(WL, Lc, sqrt_cy)
    good = np.ones(N, dtype=bool)
    good[stuck] = False
    gidx = np.nonzero(good)[0]
    sols = [[] for _ in range(N)]
    pid = np.zeros(0, dtype=np.int64)
    if gidx.size:
        Bg, Big, WLg = Bred[gidx], Bredinv[gidx], WL[gidx]
        V = _embed(Bg, WLg, Lc, sqrt_cy)
        G = V @ V.transpose(0, 2, 1)                       # reduced Gram, true units
        Bfull = np.einsum("nij,jk->nik", Bg, B0a)
        one = 1 << PB
        Vv = np.matmul(U[gidx][:, None, :], Big.astype(object))[:, 0, :]      # Bredinv^T u
        fracs = np.empty((2 * gidx.size, 8))
        zints = np.empty((2 * gidx.size, 8), dtype=np.int64)
        for sh in range(2):
            zc = (Vv * rc[sh]) >> (PB + 1)                                     # = Bredinv^T y0
            zint = (zc + (one >> 1)) >> PB
            zints[sh::2] = _wrap64(zint)
            fracs[sh::2] = ((zc - (zint << PB)) / one).astype(float)
        # x_centre = Bfull^T zint: the true result is small, so int64 wrap-around gives it exactly
        with np.errstate(over="ignore"):
            xc_arr = np.einsum("ni,nij->nj", zints, np.repeat(Bfull, 2, axis=0))
        Gs = np.repeat(G, 2, axis=0)
        Gs[1::2] *= 0.5
        pid, off = _enumerate(Gs, fracs, bound)
        for a in range(0, pid.size, CHUNK):
            pc, oc = pid[a:a + CHUNK], off[a:a + CHUNK]
            q = pc // 2
            X = xc_arr[pc] + np.einsum("ni,nij->nj", oc, Bfull[q])
            shell = k + (pc % 2)
            ok = ((X * X).sum(axis=1) == (np.int64(1) << shell)) & (_bform_rows(X) == 0)
            for r in np.nonzero(ok)[0]:
                sols[gidx[q[r]]].append((int(shell[r]), tuple(int(v) for v in X[r])))
    for t in stuck:                                     # rare: exact scalar search, both shells
        for shell in (k, k + 1):                        # (aligned_points_shells stops after a hit on k)
            xs = L2.aligned_points(cols[t], shell, eps, max_solutions=10 ** 6, node_budget=10 ** 8,
                                   bound=bound)[0]
            sols[t] += [(shell, x) for x in xs]
    stats = {"lll_rounds": rounds, "lll_stuck": int(stuck.size), "nodes_leaf": int(pid.size)}
    return sols, stats
''',
            'batch_coset': r'''"""Coset-shared batched search: one lattice reduction per base word for its three coset prefixes.

The level-t' prefixes come in coset triples B, B*H, B*SH (synth._coset_prefixes). With the SU(2)
representatives H' = w^2 H and (SH)' = w S H, the maps M = sqrt2 * C' on (u, v) are integral
8x8 matrices of index 16 with M M^dag = 2: an inner point x of sibling B*C on shell j maps to
y = M x on shell j + 1, and the sibling's search region maps exactly onto B's region (cap,
transverse ball and sigma2 ball all scale by sqrt2; the cap direction maps to +-B's). So:
  * sibling B*C on shell k  = B on shell k + 1 restricted to M Z^8  (already inside B's own
    shell-(k+1) search: nothing to do);
  * sibling B*C on shell k+1 = B on shell k + 2 restricted to M Z^8.
In B's reduced coordinates z the sublattice contains 2 Z^8, so it is {z : z mod 2 in S} for a
rank-4 subspace S of GF(2)^8; its basis (four 0/1 vectors and four 2 e_j) is reduced in B's
reduced (well-conditioned) coordinates is enumerated directly (no reduction needed). One full LLL per
base word instead of three; the solutions and their order are the same as searching the three
prefixes separately (the final unitary is U = B * Y, Y the SU(2) matrix of y)."""

import numpy as np

import _qhg_batch as BT
import _qhg_lattice2 as L2
from _qhg_zomega import zmul, zpow_omega

PB = BT.PB


def _enum_plain(Gs, fr, bound, Bmat, ycent, shell):
    return BT._enumerate(Gs, fr, bound)


ENUM = _enum_plain          # swap for enum_prune.enumerate_pruned (same interface + extra data)


def _mulmat(alpha):
    """4x4 integer matrix of multiplication by alpha in Z[w] (coordinates of 1, w, w^2, w^3)."""
    return np.array([zmul(alpha, tuple(int(i == j) for i in range(4))) for j in range(4)], dtype=np.int64).T


def _blockmat(m):
    return np.vstack([np.hstack([_mulmat(m[0][0]), _mulmat(m[0][1])]),
                      np.hstack([_mulmat(m[1][0]), _mulmat(m[1][1])])])


def _neg(a):
    return tuple(-t for t in a)


W1, W2, W3 = zpow_omega(1), zpow_omega(2), zpow_omega(3)
M_SIB = (_blockmat(((W2, W2), (W2, _neg(W2)))),          # sqrt2 * w^2 H
         _blockmat(((W1, W1), (W3, _neg(W3)))))          # sqrt2 * w S H   (w i = w^3)


_OM = np.exp(1j * np.pi / 4)
_H2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
C_SIB = (_OM ** 2 * _H2, _OM * np.diag([1, 1j]) @ _H2)             # SU(2) forms of H, SH


def _fallback(col, k, eps, plan):
    """Exact per-prefix searches for one base whose batched reduction failed (never observed;
    kept so a failure cannot crash a run). plan: [(out, sib, M, C', inner shells kept)], where the
    prefix is B C (C' its SU(2) form, M = sqrt2^lde C' on (u, v)); its target is the first column of
    C'^dag R_B, solved by batch.search (which has its own exact scalar fallback), and every inner
    point x is reported as y = M x in B's coordinates, so U = B SU2(y) as in the batched paths."""
    c0, c1 = complex(col[0]), complex(col[1])
    nrm = np.sqrt(abs(c0) ** 2 + abs(c1) ** 2)
    RB = np.array([[c0, -np.conj(c1)], [c1, np.conj(c0)]]) / nrm
    out = []
    for dest, sib, M, Cp, keep in plan:
        cc = (np.conj(Cp).T @ RB)[:, 0]
        sols, _ = BT.search([(complex(cc[0]), complex(cc[1]))], k, eps)
        lde = 0 if M is None else int(round(np.log2(abs(np.linalg.det(M.astype(float)))) / 4))
        for shell, x in sols[0]:
            if shell - k not in keep:
                continue
            y = x if M is None else tuple(int(t) for t in (M @ np.array(x, dtype=np.int64)))
            out.append((dest, (sib, shell + lde, y)))
    return out


def _rref_gf2(A):
    """Row-reduce batched 8x8 GF(2) matrices; returns (R, pivots (n, 8) with -1 padding, rank)."""
    M = (A & 1).astype(np.uint8).copy()
    n = M.shape[0]
    r = np.zeros(n, dtype=np.int64)
    piv = np.full((n, 8), -1, dtype=np.int64)
    rows = np.arange(8)
    for c in range(8):
        cand = (M[:, :, c] == 1) & (rows[None, :] >= r[:, None])
        has = cand.any(axis=1)
        if not has.any():
            continue
        idx = np.nonzero(has)[0]
        prow = np.argmax(cand[idx], axis=1)
        rr = r[idx]
        a, b = M[idx, rr].copy(), M[idx, prow].copy()
        M[idx, prow] = a
        M[idx, rr] = b
        prow_vec = M[idx, rr]                                  # pivot rows (k, 8)
        col = M[idx, :, c].copy()
        col[np.arange(idx.size), rr] = 0
        M[idx] ^= col[:, :, None] & prow_vec[:, None, :]
        piv[idx, rr] = c
        r[idx] += 1
    return M, piv, r


def _lll_fixed(K, Lr, delta=0.99, max_rounds=200):
    """Lockstep LLL of integer bases K (rows) in the metric G = Lr Lr^T (well-conditioned)."""
    Ka = K.copy()
    n = Ka.shape[0]
    embed = lambda B, L: np.einsum("nij,njk->nik", B.astype(float), L)
    R = np.linalg.qr(embed(Ka, Lr).transpose(0, 2, 1), mode="r")
    active = np.arange(n)
    calm = np.zeros(n, dtype=np.int64)
    fresh = np.ones(n, dtype=bool)
    out = Ka
    Ba, La = Ka.copy(), Lr
    rounds = 0
    while active.size and rounds < max_rounds:
        rounds += 1
        BT._size_reduce(Ba, R)
        sw = BT._swap_pass(Ba, R, 0, delta)
        si = np.nonzero(sw)[0]
        if si.size:
            Bs, Rs = Ba[si], R[si]
            BT._size_reduce_adjacent(Bs, Rs)
            Ba[si], R[si] = Bs, Rs
        sw |= BT._swap_pass(Ba, R, 1, delta)
        calm = np.where(sw, 0, calm + 1)
        fresh &= ~sw
        done = (calm >= 2) & fresh
        refresh = (calm >= 2) & ~fresh
        if refresh.any():
            ri = np.nonzero(refresh)[0]
            R[ri] = np.linalg.qr(embed(Ba[ri], La[ri]).transpose(0, 2, 1), mode="r")
            fresh[ri] = True
            calm[ri] = 0
        if done.any():
            out[active[done]] = Ba[done]
            stay = ~done
            active, Ba, R, La = active[stay], Ba[stay], R[stay], La[stay]
            calm, fresh = calm[stay], fresh[stay]
    if active.size:
        out[active] = Ba
    return out, rounds


def search(base_cols, k, eps, bound=3.0):
    """For each base column (target of B), solutions for its three coset prefixes:
    sols[g] = [(sib, shell, y)], sib 0 = B (shells k, k+1), 1 = B*H, 2 = B*SH (shell k+2 of B,
    in the sublattice). Final unitary for any entry: B * SU2(y on that shell)."""
    bf = BT._base_float(k, eps)
    b, Lc, sqrt_cy = bf["b"], bf["Lc"], bf["sqrt_cy"]
    N = len(base_cols)
    B0a = bf["B0"]
    S = np.array([L2._s1e_fp(p) for p in base_cols], dtype=object)
    U = S @ bf["B0inv_obj"]
    v = S >> (PB - BT.W_BITS)
    half, mask = 1 << (BT.LIMB - 1), (1 << BT.LIMB) - 1
    l0 = ((v + half) & mask) - half
    v = (v - l0) >> BT.LIMB
    l1 = ((v + half) & mask) - half
    l2 = (v - l1) >> BT.LIMB
    SL = np.stack([l0.astype(np.int64), l1.astype(np.int64), l2.astype(np.int64)], axis=2)
    WL = np.einsum("ij,njl->nil", B0a, SL)
    l0, l1, l2 = BT._normalise_limbs(WL[:, :, 0], WL[:, :, 1], WL[:, :, 2])
    WL = np.stack([l0, l1, l2], axis=2)
    rc = (b["Rc_fp"], (b["Rc_fp"] * L2.SQ2_FP) >> PB, b["Rc_fp"] << 1)       # shells k, k+1, k+2
    Bred, Bredinv, rounds, stuck = BT.batched_lll(WL, Lc, sqrt_cy)
    good = np.ones(N, dtype=bool)
    good[stuck] = False
    gidx = np.nonzero(good)[0]
    sols = [[] for _ in range(N)]
    stats = {"lll_rounds": rounds, "lll_stuck": int(stuck.size), "nodes_leaf": 0, "sub_rounds": 0}
    if gidx.size:
        n = gidx.size
        Bg, Big, WLg = Bred[gidx], Bredinv[gidx], WL[gidx]
        V = BT._embed(Bg, WLg, Lc, sqrt_cy)
        G = V @ V.transpose(0, 2, 1)                       # reduced Gram (shell k), true units
        Lr = np.linalg.cholesky(G)
        Bfull = np.einsum("nij,jk->nik", Bg, B0a)
        one = 1 << PB
        Vv = np.matmul(U[gidx][:, None, :], Big.astype(object))[:, 0, :]      # Bredinv^T u
        zcs = [(Vv * r) >> (PB + 1) for r in rc]
        zint = [(z + (one >> 1)) >> PB for z in zcs]
        frac = [((z - (zi << PB)) / one).astype(float) for z, zi in zip(zcs, zint)]
        zint = [BT._wrap64(zi) for zi in zint]
        # problems: 4 per base -> (Gram, frac, centre point, row matrix, shell)
        P = 4 * n
        Gs = np.empty((P, 8, 8))
        fr = np.empty((P, 8))
        ycent = np.empty((P, 8), dtype=np.int64)
        Bmat = np.empty((P, 8, 8), dtype=np.int64)
        shell = np.empty(P, dtype=np.int64)
        with np.errstate(over="ignore"):
            for s in range(2):                               # sibling 0 on shells k, k+1
                Gs[s::4] = G * (0.5 ** s)
                fr[s::4] = frac[s]
                ycent[s::4] = np.einsum("ni,nij->nj", zint[s], Bfull)
                Bmat[s::4] = Bfull
                shell[s::4] = k + s
            # Bfull^-T mod 2 = Bredinv^T B0inv^T mod 2
            B0inv2 = (np.array(b["B0inv"], dtype=np.int64) & 1)
            Tinv = (np.einsum("nji,kj->nik", Big & 1, B0inv2) & 1)             # (Bfull^-T) mod 2
            Ks, R2s, pivs = [], [], []
            for M in M_SIB:
                A = (Tinv @ (M & 1)) & 1                     # columns span S (in z mod 2)
                R2, piv, rank = _rref_gf2(A.transpose(0, 2, 1))
                if (rank != 4).any():
                    raise RuntimeError("sublattice rank != 4")
                K = np.zeros((n, 8, 8), dtype=np.int64)
                K[:, :4] = R2[:, :4].astype(np.int64)        # echelon rows (0/1)
                pivmask = np.zeros((n, 8), dtype=bool)
                pivmask[np.arange(n)[:, None], piv[:, :4]] = True
                nonpiv = np.argsort(pivmask, axis=1, kind="stable")[:, :4]       # 4 non-pivot columns
                K[np.arange(n)[:, None], 4 + np.arange(4)[None, :], nonpiv] = 2
                Ks.append(K)
                R2s.append(R2)
                pivs.append(piv)
            # The echelon basis (entries <= 2, in B's reduced, well-conditioned coordinates) is
            # enumerated directly: reducing it costs more than the ~1/3 extra internal BFS nodes
            # it saves (leaf counts are basis-independent; the leaf level stays the largest).
            for si in range(2):
                Kred, R2, piv = Ks[si], R2s[si], pivs[si]
                # centre: z0 in the sublattice near zint (shell k+2), offset d small
                r = (zint[2] & 1).astype(np.uint8)
                for i in range(4):
                    p = piv[:, i]
                    flip = r[np.arange(n), p] == 1
                    r[flip] ^= R2[flip, i]
                z0 = zint[2] - r.astype(np.int64)
                d = frac[2] + r
                oc = np.linalg.solve(Kred.transpose(0, 2, 1).astype(float), d[:, :, None])[:, :, 0]
                oi = np.rint(oc).astype(np.int64)
                zz = z0 + np.einsum("ni,nij->nj", oi, Kred)
                q = 2 + si
                E = np.einsum("nij,njk->nik", Kred.astype(float), Lr)
                Gs[q::4] = (E @ E.transpose(0, 2, 1)) * 0.25
                fr[q::4] = oc - oi
                ycent[q::4] = np.einsum("ni,nij->nj", zz, Bfull)
                Bmat[q::4] = Kred @ Bfull
                shell[q::4] = k + 2
        pid, off = ENUM(Gs, fr, bound, Bmat, ycent, shell)
        stats["nodes_leaf"] = int(pid.size)
        for a in range(0, pid.size, BT.CHUNK):
            pc, oc_ = pid[a:a + BT.CHUNK], off[a:a + BT.CHUNK]
            X = ycent[pc] + np.einsum("ni,nij->nj", oc_, Bmat[pc])
            sh = shell[pc]
            ok = ((X * X).sum(axis=1) == (np.int64(1) << sh)) & (BT._bform_rows(X) == 0)
            for r_ in np.nonzero(ok)[0]:
                g = gidx[pc[r_] // 4]
                q = int(pc[r_] % 4)
                sols[g].append((0 if q < 2 else q - 1, int(sh[r_]), tuple(int(t) for t in X[r_])))
    for t in stuck:                                         # rare: three separate exact searches
        plan = [(t, 0, None, np.eye(2), {0, 1})] + [(t, 1 + i, M_SIB[i], C_SIB[i], {1}) for i in range(2)]
        for dest, entry in _fallback(base_cols[t], k, eps, plan):
            sols[dest].append(entry)
    return sols, stats


# ---- pairs of base words: B and B' = B X with X = T^dag H S H T (lde 1, even det) ------------
#
# Consecutive bases in synth._coset_prefixes differ only in their last free block
# (Q HT HT and Q HST HT), and B^-1 B' = T^dag (HSH) T up to phase (checked on all pairs of
# levels t' = 8, 10, 12). M_X = sqrt2 w^-1 X is integral of index 16, so B' and its coset
# siblings are sublattice problems of B's reduced lattice:
#   B'      shell k+1  =  B shell k+2  in  L_X              (B' shell k lies in B's shell k+1)
#   B' C    shell k+1  =  B shell k+3  in  L_XC = M_X M_C Z^8  (C' = w^2 H, w S H)
# In B's reduced coordinates z, L_X = {z : z mod 2 in S_X}; with K_X its echelon basis and
# z = K_X^T o, L_XC = {o : o mod 2 in S_XC} where S_XC = colspace((U M_C) mod 2) and
# U = (K_X^T)^-1 T M_X (unimodular), U mod 2 from 16 U = adj(K_X^T) T M_X modulo 32.

def _mat_from_word_scaled(word_mat):
    return _blockmat(word_mat)


def _M_X():
    from _qhg_zomega import uword, umul, ureduce
    k, (a, b, c, d) = ureduce(uword("tHSHT"))           # lde 1 after reduction
    assert k == 1
    winv = zpow_omega(7)                                # w^-1 makes det 1 (det X = w^2)
    return _blockmat(((zmul(winv, a), zmul(winv, b)), (zmul(winv, c), zmul(winv, d))))


M_XPAIR = _M_X()


def _echelon_basis(A2):
    """Echelon basis (rows) of {z : z mod 2 in colspace(A2)} with A2 an (n,8,8) GF(2) matrix of rank 4."""
    n = A2.shape[0]
    R2, piv, rank = _rref_gf2(A2.transpose(0, 2, 1))
    if (rank != 4).any():
        raise RuntimeError("sublattice rank != 4")
    K = np.zeros((n, 8, 8), dtype=np.int64)
    K[:, :4] = R2[:, :4].astype(np.int64)
    pivmask = np.zeros((n, 8), dtype=bool)
    pivmask[np.arange(n)[:, None], piv[:, :4]] = True
    nonpiv = np.argsort(pivmask, axis=1, kind="stable")[:, :4]
    K[np.arange(n)[:, None], 4 + np.arange(4)[None, :], nonpiv] = 2
    return K, R2, piv


def _coset_rep(vec2, R2, piv):
    """vec2 (n,8) in GF(2): reduce by the RREF rows so pivot columns are 0 (returns r, vec2 - r in S)."""
    n = vec2.shape[0]
    r = (vec2 & 1).astype(np.uint8)
    for i in range(4):
        p = piv[:, i]
        flip = r[np.arange(n), p] == 1
        r[flip] ^= R2[flip, i]
    return r.astype(np.int64)


def _adj_det(K):
    """Exact adjugate and determinant of small integer matrices (det +-16 here)."""
    det = np.rint(np.linalg.det(K.astype(float))).astype(np.int64)
    adj = np.rint(np.linalg.inv(K.astype(float)) * det[:, None, None]).astype(np.int64)
    ok = (K @ adj == det[:, None, None] * np.eye(8, dtype=np.int64)).all(axis=(1, 2))
    if not ok.all():
        raise RuntimeError("adjugate not exact")
    return adj, det


def search_pairs(base_cols, k, eps, bound=3.0):
    """base_cols: targets of B_0, B'_0, B_1, B'_1, ... (pairs, even length). One reduction per pair.
    Returns sols[g] for every base g (pair members included): [(sib, shell, y)] in B-pair-leader
    coordinates for the leader... both members' entries are expressed as U = Bleader * SU2(y)."""
    assert len(base_cols) % 2 == 0
    lead = base_cols[0::2]
    bf = BT._base_float(k, eps)
    b, Lc, sqrt_cy = bf["b"], bf["Lc"], bf["sqrt_cy"]
    N = len(lead)
    B0a = bf["B0"]
    S = np.array([L2._s1e_fp(p) for p in lead], dtype=object)
    U = S @ bf["B0inv_obj"]
    v = S >> (PB - BT.W_BITS)
    half, mask = 1 << (BT.LIMB - 1), (1 << BT.LIMB) - 1
    l0 = ((v + half) & mask) - half
    v = (v - l0) >> BT.LIMB
    l1 = ((v + half) & mask) - half
    l2 = (v - l1) >> BT.LIMB
    SL = np.stack([l0.astype(np.int64), l1.astype(np.int64), l2.astype(np.int64)], axis=2)
    WL = np.einsum("ij,njl->nil", B0a, SL)
    l0, l1, l2 = BT._normalise_limbs(WL[:, :, 0], WL[:, :, 1], WL[:, :, 2])
    WL = np.stack([l0, l1, l2], axis=2)
    rc_fp = b["Rc_fp"]
    rc = (rc_fp, (rc_fp * L2.SQ2_FP) >> PB, rc_fp << 1, ((rc_fp << 1) * L2.SQ2_FP) >> PB)   # k..k+3
    Bred, Bredinv, rounds, stuck = BT.batched_lll(WL, Lc, sqrt_cy)
    if stuck.size:                                    # rare: good pairs batched, failed ones exactly
        bad = set(int(g) for g in stuck)
        good = [g for g in range(N) if g not in bad]
        out = [[] for _ in range(2 * N)]
        st = {"lll_rounds": rounds, "nodes_leaf": 0, "fallback_pairs": len(bad)}
        if good:
            sub, st2 = search_pairs([c for g in good for c in (base_cols[2 * g], base_cols[2 * g + 1])],
                                    k, eps, bound)
            st["nodes_leaf"] = st2["nodes_leaf"]
            for j, g in enumerate(good):
                out[2 * g], out[2 * g + 1] = sub[2 * j], sub[2 * j + 1]
        Xp = _OM ** -1 * (np.diag([1, _OM ** -1]) @ _H2 @ np.diag([1, 1j]) @ _H2 @ np.diag([1, _OM]))
        for g in bad:
            plan = [(2 * g, 0, None, np.eye(2), {0, 1})]
            plan += [(2 * g, 1 + i, M_SIB[i], C_SIB[i], {1}) for i in range(2)]
            plan += [(2 * g + 1, 0, M_XPAIR, Xp, {1})]
            plan += [(2 * g + 1, 1 + i, M_XPAIR @ M_SIB[i], Xp @ C_SIB[i], {1}) for i in range(2)]
            for dest, entry in _fallback(base_cols[2 * g], k, eps, plan):
                out[dest].append(entry)
        return out, st
    stats = {"lll_rounds": rounds, "nodes_leaf": 0}
    sols = [[] for _ in range(2 * N)]
    n = N
    V = BT._embed(Bred, WL, Lc, sqrt_cy)
    G = V @ V.transpose(0, 2, 1)
    Bfull = np.einsum("nij,jk->nik", Bred, B0a)
    one = 1 << PB
    Vv = np.matmul(U[:, None, :], Bredinv.astype(object))[:, 0, :]
    zcs = [(Vv * r) >> (PB + 1) for r in rc]
    zint_o = [(z + (one >> 1)) >> PB for z in zcs]
    frac = [((z - (zi << PB)) / one).astype(float) for z, zi in zip(zcs, zint_o)]
    zint = [BT._wrap64(zi) for zi in zint_o]
    B0inv = np.array(b["B0inv"], dtype=np.int64)
    with np.errstate(over="ignore"):
        T2 = np.einsum("nji,kj->nik", Bredinv & 1, B0inv & 1) & 1                   # T mod 2
        T32 = np.einsum("nji,kj->nik", Bredinv & 31, B0inv & 31) & 31               # T mod 32
    # problem list per pair: (basis rows K in z-coords or None for full, shell, sib)
    probs = []                                         # (K, shell index, sib, z0, d)
    eye = np.tile(np.eye(8, dtype=np.int64), (n, 1, 1))
    # full lattice: shells k, k+1 (sib 0)
    for s in range(2):
        probs.append((eye, s, 0, zint[s], frac[s]))
    # B's siblings: shell k+2 in L_H, L_SH (sibs 1, 2)
    for si, M in enumerate(M_SIB):
        K, R2, piv = _echelon_basis((T2 @ (M & 1)) & 1)
        r = _coset_rep(zint[2], R2, piv)
        probs.append((K, 2, 1 + si, zint[2] - r, frac[2] + r))
    # B': shell k+2 in L_X (sib 3)
    KX, RX, pivX = _echelon_basis((T2 @ (M_XPAIR & 1)) & 1)
    rX = _coset_rep(zint[2], RX, pivX)
    probs.append((KX, 2, 3, zint[2] - rX, frac[2] + rX))
    # B' siblings: shell k+3 in L_XC (sibs 4, 5)
    adjX, detX = _adj_det(KX.transpose(0, 2, 1))
    with np.errstate(over="ignore"):
        TMX32 = (T32 @ (M_XPAIR & 31)) & 31
        U2 = (((adjX & 31) @ TMX32) & 31) >> 4                                     # U mod 2
        rX3 = _coset_rep(zint[3], RX, pivX)
        z0X3 = zint[3] - rX3                                                       # in L_X
        o_int2 = ((((adjX & 31) @ ((z0X3 & 31)[:, :, None]))[:, :, 0] & 31) >> 4)  # coords mod 2
    for si, M in enumerate(M_SIB):
        K2, R22, piv2 = _echelon_basis((U2 @ (M & 1)) & 1)                         # in o-coords
        r2 = _coset_rep(o_int2, R22, piv2)
        with np.errstate(over="ignore"):
            z0 = z0X3 - np.einsum("ni,nij->nj", r2, KX)                            # in L_XC
        d = frac[3] + rX3 + np.einsum("ni,nij->nj", r2, KX).astype(float)
        probs.append((K2 @ KX, 3, 4 + si, z0, d))
    P = len(probs) * n
    Gs = np.empty((P, 8, 8))
    fr = np.empty((P, 8))
    ycent = np.empty((P, 8), dtype=np.int64)
    Bmat = np.empty((P, 8, 8), dtype=np.int64)
    shell = np.empty(P, dtype=np.int64)
    sib = np.empty(P, dtype=np.int64)
    m = len(probs)
    Lr = np.linalg.cholesky(G)
    with np.errstate(over="ignore"):
        for q, (K, s, sb, z0, d) in enumerate(probs):
            oc = np.linalg.solve(K.transpose(0, 2, 1).astype(float), d[:, :, None])[:, :, 0]
            oi = np.rint(oc).astype(np.int64)
            zz = z0 + np.einsum("ni,nij->nj", oi, K)
            E = np.einsum("nij,njk->nik", K.astype(float), Lr)
            Gs[q::m] = (E @ E.transpose(0, 2, 1)) * (0.5 ** s)
            fr[q::m] = oc - oi
            ycent[q::m] = np.einsum("ni,nij->nj", zz, Bfull)
            Bmat[q::m] = K @ Bfull
            shell[q::m] = k + s
            sib[q::m] = sb
    pid, off = ENUM(Gs, fr, bound, Bmat, ycent, shell)
    stats["nodes_leaf"] = int(pid.size)
    for a in range(0, pid.size, BT.CHUNK):
        pc, oc_ = pid[a:a + BT.CHUNK], off[a:a + BT.CHUNK]
        X = ycent[pc] + np.einsum("ni,nij->nj", oc_, Bmat[pc])
        sh = shell[pc]
        ok = ((X * X).sum(axis=1) == (np.int64(1) << sh)) & (BT._bform_rows(X) == 0)
        for r_ in np.nonzero(ok)[0]:
            g = pc[r_] // m
            sb = int(sib[pc[r_]])
            member = 0 if sb < 3 else 1
            sols[2 * g + member].append((sb if sb < 3 else sb - 3, int(sh[r_]), tuple(int(t) for t in X[r_])))
    return sols, stats
''',
            'synth': r'''"""Pure-Python near-optimal Clifford+T synthesis of general single-qubit unitaries.

Prefix split (Morisaki-Sano-Akibue; cf. cyclosynth): U = (U_L C) U_R with U_L a
Matsumoto-Amano block product carrying t' T gates, C a Clifford, and U_R an exact SU(2)
element found by lattice.aligned_points on shell 2^lde. The T-count parity of U equals the
parity of the determinant exponent of U_L C (det U_R = 1), so even-T words come from even t'.
"""

import cmath
import math
import sys

import numpy as np
from pathlib import Path


import _qhg_lattice2 as lattice
from _qhg_zomega import (GATES, ONE, ZERO, umul, udagger, ufloat, ureduce, uword, zconj, zmul, zneg,
                    zpow_omega)

# ---- Clifford table and prefix sets -----------------------------------------------------------

def _canonical(U):
    k, m = ureduce(U)
    best = None
    for j in range(8):
        w = zpow_omega(j)
        key = (k,) + tuple(c for e in m for c in zmul(e, w))
        if best is None or key < best:
            best = key
    return best


_CLIFFORDS = None


def cliffords():
    """The 24 single-qubit Cliffords mod phase, as (word, exact unitary)."""
    global _CLIFFORDS
    if _CLIFFORDS is None:
        seen, out = set(), []
        frontier = [""]
        while frontier and len(out) < 24:
            nxt = []
            for w in frontier:
                key = _canonical(uword(w))
                if key in seen:
                    continue
                seen.add(key)
                out.append((w, uword(w)))
                nxt += [w + "H", w + "S"]
            frontier = nxt
        _CLIFFORDS = out
    return _CLIFFORDS


_PREFIX_CACHE = {}
_MONOMIALS = None


def monomials():
    """The 8 monomial Cliffords <S, X> mod phase. Right-multiplying a prefix by one maps the
    inner lattice to itself by a signed permutation that preserves shell, B-form and search
    region, so P and P*M give the same final unitaries (checked by check_coset_symmetry.py)."""
    global _MONOMIALS
    if _MONOMIALS is None:
        _MONOMIALS = [uword(w) for w in ("", "S", "SS", "SSS", "X", "XS", "XSS", "XSSS")]
    return _MONOMIALS


def prefix_set(t_prime, coset=False):
    """(word, exact unitary, float matrix) for L_{t'} times every Clifford, deduplicated by phase;
    with coset=True, one representative per right coset P*<S, X> (8x fewer, no solution lost)."""
    if (t_prime, coset) in _PREFIX_CACHE:
        return _PREFIX_CACHE[(t_prime, coset)]
    HT = umul(GATES["H"], GATES["T"])
    HST = umul(umul(GATES["H"], GATES["S"]), GATES["T"])
    if coset:
        out = _coset_prefixes(t_prime, HT, HST)
        _PREFIX_CACHE[(t_prime, coset)] = out
        return out
    bases = []
    if t_prime == 0:
        bases.append(("", GATES["I"]))
    else:
        def grow(word, U, n):
            if n == 0:
                bases.append((word, U))
                return
            grow(word + "HT", umul(U, HT), n - 1)
            grow(word + "HST", umul(U, HST), n - 1)
        grow("", GATES["I"], t_prime)
        grow("T", GATES["T"], t_prime - 1)
    seen, out = set(), []
    for word, U in bases:
        for cw, C in cliffords():
            V = umul(U, C)
            key = _canonical(V)
            if key in seen:
                continue
            if coset:
                seen.update(_canonical(umul(V, M)) for M in monomials())
            else:
                seen.add(key)
            out.append((word + cw, V, ufloat(V)))
    _PREFIX_CACHE[(t_prime, coset)] = out
    return out


def _coset_prefixes(t_prime, HT, HST):
    """One prefix per right coset of <S, X>, generated directly (no hashing).

    A last block HST = HT*S folds into the Clifford, so the last block is fixed to HT; the 24
    Cliffords fall into 3 right cosets with representatives I, H, SH. With Matsumoto-Amano
    normal forms unique, the 2.25 * 2^t' results are distinct; checked against the hash-deduped
    set for t' <= 10."""
    reps = [(w, uword(w)) for w in ("", "H", "SH")]
    bases = []
    if t_prime == 0:
        bases.append(("", GATES["I"]))
    else:
        def grow(word, U, n):
            if n == 0:
                bases.append((word + "HT", umul(U, HT)))
                return
            grow(word + "HT", umul(U, HT), n - 1)
            grow(word + "HST", umul(U, HST), n - 1)
        grow("", GATES["I"], t_prime - 1)
        if t_prime == 1:
            bases.append(("T", GATES["T"]))
        else:
            grow("T", GATES["T"], t_prime - 2)
    out = []
    for word, U in bases:
        for cw, C in reps:
            V = umul(U, C)
            out.append((word + cw, V, ufloat(V)))
    return out


# ---- exact decomposition into gates -----------------------------------------------------------

def _lde(U):
    return ureduce(U)[0]


def _sde_abs2(U):
    """sqrt(2)-denominator exponent of |U_00|^2 (KMM: drops by exactly 1 per H T^m syllable)."""
    from _qhg_zomega import divisible_by_sqrt2, div_sqrt2
    k, m = ureduce(U)
    n = zmul(m[0], zconj(m[0]))
    if n == ZERO:
        return -1
    c = 0
    while divisible_by_sqrt2(n):
        n = div_sqrt2(n)
        c += 1
    return 2 * k - c


_T_POW = None


def _clifford_with_phase(U):
    """Word for U if U is a Clifford times a global phase, else None."""
    U = ureduce(U)
    if U[0] != 0 and _canonical(U)[0] != 0:
        return None
    key = _canonical(U)
    for cw, C in cliffords():
        if _canonical(C) == key:
            for j in range(8):
                if ureduce((C[0], tuple(zmul(e, zpow_omega(j)) for e in C[1]))) == U:
                    return cw + "W" * j
    return None


def decompose(U):
    """Gate word (H, T, S, X, Z, W letters; leftmost = leftmost factor) for an exact unitary."""
    global _T_POW
    if _T_POW is None:
        _T_POW = [uword(""), uword("t"), uword("tt"), uword("ttt")]
    U = ureduce(U)
    word = []
    H = GATES["H"]
    names = {0: "", 1: "T", 2: "S", 3: "TS"}
    while True:
        tail = _clifford_with_phase(U)
        if tail is not None:
            word.append(tail)
            return "".join(word)
        # U = T^m * Clifford ?
        for m in range(1, 4):
            tail = _clifford_with_phase(umul(_T_POW[m], U))
            if tail is not None:
                word.append(names[m] + tail)
                return "".join(word)
        s = _sde_abs2(U)
        if s >= 4:
            for m in range(4):
                V = ureduce(umul(H, umul(_T_POW[m], U)))
                if _sde_abs2(V) == s - 1:
                    word.append(names[m] + "H")
                    U = V
                    break
            else:
                raise RuntimeError(f"no syllable reduces sde (sde={s}, lde={U[0]})")
            continue
        # small-sde base case: breadth-first over up to 6 syllables
        frontier = [("", U)]
        for _ in range(6):
            nxt = []
            for pre, V in frontier:
                for m in range(4):
                    V2 = ureduce(umul(H, umul(_T_POW[m], V)))
                    p2 = pre + names[m] + "H"
                    tail = _clifford_with_phase(V2)
                    if tail is None:
                        for m2 in range(1, 4):
                            tail = _clifford_with_phase(umul(_T_POW[m2], V2))
                            if tail is not None:
                                tail = names[m2] + tail
                                break
                    if tail is not None:
                        word.append(p2 + tail)
                        return "".join(word)
                    nxt.append((p2, V2))
            frontier = nxt
        raise RuntimeError(f"base case not resolved (sde={s}, lde={U[0]})")


def tcount(word):
    return sum(g in "Tt" for g in word)


# ---- distance -----------------------------------------------------------------------------------

def op_distance(A, B):
    """Operator norm of A - e^{i phi} B with phi the phase of tr(B^dag A) (2x2 unitaries).

    Numerically stable for tiny distances: the difference matrix is formed first and its
    largest singular value computed from D^dag D (no cancellation between O(1) quantities)."""
    tr = sum(B[r][i].conjugate() * A[r][i] for r in range(2) for i in range(2))
    ph = tr / abs(tr) if abs(tr) > 0 else 1.0
    D = [[A[i][j] - ph * B[i][j] for j in range(2)] for i in range(2)]
    a = abs(D[0][0]) ** 2 + abs(D[1][0]) ** 2
    d = abs(D[0][1]) ** 2 + abs(D[1][1]) ** 2
    b = D[0][0].conjugate() * D[0][1] + D[1][0].conjugate() * D[1][1]
    lam = (a + d) / 2 + math.sqrt(((a - d) / 2) ** 2 + abs(b) ** 2)
    return math.sqrt(lam)


# ---- the synthesiser ----------------------------------------------------------------------------

def optimal_t_prime(t, eps):
    thr = 2.5 * math.log2(1 / eps)
    return 0 if t <= thr else int(math.ceil(t - thr))


_FL_ARRAYS = {}


def _fl_array(tp, coset):
    key = (tp, coset)
    if key not in _FL_ARRAYS:
        _FL_ARRAYS[key] = np.array([F for _, _, F in prefix_set(tp, coset)], dtype=complex)
    return _FL_ARRAYS[key]








class LockstepWords:
    """Even-T word searches for many targets at once (structural batching, one process).

    Every round, each active job contributes its next chunk of base words to one
    batch_coset.search call per inner shell; each job keeps its own level and position and the
    earliest-accepted rule of synthesize_cosets, so every job returns the same word as
    synthesize_cosets(W, eps) would (chunking does not change which candidate comes first)."""

    def __init__(self, eps, chunk_bases=64, even_t=True, round_bases=1024, pairs=False):
        self.eps, self.chunk, self.even_t, self.pairs = eps, chunk_bases, even_t, pairs
        self.round_bases = round_bases           # with few active jobs, larger chunks keep batches big
        self.jobs = {}
        self.next_id = 0
        log2e = math.log2(1 / eps)
        self.t_start = max(0, int(2.8 * log2e) if log2e >= 6 * math.log2(10) else int(1.5 * log2e))
        self.t_max = int(3.1 * log2e) + 2
        self.stats = {"rounds": 0, "prefixes": 0, "calls": 0}

    def add(self, W, tag):
        jid = self.next_id
        self.next_id += 1
        job = {"W": [[complex(W[i][j]) for j in range(2)] for i in range(2)], "tag": tag,
               "t": self.t_start - 1, "a": 0, "nb": 0}
        self._next_level(job)
        self.jobs[jid] = job
        return jid

    def _next_level(self, job):
        eps = self.eps
        t = job["t"] + 1
        while t <= self.t_max:
            tp = optimal_t_prime(t, eps)
            if not (self.even_t and tp % 2):
                break
            t += 1
        if t > self.t_max:
            raise RuntimeError("no word found up to t_max")
        tp = optimal_t_prime(t, eps)
        pref = prefix_set(tp, True)
        FL = _fl_array(tp, True)[0::3]
        Wa = np.array(job["W"], dtype=complex)
        Rm = np.einsum("nri,rj->nij", FL.conj(), Wa)
        dR = Rm[:, 0, 0] * Rm[:, 1, 1] - Rm[:, 0, 1] * Rm[:, 1, 0]
        ph = np.exp(-0.5j * np.angle(dR))
        job.update(t=t, tp=tp, lde=(t - tp) // 2 + 1, pref=pref, a=0, nb=len(pref) // 3,
                   cols=np.stack([Rm[:, 0, 0] * ph, Rm[:, 1, 0] * ph], axis=1))

    def run_round(self):
        """Advance every active job by one chunk; returns [(tag, word)] for jobs that finished."""
        import _qhg_batch_coset as batch_coset
        self.stats["rounds"] += 1
        by_k = {}
        for jid, job in self.jobs.items():
            by_k.setdefault(job["lde"], []).append(jid)
        done = []
        chunk = max(self.chunk, -(-self.round_bases // max(1, len(self.jobs))))
        chunk += chunk % 2
        for k, jids in by_k.items():
            groups = {"pair": ([], []), "single": ([], [])}
            for jid in jids:
                job = self.jobs[jid]
                a = job["a"]
                npaired = 2 * (job["nb"] // 2) if self.pairs else 0
                if a < npaired:
                    b, kind = min(npaired, a + chunk), "pair"
                else:
                    b, kind = min(job["nb"], a + chunk), "single"
                cols, spans = groups[kind]
                spans.append((jid, a, b, len(cols), kind))
                cols += [(complex(c0), complex(c1)) for c0, c1 in job["cols"][a:b]]
            results = []
            for kind, (cols, spans) in groups.items():
                if not cols:
                    continue
                if kind == "pair":
                    sols, _ = batch_coset.search_pairs(cols, k, self.eps)
                else:
                    sols, _ = batch_coset.search(cols, k, self.eps)
                self.stats["calls"] += 1
                self.stats["prefixes"] += 3 * len(cols)
                results += [(span, sols) for span in spans]
            for (jid, a, b, off, kind), sols in results:
                job = self.jobs[jid]
                word = None
                for g in range(b - a):
                    lead = a + g - (g % 2 if kind == "pair" else 0)
                    UL = job["pref"][3 * lead][1]
                    for sib, shell, x in sorted(sols[off + g], key=lambda e: (e[0], e[1])):
                        u, v = tuple(x[:4]), tuple(x[4:])
                        U = umul(UL, (shell, (u, zneg(zconj(v)), v, zconj(u))))
                        if op_distance(ufloat(U), job["W"]) >= self.eps:
                            continue
                        w = decompose(U)
                        if self.even_t and tcount(w) % 2:
                            continue
                        word = w
                        break
                    if word is not None:
                        break
                if word is not None:
                    done.append((job["tag"], word))
                    del self.jobs[jid]
                else:
                    job["a"] = b
                    if b >= job["nb"]:
                        self._next_level(job)
        return done


''',
            'phase_factorization': r'''"""Numerical 11-mask SU(2) multiplexor factors with output Z-phase carry.

No gate synthesis is performed here. Failure is explicit so callers can use
the existing 15-mask construction. Local rank does not imply global coverage.
"""
import hashlib
import time

import numpy as np


MASKS = (1, 2, 4, 8, 3, 5, 6, 9, 10, 12, 7)
_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)


class PhaseFactorizationError(RuntimeError):
    pass


def _exp(vector):
    angle = np.linalg.norm(vector)
    return (np.cos(angle) * np.eye(2)
            - 1j * np.sinc(angle / np.pi) * np.einsum("a,aij->ij", vector, _PAULI))


def _log(matrix):
    scalar = np.trace(matrix).real / 2
    vector = (0.5j * np.einsum("aij,ji->a", _PAULI, matrix)).real
    sine = np.linalg.norm(vector)
    if sine < 1e-300:
        if scalar < 0:
            raise PhaseFactorizationError("ambiguous SU(2) logarithm at -I")
        return np.zeros(3)
    return vector * (np.arctan2(sine, scalar) / sine)


_SELECTED = np.array([[(x & m).bit_count() % 2 for m in MASKS] for x in range(16)], bool)


def reconstruct(common, betas, phases):
    """D(phases[x]) common prod beta_j**parity(x,mask_j), D(z)=exp(-izZ)."""
    out = np.tile(common, (16, 1, 1))
    for selected, beta in zip(_SELECTED.T, betas):
        out = np.where(selected[:, None, None], out @ beta, out)
    out[:, 0, :] *= np.exp(-1j * phases[:, None])
    out[:, 1, :] *= np.exp(1j * phases[:, None])
    return out


def _adjoint(matrices):
    result = np.empty((16, 3, 3))
    for b in range(3):
        image = matrices.conj().transpose(0, 2, 1) @ _PAULI[b] @ matrices
        result[:, :, b] = (0.5 * np.einsum("aij,xji->xa", _PAULI, image)).real
    return result


def _jacobian(state):
    common, betas, _ = state
    result = np.zeros((16, 3, 52))
    suffix = np.tile(np.eye(2, dtype=complex), (16, 1, 1))
    for j in range(10, -1, -1):
        result[:, :, 3+3*j:6+3*j] = _adjoint(suffix) * _SELECTED[:, j, None, None]
        suffix = np.where(_SELECTED[:, j, None, None], betas[j] @ suffix, suffix)
    result[:, :, :3] = _adjoint(suffix)
    z_axes = _adjoint(common @ suffix)[:, :, 2]
    for x in range(16):
        result[x, :, 36+x] = z_axes[x]
    return result.reshape(48, 52)


def _update(state, step, scale):
    common, betas, phases = state
    return (common @ _exp(scale * step[:3]),
            [beta @ _exp(scale * step[3+3*j:6+3*j]) for j, beta in enumerate(betas)],
            phases + scale * step[36:])


def _residual(state, target):
    relative = target.conj().transpose(0, 2, 1) @ reconstruct(*state)
    return np.concatenate([_log(matrix) for matrix in relative])


def _attempt(target, rng, deadline):
    state = (_exp(rng.normal(size=3)), [_exp(rng.normal(size=3)) for _ in MASKS],
             rng.normal(size=16))
    start = reconstruct(*state)
    directions = [_log(a.conj().T @ b) for a, b in zip(start, target)]
    position, ds, accepted = 0.0, 0.05, 0
    while position < 1 - 1e-15:
        if time.monotonic() > deadline or accepted >= 128:
            raise PhaseFactorizationError("11-mask continuation work limit")
        next_position = min(1.0, position + ds)
        stage = np.array([a @ _exp(next_position * v) for a, v in zip(start, directions)])
        trial, converged = state, False
        for _ in range(30):
            if time.monotonic() > deadline:
                raise PhaseFactorizationError("11-mask continuation time limit")
            r = _residual(trial, stage)
            if np.max(abs(r)) < 3e-14:
                converged = True
                break
            step = np.linalg.lstsq(_jacobian(trial), -r, rcond=1e-11)[0]
            length = max(np.linalg.norm(step[3*j:3*j+3]) for j in range(12))
            scale = min(1.0, 0.5 / max(length, 1e-300))
            for _ in range(10):
                candidate = _update(trial, step, scale)
                if np.linalg.norm(_residual(candidate, stage)) < np.linalg.norm(r):
                    trial = candidate
                    break
                scale *= 0.5
            else:
                break
        if converged:
            state, position = trial, next_position
            accepted += 1
            ds = min(0.15, ds * 1.3)
        else:
            ds *= 0.5
            if ds < 1e-5:
                raise PhaseFactorizationError("11-mask continuation stalled")
    return state


def factorize_phase(target, *, attempts=2, max_seconds=5.0):
    """Return R, betas, output-Z angles, measured block operator error.

    A target-derived seed makes repeat runs reproducible. Two generic starts
    share a bounded time allowance; callers handle failure with 15 masks.
    """
    target = np.ascontiguousarray(target, dtype=np.complex128)
    if target.shape != (16, 2, 2) or not np.all(np.isfinite(target)):
        raise ValueError("11-mask factorization needs 16 finite 2x2 blocks")
    seed = int.from_bytes(hashlib.sha256(target.tobytes()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + max_seconds
    reason = "no attempts"
    for _ in range(attempts):
        try:
            state = _attempt(target, rng, deadline)
            error = max(float(np.linalg.norm(a-b, 2))
                        for a, b in zip(reconstruct(*state), target))
            if not np.isfinite(error) or error > 1e-12:
                raise PhaseFactorizationError(f"11-mask reconstruction error {error:.2e}")
            return (*state, error)
        except (PhaseFactorizationError, np.linalg.LinAlgError) as exc:
            reason = str(exc)
        if time.monotonic() > deadline:
            break
    raise PhaseFactorizationError(reason)
''',
            'gate_cleanup': r'''"""Exact local Clifford+T cleanup; no approximate synthesis or angle rounding.

Port of the reference repository's gate_cleanup.py with the external exact-synthesis library
replaced by exact Z[w] arithmetic from zomega (canonical keys of exact unitaries, 4x4 dyadic
Clifford matrices) and the integer Matsumoto-Amano normal form from normal_form (string-identical
to the library's). The cleanup logic itself is unchanged."""
from collections import deque
from functools import lru_cache

from _qhg_normal_form import ma_normal_form
from _qhg_zomega import ONE, ZERO, SQRT2, div_sqrt2, divisible_by_sqrt2, ureduce, uword, zadd, zmul, zneg, zpow_omega


_CLIFFORDS = frozenset("HSXYZ")
_SINGLE = _CLIFFORDS | {"T"}
_BASIC = {"Z": "SS", "Y": "WWXSS"}


def _basic(word):
    return "".join(_BASIC.get(gate, gate) for gate in word)


def _key(word):
    """Canonical exact matrix (global phase included) of a word over H, S, T, X, W."""
    return ureduce(uword(word))


@lru_cache(maxsize=1)
def _clifford_table():
    """Shortest Clifford words: physical gates cost one, scalar W costs zero."""
    table = {_key(""): (0, "")}
    queue = deque([""])
    while queue:
        current = queue.popleft()
        cost, word = table[_key(_basic(current))]
        for gate in "WHSXYZ":
            candidate_word = gate + word
            key = _key(_basic(candidate_word))
            candidate_cost = cost + (gate != "W")
            if key in table and table[key][0] <= candidate_cost:
                continue
            table[key] = (candidate_cost, candidate_word)
            if gate == "W":
                queue.appendleft(candidate_word)
            else:
                queue.append(candidate_word)
    if len(table) != 192:
        raise RuntimeError(f"unexpected Clifford group size: {len(table)}")
    return table


@lru_cache(maxsize=1024)
def _short_clifford(word):
    _, result = _clifford_table()[_key(_basic(word))]
    return result.replace("W", ""), result.count("W") % 8


def _clifford_chunks(word):
    """Simplify Clifford runs in a matrix-order word; return scalar W exponent."""
    out, run, phase = [], [], word.count("W")
    for gate in word.replace("W", "") + "!":
        if gate in _CLIFFORDS:
            run.append(gate)
            continue
        if run:
            reduced, scalar = _short_clifford("".join(run))
            out.append(reduced)
            phase += scalar
            run.clear()
        if gate != "!":
            out.append(gate)
    return "".join(out), phase % 8


@lru_cache(maxsize=8192)
def _reduce_run(time_word):
    """Choose an exact shorter representative, with no increase in T-count."""
    matrix_word = time_word[::-1]
    candidates = [(matrix_word, 0), _clifford_chunks(matrix_word)]
    normal = ma_normal_form(matrix_word)
    candidates.append(_clifford_chunks(normal))
    original_t = time_word.count("T")
    viable = [(word, phase) for word, phase in candidates
              if len(word) <= len(time_word) and word.count("T") <= original_t]
    word, phase = min(viable, key=lambda item: (len(item[0]), item[0].count("T"), item[0]))
    return word[::-1], phase


def _pass(gates):
    pending, nodes, last = {}, [], {}
    phase = 0

    def append(gate):
        name, wires = gate[:2]
        prior = last.get(wires[0])
        if name in {"H", "X", "Y", "Z", "CNOT"} and prior is not None:
            previous, links = nodes[prior]
            if previous == gate and all(last.get(wire) == prior for wire in wires):
                nodes[prior] = (None, links)
                for wire, link in zip(wires, links):
                    if link is None:
                        last.pop(wire, None)
                    else:
                        last[wire] = link
                return
        links = tuple(last.get(wire) for wire in wires)
        index = len(nodes)
        nodes.append((gate, links))
        for wire in wires:
            last[wire] = index

    def flush(wire, commuting=frozenset()):
        nonlocal phase
        run = pending.pop(wire, "")
        if not run:
            return
        run, scalar = _reduce_run(run)
        phase = (phase + scalar) % 8
        split = len(run)
        while split and run[split - 1] in commuting:
            split -= 1
        for gate in run[:split]:
            append((gate, (wire,)))
        if split < len(run):
            pending[wire] = run[split:]

    for gate in gates:
        name, wires = gate[:2]
        if name in _SINGLE and len(wires) == 1:
            wire = wires[0]
            pending[wire] = pending.get(wire, "") + name
        elif name == "CNOT":
            control, target = wires
            # Z-axis phase gates on controls and X on targets commute with CX.
            flush(control, frozenset("TSZ"))
            flush(target, frozenset("X"))
            append(gate)
        else:
            for wire in wires:
                flush(wire)
            append(gate)
    for wire in list(pending):
        flush(wire)
    return [gate for gate, _ in nodes if gate is not None], phase


def _pauli_pass(gates):
    """Propagate an exact X^x Z^z frame, removing internal Pauli gates.

    The invariant is processed U = omega^phase * P_frame * emitted U.
    No Hermitian-Pauli convention is assumed: XZ = -iY is tracked explicitly.
    """
    frame, out, phase = {}, [], 0
    for gate in gates:
        name, wires = gate[:2]
        if name == "CNOT":
            control, target = wires
            xc, zc = frame.get(control, (0, 0))
            xt, zt = frame.get(target, (0, 0))
            frame[control] = (xc, zc ^ zt)
            frame[target] = (xt ^ xc, zt)
            out.append(gate)
            continue
        if len(wires) != 1:
            raise ValueError(f"unsupported cleanup gate: {gate}")
        wire = wires[0]
        x, z = frame.get(wire, (0, 0))
        if name == "X":
            x ^= 1
        elif name == "Z":
            phase += 4 * x
            z ^= 1
        elif name == "Y":
            phase += 2 + 4 * x
            x, z = x ^ 1, z ^ 1
        elif name == "H":
            phase += 4 * x * z
            x, z = z, x
            out.append(gate)
        elif name == "S":
            phase += 2 * x
            z ^= x
            out.append(gate)
        elif name == "T":
            # T X = omega X T†; T† = Z S T. Absorb Z into the frame.
            phase += x
            z ^= x
            out.append(gate)
            if x:
                out.append(("S", (wire,)))
        elif name == "RZ":
            out.append(("RZ", wires, -gate[2] if x else gate[2]))
        else:
            raise ValueError(f"unsupported cleanup gate: {gate}")
        frame[wire] = (x, z)
    for wire, (x, z) in frame.items():
        if x and z:
            out.append(("Y", (wire,)))
            phase += 6  # XZ = -iY.
        elif x:
            out.append(("X", (wire,)))
        elif z:
            out.append(("Z", (wire,)))
    return out, phase % 8


_TWO_GENERATORS = tuple(
    (name, (wire,)) for wire in (0, 1) for name in "HSXYZ"
) + (("CNOT", (0, 1)), ("CNOT", (1, 0)))
_IDENTITY_ACTION = ((1, 0, 0), (0, 1, 0), (2, 0, 0), (0, 2, 0))


def _conjugate_action(action, gate):
    """Exact images of X0,Z0,X1,Z1, represented by i^p X^x Z^z."""
    name, wires = gate
    result = []
    for x, z, phase in action:
        if name == "CNOT":
            c, t = wires
            if x & (1 << c):
                x ^= 1 << t
            if z & (1 << t):
                z ^= 1 << c
        else:
            mask = 1 << wires[0]
            a, b = bool(x & mask), bool(z & mask)
            if name == "H":
                phase += 2 * a * b
                if a != b:
                    x ^= mask
                    z ^= mask
            elif name == "S":
                phase += a
                if a:
                    z ^= mask
            elif name == "X":
                phase += 2 * b
            elif name == "Z":
                phase += 2 * a
            elif name == "Y":
                phase += 2 * (a != b)
        result.append((x, z, phase % 4))
    return tuple(result)


@lru_cache(maxsize=1)
def _two_clifford_table():
    """Shortest words in the 11,520 projective two-qubit Clifford elements."""
    table = {_IDENTITY_ACTION: ()}
    queue = deque([_IDENTITY_ACTION])
    while queue:
        state = queue.popleft()
        word = table[state]
        for gate in _TWO_GENERATORS:
            next_state = _conjugate_action(state, gate)
            if next_state not in table:
                table[next_state] = word + (gate,)
                queue.append(next_state)
                if len(table) > 11520:
                    raise RuntimeError("two-qubit Clifford action escaped the finite group")
    if len(table) != 11520:
        raise RuntimeError(f"unexpected two-qubit Clifford group size: {len(table)}")
    return table


def _dyadic_reduce(entry):
    num, k = entry
    if num == ZERO:
        return ZERO, 0
    while k > 0 and divisible_by_sqrt2(num):
        num, k = div_sqrt2(num), k - 1
    return num, k


def _dyadic_add(a, b, sign=1):
    """(a + sign b) for entries num / sqrt2^k, on the larger of the two exponents."""
    (na, ka), (nb, kb) = a, b
    while ka < kb:
        na, ka = zmul(na, SQRT2), ka + 1
    while kb < ka:
        nb, kb = zmul(nb, SQRT2), kb + 1
    return zadd(na, nb if sign > 0 else zneg(nb)), ka


def _omega_times(entry, power):
    return zmul(entry[0], zpow_omega(power)), entry[1]


def _exact_two_matrix(gates):
    """Evaluate a small Clifford word using exact dyadic cyclotomic entries (num / sqrt2^k)."""
    rows = [[(ONE if i == j else ZERO, 0) for j in range(4)] for i in range(4)]
    for name, wires in gates:
        if name == "CNOT":
            c, t = wires
            rows = [rows[i ^ (1 << t) if i & (1 << c) else i] for i in range(4)]
            continue
        mask = 1 << wires[0]
        if name in {"H", "X", "Y"}:
            for i in range(4):
                if i & mask:
                    continue
                a, b = rows[i], rows[i | mask]
                if name == "H":
                    plus = [_dyadic_add(x, y) for x, y in zip(a, b)]
                    minus = [_dyadic_add(x, y, -1) for x, y in zip(a, b)]
                    rows[i] = [(num, k + 1) for num, k in plus]
                    rows[i | mask] = [(num, k + 1) for num, k in minus]
                elif name == "X":
                    rows[i], rows[i | mask] = b, a
                else:
                    rows[i] = [_omega_times(x, 6) for x in b]
                    rows[i | mask] = [_omega_times(x, 2) for x in a]
        else:
            power = 2 if name == "S" else 4
            for i in range(4):
                if i & mask:
                    rows[i] = [_omega_times(x, power) for x in rows[i]]
    return [_dyadic_reduce(entry) for row in rows for entry in row]


@lru_cache(maxsize=8192)
def _short_two_clifford(gates):
    action = _IDENTITY_ACTION
    for gate in gates:
        action = _conjugate_action(action, gate)
    reduced = _two_clifford_table()[action]
    if len(reduced) >= len(gates):
        return gates, 0
    old, new = _exact_two_matrix(gates), _exact_two_matrix(reduced)
    for phase in range(8):
        if all(a == _omega_times(b, phase) for a, b in zip(old, new)):
            return reduced, phase
    raise RuntimeError("two-qubit Clifford rewrite failed exact phase verification")


def _two_qubit_pass(gates):
    """Resynthesize zero-T blocks using at most two wires; preserve all T/RZ barriers."""
    out, block, wires, phase = [], [], [], 0

    def flush():
        nonlocal phase
        if not block:
            return
        if len(wires) == 1:
            word, scalar = _short_clifford("".join(gate[0] for gate in reversed(block)))
            out.extend((name, (wires[0],)) for name in reversed(word))
            phase = (phase + scalar) % 8
            block.clear()
            wires.clear()
            return
        mapping = {wire: i for i, wire in enumerate(wires)}
        symbolic = tuple((name, tuple(mapping[w] for w in ws)) for name, ws in block)
        reduced, scalar = _short_two_clifford(symbolic)
        out.extend((name, tuple(wires[i] for i in ws)) for name, ws in reduced)
        phase = (phase + scalar) % 8
        block.clear()
        wires.clear()

    for gate in gates:
        name, ws = gate[:2]
        is_clifford = name in _CLIFFORDS or name == "CNOT"
        new_wires = [wire for wire in ws if wire not in wires]
        if not is_clifford or len(wires) + len(new_wires) > 2:
            flush()
            new_wires = list(ws)
        if is_clifford:
            wires.extend(wire for wire in new_wires if wire not in wires)
            block.append(gate)
        else:
            out.append(gate)
    flush()
    return out, phase


def cleanup_gates(gates):
    """Return (reduced time-order gates, scalar phase in radians).

    Original U = exp(i*phase) * reduced U. All Clifford rewrites use exact
    cyclotomic arithmetic. RZ angles are kept unchanged. Iterate only while
    gate count decreases, so the cleanup terminates without a precision knob.
    """
    from math import pi

    current, phase = list(gates), 0
    while True:
        local, local_phase = _pass(current)
        framed, frame_phase = _pauli_pass(current)
        framed, final_phase = _pass(framed)
        two_local, two_local_phase = _two_qubit_pass(local)
        two_framed, two_frame_phase = _two_qubit_pass(framed)
        candidates = [(current, 0), (local, local_phase),
                      (framed, (frame_phase + final_phase) % 8),
                      (two_local, (local_phase + two_local_phase) % 8),
                      (two_framed, (frame_phase + final_phase + two_frame_phase) % 8)]
        original_t = sum(gate[0] == "T" for gate in current)
        candidates = [(ops, scalar) for ops, scalar in candidates
                      if len(ops) <= len(current)
                      and sum(gate[0] == "T" for gate in ops) <= original_t]
        reduced, scalar = min(candidates, key=lambda item: (
            len(item[0]), sum(gate[0] == "T" for gate in item[0]),
        ))
        phase = (phase + scalar) % 8
        if len(reduced) >= len(current):
            return reduced, phase * pi / 4
        current = reduced
''',
            'gold_local': r'''#!/usr/bin/env python3
"""
Local unitary-synthesis compiler for QHack 2026 Question 8.

Recursive cosine-sine decomposition supplies uniformly controlled one-qubit
gates. Parity-controlled paired words share Clifford+T implementations; an
optional 11-mask factorization carries output Z phases into subsequent blocks.
The original 15-mask factorization is retained as a numerical fallback and
for the final multiplexor. Common-word folding jointly compiles the final pair.
The final diagonal retains 2^(n-1)-1 arbitrary RZ rotations.

In this bundle the words come from the pure-Python lattice search
(synth.LockstepWords); the command-line driver and external-synthesiser
entry points are removed.
The construction builds on Yamazaki, Akibue, and Sano (arXiv:2603.14202) and
Morisaki, Sano, and Akibue (arXiv:2510.05816).

--eps is a per-word tolerance; final acceptance uses the grader's full matrix
comparison. --error-budget selects a conservative allocation that may increase
cost. At five qubits phase11 plans 376 source words, or up to 496 with fallbacks.
"""
import argparse, functools, itertools as _it, sys, time
from pathlib import Path
import numpy as np



# ======================================================================
# words.py
# ======================================================================
"""Single-qubit Clifford+T word utilities.
Convention: a word is a string over {H,S,T,X,Y,Z,s,t,W}; s=S^dag, t=T^dag, W=global phase e^{i pi/4}.
The string 'ABC' means the matrix A @ B @ C (leftmost = applied last)."""
import numpy as np

OM = np.exp(1j * np.pi / 4)
GATE = {
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "S": np.diag([1, 1j]), "s": np.diag([1, -1j]),
    "T": np.diag([1, OM]), "t": np.diag([1, np.conj(OM)]),
    "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]),
    "W": np.eye(2) * OM, "I": np.eye(2),
}

def mat(word):
    M = np.eye(2, dtype=complex)
    for g in word:
        M = M @ GATE[g]
    return M

INV = {"H": "H", "S": "s", "s": "S", "T": "t", "t": "T", "X": "X", "Y": "Y", "Z": "Z", "I": "I"}
def inverse(word):
    """Inverse word (exact up to global phase W^-k, tracked separately by callers via skeleton phase)."""
    out = []
    for g in reversed(word):
        if g == "W":
            out.append("WWWWWWW")
        else:
            out.append(INV[g])
    return "".join(out)

def tcount(word):
    return sum(g in "Tt" for g in word)

def skeleton(word):
    """Reduce a word to  phase * T^e0 H T^e1 H ... H T^er  (e_i mod 8).
    Returns (exps, phase_k) with global phase OM**phase_k.  Exact identities used:
    S=T^2, Z=T^4, s=T^6, t=T^7, X = H T^4 H, Y = i X Z = OM^2 * H T^4 H T^4, W = OM."""
    toks = []      # list of ints (T powers) and 'H'
    ph = 0
    for g in word:
        if g == "H": toks.append("H")
        elif g == "T": toks.append(1)
        elif g == "t": toks.append(7)
        elif g == "S": toks.append(2)
        elif g == "s": toks.append(6)
        elif g == "Z": toks.append(4)
        elif g == "X": toks += ["H", 4, "H"]
        elif g == "Y": toks += ["H", 4, "H", 4]; ph += 2
        elif g == "W": ph += 1
        elif g == "I": pass
        else: raise ValueError(g)
    # normalise into alternating exps: e0 H e1 H ... (merge powers, cancel H H)
    exps = [0]
    for tk in toks:
        if tk == "H":
            if len(exps) >= 2 and exps[-1] % 8 == 0:
                exps.pop()               # ... H T^0 H  ->  ... (H H = I)
            else:
                exps.append(0)
        else:
            exps[-1] = (exps[-1] + tk) % 8
    return [e % 8 for e in exps], ph % 8

def skeleton_word(exps, ph=0):
    """Word realising OM^ph * T^e0 H T^e1 H ... H T^er using only H,S,T,Z (T^k as T/S/Z combos)."""
    def tp(k):
        k %= 8
        return ("T" if k & 1 else "") + ("S" if k & 2 else "") + ("Z" if k & 4 else "")
    w = "W" * (ph % 8)
    for i, e in enumerate(exps):
        if i: w += "H"
        w += tp(e)
    return w



# ======================================================================
# parity.py
# ======================================================================
"""Parity factorisation of a uniformly controlled SU(2) gate (Yamazaki-Akibue eq. 127 form):
    V_x = R . prod_{j=1..q} beta_j^{p_j(x)}      (ordered product, j=1 leftmost),
where x ranges over 2^c control patterns, masks m_j = 1..2^c-1, p_j(x) = popcount(m_j & x) mod 2.
R = V_0. Solved by continuation from the identity with Newton steps (analytic structure, 3 params per beta)."""
import numpy as np

PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]

def su2_exp(v):
    th = np.linalg.norm(v)
    if th < 1e-300:
        return np.eye(2, dtype=complex)
    n = v / th
    return np.cos(th) * np.eye(2) - 1j * np.sin(th) * (n[0] * PAULI[0] + n[1] * PAULI[1] + n[2] * PAULI[2])

def su2_log(U):
    """v with su2_exp(v) = U (U in SU(2)), principal branch, full precision near identity."""
    a = np.real(U[0, 0] + U[1, 1]) / 2
    v = np.array([-np.imag(U[0, 1] + U[1, 0]) / 2,
                  np.real(U[1, 0] - U[0, 1]) / 2,
                  -np.imag(U[0, 0] - U[1, 1]) / 2])
    s = np.linalg.norm(v)
    if s < 1e-300:
        return np.zeros(3)
    th = np.arctan2(s, a)
    return th * v / s

def parities(c):
    q = 2 ** c - 1
    P = np.zeros((2 ** c, q), dtype=int)
    for x in range(2 ** c):
        for j, m in enumerate(range(1, 2 ** c)):
            P[x, j] = bin(x & m).count("1") & 1
    return P

def model(R, betas, P):
    out = []
    for x in range(P.shape[0]):
        M = R.copy()
        for j in range(P.shape[1]):
            if P[x, j]:
                M = M @ betas[j]
        out.append(M)
    return out

def factorize(V, max_steps=400, tol=1e-13):
    """V: array (2^c, 2, 2) of SU(2) blocks. Returns (R, betas[q]) with model == V to ~1e-13."""
    n = V.shape[0]; c = int(round(np.log2(n))); q = n - 1
    P = parities(c)
    R = V[0].copy()
    Rinv = R.conj().T
    L = [su2_log(Rinv @ V[x]) for x in range(n)]          # targets relative to R
    th = np.zeros((q, 3))                                 # beta_j = exp(th_j)
    def resid(th, s):
        betas = [su2_exp(t) for t in th]
        r = []
        for x in range(1, n):
            M = np.eye(2, dtype=complex)
            for j in range(q):
                if P[x, j]:
                    M = M @ betas[j]
            Tgt = su2_exp(s * L[x])
            r.append(su2_log(Tgt.conj().T @ M))
        return np.concatenate(r)
    def jac(th, s, h=1e-7):
        f0 = resid(th, s); J = np.zeros((f0.size, th.size))
        flat = th.ravel()
        for k in range(flat.size):
            t2 = flat.copy(); t2[k] += h
            J[:, k] = (resid(t2.reshape(q, 3), s) - f0) / h
        return f0, J
    s = 0.0; ds = 0.05
    while s < 1.0 - 1e-15:
        s_new = min(1.0, s + ds)
        th_try = th.copy(); ok = False
        for it in range(30):
            f, J = jac(th_try, s_new)
            if np.abs(f).max() < (tol if s_new == 1.0 else 1e-10):
                ok = True; break
            try:
                step = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                break
            th_try = th_try + step.reshape(q, 3)
            if np.abs(step).max() > 1.0:
                break
        if ok:
            th, s = th_try, s_new
            ds = min(0.25, ds * 1.5)
        else:
            ds /= 2
            if ds < 1e-6:
                raise RuntimeError("parity factorisation continuation failed")
    betas = [su2_exp(t) for t in th]
    # polish with exact-ish Newton at s=1
    return R, betas

# ======================================================================
# pair.py
# ======================================================================
"""Parity-controlled pair P(A,B) = |0><0|_c (x) A + |1><1|_c (x) B with T-cost ~ max(T(A),T(B)) (+ small overhead).
Layer rule (Yamazaki-Akibue Thm 2 idea): P(T^x, T^y) with x = y mod 2 costs 1 T if odd, 0 or 2 if even,
up to a phase on the control which is returned (to be absorbed downstream for free)."""
import numpy as np

V_CH = "SHT"           # V X V^dag = H exactly; controlled-H = (I(x)V) CNOT (I(x)V^dag), T-cost 2

def dcost(x, y):
    k1 = ((x + y) // 2) % 8
    d = ((y - x) // 2) % 8
    return (k1 & 1) + (d & 1)

def diagonal_params(x, y):
    """Equivalent half-angle lifts; a control-only phase is deferred by callers."""
    a = ((x + y) // 2) % 8
    d = ((y - x) // 2) % 8
    def gate_count(pair):
        k, delta = pair
        return k.bit_count() + (2 + ((-delta) % 8).bit_count() if delta else 0)
    return min(((a, d), ((a + 4) % 8, (d + 4) % 8)), key=gate_count)

def diagonal_gate_count(x, y):
    a, d = diagonal_params(x, y)
    return a.bit_count() + (2 + ((-d) % 8).bit_count() if d else 0)

def align(a, b, band=2):
    """DP alignment of skeleton segments a (word A) and b (word B), matrix order.
    Returns (cost, ops) with ops in matrix order: ('D',x,y) | ('H',) | ('HA',) (H only in A) | ('HB',)."""
    N, M = len(a) - 1, len(b) - 1
    INF = 10 ** 9
    sys.setrecursionlimit(10000)
    @functools.lru_cache(maxsize=None)
    def best(i, j, ra, rb):
        if abs(i - j) > band:
            return INF, INF, ()
        res = (INF, INF, ())
        if i == N and j == M:
            if (ra - rb) % 2 == 0:
                return dcost(ra, rb), diagonal_gate_count(ra, rb), (("D", ra, rb),)
            return res
        if i < N and j < M and (ra - rb) % 2 == 0:
            c, ng, ops = best(i + 1, j + 1, a[i + 1], b[j + 1])
            c += dcost(ra, rb)
            ng += diagonal_gate_count(ra, rb) + 1
            if (c, ng) < res[:2]:
                res = (c, ng, (("D", ra, rb), ("H",)) + ops)
        if i < N:
            for x in range(ra % 2, 8, 2):
                c, ng, ops = best(i + 1, j, a[i + 1], (rb - x) % 8)
                c += dcost(ra, x) + 2
                ng += diagonal_gate_count(ra, x) + 12
                if (c, ng) < res[:2]:
                    res = (c, ng, (("D", ra, x), ("HA",)) + ops)
        if j < M:
            for x in range(rb % 2, 8, 2):
                c, ng, ops = best(i, j + 1, (ra - x) % 8, b[j + 1])
                c += dcost(x, rb) + 2
                ng += diagonal_gate_count(x, rb) + 10
                if (c, ng) < res[:2]:
                    res = (c, ng, (("D", x, rb), ("HB",)) + ops)
        return res
    out = best(0, 0, a[0], b[0])
    best.cache_clear()
    return out[0], out[2]

def split_options(word, max_pos=3):
    """Ways to write word = X Y at a T-exponent position, giving (A,B) = (X^-1, Y) skeletons.
    Only near-balanced split points (equal T in both halves) and one exponent split per parity class."""
    f, ph = skeleton(word)
    nodd = sum(e & 1 for e in f)
    cands = []
    for j in range(len(f)):
        left_odd = sum(e & 1 for e in f[:j])
        for alpha in range(8):
            beta = (f[j] - alpha) % 8
            ta = left_odd + (alpha & 1); tb = nodd - left_odd - (f[j] & 1) + (beta & 1)
            if ta != tb:
                continue
            cands.append((abs(j - len(f) / 2), j, alpha))
    cands.sort()
    opts, seen = [], set()
    for _, j, alpha in cands:
        key = (j, alpha & 1, ((f[j] - alpha) % 8) & 1)
        if key in seen: continue
        seen.add(key)
        a = [(-alpha) % 8] + [(-e) % 8 for e in reversed(f[:j])]
        b = [(f[j] - alpha) % 8] + f[j + 1:]
        opts.append((a, b, ph))
        if len(opts) >= max_pos * 2: break
    return opts

def _emit_pair_ops(ops, ctrl_phase=0):
    """Emit a skeleton alignment, deferring its exact control phase."""
    gates = []            # time order list of (name, wires)
    def tpow(k, w):
        k %= 8
        out = []
        if k & 1: out.append(("T", (w,)))
        if k & 2: out.append(("S", (w,)))
        if k & 4: out.append(("Z", (w,)))
        return out
    def word_gates(wd, w):   # time order gates for word wd (matrix order string)
        out = []
        for g in reversed(wd):
            if g == "t": out += tpow(7, w)
            elif g == "s": out += tpow(6, w)
            else: out.append((g, (w,)))
        return out
    for op in reversed(ops):              # time order = reverse matrix order
        if op[0] == "D":
            x, y = op[1], op[2]
            k1, d = diagonal_params(x, y)
            gates += tpow(k1, "q")
            if d % 8:
                gates += [("CNOT", ("c", "q"))] + tpow(-d, "q") + [("CNOT", ("c", "q"))]
            ctrl_phase += d
        elif op[0] == "H":
            gates.append(("H", ("q",)))
        else:
            ch = word_gates(inverse(V_CH), "q") + [("CNOT", ("c", "q"))] + word_gates(V_CH, "q")
            if op[0] == "HA":                 # H only when parity = 0
                gates += [("X", ("c",))] + ch + [("X", ("c",))]
            else:
                gates += ch
    tc = sum(g[0] == "T" for g in gates)
    return {"gates": gates, "t": tc, "ctrl_phase": ctrl_phase % 8}

def compile_pair(word, band=2):
    """Best split pair with A^-1 B == word; retain exact deferred phases."""
    best = None
    for a, b, phB in split_options(word):
        cost, ops = align(a, b, band)
        ng = sum(diagonal_gate_count(op[1], op[2]) if op[0] == "D"
                 else {"H": 1, "HA": 12, "HB": 10}[op[0]] for op in ops)
        if best is None or (cost, ng) < best[:2]:
            best = (cost, ng, ops, a, phB)
    if best is None or best[0] >= 10 ** 9:
        raise RuntimeError("no parity-compatible split-word alignment")
    _, _, ops, a, phB = best
    result = _emit_pair_ops(ops, phB)
    result["A"] = skeleton_word(a)
    return result

def compile_word_pair(word0, word1, band=2):
    """Compile two full words; P(w0,w1)=exp(i global) D(ctrl) raw gates."""
    a, ph0 = skeleton(word0)
    b, ph1 = skeleton(word1)
    if sum(e & 1 for e in a) % 2 != sum(e & 1 for e in b) % 2:
        raise ValueError("full paired words must have matching T parity")
    band = max(band, abs(len(a) - len(b)) + 1)
    cost, ops = align(a, b, band)
    if cost >= 10 ** 9:
        cost, ops = align(a, b, max(len(a), len(b)))
    if cost >= 10 ** 9:
        raise RuntimeError("no full-word pair alignment")
    result = _emit_pair_ops(ops, ph1 - ph0)
    result["global"] = ph0 * np.pi / 4
    return result


# ---- Clifford conjugation variants (free): P(A'K^-1, A'K^-1 w) = P(A', A' K^-1 w K) (I (x) K^-1) ----
_CLIFF = []
def cliffords():
    if not _CLIFF:
        mats = []
        for L in range(0, 6):
            for w in _it.product("HS", repeat=L):
                w = "".join(w); M = mat(w)
                if not any(abs(abs(np.trace(M.conj().T @ m)) - 2) < 1e-9 for m in mats):
                    _CLIFF.append(w); mats.append(M)
    return _CLIFF


def compile_word_pair_best(word0, word1):
    """Search simultaneous Clifford conjugations of two full words."""
    best = None
    for K in cliffords():
        Kinv = inverse(K)
        result = compile_word_pair(ma_normal_form(Kinv + word0 + K),
                                   ma_normal_form(Kinv + word1 + K))
        pre = [(name, ("q",)) for name, _ in word_gates(Kinv, 0)]
        post = [(name, ("q",)) for name, _ in word_gates(K, 0)]
        result["gates"] = pre + result["gates"] + post
        score = result["t"], len(result["gates"])
        if best is None or score < best[0]:
            best = score, result
    return best[1]

# ======================================================================
# synths.py
# ======================================================================
"""Single-qubit synthesizer backends: word = synth(U, eps) with mat(word) ~ U up to global phase."""
import numpy as np



_CYC = {}

# ======================================================================
# mux.py
# ======================================================================
"""Uniformly controlled SU(2) gate via parity factorisation + Yamazaki-Akibue paired words (Thm 4).
    V_x = R . prod_j beta_j^{p_j(x)},   implemented as   U~ = C . P_1 ... P_q   (P_q first in time),
    P_j = P_{p_j}(A_j, A_j W^_j),   W_j = G_{j+1} beta_j G_{j+1}^-1,   G_j = A_j G_{j+1},   C ~ R G_1^-1."""
import numpy as np


def phase_of(word, U):
    """gamma with mat(word) ~ exp(i gamma) U."""
    return float(np.angle(np.trace(U.conj().T @ mat(word))))

def word_error(word, U):
    """Phase-aligned operator-norm error, stable for errors near 1e-8.

    The trace overlap alone loses small errors quadratically and can round to
    exactly one even when the matrix error exceeds the grader's tolerance.
    """
    M = mat(word)
    gamma = np.angle(np.trace(U.conj().T @ M))
    return float(np.linalg.norm(np.exp(-1j * gamma) * M - U, 2))



def factor_mux_blocks(V, mux_method="baseline", allow_output_z=True, verbose=False):
    """Return factors, actual masks, and output-Z carry; fall back before synthesis."""
    if mux_method not in ("baseline", "phase11"):
        raise ValueError(f"unknown multiplexor method: {mux_method}")
    count = len(V)
    c = int(round(np.log2(count)))
    if mux_method == "phase11" and c == 4 and allow_output_z:
        from _qhg_phase_factorization import MASKS, PhaseFactorizationError, factorize_phase
        try:
            R, betas, z, error = factorize_phase(V)
            if verbose:
                print(f"    phase11: 11 masks, factor error={error:.2e}", flush=True)
            return R, betas, MASKS, z, error, "phase11"
        except PhaseFactorizationError as exc:
            if verbose:
                print(f"    phase11 fallback to 15 masks: {exc}", flush=True)
    R, betas = factorize(V)
    error = max(float(np.linalg.norm(M - target, 2))
                for M, target in zip(model(R, betas, parities(c)), V))
    if error > 1e-11:
        raise RuntimeError(f"parity factorization matrix error {error:.2e} exceeds 1e-11")
    return R, betas, tuple(range(1, count)), np.zeros(count), error, "baseline"


# ======================================================================
# emit.py
# ======================================================================
"""Turn synthesized multiplexors into concrete gate lists and simulate small cases."""
import numpy as np

def word_gates(word, w):
    """Time-ordered gates for a matrix-order word on wire w (t,s expanded into T/S/Z)."""
    out = []
    for g in reversed(word):
        if g == "t": out += [("T", (w,)), ("S", (w,)), ("Z", (w,))]
        elif g == "s": out += [("S", (w,)), ("Z", (w,))]
        elif g == "W": pass                       # global phase handled by caller
        else: out.append((g, (w,)))
    return out

def word_phase(word):
    return np.pi / 4 * word.count("W")

def mux_gates(res, target, controls):
    """controls: list of wires, big-endian w.r.t. mask bits. Returns (gates time order, global phase rad)."""
    c = len(controls); gates = []
    gph = res["global"] + word_phase(res["C"])
    for mask, pg in res["pairs"]:
        mem = [controls[i] for i in range(c) if (mask >> (c - 1 - i)) & 1]
        pw = mem[-1]
        ladder = [("CNOT", (o, pw)) for o in mem[:-1]]
        gates += ladder
        for name, ws in pg:
            gates.append((name, tuple(pw if x == "c" else target for x in ws)))
        gates += ladder[::-1]
    gates += word_gates(res["C"], target)
    return gates, gph


# ======================================================================
# flagsynth.py
# ======================================================================
"""Recursive CSD circuit with parity-controlled paired-word multiplexors."""
import numpy as np

def csd_items(u_mat, cs_decomposition):
    n = int(round(np.log2(len(u_mat))))
    def rec(blocks, k):
        if k == n - 1:
            return [("G", blocks)]
        lefts, rights, cs = [], [], []
        for block in blocks:
            l0, l1, th, r0, r1 = cs_decomposition(block)
            lefts += [l0, l1]; rights += [r0, r1]; cs.append(2 * np.asarray(th))
        return rec(rights, k + 1) + [("CS", k, np.concatenate(cs))] + rec(lefts, k + 1)
    return rec([np.array(u_mat, dtype=complex)], 0), n

def Ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)

def full_index(n, t, controls, x, tb):
    idx = 0
    for i, w in enumerate(controls):
        if (x >> (len(controls) - 1 - i)) & 1:
            idx |= 1 << (n - 1 - w)
    if tb:
        idx |= 1 << (n - 1 - t)
    return idx

def uc_rz(angles, controls, target, out):
    """Gray-code uniformly controlled RZ; skips (near-)zero rotations."""
    angles = np.asarray(angles, float); k = len(controls)
    if k == 0:
        if abs(angles[0]) > 1e-13: out.append(("RZ", (target,), float(angles[0])))
        return
    size = 2 ** k
    gray = [i ^ (i >> 1) for i in range(size)]
    signs = np.array([[(-1) ** bin(j & g).count("1") for j in range(size)] for g in gray])
    th = signs @ angles / size
    for i in range(size):
        if abs(th[i]) > 1e-13: out.append(("RZ", (target,), float(th[i])))
        bit = (gray[i] ^ gray[(i + 1) % size]).bit_length() - 1
        out.append(("CNOT", (controls[k - 1 - bit], target)))

def diagonal_gates(phases, wires, out):
    """diag(exp(i*phases)) on wires (big-endian); returns leftover global phase (radians)."""
    phases = np.asarray(phases, float); m = len(wires)
    while m > 0:
        even, odd = phases[0::2], phases[1::2]
        uc_rz(odd - even, wires[:m - 1], wires[m - 1], out)
        phases = (even + odd) / 2; m -= 1
    return float(phases[0])

def optimize_circuit(gates, gph, verbose=False):
    """Reduce physical gates exactly, retaining all removed scalar phases."""
    from _qhg_gate_cleanup import cleanup_gates
    started = time.monotonic()
    if verbose:
        print(f"  cleaning {len(gates) + 1:,} emitted gates (including global phase)", flush=True)
    reduced, phase = cleanup_gates(gates)
    if verbose:
        print(f"  gate cleanup: {len(gates) + 1:,} -> {len(reduced) + 1:,}; "
              f"T={sum(g[0] == 'T' for g in reduced)} "
              f"({time.monotonic() - started:.1f}s)", flush=True)
    return reduced, gph + phase



# ======================================================================
# driver
# ======================================================================
_CHECKED = {}


''',
            'fast_pairs': r'''"""Faster, output-identical replacements for gold_local's pair compilers.

install(gold_local, workers=4) swaps in:
  * align: the same dynamic programme with dcost / diagonal_gate_count precomputed as 8x8 tables
    (identical loop order and strict tie-breaking, so identical results);
  * compile_pair_best / compile_word_pair_best: the per-Clifford work (normal form + alignment)
    runs in a process pool; results are collected in the original Clifford order and reduced
    with the original selection rule, so the chosen pair is identical.
Optionally full_cliffords=True also fixes cliffords() to return all 24 single-qubit Cliffords
(the original enumerates H/S words only up to length 5 and finds 21). That option changes
outputs and is measured separately.
"""

import functools
import itertools
import sys

_G = None          # gold_local module (set by install, and in each worker)
_POOL = None


def _tables(G):
    dc = [[G.dcost(x, y) for y in range(8)] for x in range(8)]
    dg = [[G.diagonal_gate_count(x, y) for y in range(8)] for x in range(8)]
    return dc, dg


def make_align(G):
    DC, DG = _tables(G)

    def align(a, b, band=2):
        """gold_local.align with table lookups; same recursion, order and tie-breaks."""
        N, M = len(a) - 1, len(b) - 1
        INF = 10 ** 9
        sys.setrecursionlimit(10000)

        @functools.lru_cache(maxsize=None)
        def best(i, j, ra, rb):
            if abs(i - j) > band:
                return INF, INF, ()
            res = (INF, INF, ())
            if i == N and j == M:
                if (ra - rb) % 2 == 0:
                    return DC[ra][rb], DG[ra][rb], (("D", ra, rb),)
                return res
            if i < N and j < M and (ra - rb) % 2 == 0:
                c, ng, ops = best(i + 1, j + 1, a[i + 1], b[j + 1])
                c += DC[ra][rb]
                ng += DG[ra][rb] + 1
                if (c, ng) < res[:2]:
                    res = (c, ng, (("D", ra, rb), ("H",)) + ops)
            if i < N:
                for x in range(ra % 2, 8, 2):
                    c, ng, ops = best(i + 1, j, a[i + 1], (rb - x) % 8)
                    c += DC[ra][x] + 2
                    ng += DG[ra][x] + 12
                    if (c, ng) < res[:2]:
                        res = (c, ng, (("D", ra, x), ("HA",)) + ops)
            if j < M:
                for x in range(rb % 2, 8, 2):
                    c, ng, ops = best(i, j + 1, (ra - x) % 8, b[j + 1])
                    c += DC[x][rb] + 2
                    ng += DG[x][rb] + 10
                    if (c, ng) < res[:2]:
                        res = (c, ng, (("D", x, rb), ("HB",)) + ops)
            return res

        out = best(0, 0, a[0], b[0])
        best.cache_clear()
        return out[0], out[2]

    return align






# ---- worker side -------------------------------------------------------------------------







# ---- parent side -------------------------------------------------------------------------

def make_compile_pair_best(G):
    def compile_pair_best(word):
        """gold_local.compile_pair_best with the per-Clifford work in the pool."""
        per_k = list(_POOL.map(_single_for, [(word, K) for K in G.cliffords()]))
        best = None
        for res, Kinv in per_k:
            pre_count = sum(2 if gate == "s" else 1 for gate in Kinv)
            if best is None or (res["t"], len(res["gates"]) + pre_count) < best[2]:
                best = (res, Kinv, (res["t"], len(res["gates"]) + pre_count))
        res, Kinv, _ = best
        pre = []
        for g in reversed(Kinv):
            if g == "s":
                pre += [("S", ("q",)), ("Z", ("q",))]
            else:
                pre.append((g, ("q",)))
        return {"gates": pre + res["gates"], "t": res["t"], "ctrl_phase": res["ctrl_phase"],
                "A": res["A"] + Kinv}
    return compile_pair_best


def make_compile_word_pair_best(G):
    def compile_word_pair_best(word0, word1):
        """gold_local.compile_word_pair_best with the per-Clifford work in the pool."""
        per_k = list(_POOL.map(_double_for, [(word0, word1, K) for K in G.cliffords()]))
        best = None
        for result, Kinv, K in per_k:
            pre = [(name, ("q",)) for name, _ in G.word_gates(Kinv, 0)]
            post = [(name, ("q",)) for name, _ in G.word_gates(K, 0)]
            result["gates"] = pre + result["gates"] + post
            score = result["t"], len(result["gates"])
            if best is None or score < best[0]:
                best = score, result
        return best[1]
    return compile_word_pair_best




def make_compile_pair_early(G, min_cliffords=2):
    """compile_pair_best with an early stop. Only the T-count is scored; over the Clifford
    conjugations the pair T-cost takes at most two values (T/2 + 2 or T/2 + 4 in 48/48 words
    measured, T = T-count of the word) and the first two Cliffords always reached the minimum.
    Stop at the first option with cost <= T/2 + 2, or after min_cliffords Cliffords if the best is
    <= T/2 + 4; otherwise search all (then identical to compile_pair_best). The tie-break on gate
    count among equal-T options can differ from the full search; the T-count does not unless a
    later Clifford beats T/2 + 4 after two failed (never observed)."""
    def compile_pair_early(word):
        T = G.tcount(word)
        best = None
        for idx, K in enumerate(G.cliffords()):
            Kinv = G.inverse(K)
            res = G.compile_pair(G.ma_normal_form(Kinv + word + K))
            pre_count = sum(2 if gate == "s" else 1 for gate in Kinv)
            key = (res["t"], len(res["gates"]) + pre_count)
            if best is None or key < best[2]:
                best = (res, Kinv, key)
            if best[2][0] <= T // 2 + 2 or (idx + 1 >= min_cliffords and best[2][0] <= T // 2 + 4):
                break
        res, Kinv, _ = best
        pre = []
        for g in reversed(Kinv):
            if g == "s":
                pre += [("S", ("q",)), ("Z", ("q",))]
            else:
                pre.append((g, ("q",)))
        return {"gates": pre + res["gates"], "t": res["t"], "ctrl_phase": res["ctrl_phase"],
                "A": res["A"] + Kinv}
    return compile_pair_early
''',
            'batched_align': r'''"""Costs of many skeleton alignments at once (gold_local.align's dynamic programme, vectorised).

align(a, b, band) minimises (T cost, gate count) lexicographically over the same moves as
gold_local.align: diagonal D(ra, rb)+H when both skeletons advance, D(ra, x)+HA or D(x, rb)+HB
when one advances. Here the value of every state (i, j, ra, rb) is computed bottom-up for P
problems at once as a key c * 2^20 + ng (lexicographic order = integer order), on band cells only.
Only the optimal (c, ng) is returned; callers re-run the scalar align for the chosen problem to
emit its ops (the scalar one picks the same (c, ng), with its own tie-break among equal paths)."""

import numpy as np

SHIFT = 20
INF = np.int64(1) << 60


def make_align_costs(G):
    DC = np.array([[G.dcost(x, y) for y in range(8)] for x in range(8)], dtype=np.int64)
    DG = np.array([[G.diagonal_gate_count(x, y) for y in range(8)] for x in range(8)], dtype=np.int64)
    par = (np.arange(8)[:, None] - np.arange(8)[None, :]) % 2 == 0                # (ra - rb) even
    KD = np.where(par, (DC << SHIFT) + DG + 1, INF)                               # diagonal move cost
    KT = np.where(par, (DC << SHIFT) + DG, INF)                                   # terminal cost
    KA = np.where(par, ((DC + 2) << SHIFT) + DG + 12, INF)                        # [ra, x]
    KB = np.where(par, ((DC + 2) << SHIFT) + DG + 10, INF)                        # [x, rb]
    r8 = np.arange(8)
    SHA = (r8[None, :] - r8[:, None]) % 8          # SHA[x, rb] = (rb - x) % 8
    SHB = (r8[None, :] - r8[:, None]) % 8          # SHB[x, ra] = (ra - x) % 8

    def align_costs(problems):
        """problems: list of (a, b, band) skeleton pairs. Returns (P, 2) array of (c, ng)."""
        P = len(problems)
        Ns = np.array([len(a) - 1 for a, _, _ in problems])
        Ms = np.array([len(b) - 1 for _, b, _ in problems])
        bands = np.array([band for _, _, band in problems])
        Nmax, Mmax, bmax = Ns.max(), Ms.max(), bands.max()
        A = np.zeros((P, Nmax + 2), dtype=np.int64)
        Bk = np.zeros((P, Mmax + 2), dtype=np.int64)
        for p, (a, b, _) in enumerate(problems):
            A[p, :len(a)] = a
            Bk[p, :len(b)] = b
        width = 2 * bmax + 1
        T = np.full((Nmax + 2, width + 2, P, 8, 8), INF, dtype=np.int64)     # T[i, j - i + bmax + 1]
        pidx = np.arange(P)
        for i in range(Nmax, -1, -1):
            for j in range(min(Mmax, i + bmax), max(0, i - bmax) - 1, -1):
                col = j - i + bmax + 1
                valid = (i <= Ns) & (j <= Ms) & (np.abs(i - j) <= bands)
                if not valid.any():
                    continue
                best = np.full((P, 8, 8), INF, dtype=np.int64)
                term = valid & (i == Ns) & (j == Ms)
                if term.any():
                    best[term] = KT
                inner = valid & ~term
                if inner.any():
                    # diagonal: (i+1, j+1) at (a[i+1], b[j+1])
                    dmask = inner & (i < Ns) & (j < Ms)
                    if dmask.any():
                        nxt = T[i + 1, col][pidx, A[:, i + 1], Bk[:, j + 1]]           # (P,)
                        cand = nxt[:, None, None] + KD[None]
                        cand[~dmask] = INF
                        best = np.minimum(best, cand)
                    # A-move: (i+1, j) at (a[i+1], (rb - x) % 8)
                    amask = inner & (i < Ns)
                    if amask.any() and col - 1 >= 0:
                        row = T[i + 1, col - 1][pidx, A[:, i + 1], :]                  # (P, 8 rb')
                        sh = row[:, SHA]                                                # (P, 8 x, 8 rb)
                        cand = (sh[:, None, :, :] + KA[None, :, :, None]).min(axis=2)  # (P, 8 ra, 8 rb)
                        cand[~amask] = INF
                        best = np.minimum(best, cand)
                    # B-move: (i, j+1) at ((ra - x) % 8, b[j+1])
                    bmask = inner & (j < Ms)
                    if bmask.any():
                        colv = T[i, col + 1][pidx, :, Bk[:, j + 1]]                    # (P, 8 ra')
                        sh = colv[:, SHB]                                               # (P, 8 x, 8 ra)
                        cand = (sh[:, :, :, None] + KB[None, :, None, :]).min(axis=1)  # (P, 8 ra, 8 rb)
                        cand[~bmask] = INF
                        best = np.minimum(best, cand)
                best = np.minimum(best, INF)
                best[~valid] = INF
                T[i, col] = best
        root = T[0, bmax + 1][pidx, A[:, 0], Bk[:, 0]]
        c = np.where(root >= INF, -1, root >> SHIFT)
        ng = np.where(root >= INF, -1, root & ((1 << SHIFT) - 1))
        return np.stack([c, ng], axis=1)

    return align_costs


def make_compile_word_pair_best(G, align_costs=None):
    """compile_word_pair_best with the 21 alignments costed in one batched pass, then the scalar
    compile for the conjugation with the least T (ties: least DP gate count plus Clifford gates).
    The T-count equals the full search's; the gate-count tie-break may differ."""
    align_costs = align_costs or make_align_costs(G)

    def compile_word_pair_best(word0, word1):
        problems, meta = [], []
        for K in G.cliffords():
            Kinv = G.inverse(K)
            a, ph0 = G.skeleton(G.ma_normal_form(Kinv + word0 + K))
            b, ph1 = G.skeleton(G.ma_normal_form(Kinv + word1 + K))
            band = max(2, abs(len(a) - len(b)) + 1)
            problems.append((a, b, band))
            meta.append((K, Kinv))
        costs = align_costs(problems)
        if (costs[:, 0] < 0).any():                       # an INF alignment: use the original
            return _ORIG[0](word0, word1)
        extra = np.array([len(G.word_gates(Kinv, 0)) + len(G.word_gates(K, 0)) for K, Kinv in meta])
        order = np.lexsort((costs[:, 1] + extra, costs[:, 0]))
        K, Kinv = meta[order[0]]
        result = G.compile_word_pair(G.ma_normal_form(Kinv + word0 + K), G.ma_normal_form(Kinv + word1 + K))
        pre = [(name, ("q",)) for name, _ in G.word_gates(Kinv, 0)]
        post = [(name, ("q",)) for name, _ in G.word_gates(K, 0)]
        result["gates"] = pre + result["gates"] + post
        return result

    _ORIG[0] = G.compile_word_pair_best
    return compile_word_pair_best


_ORIG = [None]
''',
            'fast_factor': r'''"""Batched replacement for gold_local.factorize (the 15-mask parity factorisation).

Same continuation, Newton iteration, forward-difference Jacobian (h = 1e-7) and tolerances as the
original; the 1 + 3q residual evaluations of each Jacobian are done in one numpy pass instead of
1 + 3q Python loops over all patterns and masks. install(gold_local) swaps it in."""

import numpy as np

PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)


def _exp(v):
    """su2_exp on (..., 3) arrays: cos|v| I - i sin|v| (n . sigma)."""
    th = np.linalg.norm(v, axis=-1)
    safe = np.where(th < 1e-300, 1.0, th)
    n = v / safe[..., None]
    out = np.einsum("...k,kab->...ab", -1j * np.sin(th)[..., None] * n, PAULI)
    out[..., 0, 0] += np.cos(th)
    out[..., 1, 1] += np.cos(th)
    small = th < 1e-300
    if small.any():
        out[small] = np.eye(2)
    return out


def _log(U):
    """su2_log on (..., 2, 2) arrays (principal branch, as gold_local.su2_log)."""
    a = np.real(U[..., 0, 0] + U[..., 1, 1]) / 2
    v = np.stack([-np.imag(U[..., 0, 1] + U[..., 1, 0]) / 2,
                  np.real(U[..., 1, 0] - U[..., 0, 1]) / 2,
                  -np.imag(U[..., 0, 0] - U[..., 1, 1]) / 2], axis=-1)
    s = np.linalg.norm(v, axis=-1)
    th = np.arctan2(s, a)
    scale = np.where(s < 1e-300, 0.0, th / np.where(s < 1e-300, 1.0, s))
    return v * scale[..., None]


def make_factorize(G):
    def factorize(V, max_steps=400, tol=1e-13):
        n = V.shape[0]
        c = int(round(np.log2(n)))
        q = n - 1
        P = G.parities(c)
        R = V[0].copy()
        Rinv = R.conj().T
        L = np.array([G.su2_log(Rinv @ V[x]) for x in range(n)])
        sel = [np.nonzero(P[x])[0] for x in range(n)]

        def resid_many(TH, s):
            """TH: (m, q, 3) -> residuals (m, 3 (n-1)), as gold_local's resid for each row."""
            betas = _exp(TH)                                      # (m, q, 2, 2)
            Tgt_h = np.conj(np.swapaxes(_exp(s * L[1:]), -1, -2))  # (n-1, 2, 2)
            out = np.empty((TH.shape[0], n - 1, 3))
            for x in range(1, n):
                M = None
                for j in sel[x]:
                    M = betas[:, j] if M is None else M @ betas[:, j]
                if M is None:
                    M = np.broadcast_to(np.eye(2, dtype=complex), (TH.shape[0], 2, 2))
                out[:, x - 1] = _log(Tgt_h[x - 1] @ M)
            return out.reshape(TH.shape[0], -1)

        def jac(th, s, h=1e-7):
            flat = th.ravel()
            TH = np.repeat(flat[None, :], flat.size + 1, axis=0)
            TH[np.arange(1, flat.size + 1), np.arange(flat.size)] += h
            F = resid_many(TH.reshape(-1, q, 3), s)
            f0 = F[0]
            return f0, ((F[1:] - f0) / h).T

        th = np.zeros((q, 3))
        s = 0.0
        ds = 0.05
        while s < 1.0 - 1e-15:
            s_new = min(1.0, s + ds)
            th_try = th.copy()
            ok = False
            for it in range(30):
                f, J = jac(th_try, s_new)
                if np.abs(f).max() < (tol if s_new == 1.0 else 1e-10):
                    ok = True
                    break
                try:
                    step = np.linalg.solve(J, -f)
                except np.linalg.LinAlgError:
                    break
                th_try = th_try + step.reshape(q, 3)
                if np.abs(step).max() > 1.0:
                    break
            if ok:
                th, s = th_try, s_new
                ds = min(0.25, ds * 1.5)
            else:
                ds /= 2
                if ds < 1e-6:
                    raise RuntimeError("parity factorisation continuation failed")
        betas = [G.su2_exp(t) for t in th]
        return R, betas

    return factorize


def install(G):
    G.factorize = make_factorize(G)
''',
            'fast_phase': r'''"""Batched replacement for phase_factorization.factorize_phase (11 masks + output-Z carry).

Same control flow as the original: target-derived RNG seed, two attempts sharing the time
allowance, continuation schedule, Newton steps by lstsq on the analytic Jacobian, back-tracking
line search, tolerances and the final 1e-12 reconstruction check. Only the per-block and per-
factor loops are batched (SU(2) logs of the 16 blocks, the 12 exponentials of an update, the 12
adjoint maps of the Jacobian), so results agree with the original up to floating-point rounding.
install(phase_factorization) swaps it in."""

import hashlib
import time

import numpy as np

_PF = None
_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
_EYE = np.eye(2, dtype=complex)


def _exp_many(vectors):
    """(k, 3) -> (k, 2, 2), as phase_factorization._exp row by row."""
    angle = np.linalg.norm(vectors, axis=1)
    out = np.einsum("ka,aij->kij", -1j * np.sinc(angle / np.pi)[:, None] * vectors, _PAULI)
    out += np.cos(angle)[:, None, None] * _EYE
    return out


def _log_many(matrices):
    """(k, 2, 2) -> (k, 3), as phase_factorization._log row by row (raises at -I)."""
    scalar = np.einsum("kii->k", matrices).real / 2
    vector = (0.5j * np.einsum("aij,kji->ka", _PAULI, matrices)).real
    sine = np.linalg.norm(vector, axis=1)
    tiny = sine < 1e-300
    if tiny.any() and (scalar[tiny] < 0).any():
        raise _PF.PhaseFactorizationError("ambiguous SU(2) logarithm at -I")
    safe = np.where(tiny, 1.0, sine)
    return np.where(tiny[:, None], 0.0, vector * (np.arctan2(sine, scalar) / safe)[:, None])


def _adjoint_many(matrices):
    """(..., 2, 2) -> (..., 3, 3): result[..., a, b] = 0.5 Re tr(P_a M^dag P_b M)."""
    image = np.einsum("...ji,bjk,...kl->...bil", matrices.conj(), _PAULI, matrices)
    return (0.5 * np.einsum("aij,...bji->...ab", _PAULI, image)).real


def _jacobian(state):
    PF = _PF
    common, betas, _ = state
    sel = PF._SELECTED
    suffixes = np.empty((12, 16, 2, 2), dtype=complex)
    suffix = np.tile(_EYE, (16, 1, 1))
    for j in range(10, -1, -1):
        suffixes[j + 1] = suffix
        suffix = np.where(sel[:, j, None, None], betas[j] @ suffix, suffix)
    suffixes[0] = suffix
    adj = _adjoint_many(suffixes)                              # (12, 16, 3, 3)
    result = np.zeros((16, 3, 52))
    result[:, :, :3] = adj[0]
    for j in range(11):
        result[:, :, 3 + 3 * j:6 + 3 * j] = adj[j + 1] * sel[:, j, None, None]
    z_axes = _adjoint_many(common @ suffix)[:, :, 2]
    result[np.arange(16), :, 36 + np.arange(16)] = z_axes
    return result.reshape(48, 52)


def _update(state, step, scale):
    common, betas, phases = state
    E = _exp_many(scale * step[:36].reshape(12, 3))
    return (common @ E[0], [beta @ E[j + 1] for j, beta in enumerate(betas)], phases + scale * step[36:])


def _residual(state, target):
    relative = target.conj().transpose(0, 2, 1) @ _PF.reconstruct(*state)
    return _log_many(relative).reshape(-1)


def _attempt(target, rng, deadline):
    PF = _PF
    state = (PF._exp(rng.normal(size=3)), [PF._exp(rng.normal(size=3)) for _ in PF.MASKS],
             rng.normal(size=16))
    start = PF.reconstruct(*state)
    directions = _log_many(np.einsum("kji,kjl->kil", start.conj(), target))
    position, ds, accepted = 0.0, 0.05, 0
    while position < 1 - 1e-15:
        if time.monotonic() > deadline or accepted >= 128:
            raise PF.PhaseFactorizationError("11-mask continuation work limit")
        next_position = min(1.0, position + ds)
        stage = start @ _exp_many(next_position * directions)
        trial, converged = state, False
        for _ in range(30):
            if time.monotonic() > deadline:
                raise PF.PhaseFactorizationError("11-mask continuation time limit")
            r = _residual(trial, stage)
            if np.max(abs(r)) < 3e-14:
                converged = True
                break
            step = np.linalg.lstsq(_jacobian(trial), -r, rcond=1e-11)[0]
            length = np.linalg.norm(step[:36].reshape(12, 3), axis=1).max()
            scale = min(1.0, 0.5 / max(length, 1e-300))
            rn = np.linalg.norm(r)
            for _ in range(10):
                candidate = _update(trial, step, scale)
                if np.linalg.norm(_residual(candidate, stage)) < rn:
                    trial = candidate
                    break
                scale *= 0.5
            else:
                break
        if converged:
            state, position = trial, next_position
            accepted += 1
            ds = min(0.15, ds * 1.3)
        else:
            ds *= 0.5
            if ds < 1e-5:
                raise PF.PhaseFactorizationError("11-mask continuation stalled")
    return state


def factorize_phase(target, *, attempts=2, max_seconds=1e9):
    """As phase_factorization.factorize_phase, but by default without the 5 s wall-clock limit:
    with it, a slow but convergent attempt (some take ~4 s) can fall back to 15 masks under
    load, making runs nondeterministic. The work limit (128 accepted continuation steps) stays."""
    PF = _PF
    target = np.ascontiguousarray(target, dtype=np.complex128)
    if target.shape != (16, 2, 2) or not np.all(np.isfinite(target)):
        raise ValueError("11-mask factorization needs 16 finite 2x2 blocks")
    seed = int.from_bytes(hashlib.sha256(target.tobytes()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + max_seconds
    reason = "no attempts"
    for _ in range(attempts):
        try:
            state = _attempt(target, rng, deadline)
            error = max(float(np.linalg.norm(a - b, 2)) for a, b in zip(PF.reconstruct(*state), target))
            if not np.isfinite(error) or error > 1e-12:
                raise PF.PhaseFactorizationError(f"11-mask reconstruction error {error:.2e}")
            return (*state, error)
        except (PF.PhaseFactorizationError, np.linalg.LinAlgError) as exc:
            reason = str(exc)
        if time.monotonic() > deadline:
            break
    raise PF.PhaseFactorizationError(reason)


def install(phase_factorization):
    global _PF
    _PF = phase_factorization
    phase_factorization.factorize_phase = factorize_phase
''',
            'decoupled': r'''"""Decoupled multiplexors: same construction as gold_local.synthesize, but no multiplexor waits on
another, so all of them can be synthesised in parallel.

In gold_local each multiplexor passes a carry phase to the next one:
    carry[x, tb] = phi[x] + ctrl_phase[x] + (2 tb - 1) output_z[x].
phi and output_z depend only on the blocks (continuous, computed up front here). ctrl_phase depends
on the synthesised words: it is a sum over masks j of corr_j * parity_j(x), with
corr_j = (multiple of pi/4) - gamma_j and gamma_j the word's global phase, which for an even-T
word is a multiple of pi/4 up to O(eps). Here each corr_j is rounded to m_j * pi/4 and applied in
place, right after its multiplexor, as a parity-controlled T^m_j (at most 1 T). The O(eps)
remainder is dropped, which adds an error of the same order as the word error.
"""

import math

import numpy as np

_G = None  # gold_local, set by init


def init(gold_local):
    global _G
    _G = gold_local




def _parity_phase_gates(mask, m, controls):
    """Phase omega^m on control patterns with odd parity of (x & mask): CNOT ladder, T^m, unladder."""
    c = len(controls)
    mem = [controls[i] for i in range(c) if (mask >> (c - 1 - i)) & 1]
    pw = mem[-1]
    ladder = [("CNOT", (o, pw)) for o in mem[:-1]]
    core = []
    m %= 8
    if m & 1:
        core.append(("T", (pw,)))
    if m & 2:
        core.append(("S", (pw,)))
    if m & 4:
        core.append(("Z", (pw,)))
    return ladder + core + ladder[::-1] if core else []


def plan(u_mat, cs_decomposition, mux_method="phase11"):
    """Pass 1 (cheap, sequential): the multiplexor targets with continuous carries only."""
    G = _G
    items, n = G.csd_items(u_mat, cs_decomposition)
    N = 2 ** n
    carry = np.zeros(N)
    jobs = []
    for item_index, it in enumerate(items):
        if it[0] == "G":
            t = n - 1
            blocks = list(it[1])
        else:
            t = it[1]
            blocks = [G.Ry(a) for a in it[2]]
        controls = [w for w in range(n) if w != t]
        c = len(controls)
        V = np.zeros((2 ** c, 2, 2), complex)
        phi = np.zeros(2 ** c)
        for x in range(2 ** c):
            i0 = G.full_index(n, t, controls, x, 0)
            i1 = G.full_index(n, t, controls, x, 1)
            B = blocks[x] @ np.diag([np.exp(1j * carry[i0]), np.exp(1j * carry[i1])])
            phi[x] = np.angle(np.linalg.det(B)) / 2
            V[x] = B / np.exp(1j * phi[x])
        allow_output_z = item_index < len(items) - 1
        factored = G.factor_mux_blocks(V, mux_method, allow_output_z)
        output_z = factored[3]
        jobs.append({"V": V, "t": t, "controls": controls, "allow_output_z": allow_output_z,
                     "factored": factored})
        carry = np.zeros(N)
        for x in range(2 ** c):
            for tb in (0, 1):
                carry[G.full_index(n, t, controls, x, tb)] = phi[x] + (2 * tb - 1) * output_z[x]
    return jobs, carry, n


def assemble(jobs, results, final_carry, n, round_tol=1e-6):
    """Pass 3: gates in time order, with the in-place discrete control phases."""
    G = _G
    gates, gph = [], 0.0
    worst_remainder = 0.0
    for job, res in zip(jobs, results):
        g, ph = G.mux_gates(res, job["t"], job["controls"])
        gates += g
        gph += ph
        for mask, corr in res["terms"]:
            m = round(corr / (np.pi / 4))
            rem = corr - m * np.pi / 4
            worst_remainder = max(worst_remainder, abs(rem))
            if abs(rem) > round_tol:
                raise RuntimeError(f"control correction {corr} is not near a multiple of pi/4")
            gates += _parity_phase_gates(mask, m, job["controls"])
    gph += G.diagonal_gates(final_carry, list(range(n)), gates)
    return gates, gph, worst_remainder




_SYNTH_FACTORY = None












# ---- lockstep variant: all multiplexors' word searches share batched lattice calls ----------

def mux_coroutine(V, eps, mux_method, fold_common, allow_output_z, factored=None):
    """synth_mux_terms as a generator: yields lists of targets needing even-T words, receives the
    words (same order), returns the same result dict. The arithmetic is synth_mux_terms'."""
    G = _G
    n = V.shape[0]
    c = int(round(np.log2(n)))
    if factored is None:
        factored = G.factor_mux_blocks(V, mux_method, allow_output_z)
    R, betas, masks, output_z, factor_error, method = factored
    q = len(masks)
    Gm = np.eye(2, dtype=complex)
    pairs_rev, terms = [], []
    error_bound = factor_error
    folded = fold_common and q > 0
    for j in range(q - 1, 0 if folded else -1, -1):
        W = Gm @ betas[j] @ Gm.conj().T
        (w,) = yield [W]
        gam = G.phase_of(w, W)
        error_bound += G.word_error(w, W)
        res = G.compile_pair_best(w)
        terms.append((masks[j], res["ctrl_phase"] * np.pi / 4 - gam))
        pairs_rev.append((masks[j], res["gates"]))
        Gm = G.mat(res["A"]) @ Gm
    if folded:
        target0 = R @ Gm.conj().T
        target1 = R @ betas[0] @ Gm.conj().T
        word0, word1 = yield [target0, target1]
        gamma0, gamma1 = G.phase_of(word0, target0), G.phase_of(word1, target1)
        last = G.compile_word_pair_best(word0, word1)
        terms.append((masks[0], last["ctrl_phase"] * np.pi / 4 - gamma1 + gamma0))
        pairs_rev.append((masks[0], last["gates"]))
        global_phase = last["global"] - gamma0
        error_bound += max(G.word_error(word0, target0), G.word_error(word1, target1))
        wc = ""
    else:
        Cmat = R @ Gm.conj().T
        (wc,) = yield [Cmat]           # (an even word; synth_mux_terms uses synth(), any parity)
        global_phase = -G.phase_of(wc, Cmat)
        error_bound += G.word_error(wc, Cmat)
    return {"pairs": pairs_rev, "C": wc, "global": global_phase, "c": c, "output_z": output_z,
            "terms": terms, "error_bound": error_bound, "factorization": method}


def synthesize_lockstep(u_mat, cs_decomposition, eps, engine, mux_method="phase11", fold_common=True,
                        optimize=True):
    """decoupled synthesize with every multiplexor's word requests served by one LockstepWords
    engine (pysynth.synth), so each round batches one chunk from every pending word."""
    G = _G
    jobs, final_carry, n = plan(np.array(u_mat, dtype=complex), cs_decomposition, mux_method)
    gens = [mux_coroutine(j["V"], eps, mux_method, fold_common, j["allow_output_z"], j["factored"])
            for j in jobs]
    results = [None] * len(jobs)
    pending = {}                                   # mux index -> [words or None]

    targets_of = {}

    def submit(i, targets):
        pending[i] = [None] * len(targets)
        targets_of[i] = targets
        for slot, W in enumerate(targets):
            engine.add(W, (i, slot))

    for i, g in enumerate(gens):
        submit(i, next(g))
    while pending:
        for (i, slot), word in engine.run_round():
            if G.word_error(word, targets_of[i][slot]) > eps * (1 + 1e-6):
                raise RuntimeError("lockstep word exceeds eps")
            pending[i][slot] = word
            if all(w is not None for w in pending[i]):
                words = pending.pop(i)
                try:
                    submit(i, gens[i].send(words))
                except StopIteration as stop:
                    results[i] = stop.value
    gates, gph, worst = assemble(jobs, results, final_carry, n)
    if optimize:
        gates, gph = G.optimize_circuit(gates, gph)
    return gates, gph, worst
''',
        }
        modules = {}
        for name in ['zomega', 'normal_form', 'lattice2', 'batch', 'batch_coset', 'synth', 'phase_factorization', 'gate_cleanup', 'gold_local', 'fast_pairs', 'batched_align', 'fast_factor', 'fast_phase', 'decoupled']:
            module = types.ModuleType("_qhg_" + name)
            sys.modules[module.__name__] = module
            exec(compile(sources[name], "<submission:" + name + ">", "exec"), module.__dict__)
            modules[name] = module
        gold = modules["gold_local"]
        modules["decoupled"].init(gold)
        gold.align = modules["fast_pairs"].make_align(gold)
        gold.ma_normal_form = modules["normal_form"].ma_normal_form
        gold.compile_pair_best = modules["fast_pairs"].make_compile_pair_early(gold)
        modules["fast_factor"].install(gold)
        modules["fast_phase"].install(modules["phase_factorization"])
        gold.compile_word_pair_best = modules["batched_align"].make_compile_word_pair_best(gold)
        state["modules"] = modules
    modules = state["modules"]

    # Recursive cosine-sine decomposition into 31 multiplexors (n = 5), each factored into parity
    # pieces paired as diag(A, A W); every W is an even-T Clifford+T word within eps (near
    # T-optimal lattice search), all multiplexors' word searches batched together. Plain NumPy
    # arrays throughout (pennylane.numpy tensors only add per-operation overhead here).
    import numpy

    def plain_csd(mat):
        return tuple(numpy.asarray(part) for part in cs_decomposition(mat))

    eps = 1e-9
    engine = modules["synth"].LockstepWords(eps, pairs=True)
    gates, global_phase, _ = modules["decoupled"].synthesize_lockstep(numpy.asarray(u_mat), plain_csd, eps, engine)
    single = {"H": qp.Hadamard, "S": qp.S, "T": qp.T, "X": qp.PauliX, "Y": qp.PauliY, "Z": qp.PauliZ}
    for gate in gates:
        name, wires = gate[0], gate[1]
        if name == "CNOT":
            qp.CNOT(wires=[int(w) for w in wires])
        elif name == "RZ":
            qp.RZ(float(gate[2]), wires=int(wires[0]))
        else:
            single[name](wires=int(wires[0]))
    qp.GlobalPhase(-float(global_phase))


# These functions are responsible for testing the solution.
def _measure(seed, num_samples):
    """Mean cost and auxiliary wires per size, and the largest error. Raises ValueError on
    a circuit the grader cannot accept (illegal gate, too many gates, too many wires)."""
    costs, aux_by_size, max_error = [], [], 0.0
    for num_qubits in TEST_SIZES:
        total, size_aux = 0, 0
        for sample in range(num_samples):
            sample_seed = seed + 1000 * num_qubits + sample
            try:
                tape = compile_circuit(submission_circ, haar_unitary(num_qubits, sample_seed))
            except IllegalGateError as exc:
                raise ValueError(f"Illegal gate. {exc}") from None
            if len(tape.operations) > MAX_GATES:
                raise ValueError(
                    f"Too many gates: {len(tape.operations):,} at {num_qubits} qubits, and "
                    f"the limit is {MAX_GATES:,}."
                )
            total_wires, aux_wires = wire_usage(tape, num_qubits)
            if total_wires > MAX_TOTAL_WIRES:  # rejected, and never simulated
                raise ValueError(
                    f"Too many wires: {total_wires} at {num_qubits} qubits, and the limit is "
                    f"{MAX_TOTAL_WIRES}."
                )
            size_aux = max(size_aux, aux_wires)
            total += tape_cost(tape, num_qubits)
            u_mat = haar_unitary(num_qubits, sample_seed)
            matrix = tape_matrix(tape, num_qubits)
            if not np.all(np.isfinite(matrix)):
                raise ValueError(
                    f"Not accurate: the matrix of your circuit at {num_qubits} qubits has NaN "
                    "or infinite entries. Check your rotation angles."
                )
            max_error = max(max_error, float(np.abs(matrix - u_mat).max()))
        costs.append(total / num_samples)
        aux_by_size.append(size_aux)
    return costs, aux_by_size, max_error


def _percent(value):
    """A percentage rounded down to one decimal, so the display never overstates a saving."""
    return f"{np.floor(10 * value) / 10:.1f}%"


def _shown(saving, cost, base):
    """A saving for display; a circuit at or past twice the baseline shows its multiple."""
    return _percent(saving) if saving > -100 else f"none, {cost / base:.1f}x the baseline"


def run(test_case_input: str) -> str:
    seed, num_samples = json.loads(test_case_input)
    try:
        costs, aux_by_size, max_error = _measure(seed, num_samples)
    except ValueError as exc:
        return json.dumps({"grade": "None", "reason": str(exc)})

    baseline = [baseline_cost(n) for n in TEST_SIZES]
    savings = [100 * (1 - c / b) for c, b in zip(costs, baseline)]
    saving = savings[-1]
    over = [(n, c, b, a) for n, c, b, a in zip(TEST_SIZES, costs, baseline, aux_by_size)
            if c > b]
    shrinks = [i for i in range(1, len(TEST_SIZES)) if savings[i] < savings[i - 1] - 1e-9]
    if max_error > ATOL:
        reason = (f"Not accurate: the largest entrywise error is {max_error:.1e}, above "
                  f"{ATOL:.0e}. Check the global phase, and that every auxiliary wire comes "
                  "back to |0>.")
    elif over:
        n, c, b, a = over[0]
        reason = f"Over the baseline at {n} qubits: cost {c:,.0f} against {b:,}."
        if a:
            reason += f" Your {a} auxiliary wire(s) multiply the cost by {AUX_WIRE_FACTOR ** a}."
    elif shrinks:
        i = shrinks[0]
        small, large = TEST_SIZES[i - 1], TEST_SIZES[i]
        reason = (f"Does not scale: you save {savings[i - 1]:.2f}% at {small} qubits but only "
                  f"{savings[i]:.2f}% at {large}. Your cost grows by a factor of "
                  f"{costs[i] / costs[i - 1]:.3f} from {small} to {large} qubits, and the "
                  f"baseline's by {baseline[i] / baseline[i - 1]:.3f}.")
    else:
        reason = None

    valid = reason is None
    reached = [g for g, need in GRADES.items() if valid and saving >= need]
    grade = max(reached, key=GRADES.get) if reached else "None"
    if valid and not reached:
        reason = (f"Valid, but below Bronze: it saves {_percent(saving)} against the baseline "
                  f"at {TEST_SIZES[-1]} qubits.")
    report = {"grade": grade}
    if reason:
        report["reason"] = reason
    accurate = max_error <= ATOL
    report["saving_at_%d_qubits" % TEST_SIZES[-1]] = (
        _shown(saving, costs[-1], baseline[-1]) if accurate
        else "not graded, the circuit is not accurate"
    )
    better = [(g, need) for g, need in GRADES.items() if need > saving]
    if better and valid:
        nxt, need = min(better, key=lambda item: item[1])
        report["next_grade"] = f"{nxt} needs {need:g}%"
    if accurate:
        report["saving_by_size"] = [_shown(s, c, b) for s, c, b in zip(savings, costs, baseline)]
    report["cost"] = [round(c, 1) if c % 1 else int(c) for c in costs]
    report["baseline"] = baseline
    report["max_error"] = f"{max_error:.1e}"
    report["aux_wires"] = max(aux_by_size)
    return json.dumps(report)


def check(solution_output: str, expected_output: str) -> None:
    report = json.loads(solution_output)
    required = json.loads(expected_output)["required_grade"]
    ranks = {"None": -1, **{g: i for i, g in enumerate(sorted(GRADES, key=GRADES.get))}}
    grade = report["grade"]
    if grade == "None":
        raise AssertionError(f"{required} needed, and your circuit has no grade. "
                             + report["reason"])
    key = "saving_at_%d_qubits" % TEST_SIZES[-1]
    assert ranks[grade] >= ranks[required], (
        f"{required} needed, you reached {grade}: you save {report[key]} against the "
        f"baseline at {TEST_SIZES[-1]} qubits, and {required} needs {GRADES[required]:g}%."
    )


# These are the public test cases
test_cases = [
    ('[2026, 2]', '{"required_grade": "Gold"}'),
    ('[7, 2]', '{"required_grade": "Gold"}')
]

# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")