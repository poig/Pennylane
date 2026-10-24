import json
import pennylane as qp
import pennylane.numpy as np

def same_gate_and_phase(u_matrix, v_matrix):
    """Decides whether two unitaries are the same gate, and exposes the phase.

    Args:
        u_matrix (np.ndarray): A 2x2 unitary matrix with complex entries.
        v_matrix (np.ndarray): A 2x2 unitary matrix with complex entries.

    Returns:
        tuple[bool, float, float]: Whether v_matrix equals u_matrix up to a
            global phase; the phase alpha in [0, 2*pi) if so and 0.0 if not;
            and the probability of measuring 1 on the control wire of the
            phase kickback circuit described in the problem statement.
    """
    dev = qp.device("default.qubit", wires=2)


    # Step 1: form M = V U^dagger. 
    u_dagger = u_matrix.conj().T
    M = np.array(v_matrix @ u_dagger, dtype=complex)
    
    same_gate = bool(np.allclose(M, M[0, 0] * np.eye(2), atol=1e-5))
    # Step 2: read alpha off the leading entry of M with np.angle, and fold it
    #         into [0, 2*pi). A bare np.angle returns (-pi, pi] and will fail the tests.
    if same_gate:
        alpha = float(np.angle(M[0, 0])) % (2 * np.pi)
        if np.isclose(alpha, 2 * np.pi, atol=1e-7):
            alpha = 0.0
    else:
        alpha = 0.0
    # Step 3: build the phase kickback circuit. 
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(wires=0)
        qp.ctrl(qp.QubitUnitary(M, wires=1), control=0)
        qp.Hadamard(wires=0)
        return qp.probs(wires=0)
    # Put your code here #
    probs = circuit()
    p_one = float(probs[1])

    return same_gate, alpha, p_one


# These functions are responsible for testing the solution.
def to_matrix(rows):
    """Turns a 2x2 nested list of [real, imaginary] pairs into a complex matrix."""
    return np.array([[complex(entry[0], entry[1]) for entry in row] for row in rows])


AUDIT = {"tapes": []}
_real_device = qp.device
STATE_READOUTS = ("StateMP", "DensityMatrixMP", "PurityMP", "VnEntropyMP", "MutualInfoMP")


def audited_device(*args, **kwargs):
    """Wraps a device so the grader can see which circuits were actually run."""
    dev = _real_device(*args, **kwargs)
    inner = dev.execute

    def execute(circuits, *args, **kwargs):
        batch = [circuits] if isinstance(circuits, qp.tape.QuantumScript) else circuits
        for tape in batch:
            AUDIT["tapes"].append(tape)
        return inner(circuits, *args, **kwargs)

    dev.execute = execute
    return dev


def violation(p_one):
    """Returns a message if the reported probability did not come from the circuit."""
    tapes = AUDIT["tapes"]
    if len(tapes) != 1:
        return ("Expected exactly one circuit execution, saw " + str(len(tapes))
                + ". Part 3 asks for the probability read off the phase kickback "
                "circuit, so that circuit has to be built and run.")
    tape = tapes[0]
    measurements = tape.measurements
    for m in measurements:
        if type(m).__name__ in STATE_READOUTS:
            return ("Disallowed measurement: " + type(m).__name__ + ". Read the "
                    "control wire with qp.probs, not the full state.")
    if len(measurements) != 1 or type(measurements[0]) is not qp.measurements.ProbabilityMP:
        return ("Return exactly one qp.probs measurement on the control wire. "
                "Saw: " + str([type(m).__name__ for m in measurements]) + ".")
    if len(measurements[0].wires) != 1:
        return ("Measure qp.probs on the single control wire, not on "
                + str(len(measurements[0].wires)) + " wires.")
    control = measurements[0].wires[0]
    controlled = [op for op in tape.operations if len(op.control_wires) > 0]
    if len(controlled) == 0:
        return ("Your circuit contains no controlled operation. A global phase is "
                "only visible once the gate carrying it is controlled.")
    for op in controlled:
        if control not in list(op.control_wires):
            return ("The controlled operation " + type(op).__name__ + " is not "
                    "controlled on the wire you measured (wire " + str(control)
                    + "). The phase is kicked back onto the control wire.")
        targets = [w for w in op.wires if w not in list(op.control_wires)]
        if len(targets) != 1:
            return ("The controlled operation " + type(op).__name__ + " acts on "
                    + str(len(targets)) + " target wires. M is a single-qubit "
                    "unitary, so it has exactly one target.")
    replay = qp.tape.QuantumScript(tape.operations, tape.measurements)
    device = _real_device("default.qubit", wires=list(tape.wires))
    probs = np.array(qp.execute([replay], device)[0]).flatten()
    if not np.isclose(float(probs[1]), float(p_one), atol=1e-5):
        return ("The probability you returned is not the one your circuit produced. "
                "Your circuit gives " + str(round(float(probs[1]), 6)) + " for index 1.")
    return None


def inconsistent(same_gate, alpha, p_one):
    """Applies the free self-check from the problem statement: P(1) = sin^2(alpha/2)."""
    if not bool(same_gate):
        return None
    predicted = float(np.sin(float(alpha) / 2.0) ** 2)
    if np.isclose(predicted, float(p_one), atol=1e-5):
        return None
    return ("You reported the same gate with alpha = " + str(round(float(alpha), 6))
            + " and P(1) = " + str(round(float(p_one), 6)) + ", but when the two "
            "matrices are the same gate P(1) must equal sin^2(alpha/2) = "
            + str(round(predicted, 6)) + ". One of the two is wrong: check the fold "
            "of alpha into [0, 2*pi) and the order of the Hadamards in the circuit.")


def run(test_case_input: str) -> str:
    case = json.loads(test_case_input)
    u_matrix = to_matrix(case["u"])
    v_matrix = to_matrix(case["v"])
    AUDIT["tapes"] = []
    qp.device = audited_device
    try:
        same_gate, alpha, p_one = same_gate_and_phase(u_matrix, v_matrix)
    finally:
        qp.device = _real_device
    # Raised here rather than deferred to check(), so that a solution which rebinds
    # check() in the module namespace cannot swallow the verdict.
    message = violation(p_one) or inconsistent(same_gate, alpha, p_one)
    if message is not None:
        raise AssertionError(message)
    return json.dumps([bool(same_gate), float(alpha), float(p_one)])


def check(solution_output: str, expected_output: str) -> None:
    got = json.loads(solution_output)
    exp = json.loads(expected_output)
    assert bool(got[0]) == bool(exp[0]), \
        "Wrong verdict -- two matrices can be the same gate without being equal."
    assert np.isclose(got[1], exp[1], atol=1e-5), \
        "Wrong phase -- did you fold np.angle into [0, 2*pi)?"
    assert np.isclose(got[2], exp[2], atol=1e-5), \
        "Wrong control probability -- check the order of the Hadamards and that " \
        "you controlled M rather than V."


# These are the public test cases
test_cases = [
    ('{"u": [[[0.7071067811865475, 0.0], [0.7071067811865475, 0.0]], [[0.7071067811865475, 0.0], [-0.7071067811865475, 0.0]]], "v": [[[-0.7071067811865475, 0.0], [-0.7071067811865475, 0.0]], [[-0.7071067811865475, 0.0], [0.7071067811865475, 0.0]]]}', '[true, 3.14159265359, 1.0]'),
    ('{"u": [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]], "v": [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [-1.0, 0.0]]]}', '[false, 0.0, 0.5]')
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