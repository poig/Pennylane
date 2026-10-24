import json
import pennylane as qp
import pennylane.numpy as np

def wrapper(n: int) -> float:
    """Builds the n-qubit GHZ state and averages its n parity checks.

    Args:
        n (int): The number of qubits. Guaranteed to satisfy n >= 2.

    Returns:
        float: The mean of the n parity values -- the global X parity together with
            the n-1 neighbouring ZZ parities. This equals +1 if and only if the
            state is the GHZ state.
    """
    dev = qp.device("default.qubit", wires=n)

    @qp.qnode(dev)
    def parities():
        return [qp.expval(observable) for observable in prepare_and_build_checks(n)]

    return float(np.mean(np.array(parities())))


def prepare_and_build_checks(n):
    """Prepares the GHZ state and lists the parity checks to measure on it.

    This runs inside the QNode above, so any gate you apply here is applied to the
    circuit, and the observables you return are the ones that get measured.

    Args:
        n (int): the number of qubits, guaranteed to be at least 2.

    Returns:
        list[Operator]: the n observables to measure -- the global X parity and the
            n-1 neighbouring ZZ parities. The order does not matter.
    """
    # Step 1: build the GHZ state -- a Hadamard on wire 0, then a CNOT fan-out
    #         from wire 0 to every other wire.
    # Step 2: build the global X parity: an X on every wire, multiplied together
    #         into a single observable with qp.prod and assign it 'checks'.
    # Step 3: build the n-1 neighbouring ZZ parities, one for each adjacent pair,
    #         indexed by the left-hand qubit of the pair and add it to 'checks'.
    # Step 4: collect all n observables into checks and return it.


    # Put your code here.
    qp.Hadamard(wires=0)
    for i in range(1, n):
        qp.CNOT(wires=[0, i])

    checks = [qp.prod(*[qp.PauliX(i) for i in range(n)])]

    for i in range(n - 1):
        checks.append(qp.prod(qp.PauliZ(i), qp.PauliZ(i + 1)))
    return checks


# These functions are responsible for testing the solution.
AUDIT = {"tapes": []}
_real_device = qp.device


def audited_device(*args, **kwargs):
    """Wraps a device so the grader can see which observables were measured."""
    dev = _real_device(*args, **kwargs)
    inner = dev.execute

    def execute(circuits, *args, **kwargs):
        batch = [circuits] if isinstance(circuits, qp.tape.QuantumScript) else circuits
        for tape in batch:
            AUDIT["tapes"].append(tape)
        return inner(circuits, *args, **kwargs)

    dev.execute = execute
    return dev


def pauli_key(observable):
    """Reduces an observable to a canonical Pauli-word string, or None."""
    try:
        sentence = qp.pauli.pauli_sentence(observable)
    except Exception:
        return None
    terms = list(sentence.items())
    if len(terms) != 1:
        return None
    word, coeff = terms[0]
    if not np.isclose(complex(coeff), 1.0):
        return None
    if len(word) == 0:
        return "identity"
    return "".join(str(w) + word[w] for w in sorted(word))


def expected_keys(n):
    """The canonical keys of the n checks the challenge asks for."""
    keys = ["".join(str(w) + "X" for w in range(n))]
    keys += [str(i) + "Z" + str(i + 1) + "Z" for i in range(n - 1)]
    return sorted(keys)


def violation(n):
    """Returns a message if the checks measured are not the ones asked for."""
    tapes = AUDIT["tapes"]
    if len(tapes) != 1:
        return ("Expected exactly one circuit execution, saw " + str(len(tapes))
                + ". All n checks are measured on the single QNode provided.")
    measurements = tapes[0].measurements
    for m in measurements:
        if type(m) is not qp.measurements.ExpectationMP:
            return ("Disallowed measurement: " + type(m).__name__ + ". Every check "
                    "must be the expectation value of an observable.")
    if len(measurements) != n:
        return ("You returned " + str(len(measurements)) + " checks for n = "
                + str(n) + ". There are exactly n of them: one global X parity "
                "and n-1 neighbouring ZZ parities.")
    keys = [pauli_key(m.obs) for m in measurements]
    if None in keys:
        return ("One of your checks is not a single Pauli word with coefficient 1. "
                "Each check is a plain product of Pauli operators.")
    if "identity" in keys:
        return ("One of your checks is the identity. Every state passes it, so it "
                "certifies nothing.")
    if sorted(keys) != expected_keys(n):
        return ("Your checks reduce to " + str(sorted(keys)) + " but the n asked "
                "for are " + str(expected_keys(n)) + ": an X on every wire, and a "
                "Z Z on each adjacent pair (i, i+1).")
    return None


NOT_GHZ = (
    "Not the GHZ state. A Z-basis histogram alone cannot see the X parity, and an "
    "X parity alone cannot see the Z-basis structure -- you need both."
)


def run(test_case_input: str) -> str:
    n = json.loads(test_case_input)
    AUDIT["tapes"] = []
    qp.device = audited_device
    try:
        value = wrapper(n)
    finally:
        qp.device = _real_device
    # Both verdicts are raised here rather than deferred to check(), so that they
    # cannot be bypassed by code that rebinds check() in the module namespace.
    message = violation(n)
    if message is not None:
        raise AssertionError(message)
    if not np.isclose(value, 1.0, atol=1e-5):
        raise AssertionError(NOT_GHZ)
    return json.dumps(value)


def check(solution_output: str, expected_output: str) -> None:
    sol = json.loads(solution_output)
    exp = json.loads(expected_output)
    assert np.isclose(sol, exp, atol=1e-5), NOT_GHZ


# These are the public test cases
test_cases = [
    ('3', '1.0'),
    ('5', '1.0')
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