import json
import pennylane as qp
import pennylane.numpy as np

# A three-qubit state is prepared for you. You do not need to change this.
def prepare(angles):
    """Prepares the state whose Pauli expectation values you have to measure.

    Args:
        angles (list[float]): four rotation angles.
    """
    qp.RY(angles[0], wires=0)
    qp.RX(angles[1], wires=1)
    qp.RY(angles[2], wires=2)
    qp.CNOT(wires=[0, 1])
    qp.RZ(angles[3], wires=1)
    qp.CNOT(wires=[1, 2])
    qp.RX(angles[3], wires=0)


def measure_pauli_word(word, angles):
    """Measures a Pauli word on the prepared state using only computational-basis
    probabilities.

    Args:
        word (str): three characters, each one of "I", "X", "Y", "Z". Character w
            is the Pauli factor acting on wire w.
        angles (list[float]): the four angles handed to prepare().

    Returns:
        tuple[list[str], list[int], float]:
            - the basis change applied to each wire, "none", "H" or "SH";
            - the eight signs, one per computational-basis outcome;
            - the expectation value of the Pauli word.
    """


    # --- Step 1: pick a basis change for each of the three wires -------------
    # Each letter maps to exactly one basis change:
    #     "I" -> "none"     "Z" -> "none"     "X" -> "H"     "Y" -> "SH"
    # Hint: A dictionary is the tidiest way to write a fixed lookup like this one.
    letter_to_basis = {"I": "none", "Z": "none", "X": "H", "Y": "SH"}
    basis = [letter_to_basis[letter] for letter in word]
    

    # --- Step 2: run one circuit and read the eight probabilities ------------
    dev = qp.device("default.qubit", wires=3)
    @qp.qnode(dev)
    def circuit():
        prepare(angles)
        for wire, change in enumerate(basis):
            if change == "H":
                qp.Hadamard(wires=wire)
            elif change == "SH":
                qp.adjoint(qp.S)(wires=wire)
                qp.Hadamard(wires=wire)
        return qp.probs(wires=[0, 1, 2])

    probabilities = circuit()
    # --- Step 3: work out the sign of each of the eight outcomes -------------
    active_wires = [wire for wire, letter in enumerate(word) if letter != "I"]
    signs = [
        int((-1) ** sum((outcome >> (2 - wire)) & 1 for wire in active_wires))
        for outcome in range(8)
    ]
    # --- Step 4: combine them into the expectation value ---------------------
    value = float(sum(p * s for p, s in zip(probabilities, signs)))
    return basis, signs, value


# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    loads, dumps = json.loads, json.dumps
    make_device, script = qp.device, qp.tape.QuantumScript
    measure_probs, execute_tapes = qp.probs, qp.execute
    matrix_of, build_reference = qp.matrix, qp.tape.make_qscript
    allowed_basis = ("none", "H", "SH")
    word, angles = loads(test_case_input)
    log = []
    tally = {"recorded": 0}

    def watcher(real_execute):
        """Wraps one device class's execute so that every circuit is recorded."""

        def execute(self, circuits, *inner_args, **inner_kwargs):
            
            results = real_execute(self, circuits, *inner_args, **inner_kwargs)
            single = isinstance(circuits, script)
            for tape in ([circuits] if single else list(circuits)):
                for measurement in tape.measurements:
                    log.append((type(measurement).__name__, tape.shots.total_shots,
                                measurement.obs is not None, tuple(measurement.wires),
                                len(tape.measurements), list(tape.operations)))
                    tally["recorded"] += 1
            return results

        return execute


    watched = []
    for holder in [qp.plugin_devices, qp.devices.__dict__]:
        for name in sorted(holder):
            try:
                candidate = holder[name]
                candidate = candidate.load() if hasattr(candidate, "load") else candidate
            except Exception:
                continue
            if not isinstance(candidate, type) or "execute" not in vars(candidate):
                continue
            real_execute = vars(candidate)["execute"]
            if callable(real_execute) and not any(candidate is seen for seen, _ in watched):
                watched.append((candidate, real_execute))
    for device_class, real_execute in watched:
        device_class.execute = watcher(real_execute)
    try:
        returned = measure_pauli_word(word, angles)
    finally:
        for device_class, real_execute in watched:
            device_class.execute = real_execute

    try:
        basis, signs, value = returned
    except (TypeError, ValueError):
        raise AssertionError(
            "Return exactly three values: the list of basis changes, the list of "
            "eight signs, and the expectation value.")
    basis = [str(change) for change in basis]
    if len(basis) != 3 or any(change not in allowed_basis for change in basis):
        raise AssertionError(
            "basis has to be three entries, each one of \"none\", \"H\" or \"SH\".")
    signs = [int(sign) for sign in signs]
    if len(signs) != 8 or any(sign not in (1, -1) for sign in signs):
        raise AssertionError("signs has to be eight entries, each +1 or -1.")


    if len(log) != tally["recorded"]:
        raise AssertionError(
            "The record of executed circuits was written to from outside the "
            "simulator. Solve the problem instead.")
    entries = [entry for entry in log if isinstance(entry, tuple) and len(entry) == 6]
    forbidden = {"StateMP": "qp.state", "DensityMatrixMP": "qp.density_matrix",
                 "VnEntropyMP": "qp.vn_entropy", "PurityMP": "qp.purity"}
    for kind, _, _, _, _, _ in entries:
        if kind in forbidden:
            raise AssertionError(
                "This circuit returned " + forbidden[str(kind)] + ". The point of the "
                "exercise is to reach the answer from measurement outcomes, so the "
                "only measurement allowed is qp.probs.")
        if kind != "ProbabilityMP":
            raise AssertionError(
                "A circuit returned " + str(kind) + ". Return qp.probs(wires=[0, 1, 2]) "
                "and combine the probabilities yourself.")

  
    reference = list(build_reference(prepare)(angles).operations)
    pauli = {"X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
             "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
             "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)}

    def same(operation, expected):
        """Is this the operation prepare() applies at this point, angle included?"""
        if (operation.name != expected.name
                or tuple(operation.wires) != tuple(expected.wires)
                or len(operation.data) != len(expected.data)):
            return False
        return all(abs(complex(applied) - complex(wanted)) < 1e-9
                   for applied, wanted in zip(np.asarray(operation.data).ravel(),
                                              np.asarray(expected.data).ravel()))

    def measure(operations):
        """Runs a recorded circuit again and returns the grader's own probabilities."""
        tape = script(list(operations), [measure_probs(wires=[0, 1, 2])])
        return [float(value) for value in np.asarray(
            execute_tapes([tape], make_device("default.qubit", wires=3))[0]).ravel()]

    problem = (
        "No analytic qp.probs over all three wires was recorded on the prepared state. "
        "Build your device with qp.device(\"default.qubit\", wires=3), call "
        "prepare(angles) first, apply your basis changes and return "
        "qp.probs(wires=[0, 1, 2]) -- " + str(len(entries))
        + " execution(s) were recorded.")
    accepted = False
    for _, shots, has_observable, measured, count, operations in entries:
        # Anything that is not a circuit this grader recorded itself is ignored.
        if not isinstance(measured, tuple) or not isinstance(operations, list):
            continue
        if not all(hasattr(operation, "wires") for operation in operations):
            continue
        if shots is not None or has_observable or count != 1:
            continue
        if measured not in ((0, 1, 2), ()):
            continue
        if len(operations) < len(reference) or not all(
                same(operation, expected)
                for operation, expected in zip(operations, reference)):
            problem = (
                "This circuit does not start with the state prepare(angles) builds. "
                "Call prepare(angles) as the first thing in your circuit, with the "
                "angles you were given, and apply your basis changes after it.")
            continue

        applied = {}
        stray = False
        for operation in operations[len(reference):]:
            wires = list(operation.wires)
            if len(wires) != 1 or wires[0] not in (0, 1, 2):
                stray = True
                break
            applied.setdefault(wires[0], []).append(operation)
        if stray:
            problem = (
                "A gate after prepare(angles) acts on more than one wire, or on a wire "
                "outside 0, 1 and 2. A basis change is a single-qubit rotation applied "
                "to each wire on its own.")
            continue

        mismatch = None
        for wire in range(3):
            gates = applied.get(wire, [])
            if (word[wire] == "I" or basis[wire] == "none") and gates:
                mismatch = (
                    "You declared \"" + basis[wire] + "\" on wire " + str(wire)
                    + " but applied " + str(len(gates)) + " gate(s) there. A wire "
                    "carrying I or Z needs no gate at all.")
                break
            if word[wire] == "I":
                continue
            rotation = np.eye(2, dtype=complex)
            for operation in gates:
                rotation = np.asarray(matrix_of(operation)) @ rotation
            measured_pauli = rotation.conj().T @ pauli["Z"] @ rotation
            if not np.allclose(measured_pauli, pauli[word[wire]], rtol=0.0, atol=1e-9):
                mismatch = (
                    "The gates on wire " + str(wire) + " do not turn a computational-"
                    "basis measurement into a measurement of " + word[wire] + ". "
                    "Conjugating Z through them has to give that Pauli factor back.")
                break
        if mismatch is not None:
            problem = mismatch
            continue

        # The number is recomputed from the grader's own execution of this circuit,
        # so a distribution reported by the solution is never used.
        try:
            distribution = measure(operations)
        except Exception:
            problem = ("This circuit could not be re-run by the grader. Use plain "
                       "gates on wires 0, 1 and 2.")
            continue
        if len(distribution) != 8:
            continue
        if abs(sum(probability * sign for probability, sign
                   in zip(distribution, signs)) - float(value)) > 1e-9:
            problem = (
                "Your value is not the parity sum of the probabilities this circuit "
                "produces. Combine them as sum(probabilities[b] * signs[b]) over the "
                "eight outcomes.")
            continue
        accepted = True
        break
    if not accepted:
        raise AssertionError(problem)


    # basis and signs are also settled by the word itself, so they are checked here
    # rather than only against the circuit that was run.
    letter_to_basis = {"I": "none", "Z": "none", "X": "H", "Y": "SH"}
    if basis != [letter_to_basis[letter] for letter in word]:
        raise AssertionError(
            "Wrong basis changes -- X needs a Hadamard, Y needs the adjoint of S and "
            "then a Hadamard, and I and Z need nothing at all.")
    carried = [wire for wire, letter in enumerate(word) if letter != "I"]
    if signs != [int((-1) ** sum((outcome >> (2 - wire)) & 1 for wire in carried))
                 for outcome in range(8)]:
        raise AssertionError(
            "Wrong signs -- an outcome's sign is -1 raised to the number of wires "
            "that came out as 1 and carry a non-identity factor. Wires carrying I "
            "are skipped, wires carrying Z are not.")
    
    return dumps([basis, signs, float(value)])


def check(solution_output: str, expected_output: str) -> None:
    got = json.loads(solution_output)
    exp = json.loads(expected_output)
    assert got[0] == exp[0], \
        "Wrong basis changes -- X needs a Hadamard, Y needs the adjoint of S and " \
        "then a Hadamard, and I and Z need nothing at all."
    assert got[1] == exp[1], \
        "Wrong signs -- an outcome's sign is -1 raised to the number of wires " \
        "that came out as 1 and carry a non-identity factor. Wires carrying I are " \
        "skipped, wires carrying Z are not."
    assert np.isclose(got[2], exp[2], rtol=0.0, atol=1e-6), \
        "Wrong expectation value -- it is the sum over the eight outcomes of the " \
        "probability times that outcome's sign."


# These are the public test cases
test_cases = [
    ('["ZIZ", [0.79, 2.75, 2.89, 0.33]]', '[["none", "none", "none"], [1, -1, 1, -1, -1, 1, -1, 1], 0.931986472711103]'),
    ('["XYI", [1.45, 0.22, 1.72, 1.4]]', '[["H", "SH", "none"], [1, 1, -1, -1, -1, -1, 1, 1], 0.9673999542626387]')
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