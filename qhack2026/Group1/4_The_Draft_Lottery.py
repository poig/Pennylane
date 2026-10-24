import json
import pennylane as qp
import pennylane.numpy as np

# The unitary whose spectrum you have to read. You do not need to change this.
def unitary_matrix(phases):
    """Builds a diagonal unitary with the given eigenphases.

    The eigenvalue on computational-basis state |k> is exp(2j * pi * phases[k]),
    so the eigenvectors are the computational-basis states themselves.

    Args:
        phases (list[float]): eigenphases in [0, 1), one per basis state. The length
            is a power of two.

    Returns:
        numpy array: the diagonal unitary matrix.
    """
    return np.diag(np.array([np.exp(2j * np.pi * phase) for phase in phases]))



def read_spectrum(phases, amplitudes, m):
    """Reads every eigenphase present in a superposition, together with its weight,
    and prices the repetitions needed to see the faintest one.

    Args:
        phases (list[float]): the eigenphases of the unitary, passed to unitary_matrix.
        amplitudes (list[float]): a normalised real state on the target wires.
        m (int): the number of estimation wires.

    Returns:
        tuple[list[float], list[float], int]:
            - the eigenphases present, ascending;
            - the weight of each, in the same order;
            - the number of repetitions needed to see the faintest one.
    """


    # Put your code here.
    n_target = int(np.log2(len(phases)))
    target_wires = list(range(n_target))
    estimation_wires = list(range(n_target, n_target + m))

    dev = qp.device("default.qubit", wires=n_target + m)

    @qp.qnode(dev)
    def circuit():
        qp.StatePrep(np.array(amplitudes, dtype=float), wires=target_wires)
        qp.QuantumPhaseEstimation(
            unitary_matrix(phases),
            target_wires=target_wires,
            estimation_wires=estimation_wires,
        )
        return qp.probs(wires=estimation_wires)

    probs = circuit()

    eigenphases = []
    weights = []
    for k, weight in enumerate(probs):
        if weight > 1e-9:
            eigenphases.append(float(k / (2 ** m)))
            weights.append(float(weight))

    faintest = min(weights)
    repetitions = int(np.ceil(np.log(0.01) / np.log(1.0 - faintest)))

    return eigenphases, weights, repetitions


# These functions are responsible for testing the solution.
#
def run(test_case_input: str) -> str:
    loads, dumps = json.loads, json.dumps
    make_device, script = qp.device, qp.tape.QuantumScript
    phases, amplitudes, m = loads(test_case_input)
    log = []

    def audited_device(*args, **kwargs):
        """A default.qubit that reports what it ran and what it returned."""
        device = make_device(*args, **kwargs)
        real_execute = device.execute

        def execute(circuits, *inner_args, **inner_kwargs):
            results = real_execute(circuits, *inner_args, **inner_kwargs)
            single = isinstance(circuits, script)
            batch = [circuits] if single else list(circuits)
            outputs = [results] if single else list(results)
            for tape, output in zip(batch, outputs):
                if len(tape.measurements) != 1:
                    continue
                measurement = tape.measurements[0]
                log.append((type(measurement).__name__, tape.shots.total_shots,
                            tuple(measurement.wires),
                            [float(value) for value in np.asarray(output).ravel()]))
            return results

        device.execute = execute
        return device

    qp.device = audited_device
    try:
        returned = read_spectrum(phases, amplitudes, m)
    finally:
        qp.device = make_device

    try:
        eigenphases, weights, repetitions = returned
    except (TypeError, ValueError):
        raise AssertionError(
            "Return exactly three values: the eigenphases, their weights, and the "
            "number of repetitions.")
    eigenphases = [float(phase) for phase in eigenphases]
    weights = [float(weight) for weight in weights]
    repetitions = int(repetitions)
    if len(eigenphases) != len(weights):
        raise AssertionError(
            "You returned " + str(len(eigenphases)) + " eigenphase(s) but "
            + str(len(weights)) + " weight(s). One weight per eigenphase.")
    if sorted(eigenphases) != eigenphases:
        raise AssertionError("Return the eigenphases in ascending order.")

    for kind, _, _, _ in log:
        if kind != "ProbabilityMP":
            raise AssertionError(
                "A circuit returned " + kind + ". The estimation register has to be "
                "read with qp.probs so that every eigenphase present shows up.")

    # Rebuild the answer from what the device actually returned. Any distribution the
    # solution measured over exactly m wires is admissible; at least one of them has
    # to reproduce the eigenphases and weights being reported.
    reproduced = False
    for kind, shots, wires, distribution in log:
        if shots is not None or len(wires) != m or len(distribution) != 2 ** m:
            continue
        peaks = {}
        for k, weight in enumerate(distribution):
            if weight > 1e-9:
                phase = k / 2 ** m
                peaks[phase] = peaks.get(phase, 0.0) + weight
        found = sorted(peaks)
        if (len(found) == len(eigenphases)
                and all(abs(a - b) < 1e-12 for a, b in zip(found, eigenphases))
                and all(abs(peaks[phase] - weight) < 1e-6
                        for phase, weight in zip(found, weights))):
            reproduced = True
            break
    if not reproduced:
        raise AssertionError(
            "The eigenphases and weights you returned do not match any distribution "
            "your circuit measured. Run phase estimation with " + str(m)
            + " estimation wires on the given superposition, return "
            "qp.probs of the estimation wires, and read the peaks off that -- "
            + str(len(log)) + " execution(s) were recorded.")
    faintest = min(weights)
    if repetitions != int(np.ceil(np.log(0.01) / np.log(1.0 - faintest))):
        raise AssertionError(
            "Wrong repetition count -- solve (1 - w)**r <= 0.01 for the faintest "
            "weight w and round up.")
    return dumps([eigenphases, weights, repetitions])


def check(solution_output: str, expected_output: str) -> None:
    got = json.loads(solution_output)
    exp = json.loads(expected_output)
    assert len(got[0]) == len(exp[0]), \
        "Wrong number of eigenphases -- keep every bin whose probability exceeds " \
        "1e-9, and remember that two eigenvectors sharing an eigenphase produce " \
        "one peak, not two."
    assert np.allclose(got[0], exp[0], rtol=0.0, atol=1e-9), \
        "Wrong eigenphases -- bin k of the estimation register corresponds to the " \
        "phase k / 2**m."
    assert np.allclose(got[1], exp[1], rtol=0.0, atol=1e-6), \
        "Wrong weights -- the probability of a peak is the squared overlap of the " \
        "input state with the corresponding eigenspace."
    assert got[2] == exp[2], \
        "Wrong repetition count -- solve (1 - w)**r <= 0.01 for the faintest " \
        "weight w and round up."


# These are the public test cases
test_cases = [
    ('[[0.1875, 0.6875], [0.8, 0.6], 4]', '[[0.1875, 0.6875], [0.6399999999999999, 0.3599999999999996], 11]'),
    ('[[0.125, 0.375, 0.625, 0.875], [0.7, 0.5, 0.5, 0.1], 3]', '[[0.125, 0.375, 0.625, 0.875], [0.4899999999999996, 0.2499999999999999, 0.25, 0.009999999999999993], 459]')
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