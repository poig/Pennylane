import json
import pennylane as qp
import pennylane.numpy as np

qre = qp.estimator

GATE_SET = {"Hadamard", "S", "CNOT", "T"}
ROTATION_PRECISION = 1e-6


def affordable_precision(t_gate_seconds, budget_seconds, max_estimation_wires=40):
    """Finds the largest quantum phase estimation that fits a wall-clock budget.

    Args:
        t_gate_seconds (float): The time to execute one logical T gate, in seconds.
        budget_seconds (float): The total wall-clock budget, in seconds.
        max_estimation_wires (int): A safety bound on the search. A correct model never
            reaches it; a model that returns exactly this value has not found a limit,
            it has run out of loop.

    Returns:
        tuple[int, float, int]: The largest affordable number of estimation wires m,
            the phase resolution 2**-m that buys, and the T-count at that size.
    """


    # Step 1: build a qre.ResourceConfig and set the single-qubit rotation precision
    config = qre.ResourceConfig()
    config.set_single_qubit_rot_precision(ROTATION_PRECISION)

    # Step 2: estimate QPE on a generic 2-qubit base unitary
    # Step 3: the runtime is the T-count times t_gate_seconds.
    best = None
    for m in range(1, max_estimation_wires + 1):
        qpe_op = qre.QPE(qre.QubitUnitary(num_wires=2), num_estimation_wires=m)
        resources = qre.estimate(qpe_op, gate_set=GATE_SET, config=config)
        t_count = int(resources.gate_counts.get("T", 0))

        runtime = t_count * t_gate_seconds
        if runtime <= budget_seconds:
            best = (m, 2.0 ** (-m), t_count)
        else:
            break
    # Put your code here #
    return best


# These functions are responsible for testing the solution.
#
# Everything the grader needs is copied into local variables at the top of run(),
# before any of your code runs, and the verdict is decided there. Your solution
# body executes inside affordable_precision, in this same module namespace, so it
# could rebind any name defined at module level -- including check(). None of the
# names the grade depends on are read from that namespace after your code runs.
def run(test_case_input: str) -> str:
    loads, dumps, close = json.loads, json.dumps, np.isclose
    estimator = qre
    real_estimate = estimator.estimate
    make_qpe, make_base = estimator.QPE, estimator.QubitUnitary
    make_config = estimator.ResourceConfig
    gate_set, rot_precision = set(GATE_SET), ROTATION_PRECISION
    t_gate, budget = loads(test_case_input)
    calls = []

    def audited(*args, **kwargs):
        """Records the T-count of every size your search actually costs."""
        resources = real_estimate(*args, **kwargs)
        calls.append(int(resources.gate_counts.get("T", 0)))
        return resources

    def reference_t_count(wires):
        """Costs one QPE size exactly as the challenge specifies it."""
        config = make_config()
        config.set_single_qubit_rot_precision(rot_precision)
        resources = real_estimate(
            make_qpe(make_base(num_wires=2), num_estimation_wires=wires),
            gate_set=gate_set,
            config=config,
        )
        return int(resources.gate_counts.get("T", 0))

    estimator.estimate = audited
    try:
        best = affordable_precision(t_gate, budget)
    finally:
        estimator.estimate = real_estimate

    if best is None:
        raise AssertionError(
            "Your function returned None, so no size was ever recorded as "
            "affordable. Every test case here has an affordable size of at least "
            "m = 1, so check that your loop starts at m = 1 and that you record "
            "the current size as the best one before moving on to the next.")
    try:
        wires, precision, t_count = best
    except (TypeError, ValueError):
        raise AssertionError(
            "Return exactly three values: the number of estimation wires, the "
            "phase resolution, and the T-count.")
    wires = int(wires)
    if wires < 1:
        raise AssertionError(
            "You reported " + str(wires) + " estimation wires. The search starts "
            "at m = 1.")
    if wires > 40:
        raise AssertionError(
            "You reported " + str(wires) + " estimation wires, past the "
            "max_estimation_wires safety bound. A correct model never reaches it.")
    if not close(float(precision), 2.0 ** -wires, rtol=1e-12, atol=0.0):
        raise AssertionError(
            "You reported " + str(wires) + " wires with a resolution of "
            + str(float(precision)) + ", but the resolution that buys is 2**-m = "
            + str(2.0 ** -wires) + ".")
    if len(calls) < 2:
        raise AssertionError(
            "The estimator was called " + str(len(calls)) + " time(s). Finding the "
            "largest affordable size means comparing sizes, so call qre.estimate "
            "inside your search rather than obtaining the T-count another way.")
    here = reference_t_count(wires)
    if int(t_count) != here:
        raise AssertionError(
            "At m = " + str(wires) + " the T-count is " + str(here) + ", but you "
            "reported " + str(int(t_count)) + ". If you are within a fraction of a "
            "percent, the rotation precision is not set to ROTATION_PRECISION. If "
            "you are off by orders of magnitude, the base unitary is wrong -- it "
            "must be qre.QubitUnitary(num_wires=2).")
    if here * t_gate > budget:
        raise AssertionError(
            "m = " + str(wires) + " costs " + str(here) + " T gates, which run for "
            + str(here * t_gate) + " s against a budget of " + str(budget) + " s. "
            "That size does not fit -- keep the largest one that does.")
    if reference_t_count(wires + 1) * t_gate <= budget:
        raise AssertionError(
            "m = " + str(wires) + " fits, but so does m = " + str(wires + 1) + ". "
            "Return the largest affordable size, not the first one you find.")
    return dumps([wires, float(precision), int(t_count)])


def check(solution_output: str, expected_output: str) -> None:
    # run() has already decided the grade; this is a redundant second comparison
    # against the stored answer.
    got, exp = json.loads(solution_output), json.loads(expected_output)
    assert got[0] == exp[0], "Wrong number of estimation wires."
    assert np.isclose(got[1], exp[1], rtol=1e-12, atol=0.0), \
        "Wrong phase resolution -- it should be 2**-m."
    assert got[2] == exp[2], "T-count mismatch."


# These are the public test cases
test_cases = [
    ('[1e-06, 1.0]', '[9, 0.001953125, 684108]'),
    ('[0.001, 60.0]', '[5, 0.03125, 42252]')
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