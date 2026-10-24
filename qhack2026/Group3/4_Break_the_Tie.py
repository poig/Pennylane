import json
import pennylane as qp
import pennylane.numpy as np

# Two helpers are given to you. You do not need to change either one.
def rotation_angle(fraction):
    """The RY angle that puts amplitude sqrt(fraction) on |1>.

    Args:
        fraction (float): a number in [0, 1].

    Returns:
        float: the angle theta with sin(theta / 2)**2 == fraction.
    """
    return float(2 * np.arcsin(np.sqrt(fraction)))

def run_circuit(num_wires, circuit):
    """Runs a circuit as an analytic QNode on a fresh default.qubit simulator.

    This exists so that you only have to write the circuit itself. There are no
    shots anywhere in this challenge, so every number that comes back is exact.

    Args:
        num_wires (int): the number of wires the circuit needs.
        circuit (callable): takes no arguments, applies operations, and returns a
            single measurement.

    Returns:
        list[float]: what the circuit measured, flattened into plain floats.
    """
    device = qp.device("default.qubit", wires=num_wires)
    return [float(value) for value in np.asarray(qp.QNode(circuit, device)()).ravel()]





def estimate_mean(probabilities, values, n):
    """Estimates the average of a quantity over a discrete distribution with
    amplitude estimation, resolving the two-fold ambiguity in the readout.

    Args:
        probabilities (list[float]): the probability of each outcome, summing to 1.
            Its length is a power of two, 2**m, where m is the number of data wires.
        values (list[float]): the value of the quantity at each outcome, all
            non-negative, same length as probabilities.
        n (int): the number of estimation wires.

    Returns:
        float: the estimated average of the quantity.
    """

    probabilities = np.array(probabilities, dtype=float)
    values = np.array(values, dtype=float)
    m = int(round(np.log2(len(probabilities))))     # number of data wires
    


    # --- Step 1: rescale the quantity into [0, 1] ----------------------------
    f_max = float(np.max(values))
    if f_max == 0.0: return 0.0
    scaled_values = values / f_max

    # --- Step 2: run the amplitude estimation --------------------------------
    N = 2**n
    target_wires = range(m + 1)
    estimation_wires = range(m + 1, m + 1 + n)

    def ae_circuit():
        qp.QuantumMonteCarlo(
            probabilities,
            lambda i: float(scaled_values[i]),
            target_wires=target_wires,
            estimation_wires=estimation_wires,
        )
        return qp.probs(wires=estimation_wires)

    ae_probs = run_circuit(m + 1 + n, ae_circuit)

    # --- Step 3: turn the readout into the two candidates --------------------
    y = int(np.argmax(ae_probs[: N // 2]))
    phi = y / N
    candidate_1 = float((1.0 - np.cos(np.pi * phi)) / 2.0)
    candidate_2 = float((1.0 + np.cos(np.pi * phi)) / 2.0)

    # --- Step 4: build the tie-breaker circuit  --------------------------
    state_vec = np.zeros(2 ** (m + 1), dtype=complex)
    for i in range(len(probabilities)):
        theta = rotation_angle(scaled_values[i])
        state_vec[2 * i] = np.sqrt(probabilities[i]) * np.cos(theta / 2.0)
        state_vec[2 * i + 1] = np.sqrt(probabilities[i]) * np.sin(theta / 2.0)

    def tie_breaker():
        qp.StatePrep(state_vec, wires=range(m + 1))
        return qp.probs(wires=m)

    tie_probs = run_circuit(m + 1, tie_breaker)
    p_one = tie_probs[1]

    # --- Step 5: break the tie -----------------------------------------------
    if abs(p_one - candidate_1) < abs(p_one - candidate_2):
        chosen = candidate_1
    else:
        chosen = candidate_2

    return float(f_max * chosen) # return the estimate mean


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    probabilities, values, n = json.loads(test_case_input)
    return json.dumps(float(estimate_mean(probabilities, values, n)))


def check(solution_output: str, expected_output: str) -> None:
    got = json.loads(solution_output)
    exp = json.loads(expected_output)
    assert np.isclose(got, exp, rtol=0.0, atol=1e-9), \
        "Wrong average. It has to be f_max times whichever of the two candidates " \
        "your tie-breaker picks out -- the amplitude-estimation estimate, not the " \
        "exact sum of p_i f_i, and not always the smaller of the two candidates."


# These are the public test cases
test_cases = [
    ('[[0.4008104395612045, 0.24310382032676792, 0.14744992052145606, 0.0894328975684541, 0.054243794362206846, 0.03290052437982574, 0.019955176756987287, 0.012103426523097706], [0.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0, 12.25], 8]', '1.160949660702857'),
    ('[[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], [0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0], 8]', '0.8745681972617296')
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