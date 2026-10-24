import json
import pennylane as qp
import pennylane.numpy as np

# Write any helper functions you need here


def symmetry_score(data, labels, s):
    """This function returns the data's symmetry score for a given symmetry s
    
    Args:
        data (np.array): An array containing binary arrays, which represent the data points
        labels (np.array(int)): The label for each data point. There may be more than two labels.
        s (np.array): A binary array representing the candidate symmetry.

    Returns:
        (float): The symmetry score, as described in the challenge statement.
    """


    # Put your code here #
    # Return the symmetry score

    n = len(s)
    N = len(data)
    ancilla = n

    dev = qp.device("default.qubit", wires=n + 1)

    @qp.qnode(dev)
    def class_circuit(state_vec):
        qp.StatePrep(state_vec, wires=range(n))
        for j in range(n):
            qp.Hadamard(wires=j)

        for j in range(n):
            if s[j] == 1:
                qp.CNOT(wires=[j, ancilla])

        return qp.probs(wires=ancilla)

    total_score = 0.0

    for r in np.unique(labels):
        class_points = data[labels == r]
        num_points = len(class_points)

        state_vec = np.zeros(2 ** n, dtype=float)
        for x in class_points:
            idx = sum(int(x[j]) * (2 ** (n - 1 - j)) for j in range(n))
            state_vec[idx] = 1.0 / np.sqrt(num_points)

        p_r = class_circuit(state_vec)[0]
        total_score += (num_points / N) * float(p_r)

    return float(total_score)


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    data, labels, s = json.loads(test_case_input)[0], json.loads(test_case_input)[1], json.loads(test_case_input)[2]
    data = np.array(data)
    labels = np.array(labels)
    s = np.array(s)
    out = symmetry_score(data, labels, s)
    return str(out)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
                                 
    assert np.isclose(solution_output, expected_output, rtol=1e-3)


# These are the public test cases
test_cases = [
    ('[[[0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1]], [0, 0, 1, 1, 2], [1,1,0]]', '0.9'),
    ('[[[0, 0, 0], [1, 1, 0], [0, 0, 1]], [0, 0, 0], [1,1,0]]', '0.833333333333333')
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