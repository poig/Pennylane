import json
import pennylane as qp
import pennylane.numpy as np

dev  = qp.device("default.qubit", wires=4)
@qp.qnode(dev)
def quantum_kernel(x1,x2):
    """
    A function implementing the quantum kernel of two datapoints
    Args:
        x1 (list(float)): The first of the datapoints from which we want to compute the kernel 
        x2 (list(float)): The second of the datapoints from which we want to compute the kernel     
    Returns: 
        (np.tensor): The value of the square of the overlap of x1 and x2: |⟨x2|x1⟩|^2
        
    """


    # Put your code here #
    # 2022 haqs memory
    qp.AngleEmbedding(x1, wires=range(4))
    qp.adjoint(qp.AngleEmbedding)(x2, wires=range(4))

    # |0000><0000|
    projector = np.zeros((2**4, 2**4))
    projector[0, 0] = 1.0
    return qp.expval(qp.Hermitian(projector, wires=range(4)))


def kernel_matrix(X1,X2):
    """
    A function that returns the matrix of pairwise kernel values for two given datapoints
    Args:
    - X1 (list(list(float))): The first list of datapoints from which we want to compute the kernel matrix
    - X2 (list(list(float))): The second list of datapoints from which we want to compute the kernel matrix
    Returns: 
        (np.array): A matrix with the kernel values of all different combinations of X1 and X2
        
    """


    # Put your code here #
    return np.array([[quantum_kernel(x1, x2) for x2 in X2] for x1 in X1])


# These functions are responsible for testing the solution
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = kernel_matrix(*ins)
    return output


def check(solution_output: str, expected_output: str) -> None:
    expected_output = json.loads(expected_output)
    assert np.allclose(solution_output.tolist(), expected_output, rtol=1e-4)


# These are the public test cases
test_cases = [
    ('[[[0.32058496, 0.07741205, 0.53200578, 0.91373749], [0.71838322, 0.45665254, 0.19871986, 0.1157332 ], [0.04515,    0.66228969, 0.69793587, 0.64664178]], [[0.32058496, 0.07741205, 0.53200578, 0.91373749], [0.71838322, 0.45665254, 0.19871986, 0.1157332 ], [0.04515,    0.66228969, 0.69793587, 0.64664178]]]', '[[1, 0.76528062, 0.87759179], [0.76528062, 1, 0.77075905], [0.87759179, 0.77075905, 1]]'),
    ('[[[0.94727029, 0.40665627, 0.05521391, 0.06570142], [0.32897296, 0.89902984, 0.59060573, 0.11193315]], [[0.11815914, 0.47388796, 0.64337692, 0.59215516], [0.69635966, 0.65039949, 0.17727646, 0.76091743]]]', '[[0.71461545, 0.85406897], [0.89085434, 0.81902278]]')
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