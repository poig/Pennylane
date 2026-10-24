import json
import pennylane as qp
import pennylane.numpy as np

dev = qp.device("default.qubit", wires=1)# Define your device


def training_circuit(theta_vec):
    """
    A function implementing the circuit A([theta_1, theta_2, theta_3]) shown in the upper part of the image. It should not return anything.

    Args:
        theta_vec (np.ndarray(float)): An array [theta_1, theta_2, theta_3] of angle
        parameters for the RZ, RY and RZ gates, respectively
        
    """


    # Put your code here #
    qp.RZ(theta_vec[0], wires=0)
    qp.RY(theta_vec[1], wires=0)
    qp.RZ(theta_vec[2], wires=0)


def encoding_circuit(x):
    """
    A function implementing the circuit S(x) shown in the lower part of the image. It should not return anything.

    Args:
        x (float): A variable to be encoded as an angle parameter for the RX gate
    
    """


    # Put your code here #
    qp.RX(x, wires=0)


@qp.qnode(dev)
def quantum_neural_network(x, theta_vec):
    """
    A QNode implementing the circuits S(x) and A([theta_1, theta_2, theta_3]) subsequently and returning the  
    expectation value of the Pauli Z observable

    Args:
        x (float): A variable to be encoded as an angle parameter for the RX gate
        theta_vec (np.ndarray(float)): A list [theta_1, theta_2, theta_3] of angle parameters for the RZ, RY and RZ gates, respectively

    Returns: 
        (np.tensor): The expectation value of the Pauli Z observable
    """


    # Put your code here #
    encoding_circuit(x)
    training_circuit(theta_vec)
    # Return the expectation value
    return qp.expval(qp.PauliZ(0))


# These functions are responsible for testing the solution
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = quantum_neural_network(ins[0],ins[1:])

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.isclose(solution_output, expected_output, rtol=1e-4)


# These are the public test cases
test_cases = [
    ('[0.4859371, 1.32682562, -1.06015833, -4.41959946]', '0.827549'),
    ('[-0.72318627, -4.91395046, -5.0609724, 0.1125593]', '0.865457')
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