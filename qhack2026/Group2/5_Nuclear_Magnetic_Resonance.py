import jax
import jax.numpy as jnp
import json
import pennylane as qp

# Write any helper functions you need here


dev = qp.device('default.qubit', wires = 1)

@qp.qnode(dev)
def nmr_evolve(B0: float, B1: float, omega: float, time: float) -> "jnp.ndarray":
    """This quantum circuit evolves the initial state |0> under the evolution operator generated
    by the nuclear magnetic resonance Hamiltonian.
    
    Args:
        B0 (float): Coefficient of the Pauli Z term in the Hamiltonian.
        B1 (float): Coefficient of the time-dependent component of the Hamiltonian.
        omega (float): Magnetic field oscillation frequency.
        time (float): Period of time during which the magnetic field acts on the qubit.

    Returns:
        (numpy.array): The probabilities of measuring each computational basis state.
    """


    # Put your code here #
    f_x = lambda p, t: p * jnp.cos(omega * t)
    f_y = lambda p, t: p * jnp.sin(omega * t)
    H = B0 * qp.PauliZ(0) + f_x * qp.PauliX(0) + f_y * qp.PauliY(0)
    qp.evolve(H)([B1, B1], t=time)
    # Return the probabilities
    return qp.probs(wires=0)


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = nmr_evolve(*ins).tolist()

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert jnp.allclose(
        jnp.array(solution_output), jnp.array(expected_output), atol=0.03
    ), "Your code does not yield the correct probabilities"


# These are the public test cases
test_cases = [
    ('[1,2,0.5,0.3]', '[0.6962061524391174, 0.3037952184677124]'),
    ('[1, 1, 0.25, 0.6]', '[0.7278291583061218, 0.272170752286911]')
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