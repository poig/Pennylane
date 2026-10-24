import jax
import jax.numpy as jnp
import json
import pennylane as qp

jax.config.update("jax_enable_x64", True)
dev = qp.device("default.qubit", wires=1)
t = jnp.array(2)

@qp.qnode(dev)
def target_state(state):
    """ A QNode that prepares the target state.
    
    Args:
    state (numpy.tensor): The 1-qubit target state.
    Returns:
    (numpy.tensor): The same state.
    """
    qp.StatePrep(state, wires=0)
    return qp.state()


def hamiltonian():
    """Creates the time-dependent Hamiltonian: H(t) = p0 t X + p1 t^2 Y + p2 t^3 Z
    
    Returns:
    (qp.pulse.parametrized_hamiltonian.ParametrizedHamiltonian): the Hamiltonian.
    """


    # Put your code here #
    with qp.QueuingManager.stop_recording():
        coeffs = [
            lambda p, t: p * t,
            lambda p, t: p * (t ** 2),
            lambda p, t: p * (t ** 3),
        ]
        ops = [qp.PauliX(0), qp.PauliY(0), qp.PauliZ(0)]
        return qp.dot(coeffs, ops)


@qp.qnode(dev)
def circuit(params):
    """A QNode that evolves the |0> state under the time-dependent Hamiltonian.
    This QNode must contain one operation only: an instance of qp.evolve.
    
    Args:
    params (list(numpy.tensor)): The parameters p0, p1, and p2 (in that order).
    Returns:
    (numpy.tensor): The evolved state.
    """


    # Put your code here #
    qp.evolve(hamiltonian())(params, t)
    return qp.state()
    


def optimal_parameters(state):
    """Optimizes the parameters p0, p1, and p2.

    Args:
        state (numpy.tensor): The 1-qubit target state.

    Returns:
        params (list(float)): The optimized parameters p0, p1, and p2 (in that order).
    """

    def cost(params):
        """Computes the negative fidelity between the output of circuit and the target state
        
        args: 
            params (list(numpy.tensor)): The parameters p0, p1, and p2 (in that order).
        returns:
            (numpy.tensor): the negative fidelity.
        """


        # Put your code here # 
        return -qp.math.fidelity_statevector(circuit(params), target_state(state))


    def update(gamma, params):
        """Updates the parameters, params, via a simple gradient descent routine.
        
        Args:
        gamma (float): the learning
        params (list(numpy.tensor)): the parameters p0, p1, and p2 (in that order)
        Returns:
        params (list(numpy.tensor)): the updated parameters.
        """

        gradient = jax.grad(cost)(params)

        for i  in range(len(params)):
            params[i] -= gradient[i] * gamma

        return params

    params = [jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)] 
    gamma = 0.1

    for i in range(20):
        params = update(gamma, params)
        if i > 10:
            gamma *= 0.9

        params = [param.tolist() for param in params]

        return params


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    state = json.loads(test_case_input)
    output = optimal_parameters(jnp.array(state))

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    tape = qp.workflow.construct_tape(circuit)([jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)])
    op_names = [op.name for op in tape.operations]

    assert op_names.count("ParametrizedEvolution") == 1, "You may only use one operation."

    assert op_names[0] == "ParametrizedEvolution", "You can only use an instance of qp.evolve in your circuit."
    assert jnp.allclose(jnp.array(expected_output), jnp.array(solution_output), rtol=1e-4), "Your calculated parameters are incorrect."


# These are the public test cases
test_cases = [
    ('[0.11775969, 0.99304212]', '[0.0, 0.062368193925808904, 0.0]'),
    ('[0.7071067812, 0.7071067812]', '[0.0, 0.26666667860250887, 0.0]')
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