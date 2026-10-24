import json
import pennylane as qp
import pennylane.numpy as np

# Write any helper functions you may need here


def approximation_ratio(edges):

    """This function calculates the approximation ratio of QAOA for
    the minimum vertex cover problem

    Args:
    - Edges (list(list(int))): A list of ordered pairs of integers, representing 
    the edges of the graph for which we need to solve the minimum vertex cover problem.

    Returns:
    - np.tensor(float): The approximation ratio of QAOA for the minimum vertex cover problem for
    the input graph.
    """


    # Write your QAOA circuit and optimization routine to find the minimum expval
    # reference: https://pennylane.ai/demos/tutorial_qaoa_intro/
    vertices = sorted(list(set(v for e in edges for v in e)))

    # Build Cost and Mixer Hamiltonians
    coeffs = []
    obs = []
    for i, j in edges:
        coeffs.extend([0.75, 0.75, 0.75])
        obs.extend([qp.PauliZ(i) @ qp.PauliZ(j), qp.PauliZ(i), qp.PauliZ(j)])
    for i in vertices:
        coeffs.append(-1.0)
        obs.append(qp.PauliZ(i))

    cost_h = qp.Hamiltonian(coeffs, obs)
    mixer_h = qp.Hamiltonian([1.0] * len(vertices), [qp.PauliX(i) for i in vertices])

    # define the QAOA layer and depth-4 circuit
    def qaoa_layer(gamma, alpha):
        qp.qaoa.cost_layer(gamma, cost_h)
        qp.qaoa.mixer_layer(alpha, mixer_h)

    dev = qp.device("default.qubit", wires=vertices)
    depth = 4

    @qp.qnode(dev)
    def cost_function(params):
        for w in vertices:
            qp.Hadamard(wires=w)
        qp.layer(qaoa_layer, depth, params[0], params[1])
        return qp.expval(cost_h)

    # discussion forum suggest random starting point
    rng = np.random.default_rng(42)
    candidates = [np.full((2, depth), 0.5, requires_grad=True)]
    for _ in range(5):
        perturbation = rng.uniform(-0.05, 0.05, size=(2, depth))
        candidates.append(np.array(0.5 + perturbation, requires_grad=True))
    
    # optimize and compute true minimum eigenvalue
    best_params = None
    best_cost = float("inf")
    opt = qp.GradientDescentOptimizer(stepsize=0.02)

    for init_params in candidates:
        params = init_params
        for _ in range(25):
            params = opt.step(cost_function, params)
        val = float(cost_function(params))
        if val < best_cost:
            best_cost = val
            best_params = params
    # converge the best candidate
    params = best_params
    for _ in range(200):
        params = opt.step(cost_function, params)
        val = float(cost_function(params))
        if val < best_cost:
            best_cost = val

    true_min = float(np.min(qp.eigvals(cost_h)))

    # Return the approximation ratio
    return np.array(best_cost / true_min)


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = approximation_ratio(ins)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=5e-2
    ), "Your code does not yield the right approximation ratio"


# These are the public test cases
test_cases = [
    ('[[0, 1], [1, 2], [0, 2], [2, 3]]', '0.58160924'),
    ('[[0, 1], [1, 2], [2, 3], [3, 0]]', '0.94746734')
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