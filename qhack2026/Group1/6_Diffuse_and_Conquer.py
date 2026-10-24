import json
import pennylane as qp
import pennylane.numpy as np

# DO NOT CHANGE THESE VARIABLES LATER IN THE CODE

phi = np.array([0.03, 0.04, 0.07, 0.09])
source = np.array([1., 2., 1., 0.5])
dt = 0.01
h = 0.2


C = 0.05
x = np.linspace(0.1, 1, 1000)
y = C / x
# replace the chebyshev.chebfit and chebyshev.cheb2poly
degrees = list(range(1, 22, 2))

weights = x

V = np.vstack([x**d for d in degrees]).T
W = np.diag(weights)

WV = W @ V
Wy = weights * y

coefficients, _, _, _ = np.linalg.lstsq(WV, Wy, rcond=None)

poly_coeffs = np.zeros(max(degrees) + 1)
for deg, coef in zip(degrees, coefficients):
    poly_coeffs[deg] = coef


phi_qsvt = qp.poly_to_angles(poly_coeffs, "QSVT")  # Write your QSVT Angles here
C =  0.05# Write your overall factor of C/x of the polynomial your phases approximate

b = phi + dt * source # Write the unnormalized vector b (phi_new)

# Feel free to write any helper functions here

D2 = (np.diag([-2.0] * 4) + np.diag([1.0] * 3, 1) + np.diag([1.0] * 3, -1)) / h**2

def qsvt(A, angles):
    projectors = [qp.PCPhase(angle, dim=4, wires=[1, 2, 3]) for angle in angles]
    qp.QSVT(qp.BlockEncode(A, wires=[1, 2, 3]), projectors)


device = qp.device("default.qubit", wires=4) # Create your device here


@qp.qnode(device)
def qsvt_diffusion(nu, k):

    """
    Implements QSVT to find the concentration phi after one time step

    Args:
    - nu (float): The diffusion coefficient
    - k (float): The reaction rate

    Returns:
    - (np.array): The concentation vector on the next time step.
    """


    # Write your QSVT code for matrix inversion here
    A = (1 + k * dt) * np.eye(4) - nu * dt * D2
    A = A / np.linalg.norm(A, ord=2)

    qp.StatePrep(b / np.linalg.norm(b), wires=[2, 3])

    qp.Hadamard(wires=0)
    qp.ctrl(qsvt, control=0, control_values=0)(A, phi_qsvt)
    qp.ctrl(qsvt, control=0, control_values=1)(A, -phi_qsvt)
    qp.Hadamard(wires=0)

    # Return the state
    return qp.state()


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    out_state = qsvt_diffusion(*ins)[:4]
    rescaled_out= out_state * np.linalg.norm(b) / C
    output = rescaled_out/ np.linalg.norm(rescaled_out)

    return str(np.real(output).tolist())

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    fidelity = np.dot(solution_output, expected_output)
    assert fidelity >=0.99, "Your circuit doesn't look quite right "


# These are the public test cases
test_cases = [
    ('[0.2,0.5]', '[0.27959746, 0.42869397, 0.56892447, 0.64371709]'),
    ('[1.0, 1.2]', '[0.28550356, 0.4595721 , 0.59350887, 0.59584262]')
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