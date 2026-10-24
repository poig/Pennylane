import json
import pennylane as qp
import pennylane.numpy as np
import scipy

# DO NOT CHANGE THE FOLLOWING CONSTANTS LATER IN THE CODE

M = 64

S10 = 100.0
S20 = 100.0

sigma1 = 0.20
sigma2 = 0.30

r = 0.05


# Write any helper functions you need here  
def payoff(i, w, T, K):
    """Calculate the basket option payoff for the i-th grid point assets"""
    grid = np.linspace(-3, 3, 8)
    i1 = i // 8
    i2 = i % 8
    z1 = grid[i1]
    z2 = grid[i2]

    # geometric Brownian motions for both assets at a future time T
    S1 = S10 * np.exp((r - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * z1)
    S2 = S20 * np.exp((r - 0.5 * sigma2**2) * T + sigma2 * np.sqrt(T) * z2)

    # Basket price and European call payoff
    basket = w * S1 + (1.0 - w) * S2
    return max(float(basket - K), 0.0)


n = 10
N = 2 ** n
m = int(np.log2(M))
target_wires = range(m + 1)
estimation_wires = range(m + 1, n + m + 1)

dev = qp.device("default.qubit", wires=(n + m + 1))

@qp.qnode(dev)
def quantum_montecarlo(rho, w, T, K):
    
    """This function uses Quantum Monte Carlo to estimate the option basket price
    given a correlation coefficient, a weight, a time to maturity, and a strike price K

    Args:
    - rho (float): correlation between the Brownian shocks of the two assets.
    - w (float): weight of the first asset in the basket.
    - K (float): strike price of the basket option.
    - T (float): time to maturity in years.

    Returns:
    - float: measurement probabilities in the estimation wires. The grader will use
    this to estimate the option price
    """


    # Write your Monte Carlo circuit here
    # Bivariate normal distribution and Payoffs over the 8x8 grid
    grid = np.linspace(-3, 3, 8)
    cov = [[1.0, rho], [rho, 1.0]]
    rv = scipy.stats.multivariate_normal([0.0, 0.0], cov)

    probs = np.array([rv.pdf([grid[i // 8], grid[i % 8]]) for i in range(M)])
    probs = probs / np.sum(probs)
    payoffs = np.array([payoff(i, w, T, K) for i in range(M)])
    payoff_max = np.max(payoffs)
    fn = lambda i: (payoffs[i] / payoff_max) if payoff_max > 0 else 0.0

    qp.QuantumMonteCarlo(
        probs,
        fn,
        target_wires=target_wires,
        estimation_wires=estimation_wires,
    )

    return qp.probs(estimation_wires)


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    phase_estimated = np.argmax(quantum_montecarlo(*ins)[:int(N / 2)]) / N
    payoffs = np.array([
        payoff(i, ins[1], ins[2], ins[3]) for i in range(M)
    ])

    payoff_max = np.max(payoffs)
    output = (1 - np.cos(np.pi * phase_estimated)) / 2 * payoff_max *np.exp(-r * ins[2])

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, atol=1e-3
    ), "Your circuit doesn't look quite right "


# These are the public test cases
test_cases = [
    ('[0.3, 1.0, 1.0, 100]', '10.556134590137553'),
    ('[0.4, 0.5, 1.0, 110]', '6.487093067342844')
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