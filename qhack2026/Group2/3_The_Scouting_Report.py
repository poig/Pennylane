import json
import pennylane as qp
import pennylane.numpy as np

def bloch_and_purity(prep, num_wires, target_wire):
    """Reconstructs the reduced state of one wire from expectation values only.

    Args:
        prep (callable): A black-box state preparation acting on `num_wires` wires.
            You may call it inside your circuits but you may not inspect it.
        num_wires (int): The total number of wires in the device.
        target_wire (int): The wire to characterise. This is not always wire 0.

    Returns:
        tuple[list[float], float]: The Bloch vector [r_x, r_y, r_z] of `target_wire`
            and its purity Tr(rho^2).
    """
    dev = qp.device("default.qubit", wires=num_wires)


    # Step 1: build one QNode per observable you need. How many do you need?
    # Put your code here #
    @qp.qnode(dev)
    def circuit_x():
        prep()
        return qp.expval(qp.PauliX(target_wire))

    @qp.qnode(dev)
    def circuit_y():
        prep()
        return qp.expval(qp.PauliY(target_wire))

    @qp.qnode(dev)
    def circuit_z():
        prep()
        return qp.expval(qp.PauliZ(target_wire))

    # Step 2: execute them and collect the Bloch components r_x, r_y and r_z of
    #         `target_wire`, which is an argument and not a fixed wire.
    # Put your code here #
    r_x = float(circuit_x())
    r_y = float(circuit_y())
    r_z = float(circuit_z())
    bloch_vector = [r_x, r_y, r_z]

    # Step 3: turn the Bloch vector into a purity.
    # Put your code here #
    purity = float(0.5 * (1.0 + (r_x**2 + r_y**2 + r_z**2)))

    # Step 4: return the Bloch vector components and purity
    return bloch_vector, purity


# These functions are responsible for testing the solution.
def build_prep(gate_list):
    """Turns a list of [gate_name, parameters, wires] entries into a preparation."""
    def prep():
        for name, parameters, wires in gate_list:
            getattr(qp, name)(*parameters, wires=wires)

    return prep


AUDIT = {"measurements": []}
_real_device = qp.device


def audited_device(*args, **kwargs):
    """Wraps a device so that every measurement it executes is recorded."""
    dev = _real_device(*args, **kwargs)
    inner = dev.execute

    def execute(circuits, *args, **kwargs):
        batch = [circuits] if isinstance(circuits, qp.tape.QuantumScript) else circuits
        for tape in batch:
            AUDIT["measurements"].extend(tape.measurements)
        return inner(circuits, *args, **kwargs)

    dev.execute = execute
    return dev


def audit_violation(target_wire):
    """Returns a message if the state was read out rather than measured."""
    seen = AUDIT["measurements"]
    for m in seen:
        if type(m) is not qp.measurements.ExpectationMP:
            return (
                "Disallowed measurement: " + type(m).__name__ + ". This challenge is "
                "qp.expval only. qp.state, qp.density_matrix, qp.purity and qp.probs "
                "all read the simulator's state directly, which hardware cannot do."
            )
    on_target = [m for m in seen if target_wire in list(m.wires)]
    if len(on_target) < 3:
        return (
            "Only " + str(len(on_target)) + " expectation value(s) on wire "
            + str(target_wire) + " reached the device. Three independent observables "
            "are needed, and they must be executed on the `dev` provided for you."
        )
    return None


def run(test_case_input: str) -> str:
    case = json.loads(test_case_input)
    gate_list = case["gates"]
    target_wire = case["target_wire"]
    num_wires = 1 + max(max(wires) for _, _, wires in gate_list)
    AUDIT["measurements"] = []
    qp.device = audited_device
    try:
        r, purity = bloch_and_purity(build_prep(gate_list), num_wires, target_wire)
    finally:
        qp.device = _real_device
    violation = audit_violation(target_wire)
    if violation is not None:
        return json.dumps({"violation": violation})
    return json.dumps([r, purity])


def check(solution_output: str, expected_output: str) -> None:
    got = json.loads(solution_output)
    if isinstance(got, dict):
        raise AssertionError(got["violation"])
    exp = json.loads(expected_output)
    assert np.allclose(got[0], exp[0], atol=1e-5), \
        "Bloch vector is wrong -- you need X, Y and Z, all three on `target_wire`."
    assert np.isclose(got[1], exp[1], atol=1e-5), \
        "Purity is wrong -- it follows from the length of the Bloch vector."


# These are the public test cases
test_cases = [
    ('{"gates": [["RY", [0.6], [0]], ["RZ", [1.3], [0]]], "target_wire": 0}', '[[0.151041200225, 0.544065877074, 0.82533561491], 1.0]'),
    ('{"gates": [["RX", [0.8], [0]], ["CRX", [1.4], [0, 1]]], "target_wire": 1}', '[[0.0, -0.149440145691, 0.87412830171], 0.893216322497]')
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