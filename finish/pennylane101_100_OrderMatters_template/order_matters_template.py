#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


def compare_circuits(angles):
    """Given two angles, compare two circuit outputs that have their order of operations flipped: RX then RY VERSUS RY then RX.

    Args:
        - angles (np.ndarray): Two angles

    Returns:
        - (float): | < \sigma^x >_1 - < \sigma^x >_2 |
    """

    # QHACK #

    # define a device and quantum functions/circuits here
    n_bits = 1
    dev = qml.device("default.qubit", wires=n_bits)

    @qml.qnode(dev)
    def apply_gate_1(angles):
        qml.RX(angles[0], wires=0)
        qml.RY(angles[1], wires=0)
        return qml.expval(qml.PauliX(wires=0))

    def apply_gate_2(angles):
        qml.RY(angles[1], wires=0)
        qml.RX(angles[0], wires=0)
        return qml.expval(qml.PauliX(wires=0))
    
    circuit_1 = qml.QNode(apply_gate_1,dev)
    circuit_2 = qml.QNode(apply_gate_2,dev)

    return np.abs(circuit_1(angles) - circuit_2(angles))

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    angles = np.array(sys.stdin.read().split(","), dtype=float)
    output = compare_circuits(angles)
    print(f"{output:.6f}")
