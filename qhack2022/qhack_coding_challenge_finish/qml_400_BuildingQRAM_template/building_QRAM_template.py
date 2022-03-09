#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.

    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #
        #qml.QFT(wires=range(4))
        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.
        #num_add = np.array([int(a) for a in str("{0:0{1}b}".format(i,4))])
        #qml.RY(thetas[6],wires=[0])
        '''for i in range(3):
            qml.Hadamard(wires=1)
            qml.Hadamard(wires=2)
            qml.Hadamard(wires=3)
            qml.RY(thetas[i],wires=[i+1])
            qml.CNOT(wires=[i,i+1])
            qml.RY(thetas[i+3],wires=[i+1])
            qml.CNOT(wires=[i,i+1])
        qml.RY(thetas[7],wires=[0])'''

        #qml.RY(thetas[7],wires=0)
        for i in range(3):
            qml.Hadamard(wires=i)
            
        for i in range(8):
            #qml.MultiControlledX(control_wires=[0, 1, 2], wires=3, control_values=str("{0:0{1}b}".format(i,3)))
            U = [[np.cos(thetas[i]/2),-np.sin(thetas[i]/2)],[np.sin(thetas[i]/2),np.cos(thetas[i]/2)]]
            qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3, control_values=str("{0:0{1}b}".format(i,3)))
            #qml.RY(thetas[i],wires=[3])
            #qml.MultiControlledX(control_wires=[0, 1, 2], wires=3, control_values=str("{0:0{1}b}".format(i,3)))
        #qml.adjoint(qml.QFT)(wires=range(4))

        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")

    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
