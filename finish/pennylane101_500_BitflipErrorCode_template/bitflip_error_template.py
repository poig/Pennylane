#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


def error_wire(circuit_output):
    """Function that returns an error readout.

    Args:
        - circuit_output (?): the output of the `circuit` function.

    Returns:
        - (np.ndarray): a length-4 array that reveals the statistics of the
        error channel. It should display your algorithm's statistical prediction for
        whether an error occurred on wire `k` (k in {1,2,3}). The zeroth element represents
        the probability that a bitflip error does not occur.

        e.g., [0.28, 0.0, 0.72, 0.0] means a 28% chance no bitflip error occurs, but if one
        does occur it occurs on qubit #2 with a 72% chance.
    """

    # QHACK #

    # process the circuit output here and return which qubit was the victim of a bitflip error!
    '''prob = np.zeros(4)
    for i in range(4):
        prob[i] = [int(a) for a in str["{0:}".format(circuit_output[0])]]'''
    '''list = np.array(circuit_output[:4])
    list_ = np.array(circuit_output[4:])'''
    '''prob = np.zeros(4)

    def count(i):
        try:
            return float('0.'+ ''.join([a for a in str(circuit_output[i])][10:]))
        except IndexError:
            return float('0.'+ ''.join([a for a in str(circuit_output[i])][:2]))


    list= []
    for i in range(len(circuit_output)):
        list.append(count(i))'''
    
    '''prob[0]=list[0]
    prob[1]=list[-1]
    prob[2]=list[-2]
    prob[3]=list[-3]'''
    circuit_output_1 = circuit_output[1]
    circuit_output[1] = circuit_output[3]
    circuit_output[3] = circuit_output_1

    return circuit_output  #"{0:.4f}".format(np.real(dev.state))



    # QHACK #


dev = qml.device("default.mixed", wires=3)


@qml.qnode(dev)
def circuit(p, alpha, tampered_wire):
    """A quantum circuit that will be able to identify bitflip errors.

    DO NOT MODIFY any already-written lines in this function.

    Args:
        p (float): The bit flip probability
        alpha (float): The parameter used to calculate `density_matrix(alpha)`
        tampered_wire (int): The wire that may or may not be flipped (zero-index)

    Returns:
        Some expectation value, state, probs, ... you decide!
    """

    qml.QubitDensityMatrix(density_matrix(alpha), wires=[0, 1, 2])

    # QHACK #

    wiress=[1,2]
    def get_value(S, i):
        try:
            return S[i]
        except IndexError:
            return S[0]
    '''if int(tampered_wire) == 1:
        qml.Hadamard(wires=int(tampered_wire))'''
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[0,2])
    # put any input processing gates here
    
    qml.BitFlip(p, wires=int(tampered_wire))

    # put any gates here after the bitflip error has occurred
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[0,2])
    qml.Toffoli(wires=[1,2,0])
    #qml.Hadamard(wires=int(tampered_wire))

    # return something!
    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    return qml.probs(wires=[1,2])#qml.probs(wires=[int(tampered_wire),get_value(wiress,int(tampered_wire))])#,qml.probs(wires=[0,1,2]),qml.probs(wires=[1,2])
    # QHACK #


def density_matrix(alpha):
    """Creates a density matrix from a pure state."""
    # DO NOT MODIFY anything in this code block
    psi = alpha * np.array([1, 0], dtype=float) + np.sqrt(1 - alpha**2) * np.array(
        [0, 1], dtype=float
    )
    psi = np.kron(psi, np.array([1, 0, 0, 0], dtype=float))
    return np.outer(psi, np.conj(psi))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    p, alpha, tampered_wire = inputs[0], inputs[1], int(inputs[2])

    error_readout = np.zeros(4, dtype=float)
    circuit_output = circuit(p, alpha, tampered_wire)
    error_readout = error_wire(circuit_output)

    print(*error_readout, sep=",")
