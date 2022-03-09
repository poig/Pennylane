#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qfunc_adder(m, wires):
    """Quantum function capable of adding m units to a basic state given as input.

    Args:
        - m (int): units to add.
        - wires (list(int)): list of wires in which the function will be executed on.
    """

    qml.QFT(wires=wires)

    # QHACK #
    #Modified Draper’s adder
    '''for i in wires:
        qml.Hadamard(wires=i)'''
    #qml.QuantumPhaseEstimation(unitary, wires, estimation_wires)

    '''for i in wires[::-1]:
        qml.Hadamard(wires=i)'''
    '''def qft_rotations(circuit, n):
    if n == 0: # Exit function if circuit is empty
        return circuit
    n -= 1 # Indexes start from 0
    circuit.h(n) # Apply the H-gate to the most significant qubit
    for qubit in range(n):
        # For each less significant qubit, we need to do a
        # smaller-angled controlled rotation: 
        circuit.cp(pi/2**(n-qubit), qubit, n)'''
    # QHACK #
    #Modified Draper’s adder
    '''num_add = np.array([int(a) for a in str("{0:0{1}b}".format(m,len(wires)))])
    num_add = list(reversed(num_add))

    for i in reversed(wires):
        if num_add[i] == 1:
            for i,j in zip(range(len(wires)),reversed(range(len(wires)))):
                qml.PhaseShift(((2**j)*m*np.pi/2**(len(wires))), wires=i)
            break'''
    for i,j in zip(range(len(wires)),reversed(range(len(wires)))):
            qml.PhaseShift(((2**(i+1))*m*np.pi/2**(len(wires))), wires=j)


            
            
    '''qml.PhaseShift(((2**3)*m*np.pi/2**(len(wires))), wires=0)
    qml.PhaseShift(((2**2)*m*np.pi/2**(len(wires))), wires=1)
    qml.PhaseShift(((2**1)*m*np.pi/2**(len(wires))), wires=2)'''
    #qml.CNOT(wires=[2,0])#i.in abs correct

    ###3
    #########
    
    ###########
    '''qml.PhaseShift(((2**1)*m*np.pi/2**(len(wires))), wires=3)
    qml.PhaseShift(((2**2)*m*np.pi/2**(len(wires))), wires=2)
    qml.PhaseShift(((2**3)*m*np.pi/2**(len(wires))), wires=1)
    qml.PhaseShift(((2**4)*m*np.pi/2**(len(wires))), wires=0)'''
    ######
    '''qml.PhaseShift(((2**4)*m*np.pi/2**(len(wires))), wires=0)
    qml.PhaseShift(((2**3)*m*np.pi/2**(len(wires))), wires=1)
    qml.PhaseShift(((2**2)*m*np.pi/2**(len(wires))), wires=2)
    qml.PhaseShift(((2**1)*m*np.pi/2**(len(wires))), wires=3)'''
    ######
    
    ####### corrct'''
    




    

    #qml.Hadamard(wires=i)
    # QHACK #

    qml.QFT(wires=wires).inv()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    m = int(inputs[0])
    n_wires = int(inputs[1])
    wires = range(n_wires)

    dev = qml.device("default.qubit", wires=wires, shots=1)

    @qml.qnode(dev)
    def test_circuit():
        # Input:  |2^{N-1}>
        qml.PauliX(wires=0)

        qfunc_adder(m, wires)
        return qml.sample()#qml.probs(wires)#,

    output = test_circuit()
    print(*output, sep=",")
