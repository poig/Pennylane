#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import QuantumPhaseEstimation


dev = qml.device("default.qubit", wires=8)

def oracle_matrix(indices):
    """Return the oracle matrix for a secret combination.

    Args:
        - indices (list(int)): A list of bit indices (e.g. [0,3]) representing the elements that are map to 1.

    Returns:
        - (np.ndarray): The matrix representation of the oracle
    """

    # QHACK #
    #index = np.ravel_multi_index(list(indices), 2*len(indices)) # Index of solution
    
    my_array = np.identity(2**4) # Create the identity matrix len(indices)
    for i in indices:
        my_array[i,i] = -1
    # QHACK #

    return my_array


def diffusion_matrix():

    # DO NOT MODIFY anything in this code block

    psi_piece = (1 / 2 ** 4) * np.ones(2 ** 4)
    ident_piece = np.eye(2 ** 4)
    return 2 * psi_piece - ident_piece


def grover_operator(indices):

    # DO NOT MODIFY anything in this code block

    return np.dot(diffusion_matrix(), oracle_matrix(indices))


dev = qml.device("default.qubit", wires=8)

@qml.qnode(dev)
def circuit(indices):
    """Return the probabilities of each basis state after applying QPE to the Grover operator

    Args:
        - indices (list(int)): A list of bits representing the elements that map to 1.

    Returns:
        - (np.tensor): Probabilities of measuring each computational basis state
    """

    # QHACK #

    target_wires = [i for i in range(0,4)]
    estimation_wires = [i for i in range(4,8)]

    # Build your circuit here
    for i in target_wires:
        qml.Hadamard(wires=i)
    QuantumPhaseEstimation(grover_operator(indices), target_wires=target_wires, estimation_wires=estimation_wires)
    

    # QHACK #

    return qml.probs (estimation_wires)

def number_of_solutions(indices):
    """Implement the formula given in the problem statement to find the number of solutions from the output of your circuit

    Args:
        - indices (list(int)): A list of bits representing the elements that map to 1.

    Returns:
        - (float): number of elements as estimated by the quantum counting algorithm
    """

    # QHACK #
    probs = np.array(circuit(indices))
    max = np.amax(probs)
    decimal = int(np.where(probs==max)[0])
    theta = decimal * (np.pi/8)
    m = 16* (np.sin(theta/2)**2)
    #max = np.amax(probs)
    '''decimal_ = np.where(probs==max)
    max_2=np.partition(probs.flatten(), -2)[-1]
    max_3=np.partition(probs.flatten(), -2)[-2]
    decimal_2 = np.where(probs==max_2)
    decimal_3 = np.where(probs==max_3)
    decimal__ = int(decimal_3[0])'''
    '''m=[]
    decimal = []
    for i in range(len(indices)):
        max = np.partition(probs.flatten(), -2)[-(i+1)]
        decimal.append(np.where(probs==max))
        theta = int(decimal[i][0]) * (np.pi/8)
        m.append(16* (np.sin(theta/2)**2))'''


    '''probs.remove(max)
    max_2 = np.amax(probs)
    decimal_ = np.where(probs==max_2)'''
    return  m
    # QHACK #

def relative_error(indices):
    """Calculate the relative error of the quantum counting estimation

    Args:
        - indices (list(int)): A list of bits representing the elements that map to 1.

    Returns: 
        - (float): relative error
    """

    # QHACK #
    #compare number of solution
    rel_err =  ((number_of_solutions(indices) - len(indices))/len(indices))*100
    #rel_err = number_of_solutions(indices)

    # QHACK #

    return rel_err

if __name__ == '__main__':
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    lst=[int(i) for i in inputs]
    output = relative_error(lst)
    print(f"{output}")