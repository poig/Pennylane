#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}


def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.

    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'

    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #
    
    '''from itertools import combinations

    gate_order = {"A":[], "B":[], "C":[], "D":[]}
    for i, (gs,v) in enumerate(graph.items()):
        for j in range(len(v)):
        #for i, j in combinations(qubits, 2):
            wire_1 = graph[i]
            wire_2 = graph[j]
            if i in j.neighbors():
                if i.row == j.row and i.col % 2 == 0:
                    gate_order["A"].append((wire_1, wire_2))
                elif i.row == j.row and j.col % 2 == 0:
                    gate_order["B"].append((wire_1, wire_2))
                elif i.col == j.col and i.row % 2 == 0:
                    gate_order["C"].append((wire_1, wire_2))
                elif i.col == j.col and j.row % 2 == 0:
                    gate_order["D"].append((wire_1, wire_2))
        count = 0
        cnot[0],cnot[1]
    for i, (gs,v) in enumerate(graph.items()):
        for j in range(len(v)):
            for qb_1, qb_2 in graph[gs]:
                qml.SWAP(wires=(qb_1, qb_2))
                count += 1'''
    '''count=0
    i = cnot[0]
    path = []
    times =0
    while not(graph[i] in graph[cnot[1]]):
        pair = []
        for j in len(graph[i]):
            if len(graph[[i][j]]) > 1 and not(graph[[i][j]] in pair):
                i=graph[[i][j]]
                pair.append(graph[[i][j]])
                path[times].append(graph[[i][j]])
                times+=1'''

        
    '''for i, (gs,v) in enumerate(graph.items()):
        for j in range(len(v)):
            print(i,graph[i][j])
    
    i = cnot[0]
    j = cnot[1]
    k= range(graph[i])[0]
    while not(graph[i] in graph[j]):
        i = graph[[i][k]]
        k+=1'''

    #Breadth-First Search
    def bfs(graph, start, end):
    # maintain a queue of paths
        queue = []
        # push the first path into the queue
        queue.append([start])
        while queue:
            # get the first path from the queue
            path = queue.pop(0)
            # get the last node from the path
            node = path[-1]
            # path found
            if node == end:
                return path
            # enumerate all adjacent nodes, construct a 
            # new path and push it into the queue
            for adjacent in graph.get(node, []):
                new_path = list(path)
                new_path.append(adjacent)
                queue.append(new_path)

    
    return (len(bfs(graph, cnot.wires[0], cnot.wires[1])) - 2) *2

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
