import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library.data_preparation import Initialize
from qiskit.visualization import plot_histogram, plot_bloch_multivector, array_to_latex
from qiskit.result import marginal_counts
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, random_statevector, partial_trace, random_unitary
# Simulation
from qiskit_aer import Aer, AerSimulator
#rs
from qiskit_ibm_runtime import QiskitRuntimeService

IBM_TOKEN = "IBM_TOKEN"
service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
backend = service.least_busy(operational=True, simulator=False)
print(backend)

sim = AerSimulator()

def oracle_single_marked(n):
    oracle = QuantumCircuit(n)

    oracle.cz(0, n-1)
    return oracle



def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc

def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mcx(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    return qc


'''
def run_grover_circuit(grover):
    transpiled_grover = transpile(grover, backend, optimization_level=3)
    transpiled_grover.draw('mpl', idle_wires=False)
    
    job = backend.run([transpiled_grover], shots=1024)
    job.job_id()
    
    results = job.result()
    answer = results.get_counts()
    
    plot_histogram(answer)

'''
def simulate_grover_circuit(grover):
    job = sim.run(grover)
    #print(job.result())
    result = job.result()
    answer = result.get_counts()
    plot_histogram(answer)


def run_grover_circuit(grover):
    transpiled_grover = transpile(grover, backend, optimization_level=3)
    transpiled_grover.draw('mpl', idle_wires=False)
    
    job = backend.run([transpiled_grover], shots=1024)
    job.job_id()
    
    results = job.result()
    answer = results.get_counts()
    
    plot_histogram(answer)

if __name__ == "__main__":
    n = 5

    qc = QuantumCircuit(n)
    qc.cz(0, 3)
    qc.cz(1, 3)
    qc.cz(2,3)
    
    oracle_ex3 = qc
    #oracle_ex3 = qc.to_gate()
    ##oracle_ex3.name = "asd"
    grover_circuit = QuantumCircuit(n)
    # Add the steps to the Grover's circuit
    grover_circuit = initialize_s(grover_circuit, range(n))
    grover_circuit = grover_circuit.compose(oracle_ex3)

    grover_circuit = grover_circuit.compose(diffuser(n))
    grover_circuit.measure_all()
    grover_circuit.draw('mpl')
    simulate_grover_circuit(grover_circuit)

    run_grover_circuit(grover_circuit)
    #grover_circuit.append(oracle_ex3, [0, 1, 2, 3])
    #grover_circuit.append(diffuser(n), [0, 1, 2, 3])
    ## Draw the circuit

    oracle = oracle_single_marked(n)
    grover_circuit2 = QuantumCircuit(n)
    grover_circuit2 = initialize_s(grover_circuit2, range(n))
    grover_circuit2 = grover_circuit2.compose(oracle)
    grover_circuit2 = grover_circuit2.compose(diffuser(n))
    grover_circuit2.measure_all()
    grover_circuit2.draw('mpl')
    simulate_grover_circuit(grover_circuit2)


    run_grover_circuit(grover_circuit)


    plt.show()
