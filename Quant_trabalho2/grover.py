import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, assemble, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Operator, Statevector, DensityMatrix
from qiskit.visualization import array_to_latex
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
import os
from dotenv import load_dotenv

load_dotenv()
IBM_TOKEN = os.getenv("IBM_TOKEN")
service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
backend = service.least_busy(operational=True, simulator=False)
print(backend)
sim = AerSimulator()

LENGTH = 4
def create_phase_oracle_1():
    # Create a phase oracle to mark the states |0101>, |0110>, |1100>, |1111> states
    oracle = QuantumCircuit(LENGTH)
    oracle.barrier()
    oracle.cz(0, 2)
    oracle.cz(1, 2)
    oracle.cz(2, 3)
    oracle.barrier()
    return oracle

def create_phase_oracle_2():
    # Create a phase oracle to mark the states |0101>, |0110>, |1101>, |1110> states
    oracle = QuantumCircuit(LENGTH)
    oracle.barrier()
    oracle.cz(0, 2)
    oracle.cz(1, 2)
    oracle.barrier()
        
    return oracle

def create_diffuser():
    diffuser = QuantumCircuit(LENGTH)
    diffuser.barrier()
    
    for qubit in range(LENGTH):
        diffuser.h(qubit)
        
    for qubit in range(LENGTH):
        diffuser.x(qubit)
        
    diffuser.h(LENGTH-1)
    diffuser.mcx(list(range(LENGTH-1)), LENGTH-1)
    diffuser.h(LENGTH-1)
    
    for qubit in range(LENGTH):
        diffuser.x(qubit)
        
    for qubit in range(LENGTH):
        diffuser.h(qubit)
    diffuser.barrier()
        
    return diffuser

def create_grover_circuit(oracle_idx):
    grover = QuantumCircuit(LENGTH)
    
    # Initialize the qubits
    for qubit in range(LENGTH):
        grover.h(qubit)
    
    # Mark the target states
    if(oracle_idx == 1):
        grover = grover.compose(create_phase_oracle_1(), list(range(LENGTH)))
    elif(oracle_idx == 2):
        grover = grover.compose(create_phase_oracle_2(), list(range(LENGTH)))
        
    grover = grover.compose(create_diffuser(), list(range(LENGTH)))
    grover.measure_all()
    
    return grover

def run_grover_circuit(grover):
    transpiled_grover = transpile(grover, backend)
    transpiled_grover.draw('mpl', idle_wires=False)
    
    job = backend.run([transpiled_grover], shots=1024)
    job.job_id()
    
    results = job.result()
    answer = results.get_counts()
    plot_histogram(answer)

def simulate_grover_circuit(grover):
    job = sim.run(grover)
    result = job.result()
    answer = result.get_counts()
    plot_histogram(answer)

grover_1 = create_grover_circuit(1)
grover_2 = create_grover_circuit(2)

print(grover_1)
print(grover_2)

simulate_grover_circuit(grover_1)
simulate_grover_circuit(grover_2)

run_grover_circuit(grover_1)
run_grover_circuit(grover_2)


plt.show()