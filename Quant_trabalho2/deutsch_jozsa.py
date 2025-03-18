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

def create_const_oracle(LENGTH):
    oracle = QuantumCircuit(LENGTH+1)

    oracle.barrier()

    output = np.random.randint(2)
    if(output == 1):
        oracle.x(LENGTH)

    oracle.barrier()

    return oracle

def create_balanced_oracle(LENGTH):
    oracle = QuantumCircuit(LENGTH+1)

    oracle.barrier()

    b = np.random.randint(1, 2**LENGTH)
    bstr = format(b, '0'+str(LENGTH)+'b')

    for qb in range(LENGTH):
        if(bstr[qb] == '1'):
            oracle.x(qb)

    for qb in range(LENGTH):
        oracle.cx(qb, LENGTH)

    for qb in range(LENGTH):
        if(bstr[qb] == '1'):
            oracle.x(qb)

    oracle.barrier()
    return oracle

def oracle_creation(LENGTH):
    # Create a constant oracle
    c_oracle = create_const_oracle(LENGTH)
    # Create a balanced oracle
    b_oracle = create_balanced_oracle(LENGTH)

    return c_oracle, b_oracle

def create_dj_circuit(LENGTH, oracle):
    dj = QuantumCircuit(LENGTH+1, LENGTH)

    # Stage 1: Initialize the first n qubits to |+> (input qubits)
    for qb in range(LENGTH):
        dj.h(qb)

    # Stage 2: Initialize the output qubit to |->
    dj.x(LENGTH)
    dj.h(LENGTH)

    # Stage 3: Apply the chosen oracle
    dj = dj.compose(oracle)

    # Stage 4: Apply Hadamard gates to the first n qubits
    for qb in range(LENGTH):
        dj.h(qb)
    dj.barrier()

    # Stage 5: Measure the first n qubits
    for qb in range(LENGTH):
        dj.measure(qb, qb)

    return dj

def simulate_dj_circuit(dj):
    job = sim.run(dj)
    result = job.result()
    answer = result.get_counts()
    plot_histogram(answer)

def run_dj_circuit(dj):
    transpiled_dj = transpile(dj, backend, optimization_level=3)
    transpiled_dj.draw('mpl', idle_wires=False)

    job = backend.run([transpiled_dj], shots=1024)
    job.job_id()

    results = job.result()
    answer = results.get_counts()

    plot_histogram(answer)

def fetch_job_results(job_id):
    #job = backend.job(job_id)
    job = service.job(job_id)
    results = job.result()
    answer = results.get_counts()
    plot_histogram(answer)
    plt.show()

LENGTH = 6
if __name__ == "__main__":
    # Create the oracles
    c_oracle, b_oracle = oracle_creation(LENGTH)


    # BALANCED ORACLE
    dj = create_dj_circuit(LENGTH, b_oracle)
    # Run a simulation of the circuit
    simulate_dj_circuit(dj)
    # Run the circuit on real hardware
    # run_dj_circuit(dj)
    fetch_job_results("cxb0tntpjw30008h8bg0")
    print(dj)

    # CONSTANT ORACLE
    dj = create_dj_circuit(LENGTH, c_oracle)
    # Run a simulation of the circuit
    simulate_dj_circuit(dj)
    # Run the circuit with real hardware
    #run_dj_circuit(dj)
    fetch_job_results("cxaz8tvpjw30008h87bg")


    plt.show()
