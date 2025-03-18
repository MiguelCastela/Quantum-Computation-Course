import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
import os
from dotenv import load_dotenv

if "IBM_TOKEN" in os.environ:
    del os.environ["IBM_TOKEN"]
load_dotenv()
IBM_TOKEN = os.getenv("IBM_TOKEN")
#print(f"IBM_TOKEN: {IBM_TOKEN}")
service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
backend = service.least_busy(operational=True, simulator=False)
sim = AerSimulator()
print(backend)

def constantOracle(len):
    oracle = QuantumCircuit(len+1)
    
    oracle.barrier()
    
    output = np.random.randint(2)
    if(output == 1):
        oracle.x(len)

    oracle.barrier()

    return oracle

def balancedOracle(len):
    oracle = QuantumCircuit(len+1)
    
    oracle.barrier()

    #Random implementation
    b = np.random.randint(1, 2**len)
    b_str = format(b, '0'+str(len)+'b')
    
    for qb in range(len):
        if(b_str[qb] == '1'):
            oracle.x(qb)

        
    for qb in range(len):
        oracle.cx(qb, len)
        
    for qb in range(len):
        if(b_str[qb] == '1'):
            oracle.x(qb)    

    oracle.barrier()

    #Static implementation 
    '''
    oracle = QuantumCircuit(len+1)

    oracle.barrier()

    for qb in range(len):
        if(qb % 2 == 0):
            oracle.x(qb)

    oracle.barrier()

    for qb in range(len):
        oracle.cx(qb, len)

    oracle.barrier()

    for qb in range(len):
        if(qb % 2 == 0):
            oracle.x(qb)    

    oracle.barrier()
    '''

    return oracle


def createOracle(len):
    constant = constantOracle(len)
    balanced = balancedOracle(len)
    
    return constant, balanced

def createCircuit_dj(len, oracle):
    dj = QuantumCircuit(len+1, len)
    
    #input qubits
    for qb in range(len):
        dj.h(qb)
        
    #output qubit
    dj.x(len)
    dj.h(len)
    
    #apply the oracle
    dj = dj.compose(oracle)
    
    #apply Hadamard gates to the first n qubits
    for qb in range(len):
        dj.h(qb)
        
    #measure the first n qubits
    for qb in range(len):
        dj.measure(qb, qb)
    
    return dj    

def simulateCircuit_dj(dj):
    job = sim.run(dj)
    results = job.result()
    graph = results.get_counts()
    plot_histogram(graph)

def runCircuit_dj(dj):
    transpiledCircuit = transpile(dj, backend, optimization_level=3)
    transpiledCircuit.draw('mpl', idle_wires=False)
    
    job = backend.run([transpiledCircuit], shots=1024)
    job.job_id()
    
    results = job.result()
    graph = results.get_counts()
    
    plot_histogram(graph)
    
def jobResults(job_id):
    job = service.job(job_id)
    results = job.result()
    graph = results.get_counts()
    plot_histogram(graph)
    plt.show()




#número de qubits 
len = 4
if __name__ == "__main__":

    #create the oracles
    constantOracle, balancedOracle = createOracle(len)

    #Deutsch-Jozsa circuit with balanced oracle
    balancedDj = createCircuit_dj(len, balancedOracle)

    #run it on real hardware
    #runCircuit_dj(balancedDj)

    #print("Deutch-Jozsa circuit with balanced oracle ")
    #balancedDj.draw('mpl', idle_wires=False)
    plt.savefig("balanced_oracle_circuit.png")

    print(balancedDj)

    #print("simulation of the Deutch-Jozsa circuit with balanced oracle ")
    #print(balancedDj)

    print("results of the simulation")
    simulateCircuit_dj(balancedDj)

    print("balanced iracle job results:")
    jobResults("cxde3j6vw7kg008atm10")



    #Deutsch-Jozsa circuit with the constant oracle
    constantDj = createCircuit_dj(len, constantOracle)

    # run it on real hardware
    #runCircuit_dj(constantDj)
    #print("Deutch-Jozsa circuit with constant oracle ")
    #constantDj.draw('mpl', idle_wires=False)
    plt.savefig("constant_oracle_circuit.png")

    print(constantDj)

    #print("simulation of the Deutch-Jozsa circuit with constant oracle ")
    #print(constantDj)

    print("results of the simulation")
    simulateCircuit_dj(constantDj)

    print("constant oracle job results:")
    jobResults("cxdejdh3ej4g0089x87g")

    #plt.show()
