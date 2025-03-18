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

def createPhaseOracle_1():
    #states: |0101>, |0110>, |1100>, |1111>
    oracle1 = QuantumCircuit(len)
    oracle1.barrier()
    oracle1.cz(0, 2)
    oracle1.cz(1, 2)
    oracle1.cz(2, 3)
    oracle1.barrier()
    return oracle1

def createPhaseOracle_2():
    #states: |0110>, |0111>, |1100>, |1101>
    oracle2 = QuantumCircuit(len)
    oracle2.barrier()
    oracle2.cz(1, 2)
    oracle2.cz(2, 3)
    oracle2.barrier()    
    return oracle2

def createDiffuser():
    diffuser = QuantumCircuit(len)
    diffuser.barrier()
    
    for qubit in range(len):
        diffuser.h(qubit)
        
    for qubit in range(len):
        diffuser.x(qubit)
        
    diffuser.h(len-1)
    diffuser.mcx(list(range(len-1)), len-1)
    diffuser.h(len-1)
    
    for qubit in range(len):
        diffuser.x(qubit)
        
    for qubit in range(len):
        diffuser.h(qubit)
    diffuser.barrier()
        
    return diffuser

def createCircuit1_grover():
    grover1 = QuantumCircuit(len)
    
    for qubit in range(len):
        grover1.h(qubit)
    
    #target states
    grover1 = grover1.compose(createPhaseOracle_1(), list(range(len)))

    grover1 = grover1.compose(createDiffuser(), list(range(len)))
    grover1.measure_all()
    
    return grover1

def createCircuit2_grover():
    grover2 = QuantumCircuit(len)
    
    for qubit in range(len):
        grover2.h(qubit)
    
    #target states
    grover2 = grover2.compose(createPhaseOracle_2(), list(range(len)))
        
    grover2 = grover2.compose(createDiffuser(), list(range(len)))
    grover2.measure_all()
    
    return grover2

def runCircuit_grover(grover):
    transpiledGrover = transpile(grover, backend)
    transpiledGrover.draw('mpl', idle_wires=False)
    
    job = backend.run([transpiledGrover], shots=1024)
    job.job_id()
    
    results = job.result()
    graph = results.get_counts()
    plot_histogram(graph)

def simulateCircuit_grover(grover):
    job = sim.run(grover)
    result = job.result()
    graph = result.get_counts()
    plot_histogram(graph)

def jobResults(job_id):
    job = service.job(job_id)
    results = job.result()
    graph = results.get_counts()
    plot_histogram(graph)


#número de qubits 
len = 4
if __name__ == "__main__":
    #create grover circuit (with oracle 1)
    grover_1 = createCircuit1_grover()

    #run it on real hardware
    #runCircuit_grover(grover_1)

    #print("grover_1 circuit for the first oracle:")
    #grover_1.draw('mpl', idle_wires=False)
    #plt.savefig("first_oracle_grover.png")

    print(grover_1)

    #print("grover_1 circuit simulation:")
    #print (grover_1)

    print("grover_1 simulation results:")
    simulateCircuit_grover(grover_1)

    print("grover_1 job results:")
    jobResults("cxdejvkgcckg008bgve0")




    # create grover circuit (with oracle 2)
    grover_2 = createCircuit2_grover()

    #run it on real hardware
    #runCircuit_grover(grover_2)

    #print("grover_2 circuit for the second oracle:")
    #grover_2.draw('mpl', idle_wires=False)
    #plt.savefig("second_oracle_grover.png")
    print(grover_2)

    #print("grover_2 circuit simulation:")
    #print (grover_2)

    print("grover_2 simulation results:")
    simulateCircuit_grover(grover_2)

    print("grover_2 job results:")
    jobResults("cxdek2cfdnwg008s0s00")

    plt.show()