from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram
from string import ascii_letters, digits
import numpy as np


NSHOTS = 1000000

sampler = StatevectorSampler()
qreg = QuantumRegister(1, 'q')  # Qubit
creg = ClassicalRegister(1, 'meas')  # Bit clássico -- para guardar resultados de medições
qc = QuantumCircuit(qreg, creg, name='polarizador')

#preparar
angulo = np.radians(22.5)

#medir
qc.ry(-2*angulo, qreg[0])

qc.measure(qreg[0], creg[0])

job = sampler.run([qc], shots=NSHOTS)
print(job.status())
job.status()
result = job.result()[0]
outcomes = result.data.meas.get_counts()
num_0 = outcomes.get('0', 0)

print(f'Número de 0s do primeiro circuito: {num_0}')

qc2 = QuantumCircuit(qreg, creg, name='polarizador')
qc2.ry(2*1*angulo, qreg[0])
qc2.ry(-2*2*angulo, qreg[0])
qc2.measure(qreg[0], creg[0])
job = sampler.run([qc2], shots=num_0)
print(job.status())
job.status()
result = job.result()[0]
outcomes = result.data.meas.get_counts()
num_0 = outcomes.get('0', 0)
print(f'Número de 0s do segundo circuito: {num_0}')


qc3 = QuantumCircuit(qreg, creg, name='polarizador')
qc3.ry(2*2*angulo, qreg[0])
qc3.ry(-2*3*angulo, qreg[0])
qc3.measure(qreg[0], creg[0])
job = sampler.run([qc3], shots=num_0)
print(job.status())
job.status()
result = job.result()[0]
outcomes = result.data.meas.get_counts()
num_0 = outcomes.get('0', 0)
print(f'Número de 0s do terceiro circuito: {num_0}')


qc4 = QuantumCircuit(qreg, creg, name='polarizador')
qc4.ry(2*3*angulo, qreg[0])
qc4.ry(-2*4*angulo, qreg[0])
qc4.measure(qreg[0], creg[0])
job = sampler.run([qc4], shots=num_0)
print(job.status())
job.status()
result = job.result()[0]
outcomes = result.data.meas.get_counts()
num_0 = outcomes.get('0', 0)
print(f'Número de 0s do quarto circuito: {num_0}')