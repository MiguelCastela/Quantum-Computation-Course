from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

# Function to encode an image in FRQI format
def frqi_encoding(image):
    n = int(np.log2(len(image)))  # Number of qubits needed
    qreg = QuantumRegister(n + 1)  # Extra qubit for grayscale
    qc = QuantumCircuit(qreg)
    
    # Initialize with Hadamard transform
    for i in range(n):
        qc.h(qreg[i])
    
    # Encode pixel intensities
    for i, pixel in enumerate(image):
        theta = 2 * np.arccos(np.sqrt(pixel))
        bin_index = format(i, f'0{n}b')
        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])
        qc.cry(theta, qreg[n-1], qreg[n])
        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])
    
    return qc

# Function to encode an image in NEQR format
def neqr_encoding(image):
    n = int(np.log2(len(image)))
    qreg = QuantumRegister(n + 8)  # Extra 8 qubits for grayscale (256 levels)
    qc = QuantumCircuit(qreg)
    
    # Initialize position qubits with Hadamard
    for i in range(n):
        qc.h(qreg[i])
    
    # Encode pixel values
    for i, pixel in enumerate(image):
        bin_pixel = format(int(pixel * 255), '08b')
        bin_index = format(i, f'0{n}b')
        
        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])
        
        for j, bit in enumerate(bin_pixel):
            if bit == '1':
                qc.mcx(qreg[:n], qreg[n + j])
        
        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])
    
    return qc

# Example grayscale image (normalized)
image = np.array([0.1, 0.5, 0.8, 0.3])

# Generate FRQI and NEQR circuits
frqi_circuit = frqi_encoding(image)
neqr_circuit = neqr_encoding(image)

# Draw the circuits     
print("FRQI Circuit:")
print(frqi_circuit.draw())

print("NEQR Circuit:")
print(neqr_circuit.draw())
