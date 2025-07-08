from qiskit_ibm_runtime import QiskitRuntimeService

# Replace with your actual API key and CRN:
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",  # or "ibm_cloud"
    token= "gliKn-hLEkrO-DuC1yZj7Vy1TIa1VXWbybDNh5PHADIC",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/1848194af4c74331ac2b36ef9f8edf7f:817ae8db-c51a-4125-bf09-a8cabd3cb849::",
    set_as_default=True,
    overwrite=True
)