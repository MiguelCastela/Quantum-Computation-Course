from qiskit_ibm_runtime import QiskitRuntimeService

# Initialize service and select backend
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)

# Fetch up to 50 completed jobs from that backend
jobs = service.jobs(limit=50, backend_name=backend.name, pending=False)

correct = 0
total = 0

for job in jobs:
    try:
        res0 = job.result()[0]
    except Exception:
        continue  # skip failed or incomplete jobs

    ev_arr = res0.data.evs
    err_arr = res0.data.stds
    ev  = float(ev_arr)  if ev_arr.shape == () else float(ev_arr[0])
    err = float(err_arr) if err_arr.shape == () else float(err_arr[0])

    pred = 1 if ev >= 0 else 0
    true_label = res0.metadata.get('true_label')
    if true_label is None:
        continue  # skip if no label provided

    total += 1
    if pred == true_label:
        correct += 1

    print(f"Job {job.job_id()}: ev={ev:.3f}±{err:.3f}, pred={pred}, true={true_label}")

if total > 0:
    accuracy = correct / total * 100
    print(f"\n✅ Accuracy over {total} samples: {accuracy:.1f}%")
else:
    print("⚠️ No jobs with stored true labels found.")
