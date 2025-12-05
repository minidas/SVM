from quantum_utils import (
    H7,H5,
    poly_log_t,
    H_stab_state_evol,
    stab_01n_op,
    ini_state_01,
    plot_x_y,
)
from qiskit.quantum_info import SparsePauliOp
import os
import pandas as pd
import matplotlib.pyplot as plt

n = 7
H = H7
stab_list = stab_01n_op
ini_state = ini_state_01
steps = 10
t = poly_log_t
limit = 0.01
turns = 5
max_round = 50
num_s = 1000
class_w = (1,1) #(separable, entangled)


t_list = list(range(1, max_round))

data_tr_vs_t_tog = {}
data_size_vs_t_tog = {}
data_alpha_vs_t_tog = {}
data_tr_vs_num_evol_tog = {}
data_size_vs_num_evol_tog = {}
data_alpha_vs_num_evol_tog = {}

test_ob = H_stab_state_evol(n, H7, stab_01n_op, ini_state_01, poly_log_t, steps)


for a in t_list:
    print(f'Round {a}/{max_round-1}: Processing {a} * 1/polylog')
    
    # Evolve state (Trotterized)
    test_ob.state_evol(ideal_state=True, den_mat_out=False, state_update=True)
    
    # Evolve stabilizer sum with truncation
    test_ob.stablt_sum_evol(approx=True, up=limit, step_by_step=False, stab_sum_update=True)
    
    
    if a % turns == 0:
        test_ob.SVM_linear(num_samples = num_s, class_weights=class_w)
        label, result = test_ob.test_EW_SVM()
        size_tog = test_ob.stabsum_tog.size

        # Store data
        current_time = a * t(n)
        data_tr_vs_t_tog[current_time] = result
        data_size_vs_t_tog[current_time] = size_tog
        data_alpha_vs_t_tog[current_time] = test_ob.svm_linear_model['model']['b']
        data_tr_vs_num_evol_tog[a] = result
        data_size_vs_num_evol_tog[a] = size_tog
        data_alpha_vs_num_evol_tog[a] = test_ob.svm_linear_model['model']['b']

        print(f'Entanglement detected')
        print(f'detected trace: {result}')

        # Check if entanglement is NOT detected (tr_tog > 0 means separable state)
        if result < 0:
            print(f'✗ Entanglement NOT detected (separable state)')
            print(f'detected trace: {result}')
            print(f'Stopping EW construction for limit={limit}')
            break

print(f"\nEvolution completed. Final results:")
print(f"  Rounds completed: {len(data_tr_vs_t_tog)}")
print(f"  Final trace value: {list(data_tr_vs_t_tog.values())[-1] if data_tr_vs_t_tog else 'N/A'}")

# Create results directory and save data + plots
results_dir = 'result'
os.makedirs(results_dir, exist_ok=True)

# Build a summary DataFrame indexed by time
if data_tr_vs_t_tog:
    times = sorted(data_tr_vs_t_tog.keys())
    rows = []
    for tt in times:
        rows.append({
            'time': tt,
            'trace': data_tr_vs_t_tog.get(tt, float('nan')),
            'size': data_size_vs_t_tog.get(tt, float('nan')),
            'alpha': data_alpha_vs_t_tog.get(tt, float('nan')),
        })
    df_time = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, 'summary_vs_time.csv')
    df_time.to_csv(csv_path, index=False)
    print(f'Saved time-series summary to {csv_path}')

    # Plot trace vs time
    plt.figure()
    plt.plot(df_time['time'], df_time['trace'], marker='o')
    plt.xlabel('time')
    plt.ylabel('trace (EW @ state)')
    plt.title('Witness trace vs time')
    plt.grid(True)
    trace_png = os.path.join(results_dir, 'trace_vs_time.png')
    plt.savefig(trace_png, bbox_inches='tight')
    plt.close()

    # Plot size vs time
    plt.figure()
    plt.plot(df_time['time'], df_time['size'], marker='o')
    plt.xlabel('time')
    plt.ylabel('stabilizer sum size')
    plt.title('Stabilizer size vs time')
    plt.grid(True)
    size_png = os.path.join(results_dir, 'size_vs_time.png')
    plt.savefig(size_png, bbox_inches='tight')
    plt.close()

    # Plot alpha vs time
    plt.figure()
    plt.plot(df_time['time'], df_time['alpha'], marker='o')
    plt.xlabel('time')
    plt.ylabel('alpha (svm b)')
    plt.title('SVM alpha vs time')
    plt.grid(True)
    alpha_png = os.path.join(results_dir, 'alpha_vs_time.png')
    plt.savefig(alpha_png, bbox_inches='tight')
    plt.close()

# Save evolution-indexed data if present
if data_tr_vs_num_evol_tog:
    evo_idxs = sorted(data_tr_vs_num_evol_tog.keys())
    rows_e = []
    for idx in evo_idxs:
        rows_e.append({
            'evol_step': idx,
            'trace': data_tr_vs_num_evol_tog.get(idx, float('nan')),
            'size': data_size_vs_num_evol_tog.get(idx, float('nan')),
            'alpha': data_alpha_vs_num_evol_tog.get(idx, float('nan')),
        })
    df_evol = pd.DataFrame(rows_e)
    csv_evol = os.path.join(results_dir, 'summary_vs_evolution.csv')
    df_evol.to_csv(csv_evol, index=False)
    print(f'Saved evolution summary to {csv_evol}')

    plt.figure()
    plt.plot(df_evol['evol_step'], df_evol['trace'], marker='o')
    plt.xlabel('evolution step')
    plt.ylabel('trace (EW @ state)')
    plt.title('Witness trace vs evolution step')
    plt.grid(True)
    trace_evo_png = os.path.join(results_dir, 'trace_vs_evolution.png')
    plt.savefig(trace_evo_png, bbox_inches='tight')
    plt.close()
