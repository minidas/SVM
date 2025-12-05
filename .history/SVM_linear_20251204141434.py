from quantum_utils import (
    H7,H5,
    poly_log_t,
    H_stab_state_evol,
    stab_01n_op,
    ini_state_01,
    plot_x_y,
)
from qiskit.quantum_info import SparsePauliOp

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
    test_ob.state_evol(ideal_state=False, den_mat_out=True, state_update=True)
    
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
        data_alpha_vs_t_tog[current_time] = test_ob.Wppt_alpha_tog
        data_tr_vs_num_evol_tog[a] = result
        data_size_vs_num_evol_tog[a] = size_tog
        data_alpha_vs_num_evol_tog[a] = test_ob.Wppt_alpha_tog

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
