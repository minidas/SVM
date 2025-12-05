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


data_tr_vs_t_sep = {}
data_tr_vs_t_ideal = {}
data_tr_vs_t_ideal_sep = {}
data_tr_vs_t_ideal_tog = {}
data_tr_vs_t_ideal_tog_sep = {}

test_ob = H_stab_state_evol(n, H7, stab_01n_op, ini_state_01, poly_log_t, steps)


for round_num in range(1, max_round + 1):
    print(f"\n--- Round {round_num} ---")
    
    # Evolve state (Trotterized)
    test_ob.state_evol(ideal_state=False, den_mat_out=True, state_update=True)
    
    # Evolve stabilizer sum with truncation
    test_ob.stablt_sum_evol(approx=True, up=limit, step_by_step=False, stab_sum_update=True)
    
    # Construct and test witness
    test_ob.W_PPT(sum=True)
    tr_tog = test_ob.test_W_PPT(sum=True)
    
    # Store data
    data_tr_vs_t_tog[round_num] = tr_tog
    
    # Check for entanglement
    if tr_tog > 0:
        print(f"✗ Round {round_num}: Entanglement NOT detected (separable state)")
        print(f"Stopping EW construction for limit={limit}")
        break
    else:
        print(f"✓ Round {round_num}: Entanglement detected (entangled state)")
        
        # Update limit tracking
        if limit not in qb_t_in_limits_tog:
            qb_t_in_limits_tog[limit] = []
        if limit not in qb_num_evol_in_limits_tog:
            qb_num_evol_in_limits_tog[limit] = []
            
        qb_t_in_limits_tog[limit].append(round_num)
        qb_num_evol_in_limits_tog[limit].append(round_num)

print(f"\nEvolution completed. Final results:")
print(f"  Rounds completed: {len(data_tr_vs_t_tog)}")
print(f"  Final trace value: {list(data_tr_vs_t_tog.values())[-1] if data_tr_vs_t_tog else 'N/A'}")
