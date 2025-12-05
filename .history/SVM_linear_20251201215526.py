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


