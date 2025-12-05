#!/usr/bin/env python3
"""
Simple noise-robustness runner that uses the commutator-filtered stabilizer
sum (for H7) to train a linear SVM and run bisection noise tests for multiple
noise models. Saves CSV and PNG outputs under `result/commutator_noise/`.

This script is intentionally lightweight and self-contained so it can be
customised further if needed.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector, partial_trace, entropy, SparsePauliOp
from quantum_utils import H7, H_stab_state_evol, stab_01n_op, ini_state_01, poly_log_t, train_linear_svm_pauli, classify_state_with_svm


def calculate_entropy(state_or_dm, n):
    try:
        if hasattr(state_or_dm, 'to_matrix'):
            dm = state_or_dm.to_matrix()
        else:
            arr = np.asarray(state_or_dm)
            if arr.ndim == 1:
                dm = np.outer(arr, arr.conj())
            else:
                dm = arr
        subsystem_size = n // 2
        rdm = partial_trace(dm, list(range(0, subsystem_size)))
        return float(np.real(entropy(rdm)))
    except Exception:
        return float('nan')


def apply_noise_from_obj(ent_obj, noise_type, p):
    if noise_type == 'white':
        return ent_obj.noisy_state(p)
    elif noise_type == 'bitflip':
        return ent_obj.noisy_bit_flip(p)
    elif noise_type == 'phaseflip':
        return ent_obj.noisy_phase_flip(p)
    elif noise_type == 'bitphase':
        return ent_obj.noisy_bit_phase_flip(p)
    elif noise_type == 'amplitude':
        return ent_obj.noisy_amplitude_damping(p)
    else:
        raise ValueError(f'Unknown noise type: {noise_type}')


def bisection_noise_for_svm(svm_model, ent_obj, n, noise_type='white', tol=1e-6, max_iter=50):
    noise_data = []

    # test clean state
    try:
        rho_clean = Statevector.to_operator(ent_obj.ini_state).to_matrix()
        lbl, score = classify_state_with_svm(ent_obj.ini_state, svm_model, n=n, return_score=True)
        ent_clean = calculate_entropy(rho_clean, n)
        lbl_num = 1 if str(lbl).lower().startswith('ent') else -1
        noise_data.append({'p_noise': 0.0, 'score': float(score), 'entropy': ent_clean, 'label': lbl_num})
        if score < 0:
            return noise_data, 0.0
    except Exception as e:
        print('Error testing clean state:', e)
        return noise_data, 0.0

    p_lo = 0.0
    p_hi = 1.0
    max_noise = 0.0

    for it in range(max_iter):
        p_mid = (p_lo + p_hi) / 2.0
        try:
            rho_noisy = apply_noise_from_obj(ent_obj, noise_type, p_mid)
            lbl, score = classify_state_with_svm(rho_noisy, svm_model, n=n, return_score=True)
            ent_mid = calculate_entropy(rho_noisy, n)
            lbl_num = 1 if str(lbl).lower().startswith('ent') else -1
            noise_data.append({'p_noise': p_mid, 'score': float(score), 'entropy': ent_mid, 'label': lbl_num})

            if p_hi - p_lo < tol:
                break

            if score > 0:
                max_noise = p_mid
                p_lo = p_mid
                print(f"  Iter {it}: p={p_mid:.6f}, score={score:.6f} detected")
            else:
                p_hi = p_mid
                print(f"  Iter {it}: p={p_mid:.6f}, score={score:.6f} not detected")

        except Exception as e:
            print(f"  Error at p={p_mid}: {e}")
            p_hi = p_mid

    return noise_data, max_noise


def main():
    parser = argparse.ArgumentParser(description='SVM noise robustness using commutator-filtered stabilizer sum (H7)')
    parser.add_argument('--n', type=int, default=7)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--limit', type=float, default=0.03)
    parser.add_argument('--turns', type=int, default=4)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--noise-types', nargs='+', default=['white'], choices=['white','bitflip','phaseflip','bitphase','amplitude'])
    parser.add_argument('--comm-npy', type=str, default='commutator_paulis_dict.npy')
    args = parser.parse_args()

    n = args.n
    num_s = args.num_samples
    limit = args.limit
    turns = args.turns
    steps = args.steps
    noise_types = args.noise_types
    comm_npy = args.comm_npy

    results_dir = os.path.join('result', 'commutator_noise')
    os.makedirs(results_dir, exist_ok=True)

    # Setup object and evolution
    test_ob = H_stab_state_evol(n, H7, stab_01n_op, ini_state_01, poly_log_t, steps)

    print('Evolving state and stabilizer sum...')
    test_ob.state_evol(ideal_state=True, den_mat_out=False, state_update=True)
    test_ob.stablt_sum_evol(approx=True, up=limit, step_by_step=True, stab_sum_update=True)

    # Apply commutator-based truncation
    truncated = test_ob.stablt_sum_truncate_by_commutator(commutator_npy=comm_npy, stab_sum_update=True)
    print(f'Truncated stabilizer sum size: {truncated.size}')

    # Prepare normalized operator for training
    phi_appro = (truncated + SparsePauliOp.from_list([('I'*n, 1.0)]) ) * (2 ** (-n))

    print('Training linear SVM...')
    svm_res = train_linear_svm_pauli(n, phi_appro, num_samples=num_s, class_weights=(1.0,1.0))

    # For each noise model, run bisection and save results
    for nt in noise_types:
        print(f'Running noise analysis for: {nt}')
        test_ob.noise_type = nt
        noise_data, max_noise = bisection_noise_for_svm(svm_res, test_ob, n, noise_type=nt)

        df = pd.DataFrame(noise_data)
        fname = f'comm_noise_n{n}_nt{nt}_samples{num_s}.csv'
        fpath = os.path.join(results_dir, fname)
        df.to_csv(fpath, index=False)
        print('Saved', fpath)

        # Plot score vs noise
        plt.figure(figsize=(8,5))
        detected = df[df['label']==1]
        notdet = df[df['label']==-1]
        if not detected.empty:
            plt.scatter(detected['p_noise'], detected['score'], c='tab:blue', label='detected')
        if not notdet.empty:
            plt.scatter(notdet['p_noise'], notdet['score'], c='tab:orange', label='not detected')
        plt.axhline(0, color='r', linestyle='--')
        plt.axvline(max_noise, color='g', linestyle='--', label=f'max={max_noise:.4f}')
        plt.xlabel('p_noise')
        plt.ylabel('SVM score')
        plt.legend()
        plt.grid(True)
        pplot = fpath.replace('.csv', '_score.png')
        plt.savefig(pplot, bbox_inches='tight')
        plt.close()

        # Plot entropy
        plt.figure(figsize=(8,5))
        plt.plot(df['p_noise'], df['entropy'], marker='s')
        plt.axvline(max_noise, color='g', linestyle='--')
        plt.xlabel('p_noise')
        plt.ylabel('Entropy')
        plt.grid(True)
        pplot2 = fpath.replace('.csv', '_entropy.png')
        plt.savefig(pplot2, bbox_inches='tight')
        plt.close()
        print('Saved plots for', nt)

    print('Done.')

if __name__ == '__main__':
    main()
