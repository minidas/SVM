from quantum_utils import H7, poly_log_t, H_stab_state_evol, stab_01n_op, ini_state_01
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_noise_filename(n, limit, turns, train_step):
    cw_str = f"n{n}_limit{limit}_turns{turns}_step{train_step}"
    filename = f"noise_wppt_{cw_str}.csv"
    return filename


def calculate_entropy(state_or_dm, n):
    try:
        if isinstance(state_or_dm, Statevector):
            full_dm = Statevector.to_operator(state_or_dm).to_matrix()
        elif hasattr(state_or_dm, 'to_matrix'):
            full_dm = state_or_dm.to_matrix()
        else:
            arr = np.asarray(state_or_dm)
            if arr.ndim == 1:
                full_dm = np.outer(arr, arr.conj())
            else:
                full_dm = arr

        # trace out second half
        subsystem_size = n // 2
        subsystem_indices = list(range(0, subsystem_size))
        rdm = partial_trace(full_dm, subsystem_indices)
        return float(np.real(entropy(rdm)))
    except Exception as e:
        print(f"Warning: Could not calculate entropy: {e}")
        return float('nan')


def find_noise_tolerance_for_witness(W, ent_obj, n, noise_tolerance=1e-6, max_iterations=50):
    """Bisection search on depolarizing white noise strength p where Tr(W rho_noisy) <= 0."""
    noise_data = []

    # test clean state
    try:
        rho_clean = Statevector.to_operator(ent_obj.ini_state).to_matrix()
        tr_clean = float(np.real(np.trace(W @ rho_clean)))
        ent_clean = calculate_entropy(rho_clean, n)
        label_clean = 1 if tr_clean <= 0 else -1
        noise_data.append({
            'p_noise': 0.0,
            'trace_witness': tr_clean,
            'entropy': ent_clean,
            'label': label_clean,
        })
        if label_clean == -1:
            print(f"  Warning: Clean state NOT detected by witness (trace={tr_clean:.6f})")
            return noise_data, 0.0
    except Exception as e:
        print(f"  Error testing clean state: {e}")
        return noise_data, 0.0

    p_low = 0.0
    p_high = 1.0
    max_noise = 0.0

    for it in range(max_iterations):
        p_mid = (p_low + p_high) / 2.0
        # noisy state: (1-p) rho + p I/2^n
        dim = 2 ** n
        rho_noisy = (1.0 - p_mid) * rho_clean + p_mid * np.eye(dim, dtype=complex) / dim
        tr_mid = float(np.real(np.trace(W @ rho_noisy)))
        ent_mid = calculate_entropy(rho_noisy, n)
        label_mid = 1 if tr_mid <= 0 else -1
        noise_data.append({'p_noise': p_mid, 'trace_witness': tr_mid, 'entropy': ent_mid, 'label': label_mid})

        if p_high - p_low < noise_tolerance:
            break

        if label_mid == 1:
            max_noise = p_mid
            p_low = p_mid
            print(f"    Iter {it}: p={p_mid:.6f}, trace={tr_mid:.6f} ✓ (detected)")
        else:
            p_high = p_mid
            print(f"    Iter {it}: p={p_mid:.6f}, trace={tr_mid:.6f} ✗ (not detected)")

    return noise_data, max_noise


def main():
    # Configuration (tweak as needed)
    n = 7
    H = H7
    steps = 10
    t = poly_log_t
    limit = 0.03
    turns = 4
    max_round = 20

    t_list = list(range(1, max_round))

    test_ob = H_stab_state_evol(n, H7, stab_01n_op, ini_state_01, poly_log_t, steps)

    results = {}
    noise_results = {}

    print(f"Starting WPPT noise robustness (n={n})")
    print("=" * 60)

    for a in t_list:
        print(f'Round {a}/{max_round-1}: Processing {a} * 1/polylog')
        test_ob.state_evol(ideal_state=True, den_mat_out=False, state_update=True)
        test_ob.stablt_sum_evol(approx=True, up=limit, step_by_step=True, stab_sum_update=True)

        if a % turns == 0:
            try:
                test_ob.W_PPT(sum=True)
                tr_tog = test_ob.test_W_PPT(sum=True)
                size_tog = test_ob.stabsum_tog.size
            except Exception as e:
                print(f"  Error constructing/testing WPPT: {e}")
                continue

            results[a] = {'time': a * t(n), 'trace': tr_tog, 'size': size_tog, 'alpha': test_ob.Wppt_alpha_tog}

            # entanglement detected if trace <= 0
            if tr_tog > 0:
                print(f'✗ Entanglement NOT detected (separable)')
                print(f'  Stopping WPPT construction for limit={limit}')
                break

            print(f'✓ Entanglement detected (trace={tr_tog:.6f})')

            # perform noise robustness on the witness operator
            W = test_ob.Wppt_tog
            print(f"  Performing noise robustness test for WPPT at step {a}...")
            noise_data, max_noise = find_noise_tolerance_for_witness(W, test_ob, n)
            noise_results[a] = {'noise_data': noise_data, 'max_noise': max_noise}
            print(f"  Maximum noise tolerance: p_noise = {max_noise:.6f}")

    # save noise_results
    results_dir = os.path.join('result', 'noise_wppt')
    os.makedirs(results_dir, exist_ok=True)

    for step_idx, step_data in noise_results.items():
        noise_data = step_data['noise_data']
        max_noise = step_data['max_noise']
        df = pd.DataFrame(noise_data)
        fname = generate_noise_filename(n, limit, turns, step_idx)
        path = os.path.join(results_dir, fname)
        df.to_csv(path, index=False)
        print(f'Saved WPPT noise results to {path}')

        # score/trace plot
        plt.figure(figsize=(8, 5))
        plt.plot(df['p_noise'], df['trace_witness'], marker='o')
        plt.axhline(0, color='r', linestyle='--')
        plt.axvline(max_noise, color='g', linestyle='--', label=f'max_noise={max_noise:.4f}')
        plt.xlabel('p_noise')
        plt.ylabel('Trace(W rho_noisy)')
        plt.title(f'WPPT Trace vs Noise (step {step_idx})')
        plt.legend()
        plt.grid(True)
        plotp = path.replace('.csv', '_trace.png')
        plt.savefig(plotp, bbox_inches='tight')
        plt.close()
        print(f'  Saved plot to {plotp}')

        # entropy plot
        plt.figure(figsize=(8, 5))
        plt.plot(df['p_noise'], df['entropy'], marker='s', color='orange')
        plt.axvline(max_noise, color='g', linestyle='--')
        plt.xlabel('p_noise')
        plt.ylabel('Entropy')
        plt.title(f'Entropy vs Noise (step {step_idx})')
        plt.grid(True)
        plotp2 = path.replace('.csv', '_entropy.png')
        plt.savefig(plotp2, bbox_inches='tight')
        plt.close()
        print(f'  Saved plot to {plotp2}')

    print('\nWPPT noise robustness analysis complete!')


if __name__ == '__main__':
    main()
