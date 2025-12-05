from quantum_utils import (
    H7, H5, H6,
    poly_log_t,
    H_stab_state_evol,
    stab_01n_op,
    ini_state_01,
    classify_state_with_svm,
)
from qiskit.quantum_info import SparsePauliOp, partial_trace, entropy
from qiskit.quantum_info import Statevector
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def extract_svm_intercept(svm_model):
    """Extract intercept (b) from sklearn SVC or cvxpy dict model."""
    if isinstance(svm_model, dict):
        model_obj = svm_model.get('model')
        if isinstance(model_obj, dict) and 'b' in model_obj:
            return float(model_obj['b'])
    
    # Try sklearn model
    if hasattr(svm_model, 'intercept_'):
        try:
            return float(np.asarray(svm_model.intercept_).ravel()[0])
        except Exception:
            return float(svm_model.intercept_)
    
    # Try sklearn model wrapped in dict
    if isinstance(svm_model, dict):
        model_obj = svm_model.get('model')
        if hasattr(model_obj, 'intercept_'):
            try:
                return float(np.asarray(model_obj.intercept_).ravel()[0])
            except Exception:
                return float(model_obj.intercept_)
    
    return float('nan')


def generate_noise_filename(kernel_type, n, limit, turns, num_samples, class_w, train_step, mar_neg=1.0, mar_pos=1.0, noise_type='white'):
    """Generate descriptive filename for noise robustness results."""
    cw_str = f"{class_w[0]:.1f}_{class_w[1]:.1f}".replace('.', 'p')
    margin_str = f"m{mar_neg:.1f}_{mar_pos:.1f}".replace('.', 'p') if (mar_neg != 1.0 or mar_pos != 1.0) else ""
    filename = (
        f"noise_{kernel_type}_{noise_type}_n{n}_limit{limit}_turns{turns}_step{train_step}_"
        f"num{num_samples}_cw{cw_str}{margin_str}.csv"
    )
    return filename


def calculate_entropy(state_or_dm, n):
    """Calculate entanglement entropy of the reduced density matrix."""
    try:
        # Convert to density matrix if needed
        # qiskit Statevector -> convert via to_operator
        from qiskit.quantum_info import Statevector as _SV
        if isinstance(state_or_dm, _SV):
            full_dm = _SV.to_operator(state_or_dm).to_matrix()
        elif hasattr(state_or_dm, 'to_matrix'):
            full_dm = state_or_dm.to_matrix()
        else:
            full_dm = np.asarray(state_or_dm)
        
        # If it's a 1D statevector, convert to density matrix
        if full_dm.ndim == 1:
            psi = full_dm
            full_dm = np.outer(psi, psi.conj())
        elif full_dm.ndim == 2 and full_dm.shape[0] != full_dm.shape[1]:
            # Unexpected shape, try to reshape or raise
            raise ValueError('Provided state has incompatible shape for density matrix')
        
        subsystem_size = n // 2
        subsystem_indices = list(range(0, subsystem_size))
        
        # Calculate reduced density matrix
        rdm = partial_trace(full_dm, subsystem_indices)
        
        # Calculate entropy
        ent = entropy(rdm)
        return float(np.real(ent))
    except Exception as e:
        print(f"Warning: Could not calculate entropy: {e}")
        return float('nan')


def test_noise_robustness(svm_model, ent_obj, n, noise_tolerance=1e-6, max_iterations=50, compute_entropy=False):
    """
    Use bisection method to find maximum noise level where model still detects entanglement.
    Entropy calculation is optional (disabled by default) to speed up runs.
    """
    noise_data = []

    # Test noise-free state first
    state_clean = ent_obj.ini_state
    try:
        label, score = classify_state_with_svm(state_clean, svm_model, n=n, return_score=True)

        # Map textual label to integer: 'entangled' -> 1, 'separable' -> -1
        lbl_num = 1 if str(label).lower().startswith('ent') else -1
        entry = {'p_noise': 0.0, 'score': float(score), 'label': lbl_num}
        if compute_entropy:
            try:
                entry['entropy'] = calculate_entropy(ent_obj.ini_state, n)
            except Exception:
                entry['entropy'] = float('nan')

        noise_data.append(entry)

        if score < 0:
            print(f"  Warning: Clean state not detected as entangled (score={score:.4f})")
            return noise_data, 0.0
    except Exception as e:
        print(f"  Error testing clean state: {e}")
        return noise_data, 0.0

    # Bisection search for maximum noise tolerance
    p_low = 0.0
    p_high = 1.0
    max_noise = 0.0

    # Determine noise type (prefer ent_obj attribute)
    noise_type = getattr(svm_model if isinstance(svm_model, dict) else {}, 'noise_type', None)
    if hasattr(ent_obj, 'noise_type') and ent_obj.noise_type is not None:
        noise_type = ent_obj.noise_type

    def apply_noise(p):
        if noise_type is None or noise_type == 'white':
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
            return ent_obj.noisy_state(p)

    for iteration in range(max_iterations):
        p_mid = (p_low + p_high) / 2.0

        try:
            state_noisy = apply_noise(p_mid)
            label, score = classify_state_with_svm(state_noisy, svm_model, n=n, return_score=True)

            entry = {'p_noise': p_mid, 'score': float(score)}
            if compute_entropy:
                try:
                    entry['entropy'] = calculate_entropy(state_noisy, n)
                except Exception:
                    entry['entropy'] = float('nan')

            lbl_num = 1 if str(label).lower().startswith('ent') else -1
            entry['label'] = lbl_num
            noise_data.append(entry)

            if p_high - p_low < noise_tolerance:
                break

            if score > 0:
                max_noise = p_mid
                p_low = p_mid
                print(f"    Iteration {iteration}: p_noise={p_mid:.6f}, score={score:.4f} ✓ (entangled)")
            else:
                p_high = p_mid
                print(f"    Iteration {iteration}: p_noise={p_mid:.6f}, score={score:.4f} ✗ (separable)")

        except Exception as e:
            print(f"    Error at p_noise={p_mid:.6f}: {e}")
            p_high = p_mid

    return noise_data, max_noise


# Parse CLI arguments
parser = argparse.ArgumentParser(description='Linear SVM noise robustness testing')
parser.add_argument('--n', type=int, default=7, help='number of qubits')
parser.add_argument('--H', type=str, choices=['H7', 'H5', 'H6'], default='H7', help='Hamiltonian choice')
parser.add_argument('--limit', type=float, default=0.03, help='truncation limit for stabilizer sum')
parser.add_argument('--turns', type=int, default=4, help='train SVM every N evolution steps')
parser.add_argument('--max-round', type=int, default=20, help='maximum evolution rounds')
parser.add_argument('--num-samples', type=int, default=1000, help='samples per SVM training')
parser.add_argument('--class-w', nargs=2, type=float, default=[1.0, 1.0], help='class weights for -1 and +1')
parser.add_argument('--mar-neg', type=float, default=1.0, help='negative margin')
parser.add_argument('--mar-pos', type=float, default=1.0, help='positive margin')
parser.add_argument('--steps', type=int, default=10, help='Trotter steps')
parser.add_argument('--noise-type', type=str, choices=['white','bitflip','phaseflip','bitphase','amplitude'], default='white', help='noise model to test')
parser.add_argument('--noise-types', nargs='+', choices=['white','bitflip','phaseflip','bitphase','amplitude'], help='List of noise models to test (overrides --noise-type)')
parser.add_argument('--save-plots', action='store_true', default=False, help='Save plots to files (default: False)')
parser.add_argument('--compute-entropy', action='store_true', default=False, help='Compute and record subsystem entropy (default: False)')

args = parser.parse_args()

# Unpack args into variables used below
n = args.n
H_choice = args.H
H = {'H7': H7, 'H5': H5, 'H6': H6}[H_choice]
stab_list = stab_01n_op
ini_state = ini_state_01
steps = args.steps
t = poly_log_t
limit = args.limit
turns = args.turns
max_round = args.max_round
num_s = args.num_samples
class_w = (args.class_w[0], args.class_w[1])
mar_neg = args.mar_neg
mar_pos = args.mar_pos
# Support multiple noise models (list). If --noise-types provided, use it; otherwise use single --noise-type
noise_type = args.noise_type
noise_types = args.noise_types if args.noise_types is not None else [noise_type]

t_list = list(range(1, max_round))
results_by_step = {}
noise_results = {}  # Store noise robustness results

test_ob = H_stab_state_evol(n, H, stab_01n_op, ini_state_01, poly_log_t, steps)
test_ob.noise_type = noise_type

print(f"Starting Linear SVM with Noise Robustness Analysis (n={n}, H={H_choice}, noise={noise_type})")
print(f"Generating {num_s} separable samples per iteration")
print("=" * 60)

for a in t_list:
    print(f'Round {a}/{max_round-1}: Processing {a} * 1/polylog')
    
    # Evolve state (Trotterized)
    test_ob.state_evol(ideal_state=True, den_mat_out=False, state_update=True)
    
    # Evolve stabilizer sum with truncation
    test_ob.stablt_sum_evol(approx=True, up=limit, step_by_step=True, stab_sum_update=True)
    
    if a % turns == 0:
        test_ob.SVM_linear(num_samples=num_s, class_weights=class_w, mar_neg=mar_neg, mar_pos=mar_pos)
        label, score = test_ob.test_EW_SVM()
        
        # Store results in single consolidated dict
        results_by_step[a] = {
            'time': a * t(n),
            'trace': score,
            'size': test_ob.stabsum_tog.size,
            'b_value': extract_svm_intercept(test_ob.svm_linear_model),
        }

    def test_noise_robustness(svm_model, ent_obj, n, noise_tolerance=1e-6, max_iterations=50, compute_entropy=False):
        # Check if entanglement is detected
        if score < 0:
            print(f'✗ Entanglement NOT detected (score={score:.4f})')
            print(f'Stopping EW construction for limit={limit}')
            break

        print(f'✓ Entanglement detected (score={score:.4f}, label={label})')
        
            # Perform noise robustness test for each requested noise model
            noise_results[a] = {}
            for nt in noise_types:
                print(f"  Performing noise robustness test for model at step {a} (noise={nt})...")
                test_ob.noise_type = nt
                noise_data, max_noise = test_noise_robustness(
                    test_ob.svm_linear_model, test_ob, n,
                    noise_tolerance=1e-6, max_iterations=50,
                    compute_entropy=args.compute_entropy,
                )
                noise_results[a][nt] = {'noise_data': noise_data, 'max_noise': max_noise}
                print(f"  Maximum noise tolerance ({nt}): p_noise = {max_noise:.6f}")

print(f"\nEvolution completed. Final results:")
print(f"  Rounds completed: {len(results_by_step)}")
if results_by_step:
    final_trace = list(results_by_step.values())[-1]['trace']
    print(f"  Final trace value: {final_trace:.4f}")

# Save noise robustness results under result/noise_linear/
# Build results directory that encodes run parameters: n, limit, turns, max_round, num_samples
limit_str = str(limit).replace('.', 'p')
            entry = {'p_noise': 0.0, 'score': float(score), 'label': lbl_num}
            if compute_entropy:
                ent_clean = calculate_entropy(ent_obj.ini_state, n)
                entry['entropy'] = ent_clean
            else:
                entry['entropy'] = float('nan')
results_dir = os.path.join('result', 'noise_linear', results_subdir)
os.makedirs(results_dir, exist_ok=True)

# Summary rows for aggregated per-noise/round max-noise values
summary_rows = []
for step_idx, per_noise_dict in noise_results.items():
    for nt, step_data in per_noise_dict.items():
        noise_data = step_data['noise_data']
        max_noise = step_data['max_noise']

        # Build DataFrame from noise test results
        df_noise = pd.DataFrame(noise_data)

        # Generate filename with configuration (includes noise type)
        noise_filename = generate_noise_filename(
            'linear', n, limit, turns, num_s, class_w, step_idx, mar_neg, mar_pos, nt
        )
        noise_path = os.path.join(results_dir, noise_filename)
        df_noise.to_csv(noise_path, index=False)
        print(f'Saved noise robustness results to {noise_path}')

        # Plot noise robustness: score vs p_noise with scatter colored by detection label
        plt.figure(figsize=(10, 6))
        df = df_noise
        detected = df[df['label'] == 1]
        not_detected = df[df['label'] == -1]

        if not detected.empty:
            plt.scatter(detected['p_noise'], detected['score'], c='tab:blue', label='detected', s=60, marker='o')
        if not not_detected.empty:
            plt.scatter(not_detected['p_noise'], not_detected['score'], c='tab:orange', label='not detected', s=60, marker='x')

        plt.axhline(y=0, color='r', linestyle='--', label='Decision boundary')
        plt.axvline(x=max_noise, color='g', linestyle='--', label=f'Max noise: {max_noise:.4f}')
        plt.xlabel('Noise Level (p_noise)')
        plt.ylabel('SVM Score')
        plt.title(f'SVM Score vs Noise Level (Model at step {step_idx}, noise={nt}, max_noise={max_noise:.4f})')
        plt.legend()
        plt.grid(True)

        plot_name = noise_filename.replace('.csv', '_score.png')
        plot_path = os.path.join(results_dir, plot_name)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f'  Saved plot to {plot_path}')

        # Append summary row for this noise type and training round
        try:
            summary_rows.append({'noise_type': nt, 'round': int(step_idx), 'max_noise': float(max_noise)})
        except Exception:
                if compute_entropy:
                    ent_noisy = calculate_entropy(state_noisy, n)
                    entry['entropy'] = ent_noisy
                else:
                    entry['entropy'] = float('nan')
            summary_rows.append({'noise_type': nt, 'round': step_idx, 'max_noise': max_noise})
        # Plot entropy vs p_noise
        plt.figure(figsize=(10, 6))
        plt.plot(df_noise['p_noise'], df_noise['entropy'], marker='s', linewidth=2, markersize=6, color='orange')
        plt.axvline(x=max_noise, color='g', linestyle='--', label=f'Max noise: {max_noise:.4f}')
        plt.xlabel('Noise Level (p_noise)')
        plt.ylabel('Entanglement Entropy')
        plt.title(f'Entropy vs Noise Level (Model at step {step_idx}, noise={nt})')
        plt.legend()
        plt.grid(True)

        plot_name_entropy = noise_filename.replace('.csv', '_entropy.png')
        plot_path_entropy = os.path.join(results_dir, plot_name_entropy)
        plt.savefig(plot_path_entropy, bbox_inches='tight')
        plt.close()
        print(f'  Saved plot to {plot_path_entropy}')

print("\nNoise robustness analysis complete!")

# Write aggregated summary CSV for quick overview
if len(summary_rows) > 0:
    summary_df = pd.DataFrame(summary_rows)
    summary_fname = f'noise_linear_summary_n{n}_samples{num_s}.csv'
    summary_path = os.path.join(results_dir, summary_fname)
    summary_df.to_csv(summary_path, index=False)
    print(f'Aggregated noise summary saved to {summary_path}')
