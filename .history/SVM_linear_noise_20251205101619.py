from quantum_utils import (
    H7, H5,
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


def generate_noise_filename(kernel_type, n, limit, turns, num_samples, class_w, train_step, mar_neg=1.0, mar_pos=1.0):
    """Generate descriptive filename for noise robustness results."""
    cw_str = f"{class_w[0]:.1f}_{class_w[1]:.1f}".replace('.', 'p')
    margin_str = f"m{mar_neg:.1f}_{mar_pos:.1f}".replace('.', 'p') if (mar_neg != 1.0 or mar_pos != 1.0) else ""
    
    filename = (
        f"noise_{kernel_type}_n{n}_limit{limit}_turns{turns}_step{train_step}_"
        f"num{num_samples}_cw{cw_str}{margin_str}.csv"
    )
    return filename


def calculate_entropy(state_or_dm, n):
    """Calculate entanglement entropy of the reduced density matrix."""
    try:
        # Convert to density matrix if needed
        if hasattr(state_or_dm, 'to_matrix'):
            full_dm = state_or_dm.to_matrix()
        else:
            full_dm = np.asarray(state_or_dm)
        
        # If it's a statevector, convert to density matrix
        if full_dm.shape[0] == full_dm.shape[1]:
            # It's a square matrix (density matrix or unitary), use as is
            pass
        else:
            # Assume it's a column vector (statevector)
            full_dm = full_dm @ full_dm.conj().T
        
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


def test_noise_robustness(svm_model, ent_obj, n, noise_tolerance=1e-6, max_iterations=50):
    """
    Use bisection method to find maximum noise level where model still detects entanglement.
    
    Parameters:
    -----------
    svm_model : dict
        Trained SVM model from train_linear_svm_pauli
    ent_obj : H_stab_state_evol
        Quantum state object with noisy_state method
    n : int
        Number of qubits
    noise_tolerance : float
        Bisection tolerance
    max_iterations : int
        Maximum bisection iterations
    
    Returns:
    --------
    noise_data : list of dicts
        Records of (p_noise, score, entropy) tested
    max_noise : float
        Maximum noise level where entanglement is still detected
    """
    noise_data = []
    
    # Test noise-free state first
    state_clean = ent_obj.ini_state
    try:
        label, score = classify_state_with_svm(state_clean, svm_model, n=n, return_score=True)
        ent_clean = calculate_entropy(ent_obj.ini_state, n)
        
        noise_data.append({
            'p_noise': 0.0,
            'score': float(score),
            'entropy': ent_clean,
            'label': int(label),
        })
        
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
    
    for iteration in range(max_iterations):
        p_mid = (p_low + p_high) / 2.0
        
        try:
            # Get noisy state (returns density matrix)
            state_noisy = ent_obj.noisy_state(p_mid)
            
            # Classify noisy state
            label, score = classify_state_with_svm(state_noisy, svm_model, n=n, return_score=True)
            
            # Calculate entropy of noisy state
            ent_noisy = calculate_entropy(state_noisy, n)
            
            noise_data.append({
                'p_noise': p_mid,
                'score': float(score),
                'entropy': ent_noisy,
                'label': int(label),
            })
            
            # Check convergence
            if p_high - p_low < noise_tolerance:
                break
            
            # Update bounds based on whether entanglement is detected
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


# Configuration
n = 7
H = H7
stab_list = stab_01n_op
ini_state = ini_state_01
steps = 10
t = poly_log_t
limit = 0.03
turns = 1
max_round = 20
num_s = 1000
class_w = (1, 1)
mar_neg = 1.0
mar_pos = 1.0

t_list = list(range(1, max_round))
results_by_step = {}
noise_results = {}  # Store noise robustness results

test_ob = H_stab_state_evol(n, H7, stab_01n_op, ini_state_01, poly_log_t, steps)

print(f"Starting Linear SVM with Noise Robustness Analysis (n={n})")
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

        # Check if entanglement is detected
        if score < 0:
            print(f'✗ Entanglement NOT detected (score={score:.4f})')
            print(f'Stopping EW construction for limit={limit}')
            break

        print(f'✓ Entanglement detected (score={score:.4f}, label={label})')
        
        # Perform noise robustness test on this model
        print(f"  Performing noise robustness test for model at step {a}...")
        noise_data, max_noise = test_noise_robustness(
            test_ob.svm_linear_model, test_ob, n, 
            noise_tolerance=1e-6, max_iterations=50
        )
        
        noise_results[a] = {
            'noise_data': noise_data,
            'max_noise': max_noise,
        }
        
        print(f"  Maximum noise tolerance: p_noise = {max_noise:.6f}")

print(f"\nEvolution completed. Final results:")
print(f"  Rounds completed: {len(results_by_step)}")
if results_by_step:
    final_trace = list(results_by_step.values())[-1]['trace']
    print(f"  Final trace value: {final_trace:.4f}")

# Save noise robustness results
results_dir = 'result_noise'
os.makedirs(results_dir, exist_ok=True)

for step_idx, step_data in noise_results.items():
    noise_data = step_data['noise_data']
    max_noise = step_data['max_noise']
    
    # Build DataFrame from noise test results
    df_noise = pd.DataFrame(noise_data)
    
    # Generate filename with configuration
    noise_filename = generate_noise_filename(
        'linear', n, limit, turns, num_s, class_w, step_idx, mar_neg, mar_pos
    )
    noise_path = os.path.join(results_dir, noise_filename)
    df_noise.to_csv(noise_path, index=False)
    print(f'Saved noise robustness results to {noise_path}')
    
    # Plot noise robustness: score vs p_noise
    plt.figure(figsize=(10, 6))
    plt.plot(df_noise['p_noise'], df_noise['score'], marker='o', linewidth=2, markersize=6)
    plt.axhline(y=0, color='r', linestyle='--', label='Decision boundary')
    plt.axvline(x=max_noise, color='g', linestyle='--', label=f'Max noise: {max_noise:.4f}')
    plt.xlabel('Noise Level (p_noise)')
    plt.ylabel('SVM Score')
    plt.title(f'SVM Score vs Noise Level (Model at step {step_idx}, max_noise={max_noise:.4f})')
    plt.legend()
    plt.grid(True)
    
    plot_name = noise_filename.replace('.csv', '_score.png')
    plot_path = os.path.join(results_dir, plot_name)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f'  Saved plot to {plot_path}')
    
    # Plot entropy vs p_noise
    plt.figure(figsize=(10, 6))
    plt.plot(df_noise['p_noise'], df_noise['entropy'], marker='s', linewidth=2, markersize=6, color='orange')
    plt.axvline(x=max_noise, color='g', linestyle='--', label=f'Max noise: {max_noise:.4f}')
    plt.xlabel('Noise Level (p_noise)')
    plt.ylabel('Entanglement Entropy')
    plt.title(f'Entropy vs Noise Level (Model at step {step_idx})')
    plt.legend()
    plt.grid(True)
    
    plot_name_entropy = noise_filename.replace('.csv', '_entropy.png')
    plot_path_entropy = os.path.join(results_dir, plot_name_entropy)
    plt.savefig(plot_path_entropy, bbox_inches='tight')
    plt.close()
    print(f'  Saved plot to {plot_path_entropy}')

print("\nNoise robustness analysis complete!")
