from quantum_utils import (
    H7, H5,
    poly_log_t,
    H_stab_state_evol,
    stab_01n_op,
    ini_state_01,
    sample_random_pure_separable_state,
    classify_state_with_svm,
)
from qiskit.quantum_info import SparsePauliOp
import os
import pandas as pd
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


def generate_results_filename(n, H_name, limit, turns, max_round, num_samples):
    """Generate descriptive filename with configuration parameters."""
    limit_str = str(limit).replace('.', 'p')
    filename = (
        f"svm_linear_n{n}_H{H_name}_limit{limit_str}_turns{turns}_maxround{max_round}_num{num_samples}.csv"
    )
    return filename

# Parse CLI arguments
parser = argparse.ArgumentParser(description='SVM linear evolution runner')
parser.add_argument('--n', type=int, default=7, help='number of qubits')
parser.add_argument('--H', type=str, default='H7', choices=['H7', 'H5'], help='Hamiltonian name')
parser.add_argument('--limit', type=float, default=0.03, help='truncation limit')
parser.add_argument('--turns', type=int, default=4, help='train/test every this many rounds')
parser.add_argument('--max-round', type=int, default=20, help='maximum rounds to evolve')
parser.add_argument('--num-samples', type=int, default=10000, help='number of separable samples to test per round')
args = parser.parse_args()

n = args.n
H_name = args.H
H = H7 if H_name == 'H7' else H5
stab_list = stab_01n_op
ini_state = ini_state_01
steps = 10
t = poly_log_t
limit = args.limit
turns = args.turns
max_round = args.max_round
num_s = args.num_samples

t_list = list(range(1, max_round))
results_by_step = {}
master_rows = []

test_ob = H_stab_state_evol(n, H, stab_01n_op, ini_state_01, poly_log_t, steps)


for a in t_list:
    print(f'Round {a}/{max_round-1}: Processing {a} * 1/polylog')
    
    # Evolve state (ideal))
    test_ob.state_evol(ideal_state=True, den_mat_out=False, state_update=True)
    
    # Evolve stabilizer sum with truncation
    test_ob.stablt_sum_evol(approx=True, up=limit, step_by_step=True, stab_sum_update=True)
    
    
    if a % turns == 0:
        test_ob.SVM_linear(num_samples=num_s, class_weights=(1, 1), mar_neg=1.0, mar_pos=1.0)
        label, score = test_ob.test_EW_SVM()
        
        # Store results in single consolidated dict
        results_by_step[a] = {
            'time': a * t(n),
            'score': score,
            'size': test_ob.stabsum_tog.size,
            'b_value': extract_svm_intercept(test_ob.svm_linear_model),
        }

        # Check if entanglement is NOT detected (score < 0 means separable state)
        if score < 0:
            print(f'✗ Entanglement NOT detected (score={score:.4f})')
            print(f'Stopping EW construction for limit={limit}')
            break

        print(f'✓ Entanglement detected (score={score:.4f}, label={label})')

        # --- Additional separable-sample test: generate num_s separable states and test SVM predictions ---
        try:
            trained = test_ob.svm_linear_model
            pauli_strings = trained.get('pauli_strings') if isinstance(trained, dict) else None
            model_obj = trained.get('model') if isinstance(trained, dict) else trained

            # build Pauli matrices for features once
            if pauli_strings is None:
                n_features = 0
                pauli_list_str = ''
            else:
                n_features = len(pauli_strings)
                pauli_list_str = ';'.join(pauli_strings)
                P_mats = [SparsePauliOp.from_list([(p, 1)], num_qubits=n).to_matrix() for p in pauli_strings]

            # batch classification to avoid huge memory use
            batch_size = 500
            success_count = 0
            total_test = int(num_s)
            for start in range(0, total_test, batch_size):
                bs = min(batch_size, total_test - start)
                X_batch = np.zeros((bs, n_features), dtype=float) if n_features > 0 else np.zeros((bs, 0))
                for i in range(bs):
                    rho_sep = sample_random_pure_separable_state(n)
                    if n_features > 0:
                        feats = [np.real(np.trace(P @ rho_sep)) for P in P_mats]
                        X_batch[i, :] = feats

                # classify batch
                if hasattr(model_obj, 'predict'):
                    preds = model_obj.predict(X_batch)
                    # separable label corresponds to -1
                    success_count += int((preds == -1).sum())
                else:
                    # cvxpy solution dict expected
                    if isinstance(model_obj, dict) and 'w' in model_obj and 'b' in model_obj:
                        w = np.asarray(model_obj['w']).reshape(-1)
                        b = float(model_obj['b'])
                        scores_batch = X_batch @ w + b
                        preds = np.where(scores_batch > 0, 1, -1)
                        success_count += int((preds == -1).sum())
                    else:
                        # fallback: use classify_state_with_svm per sample
                        for i in range(bs):
                            lbl = classify_state_with_svm(sample_random_pure_separable_state(n, return_statevec=False), trained, n=n)
                            if isinstance(lbl, list):
                                lbl = lbl[0]
                            if str(lbl).lower().startswith('sep'):
                                success_count += 1

            # prepare summary row for combined CSV
            success_str = f"{success_count}/{total_test}"
            success_rate = float(success_count) / float(total_test) if total_test > 0 else 0.0
            combined_row = {
                'round': int(a),
                'time': float(a * t(n)),
                'size': int(test_ob.stabsum_tog.size),
                'score': float(score),
                'b_value': float(extract_svm_intercept(test_ob.svm_linear_model)),
                'separable_success': success_str,
                'separable_success_rate': success_rate,
            }
            master_rows.append(combined_row)
            print(f'  Appended separable-test summary for round {a} ({success_str})')
        except Exception as e:
            print(f'  Warning: separable-sample test failed: {e}')

print(f"\nEvolution completed. Final results:")
print(f"  Rounds completed: {len(results_by_step)}")
if results_by_step:
    final_score = list(results_by_step.values())[-1]['score']
    print(f"  Final score value: {final_score:.4f}")

# Create results directory and save data + plots
results_dir = os.path.join('result', 'SVM_linear_result')
os.makedirs(results_dir, exist_ok=True)

# If there are per-round combined rows collected, write them into a single CSV
if master_rows:
    df_all = pd.DataFrame(master_rows)
    csv_filename = generate_results_filename(n, H_name, limit, turns, max_round, num_s)
    csv_path = os.path.join(results_dir, csv_filename)
    df_all.to_csv(csv_path, index=False)
    print(f'Saved combined SVM linear results to {csv_path}')

# Also save the lightweight summary extracted from results_by_step (if any)
if results_by_step:
    df_summary = pd.DataFrame(list(results_by_step.values()))
    summary_path = os.path.join(results_dir, f'summary_n{n}_H{H_name}.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f'Saved summary to {summary_path}')
