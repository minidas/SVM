from quantum_utils import (
    H7, H5,
    poly_log_t,
    H_stab_state_evol,
    stab_01n_op,
    ini_state_01,
    plot_x_y,
    sample_random_pure_separable_state,
    classify_state_with_svm,
)
from qiskit.quantum_info import SparsePauliOp
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


def generate_results_filename(kernel_type, n, limit, turns, max_round, num_samples, class_w, mar_neg=1.0, mar_pos=1.0):
    """Generate descriptive filename with all configuration parameters."""
    cw_str = f"{class_w[0]:.1f}_{class_w[1]:.1f}".replace('.', 'p')
    margin_str = f"m{mar_neg:.1f}_{mar_pos:.1f}".replace('.', 'p') if (mar_neg != 1.0 or mar_pos != 1.0) else ""
    
    filename = (
        f"svm_{kernel_type}_n{n}_limit{limit}_turns{turns}_maxround{max_round}_"
        f"num{num_samples}_cw{cw_str}{margin_str}.csv"
    )
    return filename

n = 7
H = H7
stab_list = stab_01n_op
ini_state = ini_state_01
steps = 10
t = poly_log_t
limit = 0.03
turns = 4
max_round = 20
num_s = 10000
class_w = (1,1) #(separable, entangled)
mar_neg = 1.0
mar_pos = 1.0


t_list = list(range(1, max_round))
results_by_step = {}

test_ob = H_stab_state_evol(n, H7, stab_01n_op, ini_state_01, poly_log_t, steps)


for a in t_list:
    print(f'Round {a}/{max_round-1}: Processing {a} * 1/polylog')
    
    # Evolve state (ideal))
    test_ob.state_evol(ideal_state=True, den_mat_out=False, state_update=True)
    
    # Evolve stabilizer sum with truncation
    test_ob.stablt_sum_evol(approx=True, up=limit, step_by_step=True, stab_sum_update=True)
    
    
    if a % turns == 0:
        test_ob.SVM_linear(num_samples=num_s, class_weights=class_w, mar_neg=mar_neg, mar_pos=mar_pos)
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

            # prepare summary row
            test_row = {
                'round': int(a),
                'num_pauli_basis': int(n_features),
                'time': float(a * t(n)),
                'pauli_basis': pauli_list_str,
                'evolved_score': float(score),
                'separable_success': f"{success_count}/{total_test}",
            }

            # save per-run aggregated CSV under parameterized folder
            limit_str = str(limit).replace('.', 'p')
            results_subdir = f'n{n}_limit{limit_str}_turns{turns}_maxround{max_round}_num{num_s}'
            test_results_dir = os.path.join('result', 'svm_linear_tests', results_subdir)
            os.makedirs(test_results_dir, exist_ok=True)
            test_csv = os.path.join(test_results_dir, f'svm_linear_separable_test_n{n}_round{a}.csv')
            pd.DataFrame([test_row]).to_csv(test_csv, index=False)
            print(f'  Saved separable-test summary to {test_csv}')
        except Exception as e:
            print(f'  Warning: separable-sample test failed: {e}')

print(f"\nEvolution completed. Final results:")
print(f"  Rounds completed: {len(results_by_step)}")
if results_by_step:
    final_score = list(results_by_step.values())[-1]['score']
    print(f"  Final score value: {final_score:.4f}")

# Create results directory and save data + plots
results_dir = 'result'
os.makedirs(results_dir, exist_ok=True)

if results_by_step:
    # Build DataFrame from consolidated results
    df = pd.DataFrame(list(results_by_step.values()))
    
    # Generate filename with configuration
    csv_filename = generate_results_filename(
        'linear', n, limit, turns, max_round, num_s, class_w, mar_neg, mar_pos
    )
    csv_path = os.path.join(results_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f'Saved results to {csv_path}')
    
    # Plot helper function
    def save_plot(x_col, y_col, xlabel, ylabel, title):
        plt.figure(figsize=(8, 5))
        plt.plot(df[x_col], df[y_col], marker='o')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        # Generate plot filename from csv filename
        plot_name = csv_filename.replace('.csv', f'_{y_col}_vs_{x_col}.png')
        png_path = os.path.join(results_dir, plot_name)
        plt.savefig(png_path, bbox_inches='tight')
        plt.close()
        print(f'  Saved plot to {png_path}')
    
    # Generate all plots
    save_plot('time', 'score', 'time', 'SVM score', 'SVM Score vs Time')
    save_plot('time', 'size', 'time', 'stabilizer size', 'Stabilizer Size vs Time')
    save_plot('time', 'b_value', 'time', 'SVM intercept (b)', 'SVM Intercept vs Time')
