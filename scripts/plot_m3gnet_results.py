# @Author : Serein
# @Time : 2026/3/12 11:51
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# 替换为你实际的模型输出目录
# 例如：'outputs/2026-03-11/11-46-17/'
run_dir = 'path_to_your_hydra_output_dir/'

y_pred = np.load(run_dir + 'test_preds.npy')
y_true = np.load(run_dir + 'test_targets.npy')

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

plt.figure(figsize=(7, 6))
plt.scatter(y_true, y_pred, alpha=0.5, color='#1f77b4', edgecolors='white', s=40)

min_val = min(np.min(y_true), np.min(y_pred)) - 0.5
max_val = max(np.max(y_true), np.max(y_pred)) + 0.5
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

metrics_text = f"MAE = {mae:.3f}\n$R^2$ = {r2:.3f}"
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('True Value', fontsize=12)
plt.ylabel('Predicted Value', fontsize=12)
plt.title('M3GNet Parity Plot', fontsize=14)

plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plot_path = run_dir + 'parity_plot.png'
plt.savefig(plot_path, dpi=300)
print(f"Parity plot saved to {plot_path}")