# plot.py

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'ieee'])

models = ['SOM (8x8)', 'SOM-VAE', 'DESOM', 'DESOM-cls', 'ViT-cls', 'ViT-SOM (ours)', 'ViT-SOM-cls (ours)']
params = [1.8, 3.7, 3.3, 7.2, 5.4, 2.5, 6.8]
purity_scores = [0.712, 0.739, 0.751, None, None, 0.817, None]
classification_acc = [None, None, None, 0.498, 0.909, None, 0.921]

unsupervised_colors = ['#729ECE', '#729ECE', '#729ECE', None, None, '#396AB1', None]
supervised_colors = [None, None, None, '#AB6856', '#AB6856', None, '#BA3E35']
unsupervised_markers = ['o', 'o', 'o', None, None, 'D', None]
supervised_markers = [None, None, None, 'o', 'o', None, 'D']

fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.set_xlabel('Model Size (Millions of Parameters)', fontsize=20, fontweight='bold')
ax1.set_ylabel('Purity Score (F-MNIST)', fontsize=20, fontweight='bold', color='black')

for i in range(len(models)):
    if purity_scores[i] is not None:
        ax1.scatter(
            params[i], purity_scores[i],
            color=unsupervised_colors[i],
            marker=unsupervised_markers[i],
            s=250, edgecolor='black'
        )
        ax1.annotate(
            models[i],
            (params[i], purity_scores[i]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=20, 
            fontweight='bold' if '(ours)' in models[i] else 'normal',
            ha='left', va='bottom', color='black'
        )

ax1.tick_params(axis='y', labelcolor='black', labelsize=20)
ax1.tick_params(axis='x', labelsize=20)
ax1.set_ylim(0.7, 0.84)

ax2 = ax1.twinx()
ax2.set_ylabel('Classification Accuracy (CIFAR-10)', fontsize=20, fontweight='bold', color='black')

for i in range(len(models)):
    if classification_acc[i] is not None:
        ax2.scatter(
            params[i], classification_acc[i],
            color=supervised_colors[i],
            marker=supervised_markers[i],
            s=250, edgecolor='black'
        )
        if models[i] == 'ViT-cls':
            ax2.annotate(
                models[i],
                (params[i], classification_acc[i]),
                xytext=(-12, 5), textcoords="offset points",
                fontsize=20, 
                fontweight='bold' if '(ours)' in models[i] else 'normal',
                ha='right', va='top', color='black'
            )
        else:
            ax2.annotate(
                models[i],
                (params[i], classification_acc[i]),
                xytext=(-6, 5), textcoords="offset points",
                fontsize=20, 
                fontweight='bold' if '(ours)' in models[i] else 'normal',
                ha='right', va='bottom', color='black'
            )

ax2.tick_params(axis='y', labelcolor='black', labelsize=20)
ax2.set_ylim(0.4, 1.0)
ax1.set_xlim(0, 8)

legend_handles = [
    plt.Line2D([0], [0], color='#729ECE', marker='o', linestyle='', markersize=20, markeredgecolor='black', label='Clustering (F-MNIST)'),
    plt.Line2D([0], [0], color='#AB6856', marker='o', linestyle='', markersize=20, markeredgecolor='black', label='Classification (CIFAR-10)')
]
legend = fig.legend(handles=legend_handles, loc='lower right', fontsize=18, frameon=True, bbox_to_anchor=(0.938, 0.08))
legend.get_frame().set_edgecolor('black')

fig.tight_layout()
plt.savefig('/app/experiments/plots/charts/combined_dual_axis_plot.pdf', dpi=300)
plt.show()