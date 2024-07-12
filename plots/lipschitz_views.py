import numpy
import torch
import matplotlib.pyplot as plt
import wandb

api = wandb.Api()

import matplotlib.pyplot as plt
import numpy as np

def plot_lipschitz_vs_views(runs: list, legends: list, views: list) -> None:
    plt.figure(figsize=(12, 6))
    lipschitz_values = []
    for run_name in runs:
        run = api.run(run_name)
        lipschitz_views = []
        for view in views:
            lipschitz = run.history(keys=[f'grad_{view}'])[f'grad_{view}']
            lipschitz = np.array(lipschitz)
            lipschitz = lipschitz[lipschitz > 1e-5]
            lipschitz = np.mean(lipschitz)
            lipschitz_views.append(lipschitz)
        lipschitz_values.append(lipschitz_views)
    
    x = np.arange(len(legends))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in range(len(views)):
        ax.bar(x + i*width, [lipschitz[i] for lipschitz in lipschitz_values], width, label=views[i])
    
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(legends)
    ax.set_xlabel('Perturbation axis', fontsize=12)
    ax.set_ylabel('Non-Zero Mean of Gradient Norm', fontsize=12)
    ax.set_title('Perturbation Axis vs Gradient Norm', fontsize=14)
    
    # Adjust the layout to prevent cut-off
    plt.tight_layout()
    
    # Move the legend outside the plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust the figure size to accommodate the legend
    plt.gcf().set_size_inches(14, 6)
    
    plt.savefig('pert_axis_vs_gradnorm.png', bbox_inches='tight', dpi=300)
    plt.show()

# Example usage:
runs = ['sagar8/grad_check_adv/0fbjs157', 'sagar8/grad_check_adv/0t848dyt', 'sagar8/grad_check_adv/cb5sp334']
legends = ['x', 'y', 'z']
views = ['agentview_image', 'robot0_eye_in_hand_image']
plot_lipschitz_vs_views(runs, legends, views)