import numpy
import torch
import matplotlib.pyplot as plt
import wandb

api = wandb.Api()


def plot_lipschitz_vs_views(runs: list, legends: list, views: list)-> None:
    plt.figure()
    lipschitz_values = []
    for run_name in runs:
        run = api.run(run_name)
        # get the lipschitz values from the run for each view
        lipschitz_views = []
        for view in views:
            lipschitz = run.history(keys=[f'Lipschitz_{view}'])[f'Lipschitz_{view}']
            # calculate the non-zeros mean for the entire steps
            lipschitz = numpy.array(lipschitz)
            # remove all the values less than 1e-5
            lipschitz = lipschitz[lipschitz > 1e-5]
            lipschitz = numpy.mean(lipschitz)
            lipschitz_views.append(lipschitz)
        lipschitz_values.append(lipschitz_views)
    
    # plot the histogram for the different legends, and each legend will have two bars for the two views
    # one beside the other
    x = numpy.arange(len(legends))
    width = 0.35
    fig, ax = plt.subplots()
    for i in range(len(views)):
        ax.bar(x + i*width, [lipschitz[i] for lipschitz in lipschitz_values], width, label=views[i])
    ax.set_xticks(x)
    ax.set_xticklabels(legends)
    ax.legend()
    plt.xlabel('Epsilon')
    plt.ylabel('Non-Zero Mean of Gradient Norm')
    plt.title('Epsilon vs Gradient Norm')
    plt.savefig('eps_vs_gradnorm.png')
             

runs = ['sagar8/ibc_pgd_experimentation/cdyk6gl8', 'sagar8/ibc_pgd_experimentation/htcmjtkm', 'sagar8/ibc_pgd_experimentation/7s66fb7g', 'sagar8/ibc_pgd_experimentation/lpkmxn87']
# runs_rand_target = ['sagar8/ibc_pgd_experimentation/nlay7vf7', 'sagar8/ibc_pgd_experimentation/313mq7ze', 'sagar8/ibc_pgd_experimentation/pglnyf5v', 'sagar8/ibc_pgd_experimentation/eullrjbh']
legends = ['0.0625', '0.125', '0.2', '0.3']
views = ['agentview_image_0', 'robot0_eye_in_hand_image_0']
plot_lipschitz_vs_views(runs, legends, views)

