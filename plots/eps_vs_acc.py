import wandb
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()

def plot_eps_vs_attack(attack_algo: dict, legend: str, save_name: str):
    metrics = {}
    plt.figure()
     # go through each run and get the test/mean_score data
    for key in attack_algo:
        run = api.run(attack_algo[key])
        # get the epsilon values from the run
        eps = run.history(keys=['Epsilon'])['Epsilon']
        acc = run.history(keys=['test/mean_score'])['test/mean_score']
        eps = np.array(eps)
        acc = np.array(acc)
        # match the epsilons with the accuracies
        eps, acc = zip(*sorted(zip(eps, acc)))    
        metrics[key] = (eps, acc)

    mean_accuracies = []
    # plot the data
    for epsilon in eps:
        accrucaries = []
        for key in metrics:
            eps, acc = metrics[key]
            accrucaries.append(acc[np.where(eps == epsilon)[0][0]])
        mean_acc = np.mean(accrucaries)
        std_acc = np.std(accrucaries)
        mean_accuracies.append(mean_acc)
        # line plot with error bars
        plt.errorbar(epsilon, mean_acc, yerr=std_acc, fmt='o', color='blue')
    # connect the dots
    plt.plot(eps, mean_accuracies, color='blue')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.title('Epsilon vs Accuracy: ')
    plt.legend([legend])
    plt.savefig(save_name)

lift_fgsm_ibc = {"train_0": 'sagar8/BC_Evaluation/xxpjevol', "train_1": 'sagar8/BC_Evaluation/8uxiz5ii', \
                    "train_2": 'sagar8/BC_Evaluation/l4rts9ex'}

lift_fgsm_lstm_gmm = {'train_0': 'sagar8/BC_Evaluation/05mpt9g6', 'train_1': 'sagar8/BC_Evaluation/kohy1544', \
                        'train_2': 'sagar8/BC_Evaluation/xvwrhjpn'}
lift_fgsm_bc = {'train_0': ''}

lift_pgd_ibc_rand_target = {'train_0': 'sagar8/BC_Evaluation/kmm0j5mt', 'train_1': 'sagar8/BC_Evaluation/6bvqb23o', \
                            'train_2': 'sagar8/BC_Evaluation/sd5foxb0'}
lift_pgd_ibc = {'train_0': 'sagar8/BC_Evaluation/v0rok3b3', 'train_1': 'sagar8/BC_Evaluation/tqchjhp8', \
                'train_2': 'sagar8/BC_Evaluation/lg9kieud'}

if __name__ == '__main__':
    # plot_eps_vs_attack(lift_fgsm_ibc, 'LIFT-FGSM-IBC', 'lift_fgsm_ibc.png')
    # plot_eps_vs_attack(lift_fgsm_lstm_gmm, 'LIFT-FGSM-LSTM-GMM', 'lift_fgsm_lstm_gmm.png')
    plot_eps_vs_attack(lift_pgd_ibc, 'LIFT-PGD-IBC', 'lift_pgd_ibc.png')
