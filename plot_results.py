"""Script to plot results as reported in the paper."""
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


def load_auc_ici(folders, dependence_on="n_samples", n_folds=10):
    """Load auc and ici (computed on test set) for a given list of folders, for all the models considered in the study.
    Input:
        - folders (list of str): folder names
        - dependence_on (str): parameter to be used to index the results ("n_samples" fo dependence on size, "ratio" for
                               dependence on balancing ratio). Default "n_samples"
        - n_folds (int): number of folds in training (Default 10)
    Return:
        - dictionaries with model names as keys and dataframe with results as values (one for auc and one for ici)
    """
    sizes_or_ratios = np.zeros(len(folders), dtype=float)
    auc = np.zeros((len(folders), n_folds))
    ici = np.zeros((len(folders), n_folds))
    auc_dict = {}
    ici_dict = {}

    for model_name in ["cnn", "xgb", "lr"]:
        for i, folder in enumerate(folders):
            path = 'results/' + model_name + '/' + folder + '/'
            with open(path + 'config.json') as file:
                config = json.load(file)
            sizes_or_ratios[i] = config[dependence_on]
            if dependence_on == "ratio" and config[dependence_on] is None:
                sizes_or_ratios[i] = 0.125

            auc[i, :] = np.load(path + "auc_test.npy")
            ici[i, :] = np.load(path + "ici_test.npy")

        auc_dict[model_name] = pd.DataFrame(auc.transpose(), columns=sizes_or_ratios).copy()
        ici_dict[model_name] = pd.DataFrame(ici.transpose(), columns=sizes_or_ratios).copy()

    return auc_dict, ici_dict


def main():
    """
    Before to run this script, one has to create a json file with the names of the directories that have to be analyzed.
    The file has to be structured as follows: keys are strings indicating the case-study ("size_dependence_original",
    "size_dependence_balanced" or "balance_dependence"); values are dictionaries with folders to be analyzed as values
    and corresponding sizes/ratios as keys.
    Example:
    {
        "size_dependence_original" : {
            "250": "N250_2023-05-29",
            "300": "N300_2023-05-29"
        },

        "size_dependence_balanced" : {
            "250": "N250_balanced05_2023-05-29",
            "300": "N300_balanced05_2023-05-29"
        },

        "balance_dependence" : {
            "0.125": "N250_2023-05-29",
            "0.25":  "N250_balanced025_2023-05-29",
            "0.375": "N250_balanced0375_2023-05-29",
            "0.5":   "N250_balanced05_2023-05-29"
        }
    }
    """
    folders_json = 'results/folders2use.json'  # json file with directory names. Change it if necessary.
    with open(folders_json) as file:
        folders_dict = json.load(file)

    # Factor to compute 95% CI from standard deviation
    n_folds = 10
    ci_factor = 1.96 / np.sqrt(n_folds)

    # Set plot configurations
    figsize = (16, 6)
    ms = 10
    lw = 2.5
    capsize = 3
    labels_size = 15
    legend_size = 14
    dpi = 600
    figures_dir = "results/figures/"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    palette = sns.color_palette("deep")
    sns.set_style("white")
    models = ["cnn", "xgb", "lr"]

    #  --- SIZE DEPENDENCE ---
    folders = folders_dict["size_dependence_original"].values()  # folder names (size dependence, original event ratio)
    auc, ici = load_auc_ici(folders)

    folders_bal = folders_dict["size_dependence_balanced"].values()  # folder names (size dependence, balanced ratio)
    auc_bal, ici_bal = load_auc_ici(folders_bal)

    # AUC plot
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True, dpi=dpi)
    for i, model in enumerate(models):  # original ratio
        ax[0].errorbar(auc[model].columns, auc[model].mean(), auc[model].std()*ci_factor, fmt='.-', markersize=ms,
                       label=model, capsize=capsize, linewidth=lw, color=palette[i])
    for i, model in enumerate(models):  # balanced
        ax[1].errorbar(auc[model].columns, auc_bal[model].mean(), auc_bal[model].std()*ci_factor, fmt='.-',
                       markersize=ms, label=model, capsize=capsize, linewidth=lw, color=palette[i])
    ax[0].set_title("Original fraction", fontsize=labels_size)
    ax[1].set_title("Balanced to 0.5", fontsize=labels_size)
    ax[0].set_ylabel("AUC", fontsize=labels_size)
    for axes in ax:
        axes.set_xlabel("# of ECGs", fontsize=labels_size)
        axes.set_xscale('log')
        axes.legend(loc="lower right", fontsize=legend_size)
        axes.grid(axis='y', alpha=0.5)
    plt.autoscale()
    fig.subplots_adjust(wspace=0.1)
    plt.savefig(figures_dir + 'auc_vs_size.png', bbox_inches='tight')

    # ICI plot
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True, dpi=dpi)
    for i, model in enumerate(models):  # plot ICI results (original ratio)
        ax[0].errorbar(ici[model].columns, ici[model].mean(), ici[model].std()*ci_factor, fmt='.-', markersize=ms,
                       label=model, capsize=capsize, linewidth=lw, color=palette[i])
    for i, model in enumerate(models):  # plot ICI results (balanced)
        ax[1].errorbar(ici[model].columns, ici_bal[model].mean(), ici_bal[model].std()*ci_factor, fmt='.-',
                       markersize=ms, label=model,  capsize=capsize, linewidth=lw, color=palette[i])
    ax[0].set_title("Original fraction", fontsize=labels_size)
    ax[1].set_title("Balanced to 0.5", fontsize=labels_size)
    ax[0].set_ylabel("ICI", fontsize=labels_size)
    for axes in ax:
        axes.set_xlabel("# of ECGs", fontsize=labels_size)
        axes.set_xscale('log')
        axes.legend(loc="best", fontsize=legend_size)
        axes.grid(axis='y', alpha=0.5)
    plt.autoscale()
    fig.subplots_adjust(wspace=0.1)
    plt.savefig(figures_dir + 'ici_vs_size.png', bbox_inches='tight')

    #  --- BALANCE RATIO DEPENDENCE ---
    folders = folders_dict["balance_dependence"].values()
    auc, ici = load_auc_ici(folders, dependence_on="ratio")
    # AUC and ICI plot
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    for i, model in enumerate(models):
        ax[0].errorbar(auc[model].columns, auc[model].mean(), auc[model].std()*ci_factor, fmt='.-', markersize=ms,
                       label=model,  capsize=capsize, linewidth=lw, color=palette[i])
    for i, model in enumerate(models):
        ax[1].errorbar(ici[model].columns, ici[model].mean(), auc[model].std()*ci_factor, fmt='.-', markersize=ms,
                       label=model,  capsize=capsize, linewidth=lw, color=palette[i])
    for axes in ax:
        axes.set_xlabel("Event fraction in training", fontsize=labels_size)
        axes.set_xscale('log')
        axes.legend(loc="best", fontsize=legend_size)
        axes.grid(axis='y', alpha=0.5)
    ax[0].set_ylabel("AUC", fontsize=labels_size)
    ax[1].set_ylabel("ICI", fontsize=labels_size)
    plt.autoscale()
    plt.savefig(figures_dir + 'auc_ici_vs_ratio.png', bbox_inches='tight')


if __name__ == "__main__":
    main()
