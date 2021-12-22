import logging
import pandas as pd
import numpy as np
from bayesian_softmax import train_bayesian_softmax_regression, save_accuracy


logging.basicConfig(level=logging.INFO)

DATASET_NAME = "miniImagenet"
N_RUNS = 1000
NUM_AUG_POINTS = 30
NUM_SAMPLES = 1000
NUM_CHAINS = 1
N_WAYS = 5
N_SHOT = 1
# ACC_FILE_NAME = "accuracy_results.txt"
ACC_FILE_NAME = "accuracy_results_final.txt"


if __name__ == "__main__":

    # path to miniImagenet 5-way 1-shot features
    miniImagenet_feats_dir = "../miniImagenet_tasks_feats"

    # to save the accuracy for each run
    non_aug_acc_list = []
    mix_aug_acc_list = []

    for run_idx in range(N_RUNS):
        logging.info(f"Current Run={run_idx} ............... ")
        # read the query data
        miniImagenet_query_data = pd.read_csv(
            f"{miniImagenet_feats_dir}/query_run_{run_idx}.csv"
        )

        # Bayesian logistic regression for the non-augmented data (support only)
        miniImagenet_support_data = pd.read_csv(
            f"{miniImagenet_feats_dir}/support_run_{run_idx}.csv"
        )
        query_acc = train_bayesian_softmax_regression(
            miniImagenet_support_data,
            miniImagenet_query_data,
            num_samples=NUM_SAMPLES,
            num_chains=NUM_CHAINS,
        )
        non_aug_acc_list.append(query_acc)

        # Bayesian logistic regression for a mix of augmented data
        # (i.e. first NUM_AUG_POINTS in X_aug)
        miniImagenet_aug_data = pd.read_csv(
            f"{miniImagenet_feats_dir}/X_aug_run_{run_idx}.csv"
        )
        query_aug_acc = train_bayesian_softmax_regression(
            miniImagenet_aug_data.iloc[:NUM_AUG_POINTS],
            miniImagenet_query_data,
            num_samples=NUM_SAMPLES,
            num_chains=NUM_CHAINS,
        )
        mix_aug_acc_list.append(query_aug_acc)

    # save accuracy for non-augmented case
    non_aug_accuracy = float(np.mean(non_aug_acc_list))
    non_aug_acc_msg = f"NON-AUGMENTED {DATASET_NAME} \
        {N_WAYS}-way {N_SHOT}-shot accuracy with {N_RUNS} runs: \t {non_aug_accuracy:.4f}"
    save_accuracy(ACC_FILE_NAME, non_aug_acc_msg)
    logging.info(
        f"Bayesian LR accuracy [{non_aug_accuracy:.4f}] for NON-AUGMENTED {DATASET_NAME} \
            {N_WAYS}-way {N_SHOT}-shot saved successfully in {ACC_FILE_NAME}"
    )

    # save accuracy for the mix-augmented data
    mix_aug_accuracy = float(np.mean(mix_aug_acc_list))
    mix_aug_acc_msg = f"mix-AUGMENTED {DATASET_NAME} \
        {N_WAYS}-way {N_SHOT}-shot accuracy with {N_RUNS} runs: \t {mix_aug_accuracy:.4f}"

    save_accuracy(ACC_FILE_NAME, mix_aug_acc_msg)
    logging.info(
        f"Bayesian LR accuracy [{mix_aug_accuracy:.4f}] for mix-AUGMENTED {DATASET_NAME} \
            {N_WAYS}-way {N_SHOT}-shot saved successfully in {ACC_FILE_NAME}"
    )
