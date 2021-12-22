import time
import logging
import pandas as pd
import numpy as np
from bayesian_softmax import train_bayesian_softmax_regression, save_accuracy


logging.basicConfig(level=logging.INFO)

DATASET_NAME = "miniImagenet"
NUM_RUNS = 1000
NUM_DATA_POINTS = [5, 15, 30, 60]
NUM_SAMPLES = 500
NUM_CHAINS = 2
N_WAYS = 5
N_SHOT = 1
# ACC_FILE_NAME = "accuracy_results.txt"
ACC_FILE_NAME = "accuracy_results_final.txt"


if __name__ == "__main__":

    # path to miniImagenet 5-way 1-shot features after applying pca
    miniImagenet_pca_dir = "./data/miniImagenet/miniImagenet_pca"

    # to save the accuracy for each run
    acc_dct = {5: [], 15: [], 30: [], 60: []}

    start = time.time()
    for run_idx in range(NUM_RUNS):
        logging.info(f"Current Run={run_idx} ............... ")

        for num_data_pnts in NUM_DATA_POINTS:

            # Bayesian logistic regression for specific number of data points

            # load the data
            train_data = pd.read_csv(
                f"{miniImagenet_pca_dir}/X_aug_pca_run_{run_idx}.csv"
            )

            query_data = pd.read_csv(
                f"{miniImagenet_pca_dir}/query+full_pca_run_{run_idx}.csv"
            )

            query_acc = train_bayesian_softmax_regression(
                train_data.iloc[:num_data_pnts],
                query_data,
                num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS,
            )
            acc_dct[num_data_pnts].append(query_acc)

    for num_data_pnts in NUM_DATA_POINTS:
        curr_acc_lst = acc_dct[num_data_pnts]

        # save accuracy
        accuracy_mean = float(np.mean(curr_acc_lst))
        accuracy_std = float(np.std(curr_acc_lst))
        accuracy_msg = f"accuracy for {num_data_pnts} data points of {DATASET_NAME} \
            {N_WAYS}-way {N_SHOT}-shot with {NUM_RUNS} runs: \t {accuracy_mean:.4f} +- {accuracy_std:.4f}"

        save_accuracy(ACC_FILE_NAME, accuracy_msg)
        logging.info(accuracy_msg)

    end = time.time()
    print(f"elapsed time for {NUM_RUNS} is: {(end - start):.4f} seconds")
