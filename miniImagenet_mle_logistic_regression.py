import time
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from bayesian_softmax import save_accuracy


logging.basicConfig(level=logging.INFO)

DATASET_NAME = "miniImagenet"
NUM_RUNS = 1000
NUM_DATA_POINTS = [5, 15, 30, 60, 3755]

N_WAYS = 5
N_SHOT = 1
# ACC_FILE_NAME = "accuracy_results.txt"
ACC_FILE_NAME = "accuracy_results_final_mle.txt"


if __name__ == "__main__":

    # path to miniImagenet 5-way 1-shot features after applying pca
    miniImagenet_dir = "./data/miniImagenet/miniImagenet_tasks_feats"

    # to save the accuracy for each run
    acc_dct = {5: [], 15: [], 30: [], 60: [], 3755:[]}

    start = time.time()
    for run_idx in range(NUM_RUNS):
        logging.info(f"Current Run={run_idx} ............... ")

        for num_data_pnts in NUM_DATA_POINTS:

            # MLE logistic regression for specific number of data points

            # load the data
            train_data = pd.read_csv(
                f"{miniImagenet_dir}/X_aug_run_{run_idx}.csv"
            )

            query_data = pd.read_csv(
                f"{miniImagenet_dir}/query_run_{run_idx}.csv"
            )

            # prepare data for MLE logistic regression classifier
            x_train_cols = train_data.columns.difference(["label"])
            x_train = train_data[x_train_cols].values
            y_train = train_data["label"].values

            x_test = query_data[x_train_cols].values
            y_test = query_data["label"].values
            lr_classifier = LogisticRegression(max_iter=1000).fit(X=x_train, y=y_train)
            query_predicts = lr_classifier.predict(x_test)
            acc_val = np.mean(query_predicts == y_test)
            acc_dct[num_data_pnts].append(acc_val)
            logging.info(f"current accuracy: acc_val")

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
