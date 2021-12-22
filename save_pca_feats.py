import os
import time
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)


def save_data_features(features_data, label_data, file_path):
    all_data = np.hstack([features_data, label_data.reshape(-1, 1)])
    cols_names = ["f#" + str(i) for i in range(1, all_data.shape[1])]
    cols_names.append("label")
    df = pd.DataFrame(data=all_data, columns=cols_names)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    start = time.time()

    N_RUNS = 1000
    NUM_COMPONENTS = 5
    miniImagenet_feats_dir = "./data/miniImagenet/miniImagenet_tasks_feats/"
    save_path = "./data/miniImagenet/miniImagenet_pca/"

    for run_idx in range(N_RUNS):
        logging.info(f"Current Run={run_idx} ............... ")

        # load the query data
        miniImagenet_query_data = pd.read_csv(
            f"{miniImagenet_feats_dir}/query_run_{run_idx}.csv"
        )

        # load the support data
        miniImagenet_support_data = pd.read_csv(
            f"{miniImagenet_feats_dir}/support_run_{run_idx}.csv"
        )

        # load the full data
        miniImagenet_aug_data = pd.read_csv(
            f"{miniImagenet_feats_dir}/X_aug_run_{run_idx}.csv"
        )

        ## PCA for both the support only version and the full data (i.e. support + augmented data)
        # prepare support data for PCA
        train_cols = miniImagenet_support_data.columns.difference(["label"])
        target_col = "label"
        x_support, y_support = (
            miniImagenet_support_data[train_cols].values,
            miniImagenet_support_data[target_col].values,
        )

        # prepare full data for PCA
        x_full, y_full = (
            miniImagenet_aug_data[train_cols].values,
            miniImagenet_aug_data[target_col].values,
        )

        # prepare query data for PCA
        x_query, y_query = (
            miniImagenet_query_data[train_cols].values,
            miniImagenet_query_data[target_col].values,
        )

        # support only: fit and transform training data with pca
        pca_support = PCA(n_components=NUM_COMPONENTS)
        x_support_transformed = pca_support.fit_transform(x_support)
        x_support_transformed = (
            x_support_transformed - x_support_transformed.mean()
        ) / x_support_transformed.std()
        # logging.info(f"[support set] Percentage of variance explained by each of \
        # the {pca_support.n_components_} components: {pca_support.explained_variance_ratio_}")
        ### SAVE the SUPPORT PCA
        support_pca_path = os.path.join(save_path, f"support_pca_run_{run_idx}.csv")
        save_data_features(x_support_transformed, y_support, support_pca_path)
        ### SAVE the QUERY for the SUPPORT PCA
        x_query_for_support_pca_transformed = pca_support.transform(x_query)
        x_query_for_support_pca_transformed = (
            x_query_for_support_pca_transformed
            - x_query_for_support_pca_transformed.mean()
        ) / x_query_for_support_pca_transformed.std()
        query_for_support_pca_path = os.path.join(
            save_path, f"query+support_pca_run_{run_idx}.csv"
        )
        save_data_features(
            x_query_for_support_pca_transformed, y_query, query_for_support_pca_path
        )

        # full data: fit and transform training data with pca
        pca_full = PCA(n_components=NUM_COMPONENTS)
        x_full_transformed = pca_full.fit_transform(x_full)
        x_full_transformed = (
            x_full_transformed - x_full_transformed.mean()
        ) / x_full_transformed.std()
        # logging.info(f"[full set] Percentage of variance explained by each of \
        # the {pca_full.n_components_} components: {pca_full.explained_variance_ratio_}")
        ### SAVE the full PCA
        full_pca_path = os.path.join(save_path, f"X_aug_pca_run_{run_idx}.csv")
        save_data_features(x_full_transformed, y_full, full_pca_path)
        ### SAVE the QUERY for the full data PCA
        x_query_for_full_pca_transformed = pca_full.transform(x_query)
        query_for_full_pca_path = os.path.join(
            save_path, f"query+full_pca_run_{run_idx}.csv"
        )
        save_data_features(
            x_query_for_full_pca_transformed, y_query, query_for_full_pca_path
        )

    end = time.time()
    logging.info(
        f"elapsed time to save the pca versions is: {(end - start):.4f} seconds"
    )
