import time
import logging
import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
import scipy.stats as stats

logging.basicConfig(level=logging.INFO)


def get_softmax_prediction(x_test, y_test, α, β, estimate_type="mean"):
    """
    get the prediction of a held-out test dataset using the posterior distribution
    of the bayesian learning (i.e. averaging over the samples of parameter values)
    :param numpy.ndarray x_test: held-out test data.     shape = (num_data_points, num_feats)
    :param numpy.ndarray y_test: held-out true labels.   shape = (num_data_points,)
    :param numpy.ndarray α: bias term.                   shape = (num_iters, num_classes)
    :param numpy.ndarray β: features weights.            shape = (num_iters, num_feats, num_classes)
    :param str           estimate_type: the averaging over parameters method.
                                    available options = ['mean', 'median', 'mode'], default value is: 'mean'
    :rtype: tuple        (prediction probabilities, prediction classes, accuracy %)
    """
    if estimate_type == "mean":
        y_pred = α.mean(axis=0) + np.dot(
            x_test, β.mean(axis=0)
        )  # first term shape: ?? , second term shape: (num_points, num_classes)

    elif estimate_type == "median":
        y_pred = np.median(α, axis=0) + np.dot(x_test, np.median(β, axis=0))

    elif estimate_type == "mode":
        α_mode = stats.mode(α, axis=0)[0].squeeze()
        β_mode = stats.mode(β, axis=0)[0].squeeze()
        y_pred = α_mode + np.dot(x_test, β_mode)
    proba = np.exp(y_pred).T / np.sum(np.exp(y_pred), axis=1)
    p_class = np.argmax(proba, axis=0)

    accuracy = np.sum(y_test == np.argmax(y_pred, axis=1)) / len(y_test)
    return proba, p_class, accuracy


def train_bayesian_softmax_regression(
    train_data, test_data, num_samples=500, num_chains=1
):
    """
    train a bayesian softmax with the default sampling in PyMC3 library

    :param pd.DataFrame train_data: dataframe of the training data (must include 'label' as column)
    :param pd.DataFrame test_data: dataframe of the test data (must include 'label' as column)
    :param int num_samples: number of samples used in the bayesian training
    :param in num_chains: number of parallel chains that will sample num_samples
                    (e.g. if num_chains=2 and num_samples=500, total number of samples=1000)
    :return: a dataframe of the events information
    :rtype: pd.DataFrame
    """

    # number of labels in the training dataset
    num_classes = len(train_data.label.unique())

    # prepare the data for the bayesian sampling
    y_train = pd.Categorical(
        train_data["label"]
    ).codes  # {0,1,2,3,4} in the case of miniImagenet
    x_train_cols = train_data.columns.difference(["label"])

    x_train = train_data[
        x_train_cols
    ].values  # shape: (number of datapoints, number of features)
    num_data_points, num_feats = x_train.shape
    logging.info(f"[Bayesian Training] number of data points: {num_data_points}")
    logging.info(f"[Bayesian Training] number of features: {num_feats}")

    # start bayesian training to learn the posterior distribution
    start = time.time()

    with pm.Model() as model_s:
        α = pm.Normal("α", mu=0, sd=5, shape=num_classes)  # bias term
        β = pm.Normal("β", mu=0, sd=5, shape=(num_feats, num_classes))  # feature weight
        μ = pm.Deterministic(
            "μ", α + pm.math.dot(x_train, β)
        )  # linear combination of the features
        θ = tt.nnet.softmax(μ)  # softmax prediction (150, 3)
        yl = pm.Categorical("yl", p=θ, observed=y_train)
        idata_samples = pm.sample(
            num_samples, chains=num_chains, target_accept=0.9, return_inferencedata=True
        )

    end = time.time()
    logging.info(
        f"elapsed time for {num_data_points} data points and {num_feats} features is: {(end - start):.4f} seconds"
    )

    # get predictions
    # first: alpha and beta are the bayesian parameters from the drawn samples
    α = idata_samples.get_values("α")
    β = idata_samples.get_values("β")

    # prediction  accuracy for training data
    y_proba, y_pred, train_accuracy = get_softmax_prediction(
        x_train, y_train, α, β, estimate_type="median"
    )
    logging.info(f"[Bayesian predictions] accuracy for training data: {train_accuracy}")

    # prediction accuracy for test data
    logging.info(f"[Bayesian predictions] predicting {len(test_data)} test data points")
    y_test = pd.Categorical(test_data["label"]).codes
    x_test_cols = test_data.columns.difference(["label"])
    x_test = test_data[
        x_test_cols
    ].values  # shape: (number of datapoints, number of features)

    y_test_proba, y_test_pred, test_accuracy = get_softmax_prediction(
        x_test, y_test, α, β, estimate_type="median"
    )
    logging.info(f"[Bayesian predictions] accuracy for test data: {test_accuracy:.4f}")

    return test_accuracy


def save_accuracy(acc_file_name, acc_msg):

    with open(acc_file_name, "a+") as file_object:
        # move read cursor to the start of file.
        file_object.seek(0)

        # if file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")

        # append text at the end of file
        file_object.write(acc_msg)
