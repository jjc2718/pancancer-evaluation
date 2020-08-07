"""
Functions for predicting mutation burden based on gene expression data.

"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

def train_model(x_train, x_test, y_train, alphas, l1_ratios, seed, n_folds=5, max_iter=1000):
    """
    Build the logic and sklearn pipelines to train x matrix based on input y

    Arguments
    ---------
    x_train: pandas DataFrame of feature matrix for training data
    x_test: pandas DataFrame of feature matrix for testing data
    y_train: pandas DataFrame of processed y matrix
    alphas: list of alphas to perform cross validation over
    l1_ratios: list of l1 mixing parameters to perform cross validation over
    n_folds: int of how many folds of cross validation to perform
    max_iter: the maximum number of iterations to test until convergence

    Returns
    ------
    The full pipeline sklearn object and y matrix predictions for training, testing,
    and cross validation
    """
    # Setup the model parameters
    reg_parameters = {
        "regression__loss": ["squared_loss"],
        "regression__penalty": ["elasticnet"],
        "regression__alpha": alphas,
        "regression__l1_ratio": l1_ratios,
    }

    estimator = Pipeline(
        steps=[
            (
                "regression",
                SGDRegressor(
                    random_state=seed,
                    loss="squared_loss",
                    max_iter=max_iter,
                    tol=1e-3,
                ),
            )
        ]
    )

    cv_pipeline = GridSearchCV(
        estimator=estimator,
        param_grid=reg_parameters,
        n_jobs=-1,
        cv=n_folds,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        iid=False
    )

    # Fit the model
    cv_pipeline.fit(X=x_train, y=y_train.log10_mut)

    # Obtain cross validation results
    y_cv = cross_val_predict(
        cv_pipeline.best_estimator_,
        X=x_train,
        y=y_train.log10_mut,
        cv=n_folds,
        method="decision_function",
    )

    # Get all performance results
    y_pred_train = cv_pipeline.predict(x_train)
    y_pred_test = cv_pipeline.predict(x_test)

    return cv_pipeline, y_pred_train, y_pred_test, y_cv

def extract_coefficients(cv_pipeline, feature_names, signal, seed):
    """
    Pull out the coefficients from the trained regression models

    Arguments
    ---------
    cv_pipeline: the trained sklearn cross validation pipeline
    feature_names: the column names of the x matrix used to train model (features)
    results: a results object output from `get_threshold_metrics`
    signal: the signal of interest
    seed: the seed used to split the data
    """
    final_pipeline = cv_pipeline.best_estimator_
    final_model = final_pipeline.named_steps["regression"]

    coef_df = pd.DataFrame.from_dict(
        {"feature": feature_names, "weight": final_model.coef_[0]}
    )

    coef_df = (
        coef_df.assign(abs=coef_df["weight"].abs())
        .sort_values("abs", ascending=False)
        .reset_index(drop=True)
        .assign(signal=signal, seed=seed)
    )

    return coef_df

def get_metrics(y_true, y_pred):
    """
    Retrieve MSE and R^2 values for predictions

    Arguments
    ---------
    y_true: an array of gold standard mutation burden
    y_pred: an array of predicted mutation burden

    Returns
    -------
    dict of MSE and R^2
    """
    return {"mse": mean_squared_error(y_true, y_pred),
            "r_squared": r2_score(y_true, y_pred)}

def summarize_results(results, gene, holdout_cancer_type, signal, seed,
                      data_type, fold_no):
    """
    Given an input results file, summarize and output all pertinent files

    Arguments
    ---------
    results: a results object output from `get_threshold_metrics`
    holdout_cancer_type: the cancer type being used as holdout data
    signal: the signal of interest
    seed: the seed used to compress the data
    data_type: the type of data (either training, testing, or cv)
    fold_no: the fold number for the external cross-validation loop
    """
    results_append_list = [
        holdout_cancer_type,
        signal,
        seed,
        data_type,
        fold_no,
    ]
    metrics_out_ = [results["mse"], results["r_squared"]] + results_append_list
    return metrics_out_

