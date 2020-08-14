"""
Functions for predicting mutation burden based on gene expression data.

"""
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline

def train_model(
    X_train,
    X_test,
    y_train,
    alphas,
    l1_ratios,
    learning_rates,
    seed,
    n_folds=4,
    max_iter=1000
):
    """Fit SGD regression model to predict y_train from X_train.
    """
    # set up the regression parameters
    # TODO: change to elastic net maybe?
    reg_parameters = {
        "regression__alpha": alphas,
        "regression__l1_ratio": l1_ratios,
        "regression__eta0": learning_rates
    }

    estimator = Pipeline(
        steps=[
            (
                "regression",
                SGDRegressor(
                    random_state=seed,
                    penalty='elasticnet',
                    learning_rate='constant',
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
    cv_pipeline.fit(X=X_train, y=y_train.log10_mut)

    # Obtain cross validation results
    # y_cv = cross_val_predict(
    #     cv_pipeline.best_estimator_,
    #     X=X_train,
    #     y=y_train.log10_mut,
    #     cv=n_folds
    # )

    # Get all performance results
    y_pred_train = cv_pipeline.predict(X_train)
    y_pred_test = cv_pipeline.predict(X_test)

    return cv_pipeline, y_pred_train, y_pred_test


def train_model_ols(x_train, x_test, y_train):
    """
    Fit ordinary least squares model to predict y_train from x_train

    Arguments
    ---------
    x_train: pandas DataFrame of feature matrix for training data
    x_test: pandas DataFrame of feature matrix for testing data
    y_train: pandas DataFrame of processed y matrix

    Returns
    ------
    Fit LinearRegression model and predictions on train/test sets
    """
    reg = LinearRegression()
    reg.fit(x_train, y_train.log10_mut)

    y_pred_train = reg.predict(x_train)
    y_cv = y_pred_train[:]
    y_pred_test = reg.predict(x_test)

    return reg, y_pred_train, y_pred_test


def extract_coefficients(cv_pipeline, feature_names, signal, seed):
    """
    Pull out the coefficients from the trained classifiers

    Arguments
    ---------
    cv_pipeline: the trained sklearn cross validation pipeline
    feature_names: the column names of the x matrix used to train model (features)
    results: a results object output from `get_threshold_metrics`
    signal: the signal of interest
    seed: the seed used to compress the data
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


def extract_coefficients_ols(model, feature_names, signal, seed):
    """
    Pull out the coefficients from the trained regression models

    Arguments
    ---------
    model: the trained sklearn LinearRegression model
    feature_names: the column names of the x matrix used to train model (features)
    signal: the signal of interest
    seed: the seed used to split the data
    """
    coef_df = pd.DataFrame.from_dict(
        {"feature": feature_names, "weight": model.coef_[0]}
    )

    coef_df = (
        coef_df.assign(abs=coef_df["weight"].abs())
        .sort_values("abs", ascending=False)
        .reset_index(drop=True)
        .assign(signal=signal, seed=seed)
    )

    return coef_df


def get_regression_metrics(y_true, y_pred):
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


def summarize_regression_results(results, holdout_cancer_type, signal, seed,
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

