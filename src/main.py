from typing import Dict, List
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from helper import standardize_data
from train import *
import preprocessing
import yaml
import pandas as pd


def classification_task(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    clf_mask: pd.DataFrame,
    clf_labels: List,
    clf_model: str,
    clf_params: Dict,
    cv_clf_params: Dict,
) -> Dict:

    clf_predictions = {}

    for label in clf_labels:

        label_mask = clf_mask.loc[label, :]

        # Select the features for the label
        X_train_selected = X_train.loc[:, label_mask]
        X_test_selected = X_test.loc[:, label_mask]

        # Standardize the data
        norm_X_train, norm_X_test = standardize_data(X_train_selected, X_test_selected)

        # Train a classifier model
        if clf_model == "RandomForestClassifier":
            clf = train_rf_classifier(
                norm_X_train,
                y_train[label],
                vital=label,
                cv_score=cv_clf_params,
                **clf_params
            )
        else:
            clf = train_svc(
                norm_X_train,
                y_train[label],
                vital=label,
                cv_score=cv_clf_params,
                **clf_params
            )

        # Make predictions
        y_pred_clf = clf.predict(norm_X_test)

        clf_predictions[label] = y_pred_clf

    return clf_predictions


def regression_task(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    regr_mask: pd.DataFrame,
    regr_labels: List,
    regr_model: str,
    regr_params: Dict,
    cv_regr_params: Dict,
) -> Dict:

    regr_predictions = {}

    for label in regr_labels:

        label_mask = regr_mask.loc[label, :]

        # Select the features for the label
        X_train_selected = X_train.loc[:, label_mask]
        X_test_selected = X_test.loc[:, label_mask]

        # Standardize the data
        norm_X_train, norm_X_test = standardize_data(X_train_selected, X_test_selected)

        # Standardize regression labels
        scaler = StandardScaler()
        norm_y_train = scaler.fit_transform(y_train[[label]])

        # Train a regression model
        regr = train_regression_model(
            norm_X_train,
            norm_y_train,
            vital=label,
            model=regr_model,
            grid_cv=None,
            cv_score=cv_regr_params,
            **regr_params
        )

        # Make predictions
        norm_y_pred_regr = regr.predict(norm_X_test)

        # Inverse transform the predictions
        y_pred_regr = scaler.inverse_transform(norm_y_pred_regr)

        regr_predictions[label] = y_pred_regr.flatten()

    return regr_predictions


if __name__ == "__main__":
    # File paths
    with open("config/paths.yaml", "r") as file:
        paths = yaml.safe_load(file)

    with open("config/params.yaml", "r") as file:
        params = yaml.safe_load(file)

    clf_model = params["clf_model"]
    regr_model = params["regr_model"]

    clf_params = params[clf_model]
    regr_params = params[regr_model]

    cv_clf_params = params["cv_clf"]
    cv_regr_params = params["cv_regr"]

    # Preprocess the data
    # preprocessing.main()

    train_features = paths["train"]["final_train_file"]
    train_labels = paths["train"]["train_labels"]
    test_features = paths["test"]["final_test_file"]

    clf_mask_file = paths["clf_mask_file"]
    regr_mask_file = paths["regr_mask_file"]

    # Load the data
    X_train = pd.read_csv(train_features)
    y_train = pd.read_csv(train_labels)
    X_test = pd.read_csv(test_features)

    # Load the masks
    clf_mask = pd.read_csv(clf_mask_file, index_col="label")
    regr_mask = pd.read_csv(regr_mask_file, index_col="label")

    # Write the results to a DataFrame
    results_df = pd.DataFrame()
    results_df["pid"] = X_test["pid"]
    regr_predictions = {}

    # Remove the "pid" column
    X_train = X_train.drop("pid", axis=1)
    X_test = X_test.drop("pid", axis=1)
    y_train = y_train.drop("pid", axis=1)

    # Split the labels into classification and regression labels
    labels = y_train.columns
    clf_labels = labels[:11]
    regr_labels = labels[11:]

    # CLASSIFICATION
    clf_predictions = classification_task(
        X_train,
        X_test,
        y_train,
        clf_mask,
        clf_labels,
        clf_model,
        clf_params,
        cv_clf_params,
    )

    # REGRESSION
    regr_predictions = regression_task(
        X_train,
        X_test,
        y_train,
        regr_mask,
        regr_labels,
        regr_model,
        regr_params,
        cv_regr_params,
    )

    predictions = {**clf_predictions, **regr_predictions}

    # Save the predictions
    results_df = pd.concat([results_df, pd.DataFrame(predictions)], axis=1)
    results_df.to_csv(paths["predictions_file"], index=False)
