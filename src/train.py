from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC, SVR


# Train a Random Forest classifier
def train_rf_classifier(X, y, vital, cv_score=None, **clf_params):

    print(f"Training classifier for {vital}")

    rf_clf = RandomForestClassifier(**clf_params)

    if cv_score:
        scores = cross_val_score(rf_clf, X, y, **cv_score)
        print(f"{vital}-ROC_AUC_score: {scores.mean()}")

    rf_clf.fit(X, y)
    return rf_clf


# Train a Support Vector Classifier
def train_svc(X, y, vital, grid_cv=None, cv_score=None, **clf_params):

    print(f"Training classifier for {vital}")

    svc_clf = SVC(**clf_params)

    if grid_cv:
        grid = GridSearchCV(svc_clf, grid_cv, scoring="roc_auc", cv=5, n_jobs=2).fit(
            X, y
        )

        # Get the best model
        svc_clf = grid.best_estimator_

        # Print the best score and parameters
        print(f"{vital}-ROC_AUC_score: {grid.best_score_}")
        print(grid.best_params_)

        return svc_clf

    else:

        if cv_score:
            scores = cross_val_score(svc_clf, X, y, **cv_score)
            print(f"{vital}-ROC_AUC_score: {scores.mean()}")

        svc_clf.fit(X, y)
        return svc_clf


# Train a Kernelized Ridge Regression model
def train_regression_model(
    X, y, vital, model="KernelRidge", grid_cv=None, cv_score=None, **regr_params
):

    # Make sure you add the appropriate parameters for the regression model in config.yaml
    assert model in ["Lasso", "Ridge", "KernelRidge", "SVR"], "Model not supported."

    if model == "Lasso":
        regr = Lasso(**regr_params)
    elif model == "Ridge":
        regr = Ridge(**regr_params)
    elif model == "KernelRidge":
        regr = KernelRidge(**regr_params)
    elif model == "SVR":
        regr = SVR(**regr_params)
    else:
        raise ValueError("Model not supported.")

    print(f"Training regression model for {vital}")

    if grid_cv:
        grid = GridSearchCV(regr, grid_cv, scoring="r2", cv=5, n_jobs=2).fit(X, y)

        # Get the best model
        regr = grid.best_estimator_

        # Print the best score and parameters
        print(f"{vital}-R2: {grid.best_score_}")
        print(grid.best_params_)

        return regr

    else:

        if cv_score:
            scores = cross_val_score(regr, X, y, **cv_score)
            print(f"{vital}-R2_score: {scores.mean()}")

        regr.fit(X, y)
        return regr
