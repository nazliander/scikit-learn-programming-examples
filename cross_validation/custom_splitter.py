import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

SEED = np.random.seed(42)


class CustomCrossValidation:

    @classmethod
    def split(cls,
              X: pd.DataFrame,
              y: np.ndarray = None,
              groups: np.ndarray = None):
        """Returns to a grouped time series split generator."""
        assert len(X) == len(groups),  (
            "Length of the predictors is not"
            "matching with the groups.")
        # The min max index must be sorted in the range
        for group_idx in range(groups.min(), groups.max()):

            training_group = group_idx
            # Gets the next group right after the training as test
            test_group = group_idx + 1
            training_indices = np.where(
                groups == training_group)[0]
            test_indices = np.where(groups == test_group)[0]
            if len(test_indices) > 0:
                # Yielding to training and testing indices for cross-validation
                # generator
                yield training_indices, test_indices


if __name__ == "__main__":
    print("\nCreating the example dataset...")
    X_experiment, y_experiment = make_regression(
        n_samples=30, n_features=5, noise=0.2)

    print("\nThe first 5 predictor values...")
    print(pd.DataFrame(X_experiment).head())

    print("\nThe first 5 target values...")
    print(pd.DataFrame(y_experiment[:5]))

    print("\nCreating the example groups,"
          "can be thought of different date indices...")
    groups_experiment = np.concatenate([np.zeros(5),  # 5 0s
                                        np.ones(10),  # 10 1s
                                        2 * np.ones(10),  # 10 2s
                                        3 * np.ones(5)  # 10 3s
                                        ]).astype(int)
    print("\nGroupings for the example dataset.."
          "\nThink that 0s are older date anchor"
          "values where as 3s the newest...")
    print(groups_experiment)

    print("\nExample split for Custom CV with date groupings...")
    for idx, (x, y) in enumerate(
        CustomCrossValidation.split(X_experiment,
                                    y_experiment,
                                    groups_experiment)):
        print(f"Split number: {idx}")
        print(f"Training indices: {x}")
        print(f"Test indices: {y}\n")

    print("\nExample split of TimeSeriesSplit with"
          "a fixed training size...")
    tscv = TimeSeriesSplit(max_train_size=10, n_splits=3)

    for idx, (x, y) in enumerate(tscv.split(X_experiment)):
        print(f"Split number: {idx}")
        print(f"Training indices: {x}")
        print(f"Test indices: {y}\n")

    print("\nLet's optimize some stuff...")
    # Instantiating the Lasso estimator
    reg_estimator = linear_model.Lasso()
    # Parameters
    parameters_to_search = {"alpha": [0.1, 1, 10]}
    # Splitter
    custom_splitter = CustomCrossValidation.split(
        X=X_experiment,
        y=y_experiment,
        groups=groups_experiment)

    # Search setup
    reg_search = GridSearchCV(
        estimator=reg_estimator,
        param_grid=parameters_to_search,
        scoring="neg_root_mean_squared_error",
        cv=custom_splitter)
    # Fitting
    best_model = reg_search.fit(
        X=X_experiment,
        y=y_experiment,
        groups=groups_experiment)

    print(f"\nBest model:\n{best_model.best_estimator_}")

    print(f"\nNumber of splits:\n{best_model.n_splits_}")
