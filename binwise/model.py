from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from .utils import convert_to_bins


class RegressionToClassificationModel(BaseEstimator, RegressorMixin):
    """An adapter that enables using classification models for regression tasks.

    This model serves as an adapter that allows you to apply any classification algorithm
    to regression problems by discretizing the continuous target into bins. It works by:
    1. Converting continuous target variables into discrete bins
    2. Training a classifier on these discretized targets
    3. Converting classification predictions back to continuous values using
       prediction probabilities and bin midpoints.

    This is particularly useful when you want to:
    - Use a classifier that cannot directly handle regression
    - Get uncertainty estimates for regression predictions

    Args:
        model_constructor: Callable that returns a classifier instance.
            The classifier must implement scikit-learn's estimator interface
            with fit() and predict_proba() methods.
        n_bins: Number of bins to use for discretizing the continuous target.
        binning_strategy: Strategy to use for binning.
            Supported strategies: 'Uniform' and 'Quantile'.

    Example:
        >>> from tabpfn import TabPFNClassifier
        >>> model = RegressionToClassificationModel(
        ...     model_constructor=lambda: TabPFNClassifier(N_ensemble_configurations=1),
        ...     n_bins=10,
        ...     binning_strategy='Uniform',
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, model_constructor, n_bins, binning_strategy):
        self.model_constructor = model_constructor
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy

        self._base_model = None
        self._label_encoder = LabelEncoder()
        self._bin_middles = None
        self._all_classes = None
        self.is_fitted_ = False

    def fit(self, X, y):
        bin_labels, _, self._bin_middles = convert_to_bins(
            y,
            self.n_bins,
            self.binning_strategy,
        )
        self._all_classes = np.arange(len(self._bin_middles))

        y_train = self._label_encoder.fit_transform(bin_labels)

        self._base_model = self.model_constructor()
        self._base_model.fit(X, y_train)
        self.is_fitted_ = True
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, ["_base_model", "_bin_middles"])
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        proba = self._base_model.predict_proba(X)

        # use label encoder to get the original class labels
        bin_values = self._label_encoder.inverse_transform(self._base_model.classes_)

        # Solve the edge case where binned class is not in training samples
        # Create a mapping of class labels to column indices
        label_to_idx = {label: idx for idx, label in enumerate(bin_values)}

        proba_all_classes = np.zeros((proba.shape[0], len(self._all_classes)))
        for idx, cls in enumerate(self._all_classes):
            if cls in label_to_idx:
                proba_all_classes[:, idx] = proba[:, label_to_idx[cls]]

        means = np.sum(proba_all_classes * self._bin_middles, axis=1)
        stds = np.sqrt(
            np.sum(
                proba_all_classes * (self._bin_middles - means[:, None]) ** 2,
                axis=1,
            )
        )

        if return_std:
            return {"mean": means, "std": stds}
        else:
            return means

    predict_proba = predict


class RegressionToClassificationEnsemble(BaseEstimator, RegressorMixin):
    """An ensemble of adapters that enables using classification models for regression tasks.

    An ensemble of base regression-to-classification adapter instances with different
    combinations of bin sizes and binning strategies. Base regression-to-classification
    adapter serves as a wrapper that allows you to apply any classification algorithm to
    regression problems by discretizing the continuous target into bins. However,
    selecting optimal binning parameters can be challenging. This ensemble version
    improves the base adapter by training multiple base adapter instances with different
    combinations of bin sizes and binning strategies, lowering the reliance on selecting
    the best bin size and binning strategy.

    Args:
        base_model_constructor: Callable that returns a classifier instance.
            The classifier must implement scikit-learn's estimator interface
            with fit() and predict_proba() methods.
        bin_sizes (list[int]): List of bin sizes for the binning ensemble.
            Each size will be used with each binning strategy.
        binning_strategies (list[str]): List of binning strategies for the ensemble.
            Supported strategies: 'Uniform' and 'Quantile'.
            Each strategy will be used with each bin size.
        subsample_ratio (float, optional): Optional data subsampling ratio. If less than 1.0,
            trains each binning configuration on a random subset of the data. Must be
            between 0 and 1. Defaults to 1.0 (use all samples).
        random_state (int, optional): Random seed for subsampling. Defaults to None.

    Example:
        >>> from tabpfn import TabPFNClassifier
        >>> # Create classifier with ensemble of 4 binning configurations
        >>> model = RegressionToClassificationEnsemble(
        ...     base_model_constructor=lambda: TabPFNClassifier(N_ensemble_configurations=1),
        ...     bin_sizes=[8, 10],  # 2 different bin sizes
        ...     binning_strategies=['Uniform', 'Quantile'],  # 2 different strategies
        ...     subsample_ratio=1.0,
        ...     random_state=42,
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        base_model_constructor,
        bin_sizes,
        binning_strategies,
        subsample_ratio=1.0,
        random_state=None,
    ):
        self.base_model_constructor = base_model_constructor
        self.bin_sizes = bin_sizes
        self.binning_strategies = binning_strategies
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state

        self._models = []
        self._rng = np.random.default_rng(self.random_state)
        self.is_fitted_ = False

    def fit(self, X, y):
        self._models = []
        for n_bins, strategy in product(self.bin_sizes, self.binning_strategies):
            model = RegressionToClassificationModel(
                model_constructor=self.base_model_constructor,
                n_bins=n_bins,
                binning_strategy=strategy,
            )
            X_subsample, y_subsample = self._random_subsample(X, y)
            model.fit(X_subsample, y_subsample)
            self._models.append(model)
        self.is_fitted_ = True
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, ["_models"])
        predictions_mean = []
        predictions_std = []

        for model in self._models:
            y_pred = model.predict(X, return_std=return_std)
            if return_std:
                predictions_mean.append(y_pred["mean"])
                predictions_std.append(y_pred["std"])
            else:
                predictions_mean.append(y_pred)

        predictions_mean = np.array(predictions_mean)
        if return_std:
            predictions_std = np.array(predictions_std)
            return {
                "mean": predictions_mean.mean(axis=0),
                "std": np.sqrt(
                    (predictions_std**2).mean(axis=0) + predictions_mean.var(axis=0)
                ),
            }
        else:
            return predictions_mean.mean(axis=0)

    predict_proba = predict

    def _random_subsample(self, X, y):
        num_samples = int(len(X) * self.subsample_ratio)
        idx = self._rng.choice(len(X), num_samples, replace=False)
        return X[idx], y[idx]
