import numpy as np
import properscoring as ps
from nonconformist.cp import IcpRegressor
from nonconformist.nc import AbsErrorErrFunc, RegressorNc
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

from binwise import RegressionToClassificationEnsemble, RegressionToClassificationModel


# Helper functions for model evaluation
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates model performance using RMSE and R2 score."""
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return rmse, r2


def evaluate_crps(model, X_test, y_test, model_name):
    """Evaluates model uncertainty using Continuous Ranked Probability Score (CRPS)."""
    y_pred = model.predict_proba(X_test, return_std=True)
    y_pred_mean, y_pred_std = y_pred["mean"], y_pred["std"]
    crps = ps.crps_gaussian(y_test, y_pred_mean, y_pred_std)
    print(f"{model_name} - CRPS: {crps.mean():.4f}")
    return crps.mean()


# We will use the diabetes dataset as it's a well-known public regression dataset with small size.
# Let's load the diabetes dataset:
dataset = load_diabetes()
X, y = dataset.data, dataset.target

# We can normalize the target variable to makes CRPS values more interpretable in this example:
y = (y - y.mean()) / y.std()

# We will split data into three parts for the fair comparison:
# - Training set: Used to train the base models
# - Calibration set: Used by methods requiring calibration (e.g., Conformal Prediction)
# - Test set: Used for final evaluation of all methods
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_full, y_train_full, test_size=0.1875, random_state=42
)


# 1. RandomForest Regressor with Inductive Conformal Prediction
# We will start with a simple baseline approach: RandomForest Regressor for predictions,
# Inductive Conformal Prediction for uncertainty estimation
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
rmse, r2 = evaluate_model(rf_reg, X_test, y_test, "RandomForest Regressor")

# Let's set up Inductive Conformal Prediction for uncertainty estimation:
nc = RegressorNc(rf_reg, AbsErrorErrFunc())
icp = IcpRegressor(nc)
icp.calibrate(X_cal, y_cal)

# We will calculate CRPS using predictions from Inductive Conformal Prediction.
# Let's generate predictions for multiple significance levels to approximate distribution:
significance_range = np.arange(0.005, 0.995, 0.005)
y_icp_pred = np.zeros((X_test.shape[0], significance_range.shape[0] * 2))

for j, alpha in enumerate(significance_range):
    predictions = icp.predict(X_test, significance=alpha)
    y_icp_pred[:, j] = predictions[:, 0]
    y_icp_pred[:, -(j + 1)] = predictions[:, 1]

y_pred_mean = np.mean(y_icp_pred, axis=1)
y_pred_std = np.std(y_icp_pred, axis=1)
icp_crps = ps.crps_gaussian(y_test, y_pred_mean, y_pred_std)
print(f"RandomForest Regressor ICP - CRPS: {icp_crps.mean():.4f}")

# Output:
# RandomForest Regressor - RMSE: 0.7016, R2: 0.4721
# RandomForest Regressor ICP - CRPS: 0.3987


# 2. RandomForest Classifier with RegressionToClassificationModel
# Let's try RandomForest Classifier implementation with our adapter
# and see how it performs for uncertainty estimation:
rf_class = RegressionToClassificationModel(
    model_constructor=lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    n_bins=10,
    binning_strategy="Uniform",
)
rf_class.fit(X_train, y_train)
rmse, r2 = evaluate_model(
    rf_class, X_test, y_test, "RandomForest with RegressionToClassificationModel"
)
crps = evaluate_crps(
    rf_class, X_test, y_test, "RandomForest with RegressionToClassificationModel"
)
# RandomForest with RegressionToClassificationModel - RMSE: 0.7120, R2: 0.4564
# RandomForest with RegressionToClassificationModel - CRPS: 0.3985

# As we can see our model performs slightly worse but already outputs better 
# uncertainty estimates. Let's see if we can improve it!


# 3. RandomForest Classifier with RegressionToClassificationEnsemble
# We will try to improve our model by using an ensemble of different bin sizes and strategies:
rf_ensemble = RegressionToClassificationEnsemble(
    base_model_constructor=lambda: RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    bin_sizes=[5, 10, 15],
    binning_strategies=["Uniform", "Quantile"],
    subsample_ratio=1.0,
    random_state=42,
)
rf_ensemble.fit(X_train, y_train)
rmse, r2 = evaluate_model(
    rf_ensemble, X_test, y_test, "RandomForest with RegressionToClassificationEnsemble"
)
crps = evaluate_crps(
    rf_ensemble, X_test, y_test, "RandomForest with RegressionToClassificationEnsemble"
)
# Output:
# RandomForest with RegressionToClassificationEnsemble - RMSE: 0.6981, R2: 0.4774
# RandomForest with RegressionToClassificationEnsemble - CRPS: 0.3889

# Now our model performs better and outputs better uncertainty estimates!


# 4. TabPFN with RegressionToClassificationModel
# But what about models that don't even have a regression implementation?
# Let's try TabPFN, which has shown good performance for small classification datasets in benchmark studies.
tabpfn_class = RegressionToClassificationModel(
    model_constructor=lambda: TabPFNClassifier(
        device="cpu", N_ensemble_configurations=1
    ),
    n_bins=10,
    binning_strategy="Uniform",
)
tabpfn_class.fit(X_train, y_train)
rmse, r2 = evaluate_model(
    tabpfn_class, X_test, y_test, "TabPFN with RegressionToClassificationModel"
)
crps = evaluate_crps(
    tabpfn_class, X_test, y_test, "TabPFN with RegressionToClassificationModel"
)
# Output:
# TabPFN with RegressionToClassificationModel - RMSE: 0.6956, R2: 0.4812
# TabPFN with RegressionToClassificationModel - CRPS: 0.3799

# TabPFN shows promising results - better prediction accuracy and improved uncertainty estimates.


# 5. TabPFN with RegressionToClassificationEnsemble
# Let's combine TabPFN with our ensemble of different bin sizes and strategies:
tabpfn_ensemble = RegressionToClassificationEnsemble(
    base_model_constructor=lambda: TabPFNClassifier(
        device="cpu", N_ensemble_configurations=1
    ),
    bin_sizes=[8, 10],
    binning_strategies=["Uniform", "Quantile"],
    subsample_ratio=1.0,
    random_state=42,
)
tabpfn_ensemble.fit(X_train, y_train)
rmse, r2 = evaluate_model(
    tabpfn_ensemble, X_test, y_test, "TabPFN with RegressionToClassificationEnsemble"
)
crps = evaluate_crps(
    tabpfn_ensemble, X_test, y_test, "TabPFN with RegressionToClassificationEnsemble"
)
# Output:
# TabPFN with RegressionToClassificationEnsemble - RMSE: 0.6828, R2: 0.5000
# TabPFN with RegressionToClassificationEnsemble - CRPS: 0.3733

# The ensemble approach improves our results even further! We get better scores across all metrics.


# So far TabPFN with our ensemble shows the best results - but we have an additional big advantage:
# we don't need to set aside data for calibration! Let's use this advantage and train our best model
# on all available training data. The test set stays the same to make sure the comparison is fair.


# 6. TabPFN with RegressionToClassificationEnsemble (Full Dataset)
# We continue with the best performing approach - TabPFN with our ensemble of different bin sizes
# and strategies. We will use this approach with all of the available training data, without
# separating a calibration dataset.

tabpfn_ensemble_full = RegressionToClassificationEnsemble(
    base_model_constructor=lambda: TabPFNClassifier(
        device="cpu", N_ensemble_configurations=1
    ),
    bin_sizes=[8, 10],
    binning_strategies=["Uniform", "Quantile"],
    subsample_ratio=1.0,
    random_state=42,
)
tabpfn_ensemble_full.fit(X_train_full, y_train_full)
rmse, r2 = evaluate_model(
    tabpfn_ensemble_full,
    X_test,
    y_test,
    "TabPFN with RegressionToClassificationEnsemble (Full Dataset)",
)
crps = evaluate_crps(
    tabpfn_ensemble_full,
    X_test,
    y_test,
    "TabPFN with RegressionToClassificationEnsemble (Full Dataset)",
)

# Output:
# TabPFN with RegressionToClassificationEnsemble (Full Dataset) - RMSE: 0.6627, R2: 0.5291
# TabPFN with RegressionToClassificationEnsemble (Full Dataset) - CRPS: 0.3611

# Using the full training dataset gives us the best results yet! This showcases a key advantage 
# of our approach - we don't need to reserve data for calibration like other uncertainty estimation
# methods do, and we see even further improvements.

# What's next?
# - Try this on your own data, especially if you have a small dataset
# - Don't forget to use proper cross-validation for more reliable results
# - Experiment with the base model parameters - maybe you can achieve even better performance