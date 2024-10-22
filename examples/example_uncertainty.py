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
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return rmse, r2


def evaluate_crps(model, X_test, y_test, model_name):
    y_pred = model.predict_proba(X_test, return_std=True)
    y_pred_mean, y_pred_std = y_pred["mean"], y_pred["std"]
    crps = ps.crps_gaussian(y_test, y_pred_mean, y_pred_std)
    print(f"{model_name} - CRPS: {crps.mean():.4f}")
    return crps.mean()


# We will use the diabetes dataset as it's a well-known public regression dataset with small size.
# Let's load the diabetes dataset:
dataset = load_diabetes()
X, y = dataset.data, dataset.target

# Normalize the target variable to makes the CRPS values more interpretable
y = (y - y.mean()) / y.std()

# Split the data into three parts:
# - Training set: Used to train the models
# - Calibration set: Used by methods that need separate calibration set (like Conformal Prediction)
# - Test set: Used to evaluate the models
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_full, y_train_full, test_size=0.1875, random_state=42
)

# List to store the results for final comparison
results = []

# 1. RandomForest Regressor (Baseline with Conformal Prediction)
# We'll use this as our baseline, applying Conformal Prediction for uncertainty estimation
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
rmse, r2 = evaluate_model(rf_reg, X_test, y_test, "RandomForest Regressor")

# Set up Conformal Prediction
nc = RegressorNc(rf_reg, AbsErrorErrFunc())
icp = IcpRegressor(nc)
icp.calibrate(X_cal, y_cal)

# Calculate CRPS using predictions from Conformal Prediction
# We generate predictions for multiple significance levels to approximate distribution
significance_range = np.arange(0.005, 0.995, 0.005)
y_icp_pred = np.zeros((X_test.shape[0], significance_range.shape[0] * 2))

for j, alpha in enumerate(significance_range):
    predictions = icp.predict(X_test, significance=alpha)
    y_icp_pred[:, j] = predictions[:, 0]
    y_icp_pred[:, -(j + 1)] = predictions[:, 1]

y_pred_mean = np.mean(y_icp_pred, axis=1)
y_pred_std = np.std(y_icp_pred, axis=1)
icp_crps = ps.crps_gaussian(y_test, y_pred_mean, y_pred_std)
print(f"RandomForest Regressor ICP CRPS: {icp_crps.mean():.4f}")

results.append(["RandomForest Regressor", rmse, r2, icp_crps.mean()])

# 2. RandomForest with RegressionToClassificationModel
# Let's try the basic version of our approach with RandomForest
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
results.append(["RandomForest with RegressionToClassificationModel", rmse, r2, crps])

# 3. RandomForest with RegressionToClassificationEnsemble
# Now, let's try the ensemble version that combines multiple bin configurations
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
results.append(["RandomForest with RegressionToClassificationEnsemble", rmse, r2, crps])

# 4. TabPFN with RegressionToClassificationModel
# Let's try TabPFN, which is designed for small datasets but only available for classification
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
results.append(["TabPFN with RegressionToClassificationModel", rmse, r2, crps])

# 5. TabPFN with RegressionToClassificationEnsemble
# Finally, we will combine TabPFN with our ensemble approach
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
results.append(["TabPFN with RegressionToClassificationEnsemble", rmse, r2, crps])

print("\n--- Models trained on full dataset (train + calibration) ---\n")

# Now let's try training models on the full training dataset (without holding out calibration set)
# This gives our approach an advantage since it doesn't need a separate calibration set

# 6. RandomForest with RegressionToClassificationModel (Full Dataset)
rf_class_full = RegressionToClassificationModel(
    model_constructor=lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    n_bins=10,
    binning_strategy="Uniform",
)
rf_class_full.fit(X_train_full, y_train_full)
rmse, r2 = evaluate_model(
    rf_class_full,
    X_test,
    y_test,
    "RandomForest with RegressionToClassificationModel (Full Dataset)",
)
crps = evaluate_crps(
    rf_class_full,
    X_test,
    y_test,
    "RandomForest with RegressionToClassificationModel (Full Dataset)",
)
results.append(
    ["RandomForest with RegressionToClassificationModel (Full Dataset)", rmse, r2, crps]
)

# 7. RandomForest with RegressionToClassificationEnsemble (Full Dataset)
rf_ensemble_full = RegressionToClassificationEnsemble(
    base_model_constructor=lambda: RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    bin_sizes=[5, 10, 15],
    binning_strategies=["Uniform", "Quantile"],
    subsample_ratio=1.0,
    random_state=42,
)
rf_ensemble_full.fit(X_train_full, y_train_full)
rmse, r2 = evaluate_model(
    rf_ensemble_full,
    X_test,
    y_test,
    "RandomForest with RegressionToClassificationEnsemble (Full Dataset)",
)
crps = evaluate_crps(
    rf_ensemble_full,
    X_test,
    y_test,
    "RandomForest with RegressionToClassificationEnsemble (Full Dataset)",
)
results.append(
    [
        "RandomForest with RegressionToClassificationEnsemble (Full Dataset)",
        rmse,
        r2,
        crps,
    ]
)

# 8. TabPFN with RegressionToClassificationModel (Full Dataset)
tabpfn_class_full = RegressionToClassificationModel(
    model_constructor=lambda: TabPFNClassifier(
        device="cpu", N_ensemble_configurations=1
    ),
    n_bins=10,
    binning_strategy="Uniform",
)
tabpfn_class_full.fit(X_train_full, y_train_full)
rmse, r2 = evaluate_model(
    tabpfn_class_full,
    X_test,
    y_test,
    "TabPFN with RegressionToClassificationModel (Full Dataset)",
)
crps = evaluate_crps(
    tabpfn_class_full,
    X_test,
    y_test,
    "TabPFN with RegressionToClassificationModel (Full Dataset)",
)
results.append(
    ["TabPFN with RegressionToClassificationModel (Full Dataset)", rmse, r2, crps]
)

# 9. TabPFN with RegressionToClassificationEnsemble (Full Dataset)
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
results.append(
    ["TabPFN with RegressionToClassificationEnsemble (Full Dataset)", rmse, r2, crps]
)

# Print final comparison table
print("\n--- Summary Table ---\n")
print(f"{'Model':<70} {'RMSE':<10} {'R2':<10} {'CRPS':<10}")
print("-" * 100)
for result in results:
    print(f"{result[0]:<70} {result[1]:<10.4f} {result[2]:<10.4f} {result[3]:<10.4f}")
