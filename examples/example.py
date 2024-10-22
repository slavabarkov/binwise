from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

from binwise import RegressionToClassificationEnsemble, RegressionToClassificationModel


# Simple helper function to help us evaluate model performance using RMSE and R2 score:
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates model performance using RMSE and R2 score."""
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")


# We will use the diabetes dataset as it's a well-known public regression dataset with small size.
# Let's load the diabetes dataset:
dataset = load_diabetes()
X, y = dataset.data, dataset.target

# Split the data into training and testing sets.
# Note: for better evaluation, consider using cross-validation instead of a single train-test split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Let's start with our experiments!


# 1. RandomForest Regressor.
# We will start with a standard RandomForest Regressor as our baseline:
rf_reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
)
rf_reg.fit(X_train, y_train)
evaluate_model(rf_reg, X_test, y_test, "RandomForest")
# Output:
# RandomForest - RMSE: 54.3324, R2: 0.4428


# 2. RandomForest Classifier with RegressionToClassificationModel.
# Let's try RandomForest Classifier implementation with our adapter and see how it performs:
rf_class = RegressionToClassificationModel(
    model_constructor=lambda: RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
    n_bins=10,
    binning_strategy="Uniform",
)
rf_class.fit(X_train, y_train)
evaluate_model(
    rf_class, X_test, y_test, "RandomForest with RegressionToClassificationModel"
)
# Output:
# RandomForest with RegressionToClassificationModel - RMSE: 53.3076, R2: 0.4636

# As we can see, we've already improved our results just by using the classification version.
# But let's explore more.


# 3. RandomForest Classifier with RegressionToClassificationEnsemble.
# Let's see if we can do even better by using an ensemble of different bin sizes and strategies:
rf_ensemble = RegressionToClassificationEnsemble(
    base_model_constructor=lambda: RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
    bin_sizes=[5, 10, 15],
    binning_strategies=["Uniform", "Quantile"],
    subsample_ratio=1.0,
    random_state=42,
)
rf_ensemble.fit(X_train, y_train)
evaluate_model(
    rf_ensemble, X_test, y_test, "RandomForest with RegressionToClassificationEnsemble"
)
# Output:
# RandomForest with RegressionToClassificationEnsemble - RMSE: 52.9668, R2: 0.4705

# As we can see, the ensemble approach gives us another boost in performance.


# 4. TabPFN with RegressionToClassificationModel.
# But what about models that don't even have a regression implementation?
# Let's try TabPFN, which has shown good performance for small classification datasets in benchmark studies.
# Note: We set N_ensemble_configurations=1 to disable TabPFN's internal ensembling
tabpfn_class = RegressionToClassificationModel(
    model_constructor=lambda: TabPFNClassifier(
        device="cpu",
        N_ensemble_configurations=1,
    ),
    n_bins=10,
    binning_strategy="Uniform",
)
tabpfn_class.fit(X_train, y_train)
evaluate_model(
    tabpfn_class, X_test, y_test, "TabPFN with RegressionToClassificationModel"
)
# Output:
# TabPFN with RegressionToClassificationModel - RMSE: 49.3873, R2: 0.5396

# As we can see, TabPFN performs well, and even better than our previous attempts.


# 5. TabPFN with RegressionToClassificationEnsemble
# Finally, let's combine TabPFN with our ensemble of different bin sizes and strategies:
tabpfn_ensemble = RegressionToClassificationEnsemble(
    base_model_constructor=lambda: TabPFNClassifier(
        device="cpu",
        N_ensemble_configurations=1,
    ),
    bin_sizes=[8, 10],
    binning_strategies=["Uniform", "Quantile"],
    subsample_ratio=1.0,
    random_state=42,
)
tabpfn_ensemble.fit(X_train, y_train)
evaluate_model(
    tabpfn_ensemble, X_test, y_test, "TabPFN with RegressionToClassificationEnsemble"
)
# Output:
# TabPFN with RegressionToClassificationEnsemble - RMSE: 49.1802, R2: 0.5435

# We managed to improve again - our best performance yet!


# What's next?
# - Check out the uncertainty_example.py to see how to get and evaluate uncertainty estimates
# - Try this on your own data, especially if you have a small dataset
# - Don't forget to use proper cross-validation for more reliable results
# - Experiment with the base model parameters - maybe you can achieve even better performance
