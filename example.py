from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, root_mean_squared_error
from tabpfn import TabPFNClassifier

from binwise import RegressionToClassificationModel, RegressionToClassificationEnsemble

# Load the dataset
dataset = load_diabetes()

# Print dataset information
print(dataset.DESCR)

X, y = dataset.data, dataset.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Function to evaluate and print results
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")


# 1. RandomForest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
evaluate_model(rf_reg, X_test, y_test, "RandomForest")

# 2. RandomForest with RegressionToClassificationModel
rf_class = RegressionToClassificationModel(
    model_constructor=lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    n_bins=10,
    binning_strategy="Uniform",
)
rf_class.fit(X_train, y_train)
evaluate_model(
    rf_class, X_test, y_test, "RandomForest with RegressionToClassificationModel"
)

# 3. RandomForest with RegressionToClassificationEnsemble
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


# 4. TabPFN with RegressionToClassificationModel
tabpfn_class = RegressionToClassificationModel(
    model_constructor=lambda: TabPFNClassifier(
        device="cpu", N_ensemble_configurations=1
    ),
    n_bins=10,
    binning_strategy="Uniform",
)
tabpfn_class.fit(X_train, y_train)
evaluate_model(
    tabpfn_class, X_test, y_test, "TabPFN with RegressionToClassificationModel"
)

# 5. TabPFN with RegressionToClassificationEnsemble
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
evaluate_model(
    tabpfn_ensemble, X_test, y_test, "TabPFN with RegressionToClassificationEnsemble"
)
