# binwise - binning adapter for regression tasks

Implementation of the binned uncertainty estimation ensemble method, as described in paper "An Efficient Model-Agnostic Approach for Uncertainty Estimation in Data-Restricted Pedometric Applications" (Barkov et al., 2024).

The Bin Adapter transforms regression tasks into classification problems, enabling the use of classification algorithms for regression and providing uncertainty estimates. This approach is particularly useful in scenarios with limited training data, such as in pedometrics and digital soil mapping.

## Features

- Use classification algorithms with scikit-learn interface for regression tasks
- Obtain uncertainty estimates for regression predictions

## Installation

```bash
pip install binwise
```

## Quick Start

Simple example of how to use the binning adapter with TabPFN:
```python
from tabpfn import TabPFNClassifier
from binwise import RegressionToClassificationEnsemble

tabpfn_ensemble = RegressionToClassificationEnsemble(
    base_model_constructor=lambda: TabPFNClassifier(N_ensemble_configurations=1),
    random_state=42,
)

tabpfn_ensemble.fit(X_train, y_train)
y_pred = tabpfn_ensemble.predict(X_test)
```

## Examples

For more detailed examples, check out the following notebooks:

[Basic Regression Prediction](examples/diabetes_prediction.ipynb): Demonstrates how to use the binning adapter for regression prediction on the diabetes dataset.

[Regression with Uncertainty Estimation](examples/diabetes_uncertainty.ipynb): Shows how to obtain uncertainty estimates along with predictions using the binning adapter.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{barkov2024uncertainty,
      title={An Efficient Model-Agnostic Approach for Uncertainty Estimation in Data-Restricted Pedometric Applications}, 
      author={Viacheslav Barkov and Jonas Schmidinger and Robin Gebbers and Martin Atzmueller},
      year={2024},
      eprint={2409.11985},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.11985}, 
}
```

## License:

This project is licensed under the AGPLv3 License - see the [LICENSE.md](LICENSE.md) file for details.