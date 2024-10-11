import numpy as np

def quantile_cut(data, num_bins):
    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.quantile(data, quantiles)
    bins = np.unique(bins)  # Remove duplicates
    return np.digitize(data, bins[1:-1]), bins

def uniform_cut(data, num_bins):
    bins = np.linspace(np.min(data), np.max(data), num_bins + 1)
    return np.digitize(data, bins[1:-1]), bins

def convert_to_bins(data, num_bins, binning_strategy="Uniform"):
    if isinstance(data, list):
        data = np.array(data)
    
    if binning_strategy == "Quantile":
        try:
            bin_labels, bins = quantile_cut(data, num_bins)
        except ValueError as ve:
            print(f"{ve} Error occurred when forming Quantile bins.")
            return None, None, None
    elif binning_strategy == "Uniform":
        try:
            bin_labels, bins = uniform_cut(data, num_bins)
        except ValueError as ve:
            print(f"{ve} Error occurred when forming Uniform bins.")
            return None, None, None
    else:
        raise ValueError(
            "Binning strategy not recognized. Choose from 'Uniform', 'Quantile'."
        )
    
    bin_ranges = {
        i: f"({bins[i]:.3f}, {bins[i+1]:.3f}]" for i in range(len(bins) - 1)
    }
    bin_middles = np.array([
        (bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)
    ])
    
    return bin_labels, bin_ranges, bin_middles