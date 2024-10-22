import numpy as np


def quantile_cut(data, num_bins):
    """Bins data into intervals based on quantiles.

    Divides the data into bins of approximately equal size by using quantile boundaries.

    Args:
        data: Array of numerical values to be binned.
        num_bins: Number of bins to create. Must be a positive integer.

    Returns:
        tuple: A tuple containing:
            - bin_labels: Array of bin assignments for each value in data (0-based indices)
            - bin_edges: Array of bin boundaries including leftmost and rightmost edges

    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
        >>> labels, edges = quantile_cut(data, 4)
        >>> print(labels)  # Shows which bin (0-3) each value falls into
        >>> print(edges)   # Shows the boundary values for the bins
    """
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(data, quantiles)
    bin_edges = np.unique(bin_edges)  # Remove duplicates
    bin_labels = np.digitize(data, bin_edges[1:-1])
    return bin_labels, bin_edges


def uniform_cut(data, num_bins):
    """Bins data into intervals of uniform width.

    Divides the data into bins of equal width spanning from the minimum to maximum value.

    Args:
        data: Array of numerical values to be binned.
        num_bins: Number of bins to create. Must be a positive integer.

    Returns:
        tuple: A tuple containing:
            - bin_labels: Array of bin assignments for each value in data (0-based indices)
            - bin_edges: Array of bin boundaries including leftmost and rightmost edges

    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
        >>> labels, edges = uniform_cut(data, 4)
        >>> print(labels)  # Shows which bin (0-3) each value falls into
        >>> print(edges)   # Shows evenly spaced boundary values for the bins
    """
    bin_edges = np.linspace(np.min(data), np.max(data), num_bins + 1)
    bin_labels = np.digitize(data, bin_edges[1:-1])
    return bin_labels, bin_edges


def convert_to_bins(data, num_bins, binning_strategy="Uniform"):
    """Converts numerical data into bins using specified binning strategy.

    This function supports two binning strategies:
    - Uniform: Creates bins of equal width
    - Quantile: Creates bins with approximately equal number of items

    Args:
        data: Numerical data to be binned. Can be a list or numpy array.
        num_bins: Number of bins to create. Must be positive.
        binning_strategy: Strategy for binning.
            Defaults to "Uniform". Options: "Uniform", "Quantile".

    Returns:
        tuple: A tuple containing:
            - bin_labels: Array of bin assignments for each value in data (0-based indices)
            - bin_ranges: Dictionary mapping bin indices to string representations of ranges
            - bin_middles: Array of middle points for each bin

    Raises:
        ValueError: If binning_strategy is not one of "Uniform" or "Quantile"

    Example:
        >>> data = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9]
        >>> labels, ranges, middles = convert_to_bins(data, 4, "Uniform")
        >>> print(ranges)  # Shows the range for each bin
        >>> print(middles)  # Shows the middle point of each bin
    """
    # Accept list and tuple, but cast them to numpy array for consistency
    if isinstance(data, (list, tuple)):
        data = np.array(data)

    if binning_strategy == "Quantile":
        bin_labels, bins = quantile_cut(data, num_bins)
    elif binning_strategy == "Uniform":
        bin_labels, bins = uniform_cut(data, num_bins)
    else:
        raise ValueError(
            "Binning strategy not recognized. Choose from 'Uniform', 'Quantile'."
        )

    # Create string representations of bin ranges with 3 decimal places
    bin_ranges = {i: f"({bins[i]:.3f}, {bins[i+1]:.3f}]" for i in range(len(bins) - 1)}
    # Calculate middle points for each bin
    bin_middles = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])

    return bin_labels, bin_ranges, bin_middles
