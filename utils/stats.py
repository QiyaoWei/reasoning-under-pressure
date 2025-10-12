import numpy as np
from typing import Tuple
from scipy import stats


def calculate_wilson_confidence_interval(successes: int, total: int, confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """Calculate confidence interval for a binomial proportion using the Wilson score interval."""
    if total == 0:
        return 0.0, 0.0, 0.0

    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)

    n = total
    p_hat = successes / n

    denominator = 1 + (z_score**2 / n)
    center = (p_hat + (z_score**2 / (2 * n))) / denominator
    margin = (z_score / denominator) * np.sqrt((p_hat * (1 - p_hat) / n) + (z_score**2 / (4 * n**2)))

    lower_bound = max(0.0, center - margin)
    upper_bound = min(1.0, center + margin)
    return lower_bound, upper_bound, float(margin)


def calculate_t_confidence_interval(mean_value: float, sample_variance: float, n: int, confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """Calculate confidence interval for a mean using the Student's t-distribution.

    Args:
        mean_value: Sample mean.
        sample_variance: Unbiased sample variance (denominator n-1).
        n: Sample size.
        confidence_level: Desired confidence level (e.g., 0.95).

    Returns:
        (lower_bound, upper_bound, margin_of_error)
    """
    if n <= 1:
        lower_bound = max(0.0, min(1.0, mean_value))
        return lower_bound, lower_bound, 0.0

    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    std_error = np.sqrt(max(0.0, sample_variance) / n)
    margin = float(t_crit * std_error)

    lower_bound = max(0.0, mean_value - margin)
    upper_bound = min(1.0, mean_value + margin)
    return lower_bound, upper_bound, margin


