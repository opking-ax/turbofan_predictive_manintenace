import math
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def phm08_score(actual_rul_list, predicted_rul_list):
    total_score = 0

    for i in range(len(predicted_rul_list)):
        error = predicted_rul_list[i] - actual_rul_list[i]

        if error < 0:
            penalty = math.exp(-error / 13) - 1
        else:
            penalty = math.exp(error / 10) - 1

        total_score += penalty

    return total_score


def evaluate_calibration(
    actual_rul, lower_bounds, upper_bounds, target_coverage=0.80
):
    within_interval = [
        (actual >= lower) and (actual <= upper)
        for actual, lower, upper in zip(actual_rul, lower_bounds, upper_bounds)
    ]

    empirical_coverage = np.mean(within_interval)
    calibration_gap = empirical_coverage - target_coverage

    return {
        "empirical_coverage": empirical_coverage,
        "target_coverage": target_coverage,
        "calibration_gap": calibration_gap,
    }


def pinball_loss(actual, predicted_quantile, q):
    errors = actual - predicted_quantile
    return np.maximum(q * errors, (q - 1) * errors)


def compute_regression_metrics(actual, prediction):
    rmse = root_mean_squared_error(actual, prediction)
    mae = mean_absolute_error(actual, prediction),
    r2 = r2_score(actual, prediction),
    phm08 = phm08_score(actual, prediction)

    return {"rsme": rmse, "mae": mae, "r2": r2, "phm08": phm08}
