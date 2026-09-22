from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from sklearn.preprocessing import MinMaxScaler


def basis_approximation_metric(derivation_coef: np.array,
                               metric_values: np.array,
                               regularization_coef: float = 0.7):

    # Get a sum of absolute values (L1 norm) for each polynom derivative
    sum_of_derivative_coef = np.array([sum(abs(x)) for x in derivation_coef])

    # Normalize metrics and der_coef
    normalized_derivative_coef = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        sum_of_derivative_coef.reshape(-1, 1))[:, 0]
    normalized_metric_values = MinMaxScaler(feature_range=(
        0, 1)).fit_transform(metric_values.reshape(-1, 1))[:, 0]

    # Alternative approach where metric = regulizer*np.exp(der_coef)*metric_val
    # exponent_of_coef = np.array([np.exp(x) for x in normalized_derivative_coef])
    # polynom_metric = exponent_of_coef * self.mse

    # Basic approach where metric = regulizer*der_coef+metric_val
    polynom_metric = regularization_coef * \
        normalized_derivative_coef + normalized_metric_values
    best_aprox_polynom_index = np.where(
        polynom_metric == min(polynom_metric))[0][0]

    return best_aprox_polynom_index
