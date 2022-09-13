from fedot.core.data.data import data_type_is_ts, data_type_is_multi_ts, data_type_is_image


def exclude_ts(preprocessing_function):
    """ Decorator for time series type checking """
    def wrapper(self, input_data, source_name, *args, **kwargs):
        if data_type_is_ts(input_data):
            return input_data
        return preprocessing_function(self, input_data, source_name, *args, **kwargs)
    return wrapper


def exclude_multi_ts(preprocessing_function):
    """ Decorator for multi time series (not to be confused with multivariate ts) type checking """
    def wrapper(self, input_data, source_name, *args, **kwargs):
        if data_type_is_multi_ts(input_data):
            return input_data
        return preprocessing_function(self, input_data, source_name, *args, **kwargs)
    return wrapper


def exclude_image(preprocessing_function):
    """ Decorator for image type checking """
    def wrapper(self, input_data, source_name, *args, **kwargs):
        if data_type_is_image(input_data):
            return input_data
        return preprocessing_function(self, input_data, source_name, *args, **kwargs)
    return wrapper

