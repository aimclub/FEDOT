def exclude_ts(preprocessing_function):
    """ Decorator for time series type checking """
    def wrapper(self, input_data, source_name, *args, **kwargs):
        if input_data.is_ts:
            return input_data
        return preprocessing_function(self, input_data, source_name, *args, **kwargs)
    return wrapper


def exclude_multi_ts(preprocessing_function):
    """ Decorator for multi time series (not to be confused with multivariate ts) type checking """
    def wrapper(self, input_data, source_name, *args, **kwargs):
        if input_data.is_multi_ts:
            return input_data
        return preprocessing_function(self, input_data, source_name, *args, **kwargs)
    return wrapper


def exclude_image(preprocessing_function):
    """ Decorator for image type checking """
    def wrapper(self, input_data, source_name, *args, **kwargs):
        if input_data.is_image:
            return input_data
        return preprocessing_function(self, input_data, source_name, *args, **kwargs)
    return wrapper

