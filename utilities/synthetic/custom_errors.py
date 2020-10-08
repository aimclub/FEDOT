class OperatingSystemValidation(Exception):
    """Raised when the operation system is not valid"""
    pass


class JsonFileExtensionValidation(Exception):
    """Raised when user want to save JSON with other extension"""
    pass


class JsonFileInvalid(Exception):
    """Raised when JSON is invalid"""
    pass
