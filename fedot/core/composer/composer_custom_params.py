class CustomParameters:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CustomParameters, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.custom_params = None

    def declare_params(self, params: dict):
        self.custom_params = params
