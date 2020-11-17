class DataOperator:
    def __init__(self, external_source):
        self.external = external_source
        self.out = None

    def input(self):
        return self.external

    def set_output(self, output):
        self.out = output
