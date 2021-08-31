class SimpleArchive(list):
    def update(self, x):
        self.extend((x))

    @property
    def items(self):
        return list(self)
