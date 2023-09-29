from inspect import stack


class MethodNotImplementError(NotImplementedError):
    def __init__(self, _class_):
        super(MethodNotImplementError, self).__init__(f'Method {stack()[1][3]} is not implemented in {_class_}.')


class AbstractMethodNotImplementError(NotImplementedError):
    def __init__(self):
        super(AbstractMethodNotImplementError, self).__init__(f'Trying to invoke abstract method.')
