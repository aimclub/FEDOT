class IDelegator:
    pass


class DelegatorFactory:
    @staticmethod
    def __init__(self, base, *args, **kwargs):
        self.base = base
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def __getattr__(self, name):
        return getattr(self.base, name)

    @classmethod
    def __prepare_dict(cls, overriden_attrs: dict):
        overriden_attrs = overriden_attrs or {}
        return {
            "__init__": DelegatorFactory.__init__,
            "__getattr__": DelegatorFactory.__getattr__,
            **overriden_attrs,
        }

    @classmethod
    def create_delegator_cls(cls, name, overriden_attrs=None, base_cls=tuple()):
        overriden_attrs = cls.__prepare_dict(overriden_attrs)
        return type(name, (*base_cls, IDelegator), overriden_attrs)

    @classmethod
    def create_delegator_inst(
        cls, base, overriden_attrs=None, cls_name=None, *args, **kwargs
    ):
        type_base = type(base)
        name = cls_name or f"{type_base.__name__}Delegator"
        created_cls = cls.create_delegator_cls(name, overriden_attrs, base_cls=(type_base,))
        return created_cls(base, *args, **kwargs)