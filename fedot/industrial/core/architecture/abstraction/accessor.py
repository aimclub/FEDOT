from functools import reduce
from typing import Any, List

from torch.nn import Module


class Accessor:
    @staticmethod
    def _set_names(root: Module):
        for name, module in root.named_modules():
            module._eigenname = name

    @classmethod
    def set_module(cls, m: Module, name: str, new: Module):
        if not name:
            return new
        *path, name = name.split('.')
        parent = reduce(getattr, path, m)
        setattr(parent, name, new)

    @classmethod
    def get_module(cls, m: Module, name: str) -> Module:
        if not name:
            return m
        return reduce(getattr, name.split('.'), m)

    @classmethod
    def __fetch_names(cls, root: Module, order: list) -> List[str]:
        cls._set_names(root)
        return [module._eigenname for module in order]

    @classmethod
    def get_names_order(cls, model: Module, *example_input) -> List[str]:
        modules_order = cls.get_layers_order(model, *example_input)
        names_order = cls.__fetch_names(model, modules_order)
        return names_order

    @classmethod
    def get_layers_order(cls, model: Module, *example_input) -> List[Module]:
        order = []
        hooks = []

        def add_hook(m):
            def forward_pre_hook(module, input):
                order.append(module)

            registered_hook = m.register_forward_pre_hook(forward_pre_hook)
            hooks.append(registered_hook)

        model.apply(add_hook)
        model(*example_input)
        [hook.remove() for hook in hooks]
        return order

    @classmethod
    def get_submodule_inputs(cls, model: Module, *example_input) -> List[Any]:
        inputs = []
        hooks = []

        def add_hook(m: Module):
            def forward_pre_hook(module: Module, input):
                inputs.append(input)

            registered_hook = m.register_forward_pre_hook(forward_pre_hook)
            hooks.append(registered_hook)

        model.apply(add_hook)
        model(*example_input)
        [hook.remove() for hook in hooks]
        return inputs

    @classmethod
    def get_name_input_mapping(cls, m: Module, *example_input: Any):
        names = cls.get_names_order(m, *example_input)
        inputs = cls.get_submodule_inputs(m, *example_input)
        return dict(zip(names, inputs))

    @staticmethod
    def is_leaf_module(m: Module) -> bool:
        for _ in m.children():
            return False
        return True