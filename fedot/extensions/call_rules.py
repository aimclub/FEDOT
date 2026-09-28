"""Select call shapes without executing extension code."""
import inspect
from dataclasses import dataclass
from typing import Tuple

from pymonad.either import Left, Right

from fedot.extensions.contracts import ExtensionContractError, ExtensionError, unwrap_extension_result


@dataclass(frozen=True)
class CallShape:
    positional: int
    keywords: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CallPlan:
    candidate_index: int


FACTORY_SHAPES = (CallShape(1), CallShape(0, ('params',)), CallShape(0))


def plan_call(signature: inspect.Signature, shapes: Tuple[CallShape, ...],
              code: str = 'unsupported_method_signature'):
    for index, shape in enumerate(shapes):
        try:
            signature.bind(*([None] * shape.positional),
                           **dict.fromkeys(shape.keywords))
        except TypeError:
            continue
        return Right(CallPlan(index))
    return Left(ExtensionError(code, 'No supported call signature.',
                               {'signature': str(signature)}))


def inspect_call(function, shapes, code='unsupported_method_signature'):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        return Left(ExtensionError(code, 'Callable signature is unavailable.', cause=exc))
    return plan_call(signature, shapes, code)


def invoke_once(function, candidates, *, signature_code='unsupported_method_signature',
                failure_code='extension_execution_failed'):
    shapes = tuple(CallShape(len(args), tuple(kwargs)) for args, kwargs in candidates)
    plan = unwrap_extension_result(inspect_call(function, shapes, signature_code))
    args, kwargs = candidates[plan.candidate_index]
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        error = ExtensionError(failure_code, 'Extension callable failed.', cause=exc)
        raise ExtensionContractError(error) from exc


def invoke_factory(factory, params, failure_code='extension_execution_failed'):
    return invoke_once(factory, (((params,), {}), ((), {'params': params}), ((), {})),
                       signature_code='invalid_factory_signature', failure_code=failure_code)
