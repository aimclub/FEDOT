"""Atomic registration decisions, independent of registry mutation."""
from dataclasses import dataclass
from typing import Tuple

from pymonad.either import Left, Right

from fedot.extensions.contracts import ExtensionError, OperationKind
from fedot.extensions.validation import validate_extension_manifest


@dataclass(frozen=True)
class RegistrationEntry:
    extension_name: str
    operation_name: str
    kind: OperationKind


@dataclass(frozen=True)
class RegistrationPlan:
    extension_names: Tuple[str, ...]
    entries: Tuple[RegistrationEntry, ...]


def plan_registration(manifests, existing=(), reserved_names=()):
    names = {manifest.name for manifest in existing}
    operations = {spec.name for manifest in existing
                  for spec in manifest.models + manifest.transforms}
    reserved = set(reserved_names)
    entries = []
    extension_names = []
    for manifest in manifests:
        validation = validate_extension_manifest(manifest)
        if validation.is_left():
            return validation
        if manifest.name in names:
            return Left(ExtensionError('duplicate_extension', 'Extension is already registered.',
                                       {'extension': manifest.name}))
        names.add(manifest.name)
        extension_names.append(manifest.name)
        for specs, kind in ((manifest.models, OperationKind.model),
                            (manifest.transforms, OperationKind.transform)):
            for spec in specs:
                if spec.name in operations or spec.name in reserved:
                    return Left(ExtensionError('operation_name_conflict', 'Operation name is already in use.',
                                               {'extension': manifest.name, 'operation': spec.name}))
                operations.add(spec.name)
                entries.append(RegistrationEntry(manifest.name, spec.name, kind))
    return Right(RegistrationPlan(tuple(extension_names), tuple(entries)))
