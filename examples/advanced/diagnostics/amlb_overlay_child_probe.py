"""No training: report effective constructors from a real sanitized AMLB child."""
import json
import os
import sys
from pathlib import Path

import fedot
from fedot.core.operations.operation_parameters import OperationParameters, get_default_params
from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import (
    FedotLightGBMClassificationImplementation,
)


def main():
    request = json.load(sys.stdin)
    model = FedotLightGBMClassificationImplementation(OperationParameters.from_operation_type('lgbm'))
    explicit = FedotLightGBMClassificationImplementation(
        OperationParameters.from_operation_type('lgbm', n_estimators=23))
    probe = dict(
        fedot_file=str(Path(fedot.__file__).resolve()),
        overlay=os.environ.get('FEDOT_OPERATION_DEFAULTS_OVERLAY'),
        defaults=get_default_params('lgbm'),
        regression_defaults=get_default_params('lgbmreg'),
        constructor_rounds=model.model.n_estimators,
        explicit_rounds=explicit.model.n_estimators,
        # Inspect only named non-secret keys, never dump the environment.
        tmpdir=os.environ.get('TMPDIR'),
        xdg_cache=os.environ.get('XDG_CACHE_HOME'),
        unretained=os.environ.get('FEDOT_DIAGNOSTIC_UNRETAINED'),
    )
    response = dict(error_message=None, predictions=None, truth=None,
                    probabilities=None, optional_columns=None, output_file=None,
                    models_count=0, training_duration=0, predict_duration=0,
                    others=dict(probe=probe))
    with open(request['config']['result_file'], 'w') as stream:
        json.dump(response, stream)


if __name__ == '__main__':
    main()
