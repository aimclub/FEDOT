from typing import Callable, List, Optional

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_validation import custom_validate, validate


def constraint_function(chain: Chain,
                        custom_rules: Optional[List[Callable]] = None):
    try:
        if not custom_rules:
            validate(chain)
        else:
            custom_validate(chain, custom_rules)
        return True
    except ValueError:
        return False
