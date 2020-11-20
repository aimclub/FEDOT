from fedot.core.chain_validation import validate
from fedot.core.composer.chain import Chain


def constraint_function(chain: Chain):
    try:
        validate(chain)
        return True
    except ValueError:
        return False
