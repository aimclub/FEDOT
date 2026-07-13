import logging
from dataclasses import dataclass
from typing import Optional

from golem.core.log import default_log


@dataclass(frozen=True)
class ValidationContext:
    logger: Optional[logging.Logger] = None

    def get_logger(self) -> logging.Logger:
        if self.logger is not None:
            return self.logger
        return default_log(prefix='FedotValidation')
