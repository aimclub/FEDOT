Memory Analytics
================

Class for monitoring Python-related system memory consumption.

Methods:
    start(): Start memory monitoring session.
    finish(): Finish memory monitoring session.
    get_measures(): Estimate Python-related system memory consumption in MiB.
    log(logger, additional_info, logging_level): Print the message about current and maximal memory consumption to the log or console.

Args:
    logger: Optional logger that should be used in output.
    additional_info: Label for the current location in the code.
    logging_level: Level of the message.
