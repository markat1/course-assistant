import logging

def configure_logging(level: str) ->  None:
    """Configure console logging and the gateway log level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("gateway").setLevel(level)