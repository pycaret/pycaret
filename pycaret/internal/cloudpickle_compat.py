"""Cloudpickle with forced protocol 4 for compatibility."""

from cloudpickle import Pickler as _CloudPickler


class Pickler(_CloudPickler):
    def __init__(self, file, protocol=None, **kwargs):
        if protocol is None or protocol > 4:
            protocol = 4
        super().__init__(file, protocol=protocol, **kwargs)
