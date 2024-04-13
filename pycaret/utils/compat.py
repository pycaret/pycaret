import os
import time

from joblib.logger import format_time


def joblib_format_load_msg(func_id, args_id, timestamp=None, metadata=None):
    """
    Helper function to format the message when loading the results.
    Vendorized from Joblib 1.3.2 to retain compatibility with Joblib 1.4.0.
    """
    signature = ""
    try:
        if metadata is not None:
            args = ", ".join(
                [
                    "%s=%s" % (name, value)
                    for name, value in metadata["input_args"].items()
                ]
            )
            signature = "%s(%s)" % (os.path.basename(func_id), args)
        else:
            signature = os.path.basename(func_id)
    except KeyError:
        pass

    if timestamp is not None:
        ts_string = "{0: <16}".format(format_time(time.time() - timestamp))
    else:
        ts_string = ""
    return "[Memory]{0}: Loading {1}".format(ts_string, str(signature))
