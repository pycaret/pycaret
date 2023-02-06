def version():
    from pycaret import version_

    return version_


def nightly_version():
    from pycaret import nightly_version_

    return nightly_version_


# Hack to lazy import __version__ from `pycaret`.
# Needed to avoid a circular dependency.
def __getattr__(name):
    if name in ("__version__", "version_"):
        return version()
    if name in ("nightly_version_"):
        return nightly_version()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
