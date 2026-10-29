from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sigmma")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "unknown"
