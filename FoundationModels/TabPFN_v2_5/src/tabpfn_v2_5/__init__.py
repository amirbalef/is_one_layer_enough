from importlib.metadata import version

from tabpfn_v2_5.classifier import TabPFNClassifier
from tabpfn_v2_5.misc.debug_versions import display_debug_info
from tabpfn_v2_5.model_loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)
from tabpfn_v2_5.regressor import TabPFNRegressor

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "__version__",
    "display_debug_info",
    "load_fitted_tabpfn_model",
    "save_fitted_tabpfn_model",
]
