"""
util.py
-------
Shared utilities for tabular ML probing experiments.
"""

import copy
import random
import typing
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from importlib import import_module
from numba import njit, prange
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


# --- Seeds ---------------------------------------------------------------
def set_seeds(seed: int) -> None:
    """Fix all random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Evaluation Helper ---------------------------------------------------------------

@dataclass
class EvalContext:
    """
    Captures which evaluation strategy was used and the resulting split sizes.

    Attributes
    ----------
    mode : str
        One of ``"standard"``, ``"holdout"``, or ``"transductive"``.
    n_train : int
        Total training-set size (before any internal split).
    n_part1 : int
        Size of the first half when ``mode="holdout"``; 0 otherwise.
    n_part2 : int
        Size of the second (held-out) half when ``mode="holdout"``; 0 otherwise.
    y_train_effective : array-like
        The label array that should be used for scoring train predictions:
        ``y_train_part2`` for holdout mode, ``y_train`` otherwise.
    """
    mode:              str
    n_train:           int
    n_part1:           int
    n_part2:           int
    y_train_effective: object  # array-like; typed as object to avoid importing pandas here

    @classmethod
    def from_config(
        cls,
        config: dict,
        y_train,
        y_train_part1=None,
        y_train_part2=None,
    ) -> "EvalContext":
        """Build an EvalContext from a run config and the actual label arrays."""
        if config.get("half_eval"):
            return cls(
                mode="holdout",
                n_train=len(y_train),
                n_part1=len(y_train_part1) if y_train_part1 is not None else 0,
                n_part2=len(y_train_part2) if y_train_part2 is not None else 0,
                y_train_effective=y_train_part2,
            )
        if config.get("full_eval"):
            return cls(
                mode="transductive",
                n_train=len(y_train),
                n_part1=0,
                n_part2=0,
                y_train_effective=y_train,
            )
        return cls(
            mode="standard",
            n_train=len(y_train),
            n_part1=0,
            n_part2=0,
            y_train_effective=y_train,
        )

    def slice_predictions(self, preds, probas):
        """
        Split a flat prediction array into (train_pred, test_pred, train_proba, test_proba).

        The array layout differs by mode:

        - ``standard``     → [train | test]
        - ``transductive`` → [train | test]
        - ``holdout``      → [part1 | part2 | test]  (part1 is discarded)
        """
        if self.mode == "holdout":
            s, e = self.n_part1, self.n_part1 + self.n_part2
            return preds[s:e], preds[e:], probas[s:e], probas[e:]
        n = self.n_train
        return preds[:n], preds[n:], probas[:n], probas[n:]

    def slice_embeddings(self, embeddings):
        """
        Return the embedding rows corresponding to the effective training split.

        - ``standard``     → all rows (train set only in the array)
        - ``transductive`` → rows after the first n_train (test portion)
        - ``holdout``      → rows after part1
        """
        if isinstance(embeddings, str):
            return embeddings  # sentinel for "not available"

        emb = embeddings.squeeze() if isinstance(embeddings, torch.Tensor) else embeddings
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()

        if self.mode == "transductive":
            return emb[self.n_train:]
        if self.mode == "holdout":
            return emb[self.n_part1:]
        return emb


def run_standard_eval(clf, X_train, y_train, X_test, y_test, task_type: str) -> dict:
    """
    Fit on the full training set, predict only on the test set.

    This is the default evaluation mode.
    """
    clf.fit(X_train, y_train)
    y_pred, y_proba = clf.predict_with_proba(X_test)
    return {
        "y_pred":       y_pred,
        "y_pred_proba": y_proba,
        "test_scores":  get_scores(task_type, y_test, y_pred, y_proba),
    }


def run_holdout_eval(clf, X_train, y_train, X_test, y_test, task_type: str, seed: int) -> dict:
    """
    Split the training set in half; fit on the first half, evaluate on the second.

    This avoids train-set leakage when measuring per-sample in-context confidence:
    the model never saw the "train" evaluation rows during fitting.

    The returned dict contains two internal keys (``_y_train_part1``,
    ``_y_train_part2``) that the caller should pop before persisting results —
    they are needed only to construct an :class:`EvalContext`.
    """
    from sklearn.model_selection import train_test_split
    X_part1, X_part2, y_part1, y_part2 = train_test_split(
        X_train, y_train, test_size=0.5, stratify=y_train, random_state=seed
    )
    clf.fit(X_part1, y_part1)
    y_pred, y_proba = clf.predict_with_proba(np.concatenate([X_part2, X_test]))
    n = len(y_part2)
    return {
        "y_pred_train":       y_pred[:n],
        "y_pred_proba_train": y_proba[:n],
        "train_scores":       get_scores(task_type, y_part2, y_pred[:n], y_proba[:n]),
        "y_pred":             y_pred[n:],
        "y_pred_proba":       y_proba[n:],
        "test_scores":        get_scores(task_type, y_test, y_pred[n:], y_proba[n:]),
        "_y_train_part1":     y_part1,   # popped by caller; needed for EvalContext
        "_y_train_part2":     y_part2,
    }


def run_transductive_eval(clf, X_train, y_train, X_test, y_test, task_type: str) -> dict:
    """
    Fit on the full training set; predict on *both* training and test rows together.

    This exposes how the model represents training points it has already seen,
    which is useful for studying in-context memorisation.
    """
    clf.fit(X_train, y_train)
    y_pred, y_proba = clf.predict_with_proba(np.concatenate([X_train, X_test]))
    n = len(y_train)
    return {
        "y_pred_train":       y_pred[:n],
        "y_pred_proba_train": y_proba[:n],
        "train_scores":       get_scores(task_type, y_train, y_pred[:n], y_proba[:n]),
        "y_pred":             y_pred[n:],
        "y_pred_proba":       y_proba[n:],
        "test_scores":        get_scores(task_type, y_test, y_pred[n:], y_proba[n:]),
    }


# --- Load Model and Configs ---------------------------------------------------------------

# Maps a model name to (module_path, class_name).
# Prefix-matched models (nanotabpfn, tabpfn_v2_hpo) are handled in
# _resolve_classifier_class below.
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "tabpfn_v1":           ("tabpfn_v1",   "TabPFNClassifier"),
    "tabpfn_v2":           ("tabpfn_v2",   "TabPFNClassifier"),
    "tabpfn_v2_5":         ("tabpfn_v2_5", "TabPFNClassifier"),
    "tabicl":              ("tabicl",      "TabICLClassifier"),
    "limix_2m":            ("limix",       "LimiXClassifier"),
    "limix_16m":           ("limix",       "LimiXClassifier"),
}


def _resolve_classifier_class(model_name: str):
    """Return the classifier class for *model_name*, using prefix fallbacks where needed."""
    if "nanotabpfn" in model_name:
        from nanotabpfn import NanoTabPFNClassifier
        return NanoTabPFNClassifier
    if "tabpfn_v2_hpo" in model_name:
        from tabpfn_v2 import TabPFNClassifier
        return TabPFNClassifier

    entry = _MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise ValueError(
            f"Unknown model {model_name!r}. "
            f"Available models: {sorted(_MODEL_REGISTRY)}"
        )
    module_name, class_name = entry
    return getattr(import_module(module_name), class_name)


def get_classifier_config_list(config_path: str) -> tuple[list, list]:
    """
    Load the config list from *config_path* and resolve each config's classifier class.

    Returns
    -------
    configs : list of dict
    classifier_classes : list of type
        Parallel list; ``classifier_classes[i]`` is the class for ``configs[i]``.
    """
    configs = import_module(config_path).generate_config_list()
    classifier_classes = [_resolve_classifier_class(cfg["model"]) for cfg in configs]
    return configs, classifier_classes


# --- Metrics  ---------------------------------------------------------------


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(((np.asarray(y_true) - np.asarray(y_pred)) ** 2).mean()))


def _prediction_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Mean Shannon entropy over a batch of probability distributions."""
    probs = np.clip(probs, eps, 1.0)
    return float(-np.sum(probs * np.log(probs), axis=-1).mean())


@njit(parallel=True)
def kl_divergence_numba(q: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """
    Mean per-sample KL divergence KL(q ‖ p), JIT-compiled with Numba.

    Parameters
    ----------
    q : ndarray of shape (n_samples, n_classes)
        The "query" distribution (typically a layer's predicted probabilities).
    p : ndarray of shape (n_samples, n_classes)
        The reference distribution (typically the final layer's predictions).
    """
    n_samples, n_classes = q.shape
    kl_total = 0.0
    for i in prange(n_samples):
        kl_i = 0.0
        for k in range(n_classes):
            qk = max(q[i, k], eps)
            pk = max(p[i, k], eps)
            kl_i += qk * (np.log(qk) - np.log(pk))
        kl_total += kl_i
    return kl_total / n_samples


def get_scores(
    task_type: str,
    y_true,
    y_pred=None,
    y_pred_proba=None,
    kl_divergence_wrt=None,
) -> dict:
    """
    Compute task-appropriate evaluation metrics.

    Parameters
    ----------
    task_type : str
        One of ``"binary"``, ``"multiclass"``, ``"regression"``.
    kl_divergence_wrt : ndarray, optional
        If provided (binary tasks only), compute KL divergence of *y_pred_proba*
        with respect to this reference distribution (e.g. the final layer's probas).
    """
    scores: dict = {}

    if task_type == "binary":
        scores["roc_auc"]            = roc_auc_score(y_true, y_pred_proba[:, 1])
        scores["balanced_accuracy"]  = balanced_accuracy_score(y_true, y_pred)
        scores["prediction_entropy"] = _prediction_entropy(y_pred_proba)
        if kl_divergence_wrt is not None:
            scores["kl_divergence"] = kl_divergence_numba(y_pred_proba, kl_divergence_wrt)

    elif task_type == "multiclass":
        scores["log_loss"]          = log_loss(y_true, y_pred_proba)
        scores["roc_auc"]          = roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro")
        scores["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    elif task_type == "regression":
        scores["rmse"] = _rmse(y_true, y_pred)

    return scores


# --- NaN-aware normalisation  ----------------------
#    Modified from https://github.com/valthom/retrieve_ft/blob/main/pfn.py

def maskmean(
    x: typing.Union[np.ndarray, torch.Tensor],
    mask: typing.Union[np.ndarray, torch.Tensor],
    dim: int,
) -> torch.Tensor:
    """Mean of *x* along *dim*, ignoring positions where *mask* is False."""
    x = torch.where(mask, x, torch.zeros_like(x))
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def maskstd(
    x: typing.Union[np.ndarray, torch.Tensor],
    mask: typing.Union[np.ndarray, torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """Std of *x* along *dim*, ignoring positions where *mask* is False."""
    num   = mask.sum(dim=dim, keepdim=True)
    mean  = maskmean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, torch.zeros_like(x))
    return ((diffs ** 2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5


def normalize_data(
    X: typing.Union[np.ndarray, torch.Tensor],
    X_train: typing.Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """Standardise *X* using the mean and std computed from *X_train* (NaN-aware)."""
    mask = ~torch.isnan(X_train)
    mean = maskmean(X_train, mask, dim=0)
    std  = maskstd(X_train, mask, dim=0) + 1e-6
    return torch.clip((X - mean) / std, min=-100, max=100)


# --- Preprocessing  ----------------------


class Preprocess:
    """
    Minimal preprocessing pipeline for tabular data.

    - Categorical columns: ordinal-encoded, with ``"MissingValue"`` imputation.
    - Continuous columns: mean imputation.
    - Target: ``LabelEncoder`` for classification, ``StandardScaler`` for regression.

    Usage
    -----
    >>> pre = Preprocess("binary")
    >>> X_train, y_train = pre.fit_transform(X_train_raw, y_train_raw)
    >>> X_test,  y_test  = pre.transform(X_test_raw, y_test_raw)
    """

    def __init__(self, task_type: str, categorical_indicator: list | None = None):
        self.task_type             = task_type
        self.is_classification     = task_type in ("multiclass", "binary")
        self.categorical_indicator = categorical_indicator  # None → auto-detect

        self.cat_columns: list = []
        self.con_columns: list = []
        self.cat_idxs:    list = []
        self.con_idxs:    list = []
        self.X_encoders:  list = []
        self.y_encoder         = None
        self.num_classes: int  = 0

    # ---- private helpers ----

    @staticmethod
    def _detect_categoricals(df: pd.DataFrame) -> list[bool]:
        return [
            isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype in (object, str)
            for c in df.columns
        ]

    def _resolve_column_splits(self, X: pd.DataFrame) -> None:
        indicator = (
            self.categorical_indicator
            if self.categorical_indicator is not None
            else self._detect_categoricals(X)
        )
        cat_mask         = np.array(indicator, dtype=bool)
        self.cat_columns = X.columns[cat_mask].tolist()
        self.con_columns = X.columns[~cat_mask].tolist()
        self.cat_idxs    = np.where(cat_mask)[0].tolist()
        self.con_idxs    = np.where(~cat_mask)[0].tolist()

    def _transform_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = X.copy()
        for i, col in enumerate(self.cat_columns):
            X[col] = X[col].astype("object").fillna("MissingValue")
            if fit:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X[col] = enc.fit_transform(X[[col]]).ravel()
                self.X_encoders.append(enc)
            else:
                X[col] = self.X_encoders[i].transform(X[[col]]).ravel()
        for col in self.con_columns:
            X[col] = X[col].fillna(X[col].mean())
        return X

    def _transform_labels(self, y, fit: bool):
        y = y.copy()
        if self.is_classification:
            if fit:
                self.y_encoder   = LabelEncoder()
                y                = self.y_encoder.fit_transform(y)
                self.num_classes = len(self.y_encoder.classes_)
            else:
                y = self.y_encoder.transform(y)
        else:
            y = np.asarray(y, dtype="float32").reshape(1, -1)
            if fit:
                self.y_encoder   = StandardScaler()
                y                = self.y_encoder.fit_transform(y).ravel()
                self.num_classes = 1
            else:
                y = self.y_encoder.transform(y).ravel()
        return y

    # ---- public API ----

    def fit_transform(self, X: pd.DataFrame, y):
        """Fit the pipeline on (X, y) and return the transformed copies."""
        self._resolve_column_splits(X)
        return self._transform_features(X, fit=True), self._transform_labels(y, fit=True)

    def transform(self, X: pd.DataFrame, y):
        """Apply the already-fitted pipeline to a new (X, y) pair."""
        return self._transform_features(X, fit=False), self._transform_labels(y, fit=False)


# ---  Embedding Utils & Similariries  ---------------------------------------------------------------

# Key string constants shared with run_experiment.py
LAYER_KEY_PREFIX  = "layer_"
ENCODER_LAYER_KEY = "encoder"


def embeddings_to_numpy(embeddings, flatten_tabpfn_v2: bool = True) -> np.ndarray:
    """
    Coerce any embedding format to a 2-D float32 numpy array.

    Parameters
    ----------
    flatten_tabpfn_v2 : bool
        TabPFN v2 produces 3-D tensors of shape ``(N, seq_len, hidden)``.
        When ``True``, flatten to ``(N, seq_len * hidden)``.
        When ``False``, take only the last token: ``(N, hidden)``.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = embeddings.squeeze()
    if embeddings.ndim == 3:
        embeddings = (
            embeddings.reshape(embeddings.shape[0], -1)
            if flatten_tabpfn_v2
            else embeddings[:, -1, :]
        )
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings after processing; got shape {embeddings.shape}")
    return embeddings.astype(np.float32)


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation."""
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def linear_cka(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Linear Centered Kernel Alignment (CKA) between representation matrices X and Y.

    Returns a scalar in [0, 1]; higher means more similar representations.
    Reference: Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    numerator   = np.sum((X.T @ Y) ** 2)
    denominator = (
        np.linalg.norm(X.T @ X, ord="fro")
        * np.linalg.norm(Y.T @ Y, ord="fro")
        + eps
    )
    return float(numerator / denominator)



# ---  Saparation Gap ---------------------------------------------------------------


def _to_int_labels(y) -> np.ndarray:
    """Convert a pandas Series or numpy array to integer class codes."""
    if hasattr(y, "to_numpy"):
        return y.astype("category").cat.codes.to_numpy()
    return np.asarray(y)


def _subsample(X: np.ndarray, max_samples: int) -> np.ndarray:
    if len(X) <= max_samples:
        return X
    return X[np.random.choice(len(X), max_samples, replace=False)]


def _pairwise_class_distances(
    embeddings: np.ndarray, y: np.ndarray, max_samples: int
) -> dict:
    """
    Compute within-class and between-class distances for the two-class case.

    Returns a dict keyed by metric name (``"euclidean"``, ``"cosine"``), each
    containing ``within_class``, ``within_class_0``, ``within_class_1``,
    and ``between_class`` scalars.
    """
    result = {}
    for metric in ("euclidean", "cosine"):
        X  = l2_normalize(embeddings) if metric == "cosine" else embeddings
        X0 = _subsample(X[y == 0], max_samples)
        X1 = _subsample(X[y == 1], max_samples)

        D0       = pairwise_distances(X0, metric=metric)
        D1       = pairwise_distances(X1, metric=metric)
        within_0 = D0[np.triu_indices_from(D0, k=1)].mean()
        within_1 = D1[np.triu_indices_from(D1, k=1)].mean()
        between  = pairwise_distances(X0, X1, metric=metric).mean()

        result[metric] = {
            "within_class":   float(0.5 * (within_0 + within_1)),
            "within_class_0": float(within_0),
            "within_class_1": float(within_1),
            "between_class":  float(between),
        }
    return result


def grouped_embedding_distances(
    all_layers_embeddings: list,
    y_train,
    y_test,
    max_samples: int = 100,
    pca: bool = True,
    n_components: float = 0.99,
    separate_y_embeddings: bool = False,
    label_distances: bool = False,
) -> list[dict]:
    """
    For each transformer layer, compute within- and between-class distances on
    the *support* set (training rows) and the *query* set (test rows).

    Parameters
    ----------
    all_layers_embeddings : list of array-like
        One entry per layer; each array has shape ``(n_train + n_test, ...)``.
    separate_y_embeddings : bool
        If ``True``, split the token dimension into feature tokens and label
        token before computing distances.
    label_distances : bool
        Only used when ``separate_y_embeddings=True``. If ``True``, use the
        label token; otherwise use the feature tokens.

    Returns
    -------
    list of dict
        One dict per layer; keys are ``"support"`` and ``"query"``, each
        containing the output of :func:`_pairwise_class_distances`.
    """
    y_train = _to_int_labels(y_train)
    y_test  = _to_int_labels(y_test)
    n_train = len(y_train)

    subsets = {
        "support": (slice(None, n_train), y_train),
        "query":   (slice(n_train, None), y_test),
    }

    # Materialise numpy arrays for every (subset, layer) pair up front
    layer_arrays: dict[str, list[np.ndarray]] = {s: [] for s in subsets}
    for embeddings in all_layers_embeddings:
        emb = embeddings.squeeze()
        if separate_y_embeddings:
            token_slice = -1 if label_distances else slice(None, -1)
            emb = emb[:, token_slice, :].reshape(len(emb), -1)
        else:
            emb = emb.reshape(len(emb), -1)
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        for subset, (slc, _) in subsets.items():
            layer_arrays[subset].append(emb[slc])

    # Fit a single PCA on a joint sample from all layers and subsets
    pca_model = None
    mu        = 0.0
    if pca:
        X_ref = np.concatenate(
            [arr for arrays in layer_arrays.values() for arr in arrays], axis=0
        )
        if len(X_ref) > 5_000:
            X_ref = X_ref[np.random.choice(len(X_ref), 5_000, replace=False)]
        mu        = X_ref.mean(axis=0, keepdims=True)
        pca_model = PCA(n_components=n_components, whiten=True).fit(X_ref - mu)
        print(f"PCA: retained {pca_model.n_components_} components")

    per_layer_results = []
    for layer_idx in range(len(all_layers_embeddings)):
        layer_res = {}
        for subset, (_, y) in subsets.items():
            emb = layer_arrays[subset][layer_idx]
            if pca_model is not None:
                emb = pca_model.transform(emb - mu)
            layer_res[subset] = _pairwise_class_distances(emb, y, max_samples)
        per_layer_results.append(layer_res)

    return per_layer_results



# ---  Probing classifiers ---------------------------------------------------------------

def _build_probing_model(model_name: str, y_train=None, model_decoder=None):
    """Instantiate a probing classifier by name."""
    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(class_weight="balanced", max_iter=1000)

    if model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)

    if model_name == "LDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis()

    if model_name == "model_decoder":
        if model_decoder is None:
            raise ValueError("model_decoder must be provided when model_name='model_decoder'")
        return _build_decoder_probing_model(model_decoder, y_train)

    raise ValueError(
        f"Unknown probing model {model_name!r}. "
        "Expected one of: 'logistic_regression', 'KNN', 'LDA', 'model_decoder'."
    )


def _build_decoder_probing_model(model_decoder, y_train):
    """
    Wrap a pre-trained decoder as a skorch ``NeuralNetClassifier`` for probing.

    The final linear layer is resized to match the number of classes in
    *y_train*, with weights copied from the original where possible.
    Class-frequency inverse weighting is applied to handle imbalanced labels.
    """
    import torch.nn as nn
    from skorch import NeuralNetClassifier

    y_enc     = LabelEncoder().fit_transform(y_train)
    n_classes = len(np.unique(y_enc))
    nn_model  = copy.deepcopy(model_decoder)

    # Resize the final linear projection to n_classes
    last_layer = nn_model[-1] if isinstance(nn_model, nn.Sequential) else nn_model.linear2
    new_layer  = torch.nn.Linear(last_layer.in_features, n_classes)
    with torch.no_grad():
        new_layer.weight[:n_classes] = last_layer.weight[:n_classes]
        new_layer.bias[:n_classes]   = last_layer.bias[:n_classes]
    if isinstance(nn_model, nn.Sequential):
        nn_model[-1]       = new_layer
    else:
        nn_model.linear2   = new_layer
    nn_model = nn_model.float()

    _, counts     = np.unique(y_enc, return_counts=True)
    inv_freq      = 1.0 / counts
    class_weights = torch.tensor(inv_freq / inv_freq.sum(), dtype=torch.float32)

    return NeuralNetClassifier(
        module=nn_model,
        criterion=torch.nn.CrossEntropyLoss,
        criterion__weight=class_weights,
        optimizer=torch.optim.Adam,
        max_epochs=30,
        lr=0.001,
        batch_size=32,
        iterator_train__shuffle=True,
        verbose=False,
    )


def _fit_probing_model(model, model_name: str, X_train: np.ndarray, y_train):
    """
    Fit *model* on (X_train, y_train).

    For LDA, falls back to the regularised ``lsqr`` solver if the standard
    fit raises an exception (e.g. singular covariance matrix).
    """
    try:
        model.fit(X_train, y_train)
    except Exception:
        if model_name == "LDA":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            model.fit(X_train, y_train)
        else:
            raise
    return model


def probing_embeddings(
    probing_model_name: str,
    embeddings: np.ndarray,
    y_train,
    y_test,
    task_type: str,
) -> dict:
    """
    Train a probing classifier on the train portion of *embeddings* and
    evaluate it on the test portion.

    Parameters
    ----------
    embeddings : array-like of shape ``(n_train + n_test, ...)``.
    """
    embeddings   = embeddings_to_numpy(embeddings, flatten_tabpfn_v2=True)
    n            = len(y_train)
    model        = _build_probing_model(probing_model_name)
    model        = _fit_probing_model(model, probing_model_name, embeddings[:n], y_train)
    y_pred       = model.predict(embeddings[n:])
    y_pred_proba = model.predict_proba(embeddings[n:])
    return get_scores(task_type, y_test, y_pred, y_pred_proba)


def layerwise_probing_embeddings(
    probing_model_name: str,
    layer_store: dict,
    y_train,
    y_test,
    task_type: str,
    model_decoder=None,
    flatten_tabpfn_v2: bool = True,
) -> dict:
    """
    For each layer, train a probing model and evaluate it on every other layer's
    embeddings (cross-layer generalisation).

    Parameters
    ----------
    layer_store : dict
        The per-layer embedding store produced by ``collect_layer_results``
        in ``run_experiment.py``. Keys matching ``LAYER_KEY_PREFIX`` are used.

    Returns
    -------
    dict
        ``scores[train_layer][eval_layer]`` contains ``{"train": ..., "test": ...}``
        plus optional ``"cosine_similarity"`` and ``"linear_cka"`` entries when
        *probing_model_name* is ``"model_decoder"``.
    """
    layer_keys = [k for k in layer_store if LAYER_KEY_PREFIX in k]
    scores: dict = {}

    for train_layer in layer_keys:
        train_emb = embeddings_to_numpy(
            layer_store[train_layer]["embeddings"], flatten_tabpfn_v2=flatten_tabpfn_v2
        )
        model = _build_probing_model(
            probing_model_name, y_train=y_train, model_decoder=model_decoder
        )
        model = _fit_probing_model(model, probing_model_name, train_emb[:len(y_train)], y_train)

        scores[train_layer] = {}
        for eval_layer in layer_keys:
            eval_emb = embeddings_to_numpy(
                layer_store[eval_layer]["embeddings"], flatten_tabpfn_v2=flatten_tabpfn_v2
            )
            entry: dict = {}
            for split_name, X_split, y_true in (
                ("train", eval_emb[:len(y_train)], y_train),
                ("test",  eval_emb[len(y_train):], y_test),
            ):
                y_pred       = model.predict(X_split)
                y_pred_proba = model.predict_proba(X_split)
                entry[split_name] = get_scores(task_type, y_true, y_pred, y_pred_proba)

            if probing_model_name == "model_decoder":
                sim                        = cosine_similarity(train_emb, eval_emb)
                entry["cosine_similarity"] = float(np.mean(np.abs(np.diag(sim))))
                entry["linear_cka"]        = linear_cka(train_emb, eval_emb)

            scores[train_layer][eval_layer] = entry

    return scores



# --- Data loading & label manipulation---------------------------------------------------------------

def load_split_data(
    dataset, task, fold: int, repeat: int
) -> tuple:
    """
    Fetch the dataset and return (X_train, y_train, X_test, y_test) DataFrames
    for the requested cross-validation fold and repeat.
    """
    X, y, _, _ = dataset.get_data(
        target=task.target_name, dataset_format="dataframe"
    )
    train_idx, test_idx = task.get_train_test_split_indices(fold=fold, repeat=repeat)
    return (
        X.iloc[train_idx], y.iloc[train_idx],
        X.iloc[test_idx],  y.iloc[test_idx],
    )


def resolve_task_type(task) -> str:
    """Return 'regression', 'binary', or 'multiclass' for an OpenML task."""
    if task.task_type == "Supervised Regression":
        return "regression"
    return "multiclass" if len(task.class_labels) > 2 else "binary"


def apply_label_shuffle(y_train, seed: int):
    """
    Randomly permute training labels in-place.
    Used as an ablation to verify that models are not trivially memorising labels.
    """
    rng = np.random.default_rng(seed)
    return y_train.iloc[rng.permutation(len(y_train))].reset_index(drop=True)


def apply_label_flip(y_train, y_test):
    """
    Cyclically shift every class label by one position (A→B, B→C, C→A, …).
    Used as an ablation to measure the impact of label identity vs. label structure.
    """
    unique_labels = y_train.unique()
    mapping = dict(zip(unique_labels, np.roll(unique_labels, shift=1)))
    return y_train.map(mapping), y_test.map(mapping)

def determine_repeats_and_folds(
    task, dataset, lite_evaluation: bool = True
) -> tuple[int, int]:
    """
    Return ``(n_repeats, n_folds)`` based on dataset size and evaluation mode.

    In lite mode (used for quick debugging) this always returns ``(1, 1)``.
    Otherwise, the number of repeats scales with dataset size:

    - < 2 500 samples   → 10 repeats
    - 2 500–250 000     → 3 repeats
    - > 250 000 samples → 1 repeat
    """
    if lite_evaluation:
        return 1, 1

    n_repeats, folds, _ = task.get_split_dimensions()
    n_samples           = dataset.qualities["NumberOfInstances"]

    if n_samples < 2_500:
        target_repeats = 10
    elif n_samples > 250_000:
        target_repeats = 1
    else:
        target_repeats = 3

    return min(target_repeats, n_repeats), folds