"""
main.py
-----------------
Entry point for running tabular ML probing experiments across OpenML benchmarks.

Typical usage
-------------
# Run one task with a specific config:
python main.py --task_id 363619 --config tabpfn_v2.config_c0

# Run all tasks in a benchmark with all default models:
python main.py --benchmark TabArena --task_id all --config config_c0
"""

import argparse
import os
import pickle
import warnings
from contextlib import nullcontext

import openml
import torch
from torch.profiler import ProfilerActivity

from util import (
    # seeds & data
    set_seeds,
    load_split_data,
    resolve_task_type,
    apply_label_shuffle,
    apply_label_flip,
    # eval runners
    EvalContext,
    run_standard_eval,
    run_holdout_eval,
    run_transductive_eval,
    # scoring & embeddings
    get_scores,
    grouped_embedding_distances,
    layerwise_probing_embeddings,
    LAYER_KEY_PREFIX,
    ENCODER_LAYER_KEY,
    # experiment setup
    get_classifier_config_list,
    determine_repeats_and_folds,
    Preprocess,
)

warnings.warn = lambda *a, **kw: None
warnings.filterwarnings("ignore")



#--- Global variables 

ENABLE_PROFILER = True

BENCHMARK_TASK_IDS: dict[str, list[int]] = {
    "TabArena": [
        363621, 363671, 363696, 363629, 363626, 363682, 363684,
        363700, 363674, 363694, 363619, 363623, 363689, 363706, 363624,
    ],
    "TabArena_multiclass": [
        363614, 363685, 363702 ,363704, 363707, 363711, #363699, 363677,
    ],
    "PMLBmini": [
        13, 27, 39, 42, 52, 54, 55, 57, 3495, 3496, 3503, 3538, 3539,
        3540, 3542, 3543, 3550, 3552, 3554, 3555, 3556, 3558, 3562, 3565,
        3568, 3570, 3722, 3819, 146188, 146196, 146208, 146210, 146236, 146240,
    ],
    "OpenMLCC18": [
        11, 14, 16, 18, 22, 23, 28, 45, 53, 2074, 2079, 3549, 
        3560, 9960, 9985, 14969, 146800, 146817, 146821, 146822
        ],
}

DEFAULT_MODELS = ["tabpfn_v1", "tabicl", "tabpfn_v2", "tabpfn_v2_5", "limix_2m", "limix_16m"]


# --- Layer naming -------------

# Layer parameter sets differ by model architecture
_TABICL_LAYER_PARAMS = [
    ENCODER_LAYER_KEY,
    {"layers_type": "predictor"},
    {"layers_type": "row"},
    {"layers_type": "col"},
]
_DEFAULT_LAYER_PARAMS = [ENCODER_LAYER_KEY, {}]


def _layer_params_for_model(model_name: str) -> list:
    """Return the parameter sets used to iterate over layers for *model_name*."""
    return _TABICL_LAYER_PARAMS if model_name == "tabicl" else _DEFAULT_LAYER_PARAMS


def _layer_key(param, layer_idx: int) -> str:
    """Build a canonical layer key string from a param dict and layer index."""
    if param == ENCODER_LAYER_KEY:
        return f"{LAYER_KEY_PREFIX}{layer_idx}_{ENCODER_LAYER_KEY}"
    suffix = f"_{param['layers_type']}" if "layers_type" in param else ""
    return f"{LAYER_KEY_PREFIX}{layer_idx}{suffix}"


# ---- Result collection-------------------------------------------

def collect_layer_results(
    clf,
    config: dict,
    results: dict,
    layer_store: dict,
    y_train,
    y_test,
    y_train_part1=None,
    y_train_part2=None,
) -> EvalContext:
    """
    Extract per-layer embeddings and predictions from *clf* and populate
    *results* and *layer_store*.

    Parameters
    ----------
    clf : fitted classifier
    config : run configuration dict
    results : top-level results dict (mutated in-place)
    layer_store : per-layer embedding store (mutated in-place)
    y_train_part1, y_train_part2 : only provided for holdout eval mode

    Returns
    -------
    EvalContext
        Captures the eval mode and split sizes needed by subsequent collect_*
        functions.
    """
    ctx = EvalContext.from_config(config, y_train, y_train_part1, y_train_part2)

    for param in _layer_params_for_model(config["model"]):
        if param == ENCODER_LAYER_KEY:
            embeddings_list         = clf.get_feature_encoder_embeddings()
            preds_list, probas_list = clf.get_feature_encoder_predictions()
        else:
            embeddings_list         = clf.get_all_layers_embeddings(**param)
            preds_list, probas_list = clf.get_all_layers_predictions(**param)

        for layer_idx, raw_embeddings in enumerate(embeddings_list):
            layer_key = _layer_key(param, layer_idx)
            train_pred, test_pred, train_proba, test_proba = ctx.slice_predictions(
                preds_list[layer_idx], probas_list[layer_idx]
            )
            emb = ctx.slice_embeddings(raw_embeddings)

            expected = len(ctx.y_train_effective) + len(y_test)
            if len(train_pred) + len(test_pred) != expected or isinstance(emb, str):
                print(
                    f"  Skipping {layer_key}: expected {expected} predictions, "
                    f"got {len(train_pred)} + {len(test_pred)}"
                )
                continue

            layer_store[layer_key] = {
                "embeddings":         emb,
                "y_pred":             test_pred,
                "y_pred_proba":       test_proba,
                "y_pred_train":       train_pred,
                "y_pred_proba_train": train_proba,
            }
            results[layer_key] = {
                "test_scores":  get_scores(
                    results["task_type"], y_test, test_pred, test_proba,
                    kl_divergence_wrt=results["y_pred_proba"],
                ),
                "train_scores": get_scores(
                    results["task_type"], ctx.y_train_effective, train_pred, train_proba
                ),
                "y_pred_proba": test_proba,
            }

    return ctx


def collect_embedding_distances(
    clf,
    config: dict,
    results: dict,
    ctx: EvalContext,
    y_test,
) -> None:
    """
    Compute within/between-class embedding distances for every layer and attach
    them to *results[layer_key]["embedding_distances"]*.

    When ``config["separate_y_embeddings"]`` is set, also compute distances
    for feature tokens and label tokens separately.
    """
    all_layer_keys: list[str] = []
    all_embeddings: list      = []

    for param in _layer_params_for_model(config["model"]):
        if param == ENCODER_LAYER_KEY:
            embs = clf.get_feature_encoder_embeddings()
        else:
            embs = clf.get_all_layers_embeddings(**param)
        all_layer_keys += [_layer_key(param, i) for i in range(len(embs))]
        all_embeddings += embs

    def _compute_and_attach(distances, key):
        for layer_key, dist in zip(all_layer_keys, distances):
            results[layer_key][key] = dist

    _compute_and_attach(
        grouped_embedding_distances(
            all_embeddings, ctx.y_train_effective, y_test,
            separate_y_embeddings=False, pca=True, n_components=0.95,
        ),
        "embedding_distances",
    )

    if config.get("separate_y_embeddings"):
        _compute_and_attach(
            grouped_embedding_distances(
                all_embeddings, ctx.y_train_effective, y_test,
                separate_y_embeddings=True, label_distances=False, pca=True, n_components=0.95,
            ),
            "feature_distances",
        )
        _compute_and_attach(
            grouped_embedding_distances(
                all_embeddings, ctx.y_train_effective, y_test,
                separate_y_embeddings=True, label_distances=True, pca=True, n_components=0.95,
            ),
            "label_distances",
        )


def collect_finetuned_decoder_results(
    clf,
    config: dict,
    results: dict,
    ctx: EvalContext,
    y_test,
) -> None:
    """
    Evaluate per-layer predictions produced by a fine-tuned decoder and store
    them under ``results[layer_key]["finetuned_decoder"]``.
    """
    all_preds, all_probas = clf.get_all_layers_predictions(decoder_type="finetuned")

    for layer_idx, (pred, proba) in enumerate(zip(all_preds, all_probas)):
        suffix    = "_predictor" if config["model"] == "tabicl" else ""
        layer_key = f"{LAYER_KEY_PREFIX}{layer_idx}{suffix}"
        train_pred, test_pred, train_proba, test_proba = ctx.slice_predictions(pred, proba)

        results[layer_key]["finetuned_decoder"] = {
            "test_scores":  get_scores(
                results["task_type"], y_test, test_pred, test_proba,
                kl_divergence_wrt=results["y_pred_proba"],
            ),
            "train_scores": get_scores(
                results["task_type"], ctx.y_train_effective, train_pred, train_proba
            ),
            "y_pred_proba": test_proba,
        }


def collect_layerwise_probing(
    clf,
    config: dict,
    results: dict,
    layer_store: dict,
    ctx: EvalContext,
    y_test,
    task_type: str,
) -> None:
    """
    Train a probing classifier on each layer's embeddings and cross-evaluate it
    on every other layer.  Results are stored in ``results["layerwise_probing"]``.
    """
    probing_models = config.get(
        "layerwise_probing_models",
        ["model_decoder", "logistic_regression", "KNN", "LDA"],
    )
    results["layerwise_probing"] = {}

    for model_name in probing_models:
        if model_name == "model_decoder":
            kwargs = dict(model_decoder=clf.get_decoder(), flatten_tabpfn_v2=False)
        else:
            kwargs = dict(flatten_tabpfn_v2=config.get("flatten_tabpfn_v2", True))

        results["layerwise_probing"][model_name] = layerwise_probing_embeddings(
            model_name, layer_store, ctx.y_train_effective, y_test, task_type, **kwargs,
        )


# -- Profiler to track GPU/CPU/Mem usage --------------------------------------------------------------------------
# Todo: tracking memory usage may lead to OOM

def _build_profiler_context():
    """Return a torch profiler context if ENABLE_PROFILER is set, else a no-op."""
    if not ENABLE_PROFILER:
        return nullcontext()
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats()
    return torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    )


def _extract_profile_stats(prof) -> dict:
    events = prof.key_averages()
    stats = {
        "cpu_s":   sum(e.cpu_time_total for e in events) / 1_000.0,
        "cpu_mem": sum(e.cpu_memory_usage for e in events) / (1024 ** 2),
    }
    if torch.cuda.is_available():
        stats["gpu_s"]   = sum(e.cuda_time for e in events) / 1_000.0
        stats["gpu_mem"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return stats


# -- Core experiment func --------------------------------------------------------------------------

def run_single_fold(
    clf,
    config: dict,
    task,
    dataset,
    repeat: int,
    fold: int,
    output_file: str,
    output_extras_dir: str,
    benchmark: str,
) -> None:
    """
    Run one cross-validation fold end-to-end and persist results to disk.

    Writes *output_file* (main results) and, if *output_extras_dir* is set,
    a companion ``*_extras.pkl`` file containing raw per-layer embeddings.
    """
    task_type = resolve_task_type(task)
    X_train, y_train, X_test, y_test = load_split_data(dataset, task, fold, repeat)

    if config.get("shuffle_train_labels"):
        y_train = apply_label_shuffle(y_train, seed=repeat)
    if config.get("flip_train_labels"):
        y_train, y_test = apply_label_flip(y_train, y_test)
    if config.get("preprocess"):
        pre = Preprocess(task_type)
        X_train, y_train = pre.fit_transform(X_train, y_train)
        X_test,  y_test  = pre.transform(X_test, y_test)

    results = {
        "task_id":    task.id,
        "repeat":     repeat,
        "fold":       fold,
        "task_type":  task_type,
        "n_classes":  1 if task.task_type == "Supervised Regression" else len(task.class_labels),
        "config":     config,
        "n_train":    len(y_train),
        "n_test":     len(y_test),
        "n_features": X_train.shape[1],
        "y_test":     y_test,
    }
    # layer_store holds raw per-layer embeddings; kept separate to avoid
    # bloating the main results pickle
    layer_store = {k: results[k] for k in ("task_id", "repeat", "fold", "config")}

    y_train_part1 = y_train_part2 = None

    with _build_profiler_context() as prof:
        # ---- Fit and predict ----
        if config.get("half_eval"):
            eval_out      = run_holdout_eval(clf, X_train, y_train, X_test, y_test, task_type, seed=repeat)
            y_train_part1 = eval_out.pop("_y_train_part1")
            y_train_part2 = eval_out.pop("_y_train_part2")
        elif config.get("full_eval"):
            eval_out = run_transductive_eval(clf, X_train, y_train, X_test, y_test, task_type)
        else:
            eval_out = run_standard_eval(clf, X_train, y_train, X_test, y_test, task_type)
        results.update(eval_out)

        results["contribution_scores"] = clf.get_all_contribution_scores()

        # ---- Per-layer embeddings & predictions ----
        ctx = collect_layer_results(
            clf, config, results, layer_store,
            y_train, y_test, y_train_part1, y_train_part2,
        )

        if config.get("embedding_distances"):
            collect_embedding_distances(clf, config, results, ctx, y_test)

        if "finetuned_decoders_path" in config:
            collect_finetuned_decoder_results(clf, config, results, ctx, y_test)

        if config.get("layerwise_probing"):
            collect_layerwise_probing(clf, config, results, layer_store, ctx, y_test, task_type)

    if ENABLE_PROFILER and prof is not None:
        results["profile"] = _extract_profile_stats(prof)

    # ---- Persist ----
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    if output_extras_dir:
        extras_dir = os.path.join(output_extras_dir, benchmark, str(task.id), f"{repeat}_{fold}")
        os.makedirs(extras_dir, exist_ok=True)
        extras_file = os.path.join(extras_dir, config["model_name"] + "_extras.pkl")
        with open(extras_file, "wb") as f:
            pickle.dump(layer_store, f)


def run_experiment(
    benchmark: str,
    task_id: str,
    config_path: str,
    output_root_dir: str,
    output_extras_dir: str,
    lite_evaluation: bool,
    rerun: bool,
) -> None:
    """
    Run experiments for one or all tasks in a benchmark.

    Parameters
    ----------
    task_id : str
        A numeric OpenML task ID, or ``"all"`` to run every task in the benchmark.
    config_path : str
        Dotted path to the config module, e.g. ``"tabpfn_v2.config_c0"``.
        The ``"configs."`` prefix is prepended automatically.
    rerun : bool
        If ``False``, skip any fold whose output file already exists.
    """
    config_path = "configs." + config_path
    task_ids    = BENCHMARK_TASK_IDS[benchmark] if task_id == "all" else [int(task_id)]
    configs, clf_classes = get_classifier_config_list(config_path)

    for tid in task_ids:
        task      = openml.tasks.get_task(tid)
        dataset   = task.get_dataset()
        n_repeats, n_folds = determine_repeats_and_folds(task, dataset, lite_evaluation)

        print(
            f"\nTask {task.id} | Dataset {dataset.id} ({dataset.name}) | "
            f"Config {config_path} | {n_repeats} repeat(s) × {n_folds} fold(s)"
        )

        for repeat in range(n_repeats):
            set_seeds(repeat)
            for fold in range(n_folds):
                fold_dir = os.path.join(output_root_dir, benchmark, str(tid), f"{repeat}_{fold}")
                os.makedirs(fold_dir, exist_ok=True)

                for config, clf_class in zip(configs, clf_classes):
                    output_file = os.path.join(fold_dir, config["model_name"] + ".pkl")
                    if os.path.exists(output_file) and not rerun:
                        print(f"  Skipping {config['model_name']} (output exists)")
                        continue

                    print(f"  repeat {repeat} | fold {fold} | {config['model_name']}")
                    clf = clf_class(**config["model_parameters"], random_state=repeat)
                    run_single_fold(
                        clf, config, task, dataset,
                        repeat, fold,
                        output_file, output_extras_dir, benchmark,
                    )


# -- CLI --------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tabular ML probing experiments on OpenML benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark",         type=str,  default="TabArena",
                        choices=list(BENCHMARK_TASK_IDS),
                        help="Which benchmark suite to use.")
    parser.add_argument("--task_id",           type=str,  default="363619",
                        help="OpenML task ID, or 'all' to run the full benchmark.")
    parser.add_argument("--config",            type=str,  default="tabpfn_v1.config_c0",
                        help="Config module path (e.g. 'tabpfn_v2.config_c0').")
    parser.add_argument("--output_root_dir",   type=str,  default="./results",
                        help="Root directory for main result pickles.")
    parser.add_argument("--output_extras_dir", type=str,  default="",
                        help="Directory for per-layer embedding pickles. Empty = disabled.")
    parser.add_argument("--lite_evaluation",   action="store_true",
                        help="Use a single fold/repeat (fast debugging mode).")
    parser.add_argument("--rerun",             action="store_true",
                        help="Re-run even if the output file already exists.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if len(args.config.split(".")) == 1:
        # No model prefix given — run every default model with this config
        for model in DEFAULT_MODELS:
            run_experiment(
                args.benchmark, args.task_id, f"{model}.{args.config}",
                args.output_root_dir, args.output_extras_dir,
                args.lite_evaluation, args.rerun,
            )
    else:
        run_experiment(
            args.benchmark, args.task_id, args.config,
            args.output_root_dir, args.output_extras_dir,
            args.lite_evaluation, args.rerun,
        )


if __name__ == "__main__":
    main()