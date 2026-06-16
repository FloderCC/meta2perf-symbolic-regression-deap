"""
Independent single-configuration GP run for symbolic regression.

This script is intended for the stage after the two-round hyperparameter search.
It runs one manually specified configuration, with optional checkpoint resume.

Key properties:
    * The configuration is specified in one CONFIG dictionary.
    * It can start from iteration 0 when RESUME_FROM_CHECKPOINT=False.
    * It can continue from its own independent checkpoint when RESUME_FROM_CHECKPOINT=True.
    * Checkpoints, logs, Hall-of-Fame CSVs, and final-population CSVs are stored
      under independent folders, so they do not overwrite or reuse the artifacts
      created by the first- or second-round scripts.
    * The checkpoint hash intentionally excludes max_iterations, config_id, and
      LOG_METRICS_EVERY, so increasing CONFIG["max_iterations"] can continue from
      the previous independent checkpoint.
"""

import hashlib
import json
import logging
import math
import operator
import os
import pickle
import random
import time
import warnings
from collections import Counter
from pathlib import Path

import multiprocessing
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.gp_custom_operators import cxOnePoint_limited
from src.utils.gp_custom_operators import genHalfAndHalf_limited


# ============================================================
# INDEPENDENT SINGLE-CONFIGURATION RUN
# ============================================================

# Set to False to start this independent test from iteration 0.
# Set to True to continue from the independent checkpoint for this same CONFIG.
RESUME_FROM_CHECKPOINT = False
SAVE_CHECKPOINT_EVERY = 500

# Optional but useful: saving the full final population can take several minutes
# because each individual is reevaluated on train/test and written to CSV.
SAVE_HOF_CSV = True
SAVE_FINAL_POPULATION_CSV = False

# Manual configuration for this independent run.
CONFIG = {
    "MAX_NODES": 400,
    "MAX_HEIGHT": 30,
    "init_pop_method": "oblesa",
    "max_iterations": 5000,
    "LOG_METRICS_EVERY": 100,
    "hof_size": 50,
    "pop_size": 2000,
    "tourn_size": 150,
    "cxpb": 0.70,
    "mutpb": 0.45,
    "constants_range": (-50.0, 50.0),
    "ELITE_SIZE": 5,
    "random_seed": 42,
}

best_config_no = "1"

# Reproducibility controls.
BASE_RANDOM_SEED = CONFIG["random_seed"]
USE_CONFIG_SPECIFIC_SEED = False

# Independent artifact folders. These keep this test separated from the
# first-round and second-round hyperparameter-search outputs.
INDEPENDENT_RUN_NAME = "exp_perf_beam_grid_search_top10_s" + best_config_no
INDEPENDENT_RESULTS_DIR = Path("results") / INDEPENDENT_RUN_NAME
INDEPENDENT_CHECKPOINT_DIR = Path("checkpoints") / INDEPENDENT_RUN_NAME
INDEPENDENT_LOG_DIR = Path("logs") / INDEPENDENT_RUN_NAME

# Initial-population caching. This is also separated from the previous phases.
CACHE_INITIAL_POPULATIONS = True
INITIAL_POP_CACHE_DIR = Path("cache/initial_populations")
INITIAL_POP_CACHE_MODE = "per_pop_size"

# Output summary for this independent run.
SEARCH_SUMMARY_PATH = INDEPENDENT_RESULTS_DIR / "summary.csv"

# Early stopping disabled by default for the independent test/final run.
EARLY_STOP_ON_TEST_R2 = False
EARLY_STOP_TEST_R2_TARGET = 0.70
STOP_CURRENT_CONFIG_ON_TARGET = False

# ============================================================
# FIXED GP CONFIGURATION
# ============================================================

FIXED_BASE_CONFIG = {
    "MAX_NODES": CONFIG["MAX_NODES"],
    "MAX_HEIGHT": CONFIG["MAX_HEIGHT"],
    "init_pop_method": CONFIG["init_pop_method"],
    "max_iterations": CONFIG["max_iterations"],
    "LOG_METRICS_EVERY": CONFIG["LOG_METRICS_EVERY"],
    "hof_size": CONFIG["hof_size"],
}

# ============================================================
# ACTIVE CONFIGURATION GLOBALS
# ============================================================

ACTIVE_CONFIG_ID = None
ACTIVE_CONFIG_HASH = None
ACTIVE_CHECKPOINT_HASH = None
cfg = None

random_seed = BASE_RANDOM_SEED
MAX_NODES = FIXED_BASE_CONFIG["MAX_NODES"]
MAX_HEIGHT = FIXED_BASE_CONFIG["MAX_HEIGHT"]
init_pop_method = FIXED_BASE_CONFIG["init_pop_method"]
max_depth = MAX_HEIGHT
max_tokens = MAX_NODES

max_iterations = CONFIG["max_iterations"]
pop_size = CONFIG["pop_size"]
tourn_size = CONFIG["tourn_size"]
cxpb = CONFIG["cxpb"]
mutpb = CONFIG["mutpb"]
constants_range = CONFIG["constants_range"]
ELITE_SIZE = CONFIG["ELITE_SIZE"]
LOG_METRICS_EVERY = CONFIG["LOG_METRICS_EVERY"]
hof_size = CONFIG["hof_size"]

checkpoint_path = None

BAD_FITNESS = 1e30
verbose = True

binary_operator_names = ["add", "sub", "mul", "div"]
unary_operator_names = ["identity", "sq2", "sq3", "exp", "log", "sqrt"]


# ============================================================
# LOGGING
# ============================================================

class GreenStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(f"\033[92m{msg}\033[0m\n")
            self.flush()
        except Exception:
            self.handleError(record)


LOGGER = logging.getLogger("beam_grid_gp")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def console_filter(record):
    return not getattr(record, "file_only", False)


def configure_logging(config_id, config_hash, append=False):
    LOGGER.handlers.clear()

    INDEPENDENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = INDEPENDENT_LOG_DIR / f"exp_perf_beam_grid_search_{config_id}_{config_hash}.txt"

    file_handler = logging.FileHandler(log_path, mode="a" if append else "w")
    console_handler = GreenStreamHandler()
    console_handler.addFilter(console_filter)

    formatter = logging.Formatter(
        "%(asctime)s %(processName)s %(levelname)s: %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


# ============================================================
# HELPERS
# ============================================================

def normalize_value_for_json(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: normalize_value_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_value_for_json(v) for v in value]
    return value


def config_signature_dict(config):
    keys = [
        "MAX_NODES",
        "MAX_HEIGHT",
        "init_pop_method",
        "max_iterations",
        "pop_size",
        "tourn_size",
        "cxpb",
        "mutpb",
        "constants_range",
        "ELITE_SIZE",
        "hof_size",
    ]
    return {k: normalize_value_for_json(config[k]) for k in keys}


def config_hash(config):
    payload = json.dumps(config_signature_dict(config), sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def checkpoint_signature_dict(config):
    """
    Stable signature for checkpoint reuse.

    This intentionally excludes max_iterations, config_id, and LOG_METRICS_EVERY.
    Therefore, the same hyperparameter configuration can be resumed when
    SEARCH_MAX_ITERATIONS is increased from, for example, 100 to 500 or 5000
    generations, or when only the logging frequency changes.
    """
    keys = [
        "random_seed",
        "MAX_NODES",
        "MAX_HEIGHT",
        "init_pop_method",
        "pop_size",
        "tourn_size",
        "cxpb",
        "mutpb",
        "constants_range",
        "ELITE_SIZE",
        "hof_size",
    ]
    return {k: normalize_value_for_json(config[k]) for k in keys}


def checkpoint_hash(config):
    payload = json.dumps(checkpoint_signature_dict(config), sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]



def stable_int_hash(obj, modulo=10**9):
    payload = json.dumps(normalize_value_for_json(obj), sort_keys=True)
    return int(hashlib.md5(payload.encode("utf-8")).hexdigest(), 16) % modulo


def make_full_config(partial_cfg=None, max_iterations_override=None):
    """Return the full manual CONFIG, optionally overriding max_iterations."""
    full = dict(CONFIG)
    if partial_cfg:
        full.update(partial_cfg)
    if max_iterations_override is not None:
        full["max_iterations"] = max_iterations_override
    return full

def set_active_config(config, config_id):
    global ACTIVE_CONFIG_ID, ACTIVE_CONFIG_HASH, ACTIVE_CHECKPOINT_HASH, cfg
    global random_seed, MAX_NODES, MAX_HEIGHT, init_pop_method
    global max_depth, max_tokens, max_iterations, pop_size, tourn_size
    global cxpb, mutpb, constants_range, ELITE_SIZE, LOG_METRICS_EVERY
    global hof_size, checkpoint_path

    cfg = config
    ACTIVE_CONFIG_ID = str(config_id)
    ACTIVE_CONFIG_HASH = config_hash(config)
    ACTIVE_CHECKPOINT_HASH = checkpoint_hash(config)

    random_seed = int(config["random_seed"])
    MAX_NODES = int(config["MAX_NODES"])
    MAX_HEIGHT = int(config["MAX_HEIGHT"])
    init_pop_method = config["init_pop_method"]

    max_depth = MAX_HEIGHT
    max_tokens = MAX_NODES

    max_iterations = int(config["max_iterations"])
    pop_size = int(config["pop_size"])
    tourn_size = int(config["tourn_size"])
    cxpb = float(config["cxpb"])
    mutpb = float(config["mutpb"])
    constants_range = tuple(config["constants_range"])

    ELITE_SIZE = int(config["ELITE_SIZE"])
    LOG_METRICS_EVERY = int(config["LOG_METRICS_EVERY"])
    hof_size = int(config["hof_size"])

    INDEPENDENT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # Stable independent checkpoint path: independent of config_id and max_iterations.
    checkpoint_path = str(
        INDEPENDENT_CHECKPOINT_DIR / f"exp_perf_beam_grid_search_independent_ckpt_{ACTIVE_CHECKPOINT_HASH}.pkl"
    )


def train_test_split_regression(X, y, test_size=0.2, b="auto", random_state=42):
    bins = np.histogram_bin_edges(y, bins=b)[:-1]
    groups = np.digitize(y, bins)
    return train_test_split(
        X, y, test_size=test_size, stratify=groups, random_state=random_state
    )


def smape_score(true, pred):
    return np.mean(np.abs(pred - true) / ((np.abs(true) + np.abs(pred)) / 2))


def is_feasible(ind):
    return len(ind) <= MAX_NODES and ind.height <= MAX_HEIGHT


def is_valid_tree(ind):
    try:
        for i in range(len(ind)):
            _ = ind.searchSubtree(i)
        return True
    except IndexError:
        return False


def get_node_depth(individual, index):
    if index < 0 or index >= len(individual):
        raise IndexError(
            f"Index {index} out of range for individual of size {len(individual)}."
        )

    stack = [0]

    for i, node in enumerate(individual):
        depth = stack.pop()

        if i == index:
            return depth

        if node.arity > 0:
            stack.extend([depth + 1] * node.arity)

    raise IndexError(f"Could not compute depth for index {index}.")


def reset_rng_for_active_config():
    np.random.seed(random_seed)
    random.seed(random_seed)


def format_duration(seconds):
    seconds = int(max(0, round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


# ============================================================
# PRIMITIVES
# ============================================================

def div(a, b):
    return a / b


def identity(a):
    return a


def sqrt(a):
    return math.sqrt(a)


def log(a):
    return math.log(a)


def exp(a):
    return math.exp(a)


def sq2(a):
    return a * a


def sq3(a):
    return a * a * a


def rand_const():
    return random.uniform(*constants_range)


# ============================================================
# DATA
# ============================================================

np.random.seed(BASE_RANDOM_SEED)
random.seed(BASE_RANDOM_SEED)
warnings.filterwarnings("ignore")

df = pd.read_csv("meta_dataset/data.csv")
df = df.drop(columns=["Seed", "Dataset", "Sample Size", "Model"])
df = df[df["MCC"] > 0]

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split_regression(
    X.values, y.values, test_size=0.2, random_state=BASE_RANDOM_SEED
)

df_train = pd.DataFrame(X_train, columns=X.columns)
df_train["MCC"] = y_train

df_test = pd.DataFrame(X_test, columns=X.columns)
df_test["MCC"] = y_test

X_TRAIN = df_train.iloc[:, :-1].values
Y_TRAIN = df_train.iloc[:, -1].values

X_TEST = df_test.iloc[:, :-1].values
Y_TEST = df_test.iloc[:, -1].values


# ============================================================
# GP SETUP
# ============================================================

def ensure_deap_creators():
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


ensure_deap_creators()

pset = gp.PrimitiveSet("MAIN", X.shape[1])

for i in range(X.shape[1]):
    pset.renameArguments(**{f"ARG{i}": f"X{i}"})

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(identity, 1)
pset.addPrimitive(sq2, 1)
pset.addPrimitive(sq3, 1)
pset.addPrimitive(exp, 1)
pset.addPrimitive(log, 1)
pset.addPrimitive(sqrt, 1)
pset.addEphemeralConstant("rand", rand_const)

toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)


def huber_np(delta, r):
    abs_r = np.abs(r)
    quad = np.minimum(abs_r, delta)
    lin = abs_r - quad
    return 0.5 * quad**2 + delta * lin


def predict_individual_safe(ind, X_data):
    try:
        func_ind = toolbox.compile(ind)
        preds = np.array([func_ind(*x) for x in X_data], dtype=float)

        if not np.all(np.isfinite(preds)):
            return None

        return preds

    except Exception:
        return None


def eval_symbreg(ind):
    preds = predict_individual_safe(ind, X_TRAIN)

    if preds is None:
        return (BAD_FITNESS,)

    loss = np.mean(huber_np(1.0, preds - Y_TRAIN))
    return (loss,)


def selTournamentFeasible(individuals, k):
    feas = [i for i in individuals if is_feasible(i)]

    if not feas:
        feas = individuals

    return tools.selTournament(feas, k, tournsize=tourn_size)


toolbox.register("evaluate", eval_symbreg)
toolbox.register("select", selTournamentFeasible)

toolbox.register(
    "mate",
    cxOnePoint_limited,
    max_nodes=MAX_NODES,
    max_height=MAX_HEIGHT,
)


def mut_uniform_bounded(individual, pset, max_nodes, max_height):
    if len(individual) == 0:
        return (individual,)

    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)

    old_subtree = gp.PrimitiveTree(individual[slice_])
    old_size = len(old_subtree)

    insertion_depth = get_node_depth(individual, index)

    max_allowed_subtree_height = max_height - insertion_depth
    max_allowed_subtree_size = max_nodes - (len(individual) - old_size)

    if max_allowed_subtree_height < 0 or max_allowed_subtree_size < 1:
        return (individual,)

    try:
        new_subtree_expr = genHalfAndHalf_limited(
            pset=pset,
            min_=0,
            max_=max_allowed_subtree_height,
            max_len=max_allowed_subtree_size,
        )
    except Exception:
        return (individual,)

    new_subtree = gp.PrimitiveTree(new_subtree_expr)

    if (
        len(new_subtree) <= max_allowed_subtree_size
        and insertion_depth + new_subtree.height <= max_height
    ):
        individual[slice_] = new_subtree

    return (individual,)


toolbox.register(
    "mutate",
    mut_uniform_bounded,
    pset=pset,
    max_nodes=MAX_NODES,
    max_height=MAX_HEIGHT,
)


def refresh_limited_operators():
    """
    Re-registers limited crossover and mutation after changing MAX_NODES/MAX_HEIGHT.
    This is kept even though the current search fixes these values.
    """
    toolbox.unregister("mate")
    toolbox.register(
        "mate",
        cxOnePoint_limited,
        max_nodes=MAX_NODES,
        max_height=MAX_HEIGHT,
    )

    toolbox.unregister("mutate")
    toolbox.register(
        "mutate",
        mut_uniform_bounded,
        pset=pset,
        max_nodes=MAX_NODES,
        max_height=MAX_HEIGHT,
    )


# ============================================================
# INITIAL POPULATION CACHE
# ============================================================

def cache_key_for_initial_population(config, requested_pop_size):
    if INITIAL_POP_CACHE_MODE != "per_pop_size":
        raise ValueError(
            "The second-round refinement script supports only "
            "INITIAL_POP_CACHE_MODE='per_pop_size'."
        )

    pop_for_cache = requested_pop_size

    payload = {
        "mode": INITIAL_POP_CACHE_MODE,
        "pop_for_cache": pop_for_cache,
        "n_vars": df_train.shape[1] - 1,
        "const_min": config["constants_range"][0],
        "const_max": config["constants_range"][1],
        "method": config["init_pop_method"],
        "binary_operators": binary_operator_names,
        "unary_operators": unary_operator_names,
        "max_depth": config["MAX_HEIGHT"],
        "max_tokens": config["MAX_NODES"],
        "seed": BASE_RANDOM_SEED if not USE_CONFIG_SPECIFIC_SEED else config["random_seed"],
    }

    key = hashlib.md5(
        json.dumps(normalize_value_for_json(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    return key, pop_for_cache


def get_cached_seed_expr_strings(config, requested_pop_size):
    """
    Returns OBLESA seed expressions from a persistent disk cache.

    By default, the cache is exact with respect to pop_size. Therefore, a
    cached 3000-individual OBLESA pool is not reused for a 2000-individual
    configuration. If the requested pop_size is different, a different cache
    key is used and OBLESA is generated again.
    """
    from src.utils.ga_initialization import create_pop_for_deap

    INITIAL_POP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    key, pop_for_cache = cache_key_for_initial_population(config, requested_pop_size)
    cache_path = INITIAL_POP_CACHE_DIR / f"oblesa_seed_exprs_{key}.json"
    meta_path = INITIAL_POP_CACHE_DIR / f"oblesa_seed_exprs_{key}.meta.json"

    if CACHE_INITIAL_POPULATIONS and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            seed_expr_strings = json.load(f)

        if len(seed_expr_strings) == requested_pop_size:
            msg = (
                f"Reusing exact OBLESA initial population from disk: {cache_path} "
                f"({len(seed_expr_strings)} cached expressions)."
            )
            LOGGER.info(msg)
            print(msg, flush=True)
            return seed_expr_strings

        if INITIAL_POP_CACHE_MODE == "max_prefix" and len(seed_expr_strings) >= requested_pop_size:
            msg = (
                f"Reusing OBLESA initial population from disk using max_prefix mode: "
                f"{cache_path} ({len(seed_expr_strings)} cached expressions; "
                f"using first {requested_pop_size})."
            )
            LOGGER.info(msg)
            print(msg, flush=True)
            return seed_expr_strings[:requested_pop_size]

        LOGGER.warning(
            f"Cached OBLESA pool size ({len(seed_expr_strings)}) does not exactly "
            f"match requested pop_size={requested_pop_size}. Regenerating {pop_for_cache}."
        )

    # Important for reproducibility: population generation starts from the same
    # seed for the same cache key, independent of the order in which configs run.
    reset_rng_for_active_config()

    msg = (
        f"Generating OBLESA initial population once for cache: {cache_path} "
        f"(pool size={pop_for_cache}, requested={requested_pop_size}, "
        f"constants_range={tuple(config['constants_range'])})."
    )
    LOGGER.info(msg)
    print(msg, flush=True)

    def fitness_function_for_deap_str(individual_as_str):
        ensure_deap_creators()
        ind = creator.Individual(
            gp.PrimitiveTree.from_string(individual_as_str, pset)
        )
        return eval_symbreg(ind)[0]

    _, seed_expr_strings = create_pop_for_deap(
        pop_size=pop_for_cache,
        n_vars=df_train.shape[1] - 1,
        const_min=config["constants_range"][0],
        const_max=config["constants_range"][1],
        method=config["init_pop_method"],
        fitness_function_for_deap_str=fitness_function_for_deap_str,
        binary_operators=binary_operator_names,
        unary_operators=unary_operator_names,
        max_depth=config["MAX_HEIGHT"],
        max_tokens=config["MAX_NODES"],
    )

    if CACHE_INITIAL_POPULATIONS:
        # Write atomically to avoid leaving a corrupted cache file if the process
        # is interrupted during serialization.
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(seed_expr_strings, f)
        os.replace(tmp_path, cache_path)

        metadata = {
            "cache_key": key,
            "cache_mode": INITIAL_POP_CACHE_MODE,
            "cached_pool_size": len(seed_expr_strings),
            "requested_pop_size": requested_pop_size,
            "constants_range": list(config["constants_range"]),
            "seed": BASE_RANDOM_SEED if not USE_CONFIG_SPECIFIC_SEED else config["random_seed"],
            "MAX_NODES": config["MAX_NODES"],
            "MAX_HEIGHT": config["MAX_HEIGHT"],
            "init_pop_method": config["init_pop_method"],
        }
        tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
        with open(tmp_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        os.replace(tmp_meta_path, meta_path)

        msg = (
            f"Saved OBLESA initial population to disk cache: {cache_path} "
            f"({len(seed_expr_strings)} expressions)."
        )
        LOGGER.info(msg)
        print(msg, flush=True)

    return seed_expr_strings[:requested_pop_size]


def build_initial_population_from_cache(config):
    seed_expr_strings = get_cached_seed_expr_strings(config, config["pop_size"])

    LOGGER.info("Seeded initial population expressions ...")

    initial_population = []

    for expr_str in seed_expr_strings:
        try:
            ind = creator.Individual(gp.PrimitiveTree.from_string(expr_str, pset))

            if is_valid_tree(ind) and is_feasible(ind):
                initial_population.append(ind)
            else:
                LOGGER.warning(f"Discarding invalid or infeasible seed: {expr_str}")
                LOGGER.warning(
                    f"Seed is invalid: valid={is_valid_tree(ind)}, "
                    f"feasible={is_feasible(ind)}"
                )
                LOGGER.warning(f"len={len(ind)}, height={ind.height}")

        except Exception as e:
            LOGGER.warning(f"Could not parse seed '{expr_str}': {e}")

    if len(initial_population) < config["pop_size"]:
        raise RuntimeError(
            f"Initial population has only {len(initial_population)} valid individuals, "
            f"but pop_size={config['pop_size']} is required."
        )

    initial_population = initial_population[: config["pop_size"]]

    sizes = [len(ind) for ind in initial_population]
    heights = [ind.height for ind in initial_population]
    unique_count = len({str(ind) for ind in initial_population})
    total = len(initial_population)
    duplicates = total - unique_count

    LOGGER.info(
        f"Initial population: {total} individuals "
        f"({unique_count} unique, {duplicates} duplicates)"
    )
    LOGGER.info(
        f"Size - min: {min(sizes)}, max: {max(sizes)}, "
        f"avg: {sum(sizes) / len(sizes):.2f}"
    )
    LOGGER.info(
        f"Height - min: {min(heights)}, max: {max(heights)}, "
        f"avg: {sum(heights) / len(heights):.2f}"
    )

    return initial_population


# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(path, population, hof, generation):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "config_id": ACTIVE_CONFIG_ID,
        "config_hash": ACTIVE_CONFIG_HASH,
        "checkpoint_hash": ACTIVE_CHECKPOINT_HASH,
        "cfg": cfg,
        "generation": generation,
        "population": population,
        "hof": hof,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)

    LOGGER.info(f"Checkpoint saved at generation {generation}: {path}")


def load_checkpoint(path):
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    stored_checkpoint_hash = checkpoint.get("checkpoint_hash")
    if stored_checkpoint_hash != ACTIVE_CHECKPOINT_HASH:
        raise ValueError(
            f"Checkpoint checkpoint_hash={stored_checkpoint_hash} does not match "
            f"current checkpoint_hash={ACTIVE_CHECKPOINT_HASH}."
        )

    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])

    LOGGER.info(f"Checkpoint loaded from generation {checkpoint['generation']}: {path}")

    return (
        checkpoint["population"],
        checkpoint["hof"],
        checkpoint["generation"],
    )


def find_existing_checkpoint_for_active_config():
    """Return the stable checkpoint path for the active configuration, if it exists."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path
    return None


# ============================================================
# ASSESSMENT HELPERS
# ============================================================

def assess_individual(ind):
    ytr_ind = predict_individual_safe(ind, X_TRAIN)
    yte_ind = predict_individual_safe(ind, X_TEST)

    if ytr_ind is None or yte_ind is None:
        train_r2_ind = None
        test_r2_ind = None
        train_huber_loss_ind = None
    else:
        train_r2_ind = r2_score(Y_TRAIN, ytr_ind)
        test_r2_ind = r2_score(Y_TEST, yte_ind)
        train_huber_loss_ind = float(np.mean(huber_np(1.0, ytr_ind - Y_TRAIN)))

    num_nodes = len(ind)
    height = ind.height

    var_counts = Counter(
        node.name
        for node in ind
        if isinstance(node, gp.Terminal)
        and hasattr(node, "name")
        and node.name.startswith("ARG")
    )

    count_vars = ";".join(
        f"{count}"
        for _, count in sorted(var_counts.items(), key=lambda x: int(x[0][3:]))
    )

    num_logs = sum(
        1 for node in ind
        if isinstance(node, gp.Primitive) and node.name == "log"
    )
    num_exps = sum(
        1 for node in ind
        if isinstance(node, gp.Primitive) and node.name == "exp"
    )
    num_sqrts = sum(
        1 for node in ind
        if isinstance(node, gp.Primitive) and node.name == "sqrt"
    )
    num_sq2 = sum(
        1 for node in ind
        if isinstance(node, gp.Primitive) and node.name == "sq2"
    )
    num_sq3 = sum(
        1 for node in ind
        if isinstance(node, gp.Primitive) and node.name == "sq3"
    )

    return [
        train_r2_ind,
        test_r2_ind,
        train_huber_loss_ind,
        num_nodes,
        height,
        count_vars,
        num_logs,
        num_exps,
        num_sqrts,
        num_sq2,
        num_sq3,
    ]


def save_hof_and_population_csvs(hof, population):
    INDEPENDENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "train_r2",
        "test_r2",
        "train_huber_loss",
        "num_nodes",
        "height",
        "count_vars",
        "num_logs",
        "num_exps",
        "num_sqrts",
        "num_sq2",
        "num_sq3",
        "expression",
    ]

    hof_path = ""
    pop_path = ""
    hof_df = pd.DataFrame(columns=columns)
    pop_df = pd.DataFrame(columns=columns)

    if SAVE_HOF_CSV:
        hof_data = []
        for ind in hof:
            ind_metrics = assess_individual(ind)
            hof_data.append(ind_metrics + [str(ind)])

        hof_df = pd.DataFrame(hof_data, columns=columns)
        hof_df = hof_df.sort_values(by="test_r2", ascending=False, na_position="last")

        hof_path = str(
            INDEPENDENT_RESULTS_DIR /
            f"exp_perf_beam_grid_search_{ACTIVE_CONFIG_ID}_{ACTIVE_CONFIG_HASH}_hall_of_fame.csv"
        )
        hof_df.to_csv(hof_path, index=False)

    if SAVE_FINAL_POPULATION_CSV:
        pop_data = []
        for ind in population:
            ind_metrics = assess_individual(ind)
            pop_data.append(ind_metrics + [str(ind)])

        pop_df = pd.DataFrame(pop_data, columns=columns)
        pop_df = pop_df.sort_values(by="test_r2", ascending=False, na_position="last")

        pop_path = str(
            INDEPENDENT_RESULTS_DIR /
            f"exp_perf_beam_grid_search_{ACTIVE_CONFIG_ID}_{ACTIVE_CONFIG_HASH}_final_population.csv"
        )
        pop_df.to_csv(pop_path, index=False)

    return hof_path, pop_path, hof_df, pop_df


# ============================================================
# GP RUNNER FOR ONE HYPERPARAMETER CONFIGURATION
# ============================================================

def run_single_gp_config(config, config_id):
    set_active_config(config, config_id)
    configure_logging(ACTIVE_CONFIG_ID, ACTIVE_CONFIG_HASH, append=RESUME_FROM_CHECKPOINT)
    refresh_limited_operators()

    reset_rng_for_active_config()

    LOGGER.info(f"Running CONFIG_ID: {ACTIVE_CONFIG_ID}")
    LOGGER.info(f"CONFIG_HASH: {ACTIVE_CONFIG_HASH}")
    LOGGER.info(f"CHECKPOINT_HASH: {ACTIVE_CHECKPOINT_HASH}")
    LOGGER.info(f"CONFIG: {cfg}")
    LOGGER.info(f"RESUME_FROM_CHECKPOINT: {RESUME_FROM_CHECKPOINT}")

    start_generation = 0
    reached_target = False
    stop_reason = "max_iterations"

    if RESUME_FROM_CHECKPOINT:
        resume_checkpoint_path = find_existing_checkpoint_for_active_config()

        if resume_checkpoint_path is not None:
            LOGGER.info(f"Checkpoint found: {resume_checkpoint_path}")
            population, hof, start_generation = load_checkpoint(resume_checkpoint_path)

            if start_generation >= max_iterations:
                LOGGER.warning(
                    f"Checkpoint generation ({start_generation}) is already >= "
                    f"max_iterations ({max_iterations}). No additional evolution is needed; "
                    f"the stored population and Hall of Fame will be reused."
                )

            LOGGER.info(
                f"Resuming evolution from generation {start_generation}/{max_iterations}"
            )
        else:
            LOGGER.info(
                f"RESUME_FROM_CHECKPOINT=True, but no checkpoint was found at "
                f"the stable path: {checkpoint_path}. Starting a new run."
            )
            population = build_initial_population_from_cache(cfg)
            hof = tools.HallOfFame(hof_size)

    else:
        population = build_initial_population_from_cache(cfg)
        hof = tools.HallOfFame(hof_size)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    pbar = tqdm(
        total=max_iterations,
        initial=start_generation,
        desc=f"GP {ACTIVE_CONFIG_ID}",
    )

    def log_best_metrics(gen, best_ind):
        ytr = predict_individual_safe(best_ind, X_TRAIN)
        yte = predict_individual_safe(best_ind, X_TEST)

        if ytr is None or yte is None:
            LOGGER.info(
                f"Generation {gen + 1}/{max_iterations} - "
                f"Best individual could not be evaluated safely on train/test. "
                f"Stored fitness: {best_ind.fitness.values[0]:.6f}, "
                f"Size: {len(best_ind)} nodes, "
                f"Height: {best_ind.height}"
            )
            return None

        train_huber_loss = float(np.mean(huber_np(1.0, ytr - Y_TRAIN)))
        train_r2 = r2_score(Y_TRAIN, ytr)
        test_r2 = r2_score(Y_TEST, yte)

        LOGGER.info(
            f"Generation {gen + 1}/{max_iterations} - "
            f"Best Huber loss: {train_huber_loss:.6f}, "
            f"Train R^2: {train_r2:.4f}, "
            f"Test R^2: {test_r2:.4f}, "
            f"Size: {len(best_ind)} nodes, "
            f"Height: {best_ind.height}"
        )

        return {
            "train_huber_loss": train_huber_loss,
            "train_r2": train_r2,
            "test_r2": test_r2,
        }

    try:
        for gen in range(start_generation, max_iterations):
            invalid = [ind for ind in population if not ind.fitness.valid]
            fits = toolbox.map(toolbox.evaluate, invalid)

            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            feasible_population = [
                ind for ind in population
                if is_valid_tree(ind) and is_feasible(ind) and ind.fitness.valid
            ]

            if feasible_population:
                hof.update(feasible_population)

            elite = []
            if len(hof) > 0:
                elite = [toolbox.clone(ind) for ind in hof[:ELITE_SIZE]]

            n_offspring = pop_size - len(elite)

            offspring = toolbox.select(population, n_offspring)
            offspring = list(map(toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    try:
                        toolbox.mate(c1, c2)

                        if not is_feasible(c1) or not is_valid_tree(c1):
                            LOGGER.warning(
                                "Crossover produced an invalid or infeasible first child."
                            )

                        if not is_feasible(c2) or not is_valid_tree(c2):
                            LOGGER.warning(
                                "Crossover produced an invalid or infeasible second child."
                            )

                        if c1.fitness.valid:
                            del c1.fitness.values
                        if c2.fitness.valid:
                            del c2.fitness.values

                    except Exception as e:
                        LOGGER.warning(f"Crossover failed. Error: {e}")

            for m in offspring:
                if random.random() < mutpb:
                    try:
                        before_len = len(m)
                        before_height = m.height
                        before_valid = is_valid_tree(m)
                        before_feasible = is_feasible(m)

                        toolbox.mutate(m)

                        after_len = len(m)
                        after_height = m.height
                        after_valid = is_valid_tree(m)
                        after_feasible = is_feasible(m)

                        if not after_valid or not after_feasible:
                            LOGGER.warning(
                                "Mutation invalid/infeasible. "
                                f"Before: len={before_len}, height={before_height}, "
                                f"valid={before_valid}, feasible={before_feasible}. "
                                f"After: len={after_len}, height={after_height}, "
                                f"valid={after_valid}, feasible={after_feasible}."
                            )

                        if m.fitness.valid:
                            del m.fitness.values

                    except Exception as e:
                        LOGGER.warning(f"Mutation failed for individual {m}. Error: {e}")

            valid_offspring = [
                ind for ind in offspring
                if is_valid_tree(ind) and is_feasible(ind)
            ]

            while len(valid_offspring) < n_offspring:
                LOGGER.info(
                    f"Refilling population: {len(valid_offspring)}/{n_offspring} "
                    f"valid non-elite offspring. Generating random trees to fill the gap."
                )

                rand_expr = genHalfAndHalf_limited(
                    pset=pset,
                    min_=0,
                    max_=MAX_HEIGHT,
                    max_len=MAX_NODES,
                )
                ind = creator.Individual(rand_expr)

                if is_valid_tree(ind) and is_feasible(ind):
                    valid_offspring.append(ind)

            population[:] = valid_offspring[:n_offspring] + elite

            invalid = [ind for ind in population if not ind.fitness.valid]
            fits = toolbox.map(toolbox.evaluate, invalid)

            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            feasible_population = [
                ind for ind in population
                if is_valid_tree(ind) and is_feasible(ind) and ind.fitness.valid
            ]

            if feasible_population:
                hof.update(feasible_population)

            pbar.update(1)

            if hof:
                best_ind = hof[0]
                best_fit = best_ind.fitness.values[0]

                should_log_full = (
                    (gen + 1) % LOG_METRICS_EVERY == 0
                    or gen == start_generation
                    or gen == max_iterations - 1
                )

                if should_log_full:
                    metrics = log_best_metrics(gen, best_ind)

                    if (
                        STOP_CURRENT_CONFIG_ON_TARGET
                        and EARLY_STOP_ON_TEST_R2
                        and metrics is not None
                        and metrics["test_r2"] >= EARLY_STOP_TEST_R2_TARGET
                    ):
                        reached_target = True
                        stop_reason = "target_test_r2_reached"
                        LOGGER.info(
                            f"Stopping current configuration early because "
                            f"Test R^2={metrics['test_r2']:.4f} >= "
                            f"{EARLY_STOP_TEST_R2_TARGET:.4f}."
                        )
                        save_checkpoint(
                            checkpoint_path,
                            population,
                            hof,
                            generation=gen + 1,
                        )
                        break
                else:
                    LOGGER.info(
                        f"Generation {gen + 1}/{max_iterations} - "
                        f"Best Huber loss: {best_fit:.6f}"
                    )
            else:
                LOGGER.info(
                    f"Generation {gen + 1}/{max_iterations} - "
                    f"No feasible individuals found."
                )

            if (
                (gen + 1) % SAVE_CHECKPOINT_EVERY == 0
                or gen == max_iterations - 1
            ):
                save_checkpoint(
                    checkpoint_path,
                    population,
                    hof,
                    generation=gen + 1,
                )

    except KeyboardInterrupt:
        LOGGER.warning("KeyboardInterrupt detected. Saving checkpoint before exiting.")
        save_checkpoint(
            checkpoint_path,
            population,
            hof,
            generation=gen + 1,
        )
        raise

    finally:
        pbar.close()
        pool.close()
        pool.join()

    if len(hof) == 0:
        raise RuntimeError("Hall of Fame is empty. No feasible individual was found.")

    safe_hof_candidates = []

    for candidate in hof:
        candidate_ytr = predict_individual_safe(candidate, X_TRAIN)
        candidate_yte = predict_individual_safe(candidate, X_TEST)

        if candidate_ytr is not None and candidate_yte is not None:
            safe_hof_candidates.append((candidate, candidate_ytr, candidate_yte))
        else:
            LOGGER.warning(
                "A Hall-of-Fame individual failed during final safe evaluation "
                f"and will be ignored. Fitness={candidate.fitness.values[0]}, "
                f"size={len(candidate)}, height={candidate.height}"
            )

    if not safe_hof_candidates:
        LOGGER.warning(
            "No Hall-of-Fame individual could be evaluated safely on both train and test. "
            "This configuration will be marked as failed and the search will continue."
        )

        result = {
            "config_id": ACTIVE_CONFIG_ID,
            "config_hash": ACTIVE_CONFIG_HASH,
            "checkpoint_hash": ACTIVE_CHECKPOINT_HASH,
            "random_seed": random_seed,
            "MAX_NODES": MAX_NODES,
            "MAX_HEIGHT": MAX_HEIGHT,
            "init_pop_method": init_pop_method,
            "max_iterations": max_iterations,
            "pop_size": pop_size,
            "tourn_size": tourn_size,
            "cxpb": cxpb,
            "mutpb": mutpb,
            "constants_range_json": json.dumps(list(constants_range)),
            "ELITE_SIZE": ELITE_SIZE,
            "LOG_METRICS_EVERY": LOG_METRICS_EVERY,
            "hof_size": hof_size,
            "best_train_r2": np.nan,
            "best_test_r2": np.nan,
            "best_train_huber_loss": BAD_FITNESS,
            "best_train_adj_r2": np.nan,
            "best_test_adj_r2": np.nan,
            "best_train_smape": np.nan,
            "best_test_smape": np.nan,
            "best_train_mae": np.nan,
            "best_test_mae": np.nan,
            "best_num_nodes": np.nan,
            "best_height": np.nan,
            "best_expression": "",
            "hof_csv": "",
            "population_csv": "",
            "checkpoint_path": checkpoint_path,
            "reached_target": False,
            "stop_reason": "no_safe_hof_candidate",
        }

        return result

    best, ytr, yte = min(
        safe_hof_candidates,
        key=lambda item: item[0].fitness.values[0],
    )

    LOGGER.info(
        "Selected final best individual as the safest Hall-of-Fame candidate "
        "with the lowest train Huber loss."
    )

    train_r2 = r2_score(Y_TRAIN, ytr)
    train_mape = smape_score(Y_TRAIN, ytr)
    train_mae = mean_absolute_error(Y_TRAIN, ytr)
    n_train = len(Y_TRAIN)
    k = df_train.shape[1] - 1
    train_adj_r2 = 1 - (1 - train_r2) * ((n_train - 1) / (n_train - k - 1))
    train_huber_loss = float(np.mean(huber_np(1.0, ytr - Y_TRAIN)))

    LOGGER.info(
        f"Train dataset ({len(Y_TRAIN)} rows): "
        f"R^2: {round(train_r2, 3)}, "
        f"Adjusted R^2: {round(train_adj_r2, 3)}, "
        f"sMAPE: {round(train_mape, 3)}, "
        f"MAE: {round(train_mae, 3)}, "
        f"Huber: {train_huber_loss:.6f}"
    )

    test_r2 = r2_score(Y_TEST, yte)
    test_mape = smape_score(Y_TEST, yte)
    test_mae = mean_absolute_error(Y_TEST, yte)
    n_test = len(Y_TEST)
    k = df_test.shape[1] - 1
    test_adj_r2 = 1 - (1 - test_r2) * ((n_test - 1) / (n_test - k - 1))

    LOGGER.info(
        f"Test dataset ({len(Y_TEST)} rows): "
        f"R^2: {round(test_r2, 3)}, "
        f"Adjusted R^2: {round(test_adj_r2, 3)}, "
        f"sMAPE: {round(test_mape, 3)}, "
        f"MAE: {round(test_mae, 3)}"
    )

    if EARLY_STOP_ON_TEST_R2 and test_r2 >= EARLY_STOP_TEST_R2_TARGET:
        reached_target = True
        stop_reason = "target_test_r2_reached_final"

    LOGGER.info(f"Best expression: {best}")
    LOGGER.info(f"Best individual size: {len(best)} nodes, height={best.height}")
    LOGGER.info(f"Best individual (train Huber loss): {best} -> {train_huber_loss}")

    hof_path, pop_path, hof_df, pop_df = save_hof_and_population_csvs(hof, population)

    result = {
        "config_id": ACTIVE_CONFIG_ID,
        "config_hash": ACTIVE_CONFIG_HASH,
        "checkpoint_hash": ACTIVE_CHECKPOINT_HASH,
        "random_seed": random_seed,
        "MAX_NODES": MAX_NODES,
        "MAX_HEIGHT": MAX_HEIGHT,
        "init_pop_method": init_pop_method,
        "max_iterations": max_iterations,
        "pop_size": pop_size,
        "tourn_size": tourn_size,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "constants_range_json": json.dumps(list(constants_range)),
        "ELITE_SIZE": ELITE_SIZE,
        "LOG_METRICS_EVERY": LOG_METRICS_EVERY,
        "hof_size": hof_size,
        "best_train_r2": train_r2,
        "best_test_r2": test_r2,
        "best_train_huber_loss": train_huber_loss,
        "best_train_adj_r2": train_adj_r2,
        "best_test_adj_r2": test_adj_r2,
        "best_train_smape": train_mape,
        "best_test_smape": test_mape,
        "best_train_mae": train_mae,
        "best_test_mae": test_mae,
        "best_num_nodes": len(best),
        "best_height": best.height,
        "best_expression": str(best),
        "hof_csv": hof_path,
        "population_csv": pop_path,
        "checkpoint_path": checkpoint_path,
        "reached_target": reached_target,
        "stop_reason": stop_reason,
    }

    return result


# ============================================================
# INDEPENDENT RUN SUMMARY
# ============================================================

def load_existing_summary():
    if SEARCH_SUMMARY_PATH.exists():
        return pd.read_csv(SEARCH_SUMMARY_PATH)
    return pd.DataFrame()


def append_result_to_summary(result):
    SEARCH_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_summary()
    row_df = pd.DataFrame([result])

    if not existing.empty and "config_hash" in existing.columns:
        existing = existing[existing["config_hash"] != result["config_hash"]]
        out = pd.concat([existing, row_df], ignore_index=True)
    else:
        out = row_df

    out = out.sort_values(
        by=["best_train_huber_loss", "best_test_r2"],
        ascending=[True, False],
        na_position="last",
    )
    out.to_csv(SEARCH_SUMMARY_PATH, index=False)


def run_independent_single_config():
    INDEPENDENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEPENDENT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    INDEPENDENT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    full_cfg = make_full_config(CONFIG)
    config_id = INDEPENDENT_RUN_NAME
    h = config_hash(full_cfg)
    ckpt_h = checkpoint_hash(full_cfg)

    print("\nIndependent single-configuration GP run", flush=True)
    print(f"CONFIG_HASH: {h}", flush=True)
    print(f"CHECKPOINT_HASH: {ckpt_h}", flush=True)
    print(f"RESUME_FROM_CHECKPOINT: {RESUME_FROM_CHECKPOINT}", flush=True)
    print(f"max_iterations: {full_cfg['max_iterations']}", flush=True)
    print(f"Independent checkpoint dir: {INDEPENDENT_CHECKPOINT_DIR}", flush=True)
    print(f"Independent results dir: {INDEPENDENT_RESULTS_DIR}", flush=True)
    print(f"CONFIG: {full_cfg}", flush=True)

    start_time = time.perf_counter()
    result = run_single_gp_config(full_cfg, config_id)

    elapsed = time.perf_counter() - start_time
    append_result_to_summary(result)

    print(
        f"\nFinished independent run: "
        f"train_huber={result['best_train_huber_loss']:.6f}, "
        f"test_r2={result['best_test_r2']:.4f}, "
        f"elapsed={format_duration(elapsed)}",
        flush=True,
    )
    print(f"Summary saved to: {SEARCH_SUMMARY_PATH}", flush=True)
    print(f"Checkpoint path: {result['checkpoint_path']}", flush=True)


def main():
    multiprocessing.freeze_support()
    run_independent_single_config()


if __name__ == "__main__":
    main()
