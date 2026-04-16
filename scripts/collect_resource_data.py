#!/usr/bin/env python3
"""
Collect training data for the CPU/I/O resource prediction model.

The default path keeps the original random configuration sampling behavior.
The OpAdviser and hybrid paths run the normal DBTuner optimization loop, then
convert its history file into the same resource_data.json format used by
train_resource_model.py.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from autotune.utils.constants import FAILED, MEMOUT, SUCCESS, TIMEOUT


RESOURCE_KEYS = [
    "cpu",
    "readIO",
    "writeIO",
    "virtualMem",
    "physicalMem",
    "dirty",
    "hit",
    "data",
]

COMPATIBILITY_INFO_KEYS = [
    "knob_config_file",
    "knob_num",
    "database",
    "workload_type",
    "workload_name",
]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def create_collection_config(output_dir: str = "resource_data") -> str:
    """Create a default configuration file for resource data collection."""
    config_content = """[database]
# Database configuration
db = mysql
database = mysql
host = localhost
port = 3306
user = root
passwd = password
dbname = sbrw
sock = /var/run/mysqld/mysqld.sock
cnf = scripts/template/experiment_normandy.cnf
mysqld = /usr/sbin/mysqld
pg_ctl = /usr/bin/pg_ctl
pgdata = /var/lib/postgresql/data
postgres = /usr/bin/postgres
knob_config_file = scripts/experiment/gen_knobs/mysql_cpu_io_dynamic_15.json
knob_num = 15

# Workload configuration
workload = sysbench
oltpbench_config_xml =
workload_type = sbrw
workload_name = sbrw
thread_num = 40
workload_threads = 40
workload_time = 120
workload_warmup_time = 30
online_mode = True
remote_mode = False
isolation_mode = False
ssh_user =
pid = 0
lhs_log =
cpu_core =

# Metrics
performance_metric = ['tps']
resource_metric = ['cpu', 'readIO', 'writeIO']
constraints = []

[tune]
# Tuning configuration for data collection
task_id = resource_data_collection
performance_metric = ['tps']
reference_point = [None, None]
constraints =
max_runs = 60
initial_runs = 10
runtime_limit = None
optimize_method = SMAC
space_transfer = False
transfer_framework = none
auto_optimizer = False
auto_optimizer_type = learned
acq_optimizer_type = local_random

# Knob selection
initial_tunable_knob_num = 15
incremental = none
incremental_every = 0
incremental_num = 5
selector_type = shap

# Other settings
tr_init = True
batch_size = 16
data_repo = repo
mean_var_file =
params =
replay_memory =
only_knob = False
only_range = False
latent_dim = 0
softmax_weight = False
model_update_interval = 1
"""

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "collection_config.ini")
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"Created default collection config: {config_path}")
    print("Please edit the database connection settings before running.")
    return config_path


def prepare_collection_args(
    config_file: str,
    num_samples: int,
    workload_time_s: int,
    workload_warmup_time_s: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from autotune.utils.config import parse_args

    args_db, args_tune = parse_args(config_file)
    args_db = dict(args_db)
    args_tune = dict(args_tune)

    db_name = args_db.get("database") or args_db.get("db") or "mysql"
    args_db["database"] = db_name
    args_db["db"] = db_name
    args_db.setdefault("remote_mode", "False")
    args_db.setdefault("isolation_mode", "False")
    args_db.setdefault("pid", "0")
    args_db.setdefault("oltpbench_config_xml", "")
    args_db.setdefault("lhs_log", "")
    args_db.setdefault("cpu_core", "")
    args_db.setdefault("ssh_user", "")
    args_db.setdefault("thread_num", args_db.get("workload_threads", "40"))
    args_db.setdefault("workload_threads", args_db.get("thread_num", "40"))

    args_db["workload_time"] = str(workload_time_s)
    args_db["workload_warmup_time"] = str(workload_warmup_time_s)
    args_db["online_mode"] = "True"

    args_tune["max_runs"] = str(num_samples)
    args_tune.setdefault("reference_point", "[None, None]")
    args_tune.setdefault("constraints", "")
    args_tune.setdefault("runtime_limit", "None")
    args_tune.setdefault("acq_optimizer_type", "local_random")
    args_tune.setdefault("auto_optimizer", "False")
    args_tune.setdefault("auto_optimizer_type", "learned")
    args_tune.setdefault("transfer_framework", "none")
    args_tune.setdefault("space_transfer", "False")
    args_tune.setdefault("mean_var_file", "")
    args_tune.setdefault("only_knob", "False")
    args_tune.setdefault("only_range", "False")
    if args_tune.get("softmax_weight", "") == "":
        args_tune["softmax_weight"] = False
    else:
        args_tune.setdefault("softmax_weight", False)
    args_tune["softmax_weight"] = parse_bool(args_tune["softmax_weight"])

    return args_db, args_tune


def build_database(args_db: Dict[str, Any]) -> Any:
    db_name = str(args_db.get("database") or args_db.get("db", "mysql")).lower()
    if db_name == "mysql":
        from autotune.database.mysqldb import MysqlDB

        return MysqlDB(args_db)
    if db_name in {"postgresql", "postgres"}:
        from autotune.database.postgresqldb import PostgresqlDB

        return PostgresqlDB(args_db)
    raise ValueError(f"Unsupported database: {args_db.get('database')}")


def build_env_and_tuner(args_db: Dict[str, Any], args_tune: Dict[str, Any]) -> Tuple[Any, Any]:
    from autotune.dbenv import DBEnv
    from autotune.tuner import DBTuner

    db = build_database(args_db)
    env = DBEnv(args_db, args_tune, db)
    tuner = DBTuner(args_db, args_tune, env)
    return env, tuner


def normalize_resource(resource: Any) -> Optional[Dict[str, float]]:
    if resource is None:
        return None

    if isinstance(resource, dict):
        normalized = {}
        for key in RESOURCE_KEYS:
            if key == "physicalMem":
                value = resource.get("physicalMem", resource.get("physical", 0))
            else:
                value = resource.get(key, 0)
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                normalized[key] = 0.0
        return normalized

    if isinstance(resource, np.ndarray):
        resource = resource.tolist()

    if isinstance(resource, (list, tuple)):
        values = list(resource)
        if len(values) < 3:
            return None
        normalized = {}
        for index, key in enumerate(RESOURCE_KEYS):
            value = values[index] if index < len(values) else 0
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                normalized[key] = 0.0
        return normalized

    return None


def valid_resource(resource: Optional[Dict[str, float]]) -> bool:
    if resource is None:
        return False
    return (
        resource.get("cpu", 0) > 0
        and resource.get("readIO", -1) >= 0
        and resource.get("writeIO", -1) >= 0
    )


def workload_info(args_db: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": args_db.get("workload_name", args_db.get("workload_type", "unknown")),
        "type": args_db.get("workload_type", args_db.get("benchmark", "unknown")),
        "threads": _safe_int(args_db.get("workload_threads", args_db.get("thread_num")), 0),
    }


def trial_state_name(trial_state: Any) -> str:
    if isinstance(trial_state, str):
        return trial_state.upper()
    if trial_state == SUCCESS:
        return "SUCCESS"
    if trial_state == TIMEOUT:
        return "TIMEOUT"
    if trial_state == MEMOUT:
        return "MEMOUT"
    if trial_state == FAILED:
        return "FAILED"
    return "UNKNOWN"


def make_resource_sample(
    configuration: Dict[str, Any],
    external_metrics: Any,
    internal_metrics: Any,
    resource: Dict[str, float],
    args_db: Dict[str, Any],
    trial_state: Any = SUCCESS,
) -> Dict[str, Any]:
    return {
        "configuration": _to_jsonable(configuration),
        "external_metrics": _to_jsonable(external_metrics),
        "internal_metrics": _to_jsonable(internal_metrics),
        "resource": _to_jsonable(resource),
        "workload": workload_info(args_db),
        "trial_state": trial_state_name(trial_state),
    }


def collection_info(
    args_db: Dict[str, Any],
    sampling_strategy: str,
    num_requested: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    info = {
        "sampling_strategy": sampling_strategy,
        "num_requested": num_requested,
        "database": args_db.get("database", args_db.get("db")),
        "knob_config_file": args_db.get("knob_config_file"),
        "knob_num": _safe_int(args_db.get("knob_num"), 0),
        "workload_type": args_db.get("workload_type"),
        "workload_name": args_db.get("workload_name"),
        "workload_threads": _safe_int(args_db.get("workload_threads", args_db.get("thread_num")), 0),
        "workload_time": _safe_int(args_db.get("workload_time"), 0),
        "workload_warmup_time": _safe_int(args_db.get("workload_warmup_time"), 0),
    }
    if extra:
        info.update(extra)
    return info


def load_existing_output(output_dir: str) -> Optional[Dict[str, Any]]:
    file_path = os.path.join(output_dir, "resource_data.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path) as f:
        return json.load(f)


def load_existing_data(output_dir: str) -> List[Dict[str, Any]]:
    existing = load_existing_output(output_dir)
    if not existing:
        return []
    data = existing.get("data", [])
    print(f"Loaded {len(data)} existing samples")
    return data


def validate_append_compatibility(existing_info: Dict[str, Any], new_info: Dict[str, Any]) -> None:
    mismatches = []
    for key in COMPATIBILITY_INFO_KEYS:
        old_value = existing_info.get(key)
        new_value = new_info.get(key)
        if old_value not in (None, "") and new_value not in (None, "") and str(old_value) != str(new_value):
            mismatches.append(f"{key}: existing={old_value!r}, new={new_value!r}")

    if mismatches:
        message = "Cannot append resource data with incompatible metadata:\n"
        message += "\n".join(f"  - {item}" for item in mismatches)
        raise ValueError(message)


def save_data(
    data: List[Dict[str, Any]],
    output_dir: str,
    intermediate: bool = False,
    append: bool = True,
    info_extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    file_name = "resource_data_intermediate.json" if intermediate else "resource_data.json"
    file_path = os.path.join(output_dir, file_name)

    existing_data = []
    existing_info = {}
    if append and not intermediate:
        existing = load_existing_output(output_dir)
        if existing:
            existing_data = existing.get("data", [])
            existing_info = existing.get("info", {})
            validate_append_compatibility(existing_info, info_extra or {})

    all_data = existing_data + data if append and not intermediate else data

    info = {
        "num_samples": len(all_data),
        "new_samples": len(data),
        "previous_samples": len(existing_data),
        "collection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "resource": ["cpu", "readIO", "writeIO"],
            "description": "CPU, Read I/O, and Write I/O resource metrics",
        },
    }
    if existing_info:
        for key, value in existing_info.items():
            if key not in info:
                info[key] = value
    if info_extra:
        info.update(info_extra)

    output = {"info": info, "data": all_data}
    with open(file_path, "w") as f:
        json.dump(_to_jsonable(output), f, indent=2)

    print(f"Saved {len(all_data)} samples to {file_path}")


def collect_random_samples(
    config_file: str,
    num_samples: int,
    output_dir: str,
    workload_time_s: int,
    workload_warmup_time_s: int,
    save_intermediate: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if num_samples <= 0:
        return [], {}

    print(f"\nCollecting {num_samples} random resource samples")
    args_db, args_tune = prepare_collection_args(
        config_file, num_samples, workload_time_s, workload_warmup_time_s
    )
    args_tune["initial_runs"] = str(num_samples)
    args_tune["space_transfer"] = "False"
    args_tune["transfer_framework"] = "none"
    args_tune["auto_optimizer"] = "False"

    env, tuner = build_env_and_tuner(args_db, args_tune)
    config_space = tuner.setup_configuration_space(
        args_db["knob_config_file"], int(args_db["knob_num"])
    )

    collected_data = []
    start_time = time.time()

    for i in range(num_samples):
        print(f"\nSample {i + 1}/{num_samples}")
        try:
            config = config_space.sample_configuration()
            config_dict = config.get_dictionary()
            print(f"Configuration: {config_dict}")

            env.apply_knobs(config_dict)
            benchmark_timeout, external_metrics, internal_metrics, resource = env.get_states(
                collect_resource=1
            )

            resource_dict = normalize_resource(resource)
            if not valid_resource(resource_dict):
                print(f"Invalid resource metrics: {resource}")
                continue

            sample = make_resource_sample(
                config_dict,
                external_metrics,
                internal_metrics,
                resource_dict,
                args_db,
                TIMEOUT if benchmark_timeout else SUCCESS,
            )
            collected_data.append(sample)

            print(
                "Resources: "
                f"CPU={resource_dict['cpu']:.2f}, "
                f"ReadIO={resource_dict['readIO']:.2f}, "
                f"WriteIO={resource_dict['writeIO']:.2f}"
            )

            if save_intermediate and (i + 1) % 10 == 0:
                print(f"Saving intermediate data ({len(collected_data)} samples)")
                save_data(
                    collected_data,
                    output_dir,
                    intermediate=True,
                    append=False,
                    info_extra=collection_info(args_db, "random", num_samples),
                )

        except Exception as e:
            print(f"Error collecting sample {i + 1}: {e}")
            continue

    elapsed = time.time() - start_time
    avg_time = elapsed / len(collected_data) if collected_data else 0
    print("\nRandom collection complete:")
    print(f"  Collected: {len(collected_data)}/{num_samples} valid samples")
    print(f"  Time: {elapsed:.1f}s total, {avg_time:.1f}s/sample")

    return collected_data, args_db


def _safe_sequence_get(values: Sequence[Any], index: int, default: Any) -> Any:
    try:
        return values[index]
    except (IndexError, TypeError):
        return default


def history_object_to_payload(history: Any) -> Dict[str, Any]:
    configurations = getattr(history, "configurations_all", None) or getattr(
        history, "configurations", []
    )
    external_metrics = getattr(history, "external_metrics", [])
    internal_metrics = getattr(history, "internal_metrics", [])
    resources = getattr(history, "resource", [])
    contexts = getattr(history, "contexts", [])
    trial_states = getattr(history, "trial_states", [])
    elapsed_times = getattr(history, "elapsed_times", [])
    iter_times = getattr(history, "iter_times", [])

    data = []
    for i, config in enumerate(configurations):
        if hasattr(config, "get_dictionary"):
            config_dict = config.get_dictionary()
        else:
            config_dict = dict(config)

        data.append(
            {
                "configuration": config_dict,
                "external_metrics": _safe_sequence_get(external_metrics, i, {}),
                "internal_metrics": _safe_sequence_get(internal_metrics, i, []),
                "resource": _safe_sequence_get(resources, i, []),
                "context": _safe_sequence_get(contexts, i, {}),
                "trial_state": _safe_sequence_get(trial_states, i, SUCCESS),
                "elapsed_time": _safe_sequence_get(elapsed_times, i, None),
                "iter_time": _safe_sequence_get(iter_times, i, None),
            }
        )

    return {"info": getattr(history, "info", {}), "data": data}


def load_history_payload(history_or_path: Any) -> Dict[str, Any]:
    if isinstance(history_or_path, str):
        with open(history_or_path) as f:
            return json.load(f)
    if isinstance(history_or_path, dict):
        return history_or_path
    return history_object_to_payload(history_or_path)


def history_to_resource_samples(
    history_or_path: Any,
    args_db: Dict[str, Any],
    include_failed: bool = False,
) -> List[Dict[str, Any]]:
    payload = load_history_payload(history_or_path)
    samples = []
    skipped = 0

    for trial in payload.get("data", []):
        state = trial.get("trial_state", SUCCESS)
        if not include_failed and trial_state_name(state) != "SUCCESS":
            skipped += 1
            continue

        resource_dict = normalize_resource(trial.get("resource"))
        if not valid_resource(resource_dict):
            skipped += 1
            continue

        sample = make_resource_sample(
            trial.get("configuration", {}),
            trial.get("external_metrics", {}),
            trial.get("internal_metrics", []),
            resource_dict,
            args_db,
            state,
        )

        context = trial.get("context")
        if context:
            sample["context"] = _to_jsonable(context)

        samples.append(sample)

    print(f"Converted {len(samples)} valid samples from OpAdviser history")
    if skipped:
        print(f"Skipped {skipped} failed or invalid history trials")
    return samples


def collect_opadviser_samples(
    config_file: str,
    num_samples: int,
    workload_time_s: int,
    workload_warmup_time_s: int,
    optimize_method: str,
    space_transfer: bool,
    transfer_framework: str,
    task_id: Optional[str],
    initial_runs: int,
    acq_optimizer_type: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if num_samples <= 0:
        return [], {}, {}

    print(f"\nCollecting {num_samples} OpAdviser-guided resource samples")
    if optimize_method == "TURBO":
        optimize_method = "TurBO"

    args_db, args_tune = prepare_collection_args(
        config_file, num_samples, workload_time_s, workload_warmup_time_s
    )

    task_id = task_id or f"resource_data_collection_opadviser_{_now_id()}"
    args_tune["task_id"] = task_id
    args_tune["max_runs"] = str(num_samples)
    args_tune["initial_runs"] = str(max(1, min(initial_runs, num_samples)))
    args_tune["optimize_method"] = optimize_method
    args_tune["space_transfer"] = "True" if space_transfer else "False"
    args_tune["transfer_framework"] = transfer_framework
    args_tune["auto_optimizer"] = "False"
    args_tune["acq_optimizer_type"] = acq_optimizer_type

    env, tuner = build_env_and_tuner(args_db, args_tune)
    history = tuner.tune()

    history_path = os.path.join("repo", f"history_{task_id}.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    if history is not None and hasattr(history, "save_json"):
        history.save_json(history_path)

    if os.path.exists(history_path):
        samples = history_to_resource_samples(history_path, args_db)
    else:
        samples = history_to_resource_samples(history, args_db)

    opadviser_info = {
        "opadviser": {
            "task_id": task_id,
            "history_path": history_path,
            "optimize_method": optimize_method,
            "space_transfer": space_transfer,
            "transfer_framework": transfer_framework,
            "initial_runs": _safe_int(args_tune.get("initial_runs"), initial_runs),
            "acq_optimizer_type": acq_optimizer_type,
        }
    }
    return samples, args_db, opadviser_info


def collect_data(
    config_file: str,
    num_samples: int = 60,
    output_dir: str = "resource_data",
    append: bool = True,
    workload_time_s: int = 120,
    workload_warmup_time_s: int = 30,
    sampling_strategy: str = "random",
    initial_random_samples: int = 10,
    opadviser_optimize_method: str = "SMAC",
    opadviser_space_transfer: bool = True,
    opadviser_transfer_framework: str = "none",
    opadviser_task_id: Optional[str] = None,
    opadviser_initial_runs: int = 3,
    opadviser_acq_optimizer_type: str = "local_random",
) -> List[Dict[str, Any]]:
    """Collect resource data using random, OpAdviser, or hybrid sampling."""
    if sampling_strategy not in {"random", "opadviser", "hybrid"}:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    print("=" * 60)
    print("Resource Data Collection")
    print("=" * 60)
    print(f"Strategy: {sampling_strategy}")
    print(f"Target samples: {num_samples}")
    print(f"Output directory: {output_dir}")

    collected_data = []
    args_db_for_info = None
    extra_info: Dict[str, Any] = {}

    if sampling_strategy == "random":
        collected_data, args_db_for_info = collect_random_samples(
            config_file,
            num_samples,
            output_dir,
            workload_time_s,
            workload_warmup_time_s,
        )

    elif sampling_strategy == "opadviser":
        collected_data, args_db_for_info, extra_info = collect_opadviser_samples(
            config_file,
            num_samples,
            workload_time_s,
            workload_warmup_time_s,
            opadviser_optimize_method,
            opadviser_space_transfer,
            opadviser_transfer_framework,
            opadviser_task_id,
            opadviser_initial_runs,
            opadviser_acq_optimizer_type,
        )

    else:
        random_count = min(max(initial_random_samples, 0), num_samples)
        opadviser_count = num_samples - random_count
        random_data, random_args_db = collect_random_samples(
            config_file,
            random_count,
            output_dir,
            workload_time_s,
            workload_warmup_time_s,
        )
        opadviser_data, opadviser_args_db, extra_info = collect_opadviser_samples(
            config_file,
            opadviser_count,
            workload_time_s,
            workload_warmup_time_s,
            opadviser_optimize_method,
            opadviser_space_transfer,
            opadviser_transfer_framework,
            opadviser_task_id,
            opadviser_initial_runs,
            opadviser_acq_optimizer_type,
        )
        collected_data = random_data + opadviser_data
        args_db_for_info = random_args_db or opadviser_args_db
        extra_info["hybrid"] = {
            "random_requested": random_count,
            "opadviser_requested": opadviser_count,
            "random_collected": len(random_data),
            "opadviser_collected": len(opadviser_data),
        }

    if args_db_for_info is None:
        args_db_for_info, _ = prepare_collection_args(
            config_file, num_samples, workload_time_s, workload_warmup_time_s
        )

    info = collection_info(args_db_for_info, sampling_strategy, num_samples, extra_info)
    save_data(collected_data, output_dir, intermediate=False, append=append, info_extra=info)
    return collected_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect resource training data for CPU/I/O prediction"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Configuration file for database and workload",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=60,
        help="Number of samples to collect",
    )
    parser.add_argument(
        "--output_dir",
        default="resource_data",
        help="Output directory for collected data",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Do not append to existing data, overwrite instead",
    )
    parser.add_argument(
        "--workload_time",
        type=int,
        default=120,
        help="Workload running time per sample in seconds",
    )
    parser.add_argument(
        "--workload_warmup_time",
        type=int,
        default=30,
        help="Workload warmup time per sample in seconds",
    )
    parser.add_argument(
        "--sampling_strategy",
        choices=["random", "opadviser", "hybrid"],
        default="random",
        help="How to choose configurations for resource data collection",
    )
    parser.add_argument(
        "--initial_random_samples",
        type=int,
        default=10,
        help="Random samples to collect before OpAdviser in hybrid mode",
    )
    parser.add_argument(
        "--opadviser_optimize_method",
        default="SMAC",
        choices=["SMAC", "MBO", "GA", "DDPG", "TPE", "TurBO", "TURBO", "auto"],
        help="Optimizer method for OpAdviser-guided collection",
    )
    parser.add_argument(
        "--opadviser_space_transfer",
        type=parse_bool,
        default=True,
        help="Whether OpAdviser should use compact-space transfer",
    )
    parser.add_argument(
        "--opadviser_transfer_framework",
        default="none",
        choices=["none", "rgpe", "workload_map"],
        help="Transfer-learning framework for OpAdviser",
    )
    parser.add_argument(
        "--opadviser_task_id",
        default=None,
        help="Task id for OpAdviser history output. Defaults to a timestamped id.",
    )
    parser.add_argument(
        "--opadviser_initial_runs",
        type=int,
        default=3,
        help="Initial random trials inside the OpAdviser optimizer",
    )
    parser.add_argument(
        "--opadviser_acq_optimizer_type",
        default="local_random",
        help="Acquisition optimizer type passed to DBTuner",
    )

    args = parser.parse_args()

    append = not args.no_append
    if append:
        existing_data = load_existing_data(args.output_dir)
        if existing_data:
            print(f"Append mode: will add new samples to {len(existing_data)} existing samples")
        else:
            print("Append mode: no existing data found, starting fresh")
    else:
        print("Overwrite mode: will replace existing data")

    config_file = args.config
    if config_file is None:
        config_file = os.path.join(args.output_dir, "collection_config.ini")
        if not os.path.exists(config_file):
            print("No config file specified. Creating default config...")
            config_file = create_collection_config(args.output_dir)
            print("\nPlease edit the config file and run again:")
            print(f"  {config_file}")
            return 1

    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        return 1

    collect_data(
        config_file=config_file,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        append=append,
        workload_time_s=args.workload_time,
        workload_warmup_time_s=args.workload_warmup_time,
        sampling_strategy=args.sampling_strategy,
        initial_random_samples=args.initial_random_samples,
        opadviser_optimize_method=args.opadviser_optimize_method,
        opadviser_space_transfer=args.opadviser_space_transfer,
        opadviser_transfer_framework=args.opadviser_transfer_framework,
        opadviser_task_id=args.opadviser_task_id,
        opadviser_initial_runs=args.opadviser_initial_runs,
        opadviser_acq_optimizer_type=args.opadviser_acq_optimizer_type,
    )

    print("\nNext steps:")
    print("1. Inspect collected data:")
    print(f"   {os.path.join(args.output_dir, 'resource_data.json')}")
    print("2. Train resource models:")
    print("   python scripts/train_resource_model.py --data_file resource_data/resource_data.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
