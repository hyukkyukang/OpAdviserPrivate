# OpAdviser: Automated Database Configuration Tuning

OpAdviser is an intelligent database tuning system that automatically optimizes database configurations (knobs) to improve performance metrics like throughput (TPS) and latency. It leverages **transfer learning** from historical tuning data across different workloads to accelerate optimization.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluating OpAdviser Performance](#evaluating-opadviser-performance)
  - [Quick Evaluation (Fast Mode)](#-quick-evaluation-fast-mode---recommended-for-first-time-users)
  - [Ultra-Fast Evaluation](#-ultra-fast-evaluation-50-minutes-total)
  - [CPU & Disk I/O Focused Experiment](#-cpu--disk-io-focused-experiment-recommended)
- [Configuration Options](#configuration-options)
- [Supported Workloads](#supported-workloads)
- [Resource Prediction Model Training](#resource-prediction-model-training)
- [Project Structure](#project-structure)

---

## Overview

### What OpAdviser Does

1. **Tunes database knobs** (e.g., `innodb_buffer_pool_size`, `innodb_io_capacity`) to maximize performance
2. **Transfers knowledge** from previously tuned workloads to accelerate new tuning tasks
3. **Automatically prunes the search space** based on workload similarity
4. **Supports multiple optimization algorithms**: Bayesian Optimization (SMAC, MBO), Reinforcement Learning (DDPG), Genetic Algorithm (GA)

### Key Innovation: Space Transfer

OpAdviser uses **RGPE (Ranking-weighted Gaussian Process Ensemble)** to:
- Measure similarity between workloads
- Identify "promising regions" in the configuration space
- Focus search on configurations that worked well for similar workloads

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OpAdviser Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   Source    │    │    RGPE     │    │   Configuration Space   │ │
│  │  Workloads  │───▶│  Similarity │───▶│       Pruning           │ │
│  │  (repo/)    │    │  Calculation│    │                         │ │
│  └─────────────┘    └─────────────┘    └───────────┬─────────────┘ │
│                                                     │               │
│                                                     ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   Target    │◀───│  Optimizer  │◀───│   Acquisition Function  │ │
│  │  Database   │    │ (SMAC/DDPG/ │    │   Maximization          │ │
│  │             │    │  GA/MBO)    │    │                         │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Benchmark Execution → Metrics Collection → History Update  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware Requirements
- **Recommended**: 32GB+ RAM, SSD storage
- MySQL 8.0+ or PostgreSQL 12+

### Software Requirements
- Python 3.8+
- MySQL Server or PostgreSQL Server
- Benchmark tools (Sysbench, OLTP-Bench)

---

## Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd OpAdviserPrivate
pip install -r requirements.txt
export PYTHONPATH="."
```

### 2. Install Sysbench (for Sysbench workloads)

```bash
git clone https://github.com/akopytov/sysbench.git
cd sysbench
git checkout ead2689ac6f61c5e7ba7c6e19198b86bd3a51d3c
./autogen.sh && ./configure && make && sudo make install
cd ..
```

### 3. Install OLTP-Bench (for TPC-C, Twitter, YCSB, etc.)

```bash
git clone https://github.com/seokjeongeum/oltpbench.git
cd oltpbench
ant bootstrap && ant resolve && ant build
chmod 777 /oltpbench/*
cd ..
```

---

## Quick Start

### Step 1: Prepare Database and Workload

Example for Sysbench Read-Write:

```bash
# Create database
mysql -uroot -ppassword -e "CREATE DATABASE IF NOT EXISTS sbrw;"

# Load data
sysbench \
    --db-driver=mysql \
    --mysql-host=localhost \
    --mysql-port=3306 \
    --mysql-user=root \
    --mysql-password=password \
    --table_size=800000 \
    --tables=300 \
    --threads=80 \
    --mysql-db=sbrw \
    oltp_read_write prepare
```

### Step 2: Run OpAdviser

```bash
python scripts/optimize.py \
    --dbname=sbrw \
    --workload=sysbench \
    --workload_type=sbrw \
    --softmax_weight \
    --transformer
```

### Step 3: Monitor Progress

Results are saved to:
- `repo/history_<task_id>.json` - Tuning history
- `logs/` - Detailed logs
- `<task_id>.png` - Convergence plot

---

## Evaluating OpAdviser Performance

To properly evaluate OpAdviser, you need to compare it against a **ground truth baseline** on your specific hardware.

### ⚡ Quick Evaluation (Fast Mode) - Recommended for First-Time Users

For quick testing (~3 hours total instead of ~30 hours):

```bash
# 1. Prepare smaller dataset
mysql -uroot -ppassword -e "DROP DATABASE IF EXISTS sbrw; CREATE DATABASE sbrw;"
sysbench \
    --db-driver=mysql \
    --mysql-host=localhost \
    --mysql-port=3306 \
    --mysql-user=root \
    --mysql-password=password \
    --table_size=100000 \
    --tables=50 \
    --threads=80 \
    --mysql-db=sbrw \
    oltp_read_write prepare

# 2. Run Ground Truth (Fast) - ~2.5 hours
python scripts/optimize.py --config=scripts/config_ground_truth_fast.ini

# 3. Run OpAdviser (Fast) - ~45 minutes  
python scripts/optimize.py --config=scripts/config_opadviser_fast.ini

# 4. Compare results
python scripts/compare_results.py \
    --opadviser=sbrw_opadviser_fast \
    --ground_truth=sbrw_ground_truth_fast \
    --plot
```

### Speed Optimization Options

| Parameter | Full Experiment | Fast Experiment | Impact |
|-----------|-----------------|-----------------|--------|
| `max_runs` | 500 (GT) / 100 (OP) | 100 (GT) / 30 (OP) | Fewer iterations |
| `knob_num` | 197 | 20 | Smaller search space |
| `workload_time` | 180s | 60s | Shorter benchmark per iteration |
| `workload_warmup_time` | 10s | 5s | Less warmup |
| `online_mode` | False | True | No MySQL restart (faster) |
| `table_size` | 800,000 | 100,000 | Smaller dataset |
| `tables` | 300 | 50 | Fewer tables |

**Time Estimates:**

| Mode | Ground Truth | OpAdviser | Total |
|------|--------------|-----------|-------|
| **Full** | ~25 hours | ~5 hours | ~30 hours |
| **Fast** | ~2.5 hours | ~45 min | ~3.5 hours |
| **Ultra-Fast** | ~35 min | ~15 min | ~50 min |

### 🚀 Ultra-Fast Evaluation (~50 minutes total)

For the quickest possible test. This uses **online mode** (no MySQL restarts) with **dynamically-changeable knobs only**.

#### Prerequisites

Before running, ensure these are set up:

```bash
# 1. Set required environment variables
export PYTHONPATH="."
export MYSQL_SOCK=/var/run/mysqld/mysqld.sock

# 2. Verify MySQL is running
service mysql status
# If not running: service mysql start

# 3. Verify sysbench is installed
sysbench --version
# Should show: sysbench 1.0.x
```

#### Step-by-Step Instructions

```bash
# Step 1: Prepare minimal dataset (~1 minute)
mysql -uroot -ppassword -e "DROP DATABASE IF EXISTS sbrw; CREATE DATABASE sbrw;"
sysbench \
    --db-driver=mysql \
    --mysql-host=localhost \
    --mysql-port=3306 \
    --mysql-user=root \
    --mysql-password=password \
    --table_size=50000 \
    --tables=20 \
    --threads=40 \
    --mysql-db=sbrw \
    oltp_read_write prepare

# Step 2: Run Ground Truth (~25 minutes, 50 iterations × 30s each)
export PYTHONPATH="." && export MYSQL_SOCK=/var/run/mysqld/mysqld.sock
python scripts/optimize.py --config=scripts/config_ground_truth_ultrafast.ini

# Step 3: Run OpAdviser (~8 minutes, 15 iterations × 30s each)
export PYTHONPATH="." && export MYSQL_SOCK=/var/run/mysqld/mysqld.sock
python scripts/optimize.py --config=scripts/config_opadviser_ultrafast.ini

# Step 4: Compare results
python scripts/compare_results.py \
    --opadviser=sbrw_opadviser_ultrafast \
    --ground_truth=sbrw_ground_truth_ultrafast \
    --plot
```

#### Ultra-Fast Configuration Details

| Setting | Ground Truth | OpAdviser |
|---------|--------------|-----------|
| Config file | `config_ground_truth_ultrafast.ini` | `config_opadviser_ultrafast.ini` |
| Knob file | `mysql_dynamic_10.json` | `mysql_dynamic_10.json` |
| Knobs | 10 (dynamically changeable) | 10 |
| Iterations | 50 | 15 |
| Benchmark time | 30s | 30s |
| `space_transfer` | False | True |
| `optimize_method` | SMAC | DDPG |
| `online_mode` | True | True |

#### Knobs Tuned in Ultra-Fast Mode

These 10 knobs can be changed without restarting MySQL:

| Knob | Description | Range |
|------|-------------|-------|
| `innodb_io_capacity` | Background I/O ops/sec | 100 - 200,000 |
| `innodb_io_capacity_max` | Max I/O capacity | 100 - 400,000 |
| `innodb_thread_concurrency` | Max threads in InnoDB | 0 - 1,000 |
| `innodb_spin_wait_delay` | Spin lock delay (µs) | 0 - 6,000 |
| `innodb_max_dirty_pages_pct` | Max dirty page % | 0 - 99 |
| `thread_cache_size` | Cached threads | 0 - 16,384 |
| `table_open_cache` | Table cache size | 1 - 100,000 |
| `sort_buffer_size` | Sort buffer | 32KB - 128MB |
| `read_buffer_size` | Read buffer | 8KB - 2GB |
| `join_buffer_size` | Join buffer | 128B - 4GB |

**⚠️ Trade-offs of Ultra-Fast Mode:**
- Fewer iterations → Less reliable "best" configuration found
- Only 10 knobs → May miss optimal configurations involving other knobs
- Online mode → Static knobs (requiring restart) are excluded

**Recommendation:** Use Ultra-Fast for initial testing and learning. For production tuning, use the CPU/IO focused or full evaluation.

---

### 🎯 Ultra-Fast CPU & I/O Focused Experiment (Recommended for Performance Tuning)

For targeted tuning of **CPU usage and Disk I/O performance**, we provide a curated set of knobs that directly impact these resources.

#### Why Focus on CPU & I/O?

- **High impact**: These knobs have the largest effect on database performance
- **More interpretable**: All knobs directly affect CPU or I/O
- **Faster convergence**: Smaller, focused search space

---

#### 🚀 Ultra-Fast CPU/IO Mode (~1.5 hours total)

Uses **15 dynamically-changeable CPU/IO knobs** with online mode (no MySQL restarts).

##### Prerequisites

```bash
# Set required environment variables
export PYTHONPATH="."
export MYSQL_SOCK=/var/run/mysqld/mysqld.sock

# Verify MySQL is running
service mysql status
```

##### Step-by-Step Instructions

```bash
# Step 1: Prepare minimal dataset (~1 minute)
mysql -uroot -ppassword -e "DROP DATABASE IF EXISTS sbrw; CREATE DATABASE sbrw;"
sysbench \
    --db-driver=mysql \
    --mysql-host=localhost \
    --mysql-port=3306 \
    --mysql-user=root \
    --mysql-password=password \
    --table_size=50000 \
    --tables=20 \
    --threads=40 \
    --mysql-db=sbrw \
    oltp_read_write prepare

# Step 2: Run Ground Truth (~60 minutes, 100 iterations × 30s each)
export PYTHONPATH="." && export MYSQL_SOCK=/var/run/mysqld/mysqld.sock
python scripts/optimize.py --config=scripts/config_cpu_io_ground_truth_ultrafast.ini

# Step 3: Run OpAdviser (~25 minutes, 40 iterations × 30s each)
export PYTHONPATH="." && export MYSQL_SOCK=/var/run/mysqld/mysqld.sock
python scripts/optimize.py --config=scripts/config_cpu_io_opadviser_ultrafast.ini

# Step 4: Compare results
python scripts/compare_results.py \
    --opadviser=sbrw_cpu_io_opadviser_ultrafast \
    --ground_truth=sbrw_cpu_io_ground_truth_ultrafast \
    --plot
```

##### CPU/IO Knobs Tuned (15 dynamically-changeable knobs)

| Category | Knob | Description |
|----------|------|-------------|
| **I/O Capacity** | `innodb_io_capacity` | Background I/O ops/sec (100-200K) |
| | `innodb_io_capacity_max` | Max I/O capacity (100-400K) |
| **Threading** | `innodb_thread_concurrency` | Max concurrent InnoDB threads (0-1000) |
| | `thread_cache_size` | Cached thread connections (0-16K) |
| **Spin/Wait** | `innodb_spin_wait_delay` | Spin lock delay µs (0-6000) |
| | `innodb_sync_spin_loops` | Spin loops before sleep (0-30K) |
| **Dirty Pages** | `innodb_max_dirty_pages_pct` | Max dirty page % (0-99) |
| | `innodb_max_dirty_pages_pct_lwm` | Dirty page low water mark (0-99) |
| **Flushing** | `innodb_flushing_avg_loops` | Flushing average loops (1-1000) |
| | `innodb_lru_scan_depth` | LRU scan depth (100-10K) |
| **Caching** | `table_open_cache` | Open table cache (1-100K) |
| | `innodb_old_blocks_time` | Old blocks time ms (0-4B) |
| **Buffers** | `sort_buffer_size` | Sort buffer (32KB-128MB) |
| | `read_buffer_size` | Sequential read buffer (8KB-2GB) |
| | `join_buffer_size` | Join buffer (128B-4GB) |

##### Ultra-Fast CPU/IO Configuration Details

| Setting | Ground Truth | OpAdviser |
|---------|--------------|-----------|
| Config file | `config_cpu_io_ground_truth_ultrafast.ini` | `config_cpu_io_opadviser_ultrafast.ini` |
| Knob file | `mysql_cpu_io_dynamic_15.json` | `mysql_cpu_io_dynamic_15.json` |
| Knobs | 15 | 15 |
| Iterations | 100 | 40 |
| Time estimate | ~60 min | ~25 min |
| `online_mode` | True | True |

---

#### ⏱️ Standard CPU/IO Mode (~10 hours total)

Uses **39 CPU/IO knobs** including static knobs (requires MySQL restarts).

##### Knobs Included (39 total)

| Category | Count | Examples |
|----------|-------|----------|
| **CPU/Threading** | 12 | `innodb_thread_concurrency`, `innodb_read_io_threads`, `innodb_write_io_threads`, `innodb_purge_threads` |
| **Disk I/O Capacity** | 6 | `innodb_io_capacity`, `innodb_flush_log_at_trx_commit`, `innodb_doublewrite` |
| **Buffer/Memory** | 8 | `innodb_buffer_pool_size`, `innodb_log_buffer_size`, `innodb_log_file_size` |
| **Flushing/Dirty Pages** | 8 | `innodb_max_dirty_pages_pct`, `innodb_adaptive_flushing`, `sync_binlog` |
| **Read Ahead/Caching** | 5 | `innodb_read_ahead_threshold`, `innodb_lru_scan_depth`, `table_open_cache` |

##### Running Standard CPU/IO Experiments

```bash
# 1. Prepare database
mysql -uroot -ppassword -e "DROP DATABASE IF EXISTS sbrw; CREATE DATABASE sbrw;"
sysbench \
    --db-driver=mysql \
    --mysql-host=localhost \
    --mysql-port=3306 \
    --mysql-user=root \
    --mysql-password=password \
    --table_size=100000 \
    --tables=50 \
    --threads=80 \
    --mysql-db=sbrw \
    oltp_read_write prepare

# 2. Run Ground Truth (CPU/IO focused) - ~6-8 hours
export PYTHONPATH="." && export MYSQL_SOCK=/var/run/mysqld/mysqld.sock
python scripts/optimize.py --config=scripts/config_cpu_io_ground_truth.ini

# 3. Run OpAdviser (CPU/IO focused) - ~2-3 hours
export PYTHONPATH="." && export MYSQL_SOCK=/var/run/mysqld/mysqld.sock
python scripts/optimize.py --config=scripts/config_cpu_io_opadviser.ini

# 4. Compare results
python scripts/compare_results.py \
    --opadviser=sbrw_cpu_io_opadviser \
    --ground_truth=sbrw_cpu_io_ground_truth \
    --plot
```

##### Standard CPU/IO Configuration Details

| Setting | Ground Truth | OpAdviser |
|---------|--------------|-----------|
| Config file | `config_cpu_io_ground_truth.ini` | `config_cpu_io_opadviser.ini` |
| Knob file | `mysql_cpu_io_40.json` | `mysql_cpu_io_40.json` |
| Knobs | 39 | 39 |
| Iterations | 150 | 50 |
| Time estimate | ~6-8 hours | ~2-3 hours |
| `space_transfer` | False | True |
| `optimize_method` | SMAC | DDPG |
| `online_mode` | False | False |

#### Detailed Knob List

<details>
<summary>Click to expand full knob list</summary>

**CPU/Threading (12 knobs)**
| Knob | Range | Description |
|------|-------|-------------|
| `innodb_thread_concurrency` | 0-1000 | Max concurrent threads in InnoDB |
| `innodb_read_io_threads` | 1-64 | Background read I/O threads |
| `innodb_write_io_threads` | 1-64 | Background write I/O threads |
| `innodb_purge_threads` | 1-32 | Purge operation threads |
| `innodb_page_cleaners` | 1-8 | Page cleaner threads |
| `innodb_spin_wait_delay` | 0-6000 | Spin lock delay (µs) |
| `innodb_sync_spin_loops` | 0-30000 | Spin loops before sleeping |
| `innodb_adaptive_max_sleep_delay` | 0-1000000 | Max adaptive sleep (µs) |
| `innodb_thread_sleep_delay` | 0-1000000 | Thread sleep delay (µs) |
| `innodb_concurrency_tickets` | 1-4B | Concurrency tickets |
| `thread_cache_size` | 0-16384 | Cached threads |
| `innodb_commit_concurrency` | 0-1000 | Commit concurrency |

**Disk I/O Capacity (6 knobs)**
| Knob | Range | Description |
|------|-------|-------------|
| `innodb_io_capacity` | 100-2M | Background I/O ops/sec |
| `innodb_io_capacity_max` | 100-40K | Max I/O capacity |
| `innodb_flush_neighbors` | 0/1/2 | Flush neighboring pages |
| `innodb_flush_log_at_trx_commit` | 0/1/2 | Durability vs performance |
| `innodb_doublewrite` | ON/OFF | Double write buffer |
| `innodb_use_native_aio` | ON/OFF | Native async I/O |

**Buffer/Memory (8 knobs)**
| Knob | Range | Description |
|------|-------|-------------|
| `innodb_buffer_pool_size` | 10GB-32GB | Main buffer pool |
| `innodb_log_buffer_size` | 256KB-4GB | Redo log buffer |
| `innodb_log_file_size` | 4MB-1GB | Redo log file size |
| `innodb_log_files_in_group` | 2-10 | Number of log files |
| `innodb_change_buffer_max_size` | 0-50% | Change buffer size |
| `key_buffer_size` | 8B-16GB | MyISAM key buffer |
| `read_buffer_size` | 8KB-2GB | Sequential read buffer |
| `sort_buffer_size` | 32KB-128MB | Sort buffer |

**Flushing/Dirty Pages (8 knobs)**
| Knob | Range | Description |
|------|-------|-------------|
| `innodb_max_dirty_pages_pct` | 0-99% | Max dirty page percentage |
| `innodb_max_dirty_pages_pct_lwm` | 0-99% | Low water mark |
| `innodb_adaptive_flushing` | ON/OFF | Adaptive flushing |
| `innodb_adaptive_flushing_lwm` | 0-70 | Adaptive flushing LWM |
| `innodb_flushing_avg_loops` | 1-1000 | Flushing average loops |
| `innodb_flush_log_at_timeout` | 1-2700s | Flush timeout |
| `innodb_flush_sync` | ON/OFF | Flush sync |
| `sync_binlog` | 0-4B | Binlog sync frequency |

**Read Ahead/Caching (5 knobs)**
| Knob | Range | Description |
|------|-------|-------------|
| `innodb_read_ahead_threshold` | 0-64 | Read ahead threshold |
| `innodb_random_read_ahead` | ON/OFF | Random read ahead |
| `innodb_lru_scan_depth` | 100-10240 | LRU scan depth |
| `innodb_old_blocks_time` | 0-4B ms | Old blocks time |
| `table_open_cache` | 1-250K | Table cache |

</details>

#### Customizing the Knob Set

To create your own custom knob configuration:

```python
import json

# Load all knobs
with open('scripts/experiment/gen_knobs/mysql_all_197_32G.json') as f:
    all_knobs = json.load(f)

# Select your knobs
my_knobs = ['innodb_buffer_pool_size', 'innodb_io_capacity', ...]

# Create filtered config
filtered = {k: all_knobs[k] for k in my_knobs if k in all_knobs}

# Save
with open('scripts/experiment/gen_knobs/my_custom_knobs.json', 'w') as f:
    json.dump(filtered, f, indent=4)
```

Then update your config file:
```ini
knob_config_file = scripts/experiment/gen_knobs/my_custom_knobs.json
knob_num = <number_of_knobs>
```

---

### Understanding the Evaluation Setup

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EVALUATION WORKFLOW                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PHASE 1: Generate Ground Truth (One-time, per workload)           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Run extensive random/SMAC search (500+ iterations)         │   │
│  │  This establishes the "best possible" TPS on your hardware  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  PHASE 2: Run OpAdviser                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Run OpAdviser with space_transfer=True (100 iterations)    │   │
│  │  This tests how well OpAdviser performs                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  PHASE 3: Compare Results                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Compare: OpAdviser TPS vs Ground Truth TPS                 │   │
│  │  Metrics: Final TPS, Convergence Speed, Iterations Needed   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Evaluation Guide

#### Step 1: Choose a Target Workload

Select one of the supported workloads to evaluate:

| Workload | Command |
|----------|---------|
| Sysbench RW | `--workload=sysbench --workload_type=sbrw` |
| Sysbench RO | `--workload=sysbench --workload_type=sbread` |
| Sysbench WO | `--workload=sysbench --workload_type=sbwrite` |
| Twitter | `--workload=oltpbench_twitter` |
| TPC-C | `--workload=oltpbench_tpcc` |
| YCSB | `--workload=oltpbench_ycsb` |

#### Step 2: Create Ground Truth Config

Create `scripts/config_ground_truth.ini`:

```ini
[database]
db = mysql
host = localhost
port = 3306
user = root
passwd = password
sock = /var/run/mysqld/mysqld.sock
cnf = scripts/template/experiment_normandy.cnf
mysqld = /usr/sbin/mysqld
knob_config_file = scripts/experiment/gen_knobs/mysql_all_197_32G.json
knob_num = 197
dbname = sbrw
workload = sysbench
oltpbench_config_xml = 
workload_type = sbrw
thread_num = 80
workload_warmup_time = 10
workload_time = 180
remote_mode = False
online_mode = False
isolation_mode = False
pid = 0

[tune]
task_id = sbrw_ground_truth
performance_metric = ['tps']
reference_point = [None, None]
constraints = 
max_runs = 500
selector_type = shap
initial_runs = 10
initial_tunable_knob_num = 197
incremental = none
optimize_method = SMAC
space_transfer = False
auto_optimizer = False
acq_optimizer_type = local_random
batch_size = 16
mean_var_file = 
transfer_framework = none
data_repo = repo
only_knob = False
only_range = False
```

**Key settings for ground truth:**
- `max_runs = 500` (or more) - Extensive search
- `space_transfer = False` - No transfer learning (pure baseline)
- `optimize_method = SMAC` - Standard Bayesian optimization

#### Step 3: Generate Ground Truth

```bash
# This takes a LONG time (~25+ hours for 500 iterations)
python scripts/optimize.py --config=scripts/config_ground_truth.ini
```

#### Step 4: Create OpAdviser Config

Create `scripts/config_opadviser.ini`:

```ini
[database]
# ... same database settings as ground truth ...
dbname = sbrw
workload = sysbench
workload_type = sbrw

[tune]
task_id = sbrw_opadviser
performance_metric = ['tps']
max_runs = 100
optimize_method = DDPG
space_transfer = True
auto_optimizer = False
transfer_framework = none
data_repo = repo
# ... other settings ...
```

**Key settings for OpAdviser:**
- `max_runs = 100` - Fewer iterations (testing efficiency)
- `space_transfer = True` - Enable transfer learning
- `optimize_method = DDPG` - Use DDPG optimizer

#### Step 5: Run OpAdviser

```bash
python scripts/optimize.py --config=scripts/config_opadviser.ini
```

Or use command-line arguments:

```bash
python scripts/optimize.py \
    --dbname=sbrw \
    --workload=sysbench \
    --workload_type=sbrw \
    --softmax_weight \
    --transformer
```

#### Step 6: Compare Results

```python
import json

# Load ground truth
with open("repo/history_sbrw_ground_truth.json") as f:
    gt_data = json.load(f)["data"]
    gt_best = max(gt_data, key=lambda x: x["external_metrics"].get("tps", 0))
    gt_tps = gt_best["external_metrics"]["tps"]

# Load OpAdviser result
with open("repo/history_sbrw_opadviser.json") as f:
    op_data = json.load(f)["data"]
    op_best = max(op_data, key=lambda x: x["external_metrics"].get("tps", 0))
    op_tps = op_best["external_metrics"]["tps"]

print(f"Ground Truth Best TPS: {gt_tps}")
print(f"OpAdviser Best TPS: {op_tps}")
print(f"OpAdviser achieved {op_tps/gt_tps*100:.1f}% of ground truth")
print(f"OpAdviser iterations: {len(op_data)}, Ground Truth iterations: {len(gt_data)}")
```

### Pre-collected Historical Data

The `repo/` directory contains **~430 pre-collected tuning histories** from various workloads and optimizers. These serve as **source knowledge** for transfer learning:

```
repo/
├── history_sysbench_smac_*.json    # Sysbench with SMAC
├── history_twitter_ddpg_*.json     # Twitter with DDPG
├── history_tpcc_mbo_*.json         # TPC-C with MBO
├── history_job_ga_*.json           # JOB with GA
└── ...
```

**Note**: These are from the original authors' hardware. For accurate transfer learning, the historical data should ideally be from similar hardware configurations.

---

## Configuration Options

### Key Configuration Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `optimize_method` | Optimization algorithm | `SMAC`, `MBO`, `DDPG`, `GA`, `TPE`, `TurBO` |
| `space_transfer` | Enable space pruning via transfer | `True`, `False` |
| `auto_optimizer` | Auto-select optimizer | `True`, `False` |
| `max_runs` | Maximum tuning iterations | Integer (e.g., 100) |
| `initial_runs` | Initial random exploration | Integer (e.g., 10) |
| `knob_num` | Number of knobs to tune | Integer (e.g., 197) |

### Optimizer Descriptions

| Optimizer | Type | Best For |
|-----------|------|----------|
| **SMAC** | Bayesian Optimization (Random Forest) | High-dimensional spaces |
| **MBO** | Bayesian Optimization (Gaussian Process) | Lower-dimensional, continuous |
| **DDPG** | Reinforcement Learning | Learning from internal metrics |
| **GA** | Genetic Algorithm | Discrete/mixed spaces |
| **TPE** | Tree-structured Parzen Estimator | Categorical-heavy configs |

---

## Supported Workloads

| Workload | Benchmark Tool | Type | Command |
|----------|----------------|------|---------|
| Sysbench RW | Sysbench | OLTP | `--workload=sysbench --workload_type=sbrw` |
| Sysbench RO | Sysbench | OLTP | `--workload=sysbench --workload_type=sbread` |
| Sysbench WO | Sysbench | OLTP | `--workload=sysbench --workload_type=sbwrite` |
| TPC-C | OLTP-Bench | OLTP | `--workload=oltpbench_tpcc` |
| Twitter | OLTP-Bench | OLTP | `--workload=oltpbench_twitter` |
| Wikipedia | OLTP-Bench | OLTP | `--workload=oltpbench_wikipedia` |
| YCSB | OLTP-Bench | Key-Value | `--workload=oltpbench_ycsb` |
| TATP | OLTP-Bench | OLTP | `--workload=oltpbench_tatp` |
| Voter | OLTP-Bench | OLTP | `--workload=oltpbench_voter` |
| TPC-H | Custom SQL | OLAP | `--workload=tpch` |
| JOB | Custom SQL | OLAP | `--workload=job` |

---

## Project Structure

```
OpAdviserPrivate/
├── autotune/                    # Core tuning library
│   ├── database/                # Database connectors (MySQL, PostgreSQL)
│   ├── optimizer/               # Optimization algorithms
│   │   ├── bo_optimizer.py      # Bayesian Optimization
│   │   ├── ddpg_optimizer.py    # DDPG (RL-based)
│   │   ├── ga_optimizer.py      # Genetic Algorithm
│   │   └── surrogate/           # Surrogate models (GP, RF, DDPG)
│   ├── pipleline/               # Main tuning pipeline
│   ├── transfer/                # Transfer learning (RGPE, workload mapping)
│   ├── selector/                # Knob selection (SHAP, FANOVA)
│   ├── utils/
│   │   └── resource_parser.py  # Resource data parsing utilities
│   ├── tuner.py                 # Main DBTuner class
│   ├── dbenv.py                 # Database environment
│   ├── dbenv_bench.py           # BenchEnv with resource prediction
│   └── knobs.py                 # Knob configuration handling
├── scripts/
│   ├── optimize.py              # Main entry point
│   ├── collect_resource_data.py # Resource data collection script
│   ├── train_resource_model.py  # Resource model training script
│   ├── evaluate_resource_model.py # Model evaluation utilities
│   ├── config.ini               # Default configuration
│   └── experiment/gen_knobs/    # Knob definition files
├── resource_data/               # Collected resource training data
├── resource_models/             # Trained resource prediction models
├── repo/                        # Historical tuning data (source knowledge)
├── logs/                        # Tuning logs
└── requirements.txt             # Python dependencies
```

---

## Resource Prediction Model Training

OpAdviser can predict CPU usage and Disk I/O for database configurations using trained Random Forest models. This enables fast evaluation without running actual benchmarks.

### Overview

The resource prediction models predict:
- **CPU Usage** (%)
- **Read I/O** (MB/s)
- **Write I/O** (MB/s)

These models achieve **<10% MAPE** (Mean Absolute Percentage Error) and can be used in `BenchEnv` for fast surrogate evaluation.

### Quick Start: Training Resource Models

#### Step 1: Collect Training Data

Collect resource data from actual benchmark runs:

```bash
# Set environment variables
export PYTHONPATH="."
export MYSQL_SOCK=/var/run/mysqld/mysqld.sock

# Collect 60 samples (~40 minutes)
python scripts/collect_resource_data.py \
    --num_samples 60 \
    --output_dir resource_data
```

**What this does:**
- Runs 60 benchmark iterations with random configurations
- Each iteration: 30s benchmark + 5s warmup (~40s total)
- Collects CPU, ReadIO, WriteIO metrics via `ResourceMonitor`
- Saves data to `resource_data/resource_data.json`

**Configuration:**
- Uses `mysql_cpu_io_dynamic_15.json` (15 dynamic knobs)
- `online_mode=True` (no MySQL restarts, faster)
- `workload_time=30s` (minimal benchmark time)

#### Step 2: Train Models

Train Random Forest models on the collected data:

```bash
python scripts/train_resource_model.py \
    --data_file resource_data/resource_data.json \
    --knob_config scripts/experiment/gen_knobs/mysql_cpu_io_dynamic_15.json \
    --knob_num 15 \
    --output_dir resource_models \
    --num_trees 100
```

**What this does:**
- Splits data: 60% train, 20% val, 20% test
- Trains 3 separate RF models (CPU, ReadIO, WriteIO)
- Evaluates MAPE on validation and test sets
- Saves models to `resource_models/resource_predictor.joblib`

**Expected Output:**
```
Training complete in X.X minutes
CPU MAPE:      X.XX% ✓
Read I/O MAPE: X.XX% ✓
Write I/O MAPE: X.XX% ✓
All models <10% MAPE: ✓ PASS
```

#### Step 3: Use Trained Models

The models are automatically loaded by `BenchEnv` when available:

```python
# In autotune/dbenv_bench.py, models are loaded automatically
# if resource_models/resource_predictor.joblib exists

# Models are used in get_states() to predict resources
external_metrics, internal_metrics, resource = env.get_states(knobs)
# resource[0] = predicted CPU
# resource[1] = predicted ReadIO
# resource[2] = predicted WriteIO
```

### Detailed Usage

#### Data Collection Options

```bash
# Custom number of samples
python scripts/collect_resource_data.py --num_samples 80

# Use existing config file
python scripts/collect_resource_data.py \
    --config scripts/config_cpu_io_opadviser_ultrafast.ini \
    --num_samples 60

# Custom output directory
python scripts/collect_resource_data.py \
    --output_dir my_resource_data \
    --num_samples 60
```

#### Training Options

```bash
# Adjust number of trees (more = better but slower)
python scripts/train_resource_model.py \
    --data_file resource_data/resource_data.json \
    --knob_config scripts/experiment/gen_knobs/mysql_cpu_io_dynamic_15.json \
    --knob_num 15 \
    --num_trees 200  # Default: 100

# Custom train/val/test split
python scripts/train_resource_model.py \
    --data_file resource_data/resource_data.json \
    --knob_config scripts/experiment/gen_knobs/mysql_cpu_io_dynamic_15.json \
    --knob_num 15 \
    --test_size 0.15 \
    --val_size 0.15
```

#### Evaluation

Evaluate trained models on new data:

```python
from scripts.evaluate_resource_model import evaluate_all_models, calculate_mape
import numpy as np

# Load test data
y_cpu_true = np.array([...])  # Actual CPU values
y_cpu_pred = np.array([...])   # Predicted CPU values
# ... same for read_io and write_io

# Evaluate
metrics = evaluate_all_models(
    y_cpu_true, y_cpu_pred,
    y_read_io_true, y_read_io_pred,
    y_write_io_true, y_write_io_pred
)

# Check MAPE
mape_cpu = calculate_mape(y_cpu_true, y_cpu_pred)
print(f"CPU MAPE: {mape_cpu:.2f}%")
```

### Model Architecture

- **Algorithm**: Random Forest with Instance Features
- **Input**: Database configuration (knobs) + Workload features
- **Output**: CPU usage, Read I/O, Write I/O
- **Workload Features**: One-hot encoded workload type + normalized thread count
- **Training Time**: ~10-15 minutes for 60 samples
- **Prediction Time**: <1ms per configuration

### File Structure

```
OpAdviserPrivate/
├── scripts/
│   ├── collect_resource_data.py      # Data collection script
│   ├── train_resource_model.py        # Training script
│   └── evaluate_resource_model.py     # Evaluation utilities
├── autotune/
│   ├── utils/
│   │   └── resource_parser.py         # Data parsing utilities
│   └── dbenv_bench.py                 # BenchEnv with resource prediction
├── resource_data/                      # Collected training data
│   └── resource_data.json
└── resource_models/                   # Trained models
    ├── resource_predictor.joblib      # Model file
    └── training_metadata.json         # Training metadata
```

### Troubleshooting

**Issue: MAPE > 10%**
- Collect more samples (try 80-100)
- Increase `num_trees` to 200
- Check data quality (filter invalid samples)

**Issue: Models not loading**
- Verify `resource_models/resource_predictor.joblib` exists
- Check file permissions
- Ensure models were trained with same knob config

**Issue: Poor predictions**
- Ensure training data covers configuration space
- Use stratified split by workload
- Check for data quality issues (outliers, missing values)

---

## Troubleshooting

### Environment Setup Issues

1. **Missing `PYTHONPATH` or `MYSQL_SOCK`**
   ```bash
   # Always set these before running
   export PYTHONPATH="."
   export MYSQL_SOCK=/var/run/mysqld/mysqld.sock
   ```

2. **MySQL authentication error** (`caching_sha2_password not supported`)
   ```bash
   # Install newer MySQL connector
   pip install mysql-connector-python
   ```

### Common Runtime Issues

1. **MySQL connection failed**
   - Verify MySQL is running: `service mysql status`
   - Check socket path: `ls -la /var/run/mysqld/mysqld.sock`
   - Start MySQL: `service mysql start`

2. **Benchmark runs instantly with 0 iterations or MAXINT values**
   - Check if sysbench supports `--warmup-time` (some versions don't)
   - Verify tables/data exist: `mysql -uroot -ppassword -e "SELECT COUNT(*) FROM sbrw.sbtest1;"`
   - Check for existing history file with max iterations already reached

3. **`IndexError: list index out of range` in `get_incumbents()`**
   - All benchmark iterations failed
   - Check MySQL logs: `tail -50 /var/log/mysql/error.log`
   - Delete corrupted history file: `rm repo/history_<task_id>.json`

4. **`psutil.AccessDenied` for MySQL process**
   - Non-fatal warning (resource monitoring issue)
   - MySQL process running as different user
   - The optimization will still work

5. **Knobs not being applied in online mode**
   - Some knobs require MySQL restart (static variables)
   - Use `online_mode = False` for full knob support
   - Or use dynamic-only knob configs (`mysql_dynamic_10.json`, `mysql_cpu_io_dynamic_15.json`)

6. **History file already exists with max iterations**
   - Delete or rename: `rm repo/history_<task_id>.json`
   - Or increase `max_runs` in config file

### Logs

Check logs in:
- `logs/` directory
- `log/tune_database_<timestamp>.log`
- MySQL error log: `/var/log/mysql/error.log`

### Verifying Your Setup

Run this quick test to verify everything is working:

```bash
# Test MySQL connection
mysql -uroot -ppassword -e "SELECT 1;"

# Test sysbench
sysbench oltp_read_write \
    --mysql-host=localhost \
    --mysql-user=root \
    --mysql-password=password \
    --mysql-db=sbrw \
    --mysql-socket=$MYSQL_SOCK \
    --tables=1 \
    --table-size=1000 \
    --time=5 \
    run

# Test OpAdviser setup
cd /root/OpAdviser
export PYTHONPATH="."
python -c "
from autotune.database.mysqldb import MysqlDB
from autotune.utils.config import parse_args
args_db, _ = parse_args('scripts/config_ground_truth_ultrafast.ini')
db = MysqlDB(args_db)
print(f'MySQL PID: {db.pid}')
print('Setup OK!')
"
```

---

## Citation

If you use OpAdviser in your research, please cite the relevant papers.

---

## License

MIT License - See [LICENSE](LICENSE) file.
