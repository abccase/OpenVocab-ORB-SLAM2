# P05 Baseline Noninferiority V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, register, execute, and independently verify the approved 15-pair-per-sequence P05 baseline noninferiority gate without committing generated run or agent artifacts.

**Architecture:** A tracked canonical protocol manifest freezes study identity, order, margins, and statistical settings. Focused Python modules validate that protocol and compute paired bootstrap bounds; thin command-line tools build the legacy oracle, run the frozen matrix, audit file access, and verify every deterministic and statistical gate. Existing ORB-SLAM2 condition runners remain the sole artifact producers and gain only an optional formal-registration identity so historical studies keep their behavior.

**Tech Stack:** Python 3 standard library, NumPy `Generator(PCG64)`, `unittest`, Git worktrees, CMake/CTest, Bash build recipe, `strace`, existing ORB-SLAM2 RGB-D executables.

**Spec:** `docs/design/P05_BASELINE_NONINFERIORITY_V2.md`

## Global Constraints

- Study ID is exactly `ovorb2_p05_baseline_noninferiority_v2`; old `equivalence` attempts remain invalid pilot evidence and are never pooled.
- Use the six datasets in `config/EXPERIMENT_MANIFEST.yaml` and repetition IDs `23011` through `23025`, yielding exactly 90 paired blocks and 180 valid statistical runs.
- Freeze within-block order with `random.Random(23010)`; every sequence has a 7/8 split of first positions and formal conditions run sequentially.
- Oracle producer is exactly `58014b7c1f2b73427b67b4e80a8cf334127f48ea`; candidate policy is exactly `HEAD_AT_REGISTRATION`.
- Use 100,000 paired resamples; recreate `numpy.random.Generator(numpy.random.PCG64(23010))` for each sequence and reuse its index matrix for both metrics.
- Quantiles use `numpy.quantile(..., method="linear")`; pass only when `pose_lower_95 >= -0.10` and `ate_ratio_upper_95 <= 1.25` for all six sequences.
- ATE is rigid SE(3) aligned translation RMSE with no scale and one-to-one timestamp association at maximum difference `0.02` seconds.
- Candidate baseline must have zero semantic access, zero removals, equal raw/used keypoints, baseline state, zero cache/policy time, and no cache root or identity.
- Run one registered candidate `strace -ff -e trace=file` probe per sequence before the statistical matrix; audit runs never enter numerical comparison.
- A duplicate valid expected-producer attempt, identity mismatch, corrupt artifact, missing run, incomplete telemetry, or hard-gate failure fails closed.
- Formal results are executed once. A statistical failure is preserved and cannot be rerun under this study ID to seek a favorable result.
- Git includes only reproducibility-critical config, source, tests, design, and build instructions. It excludes reports, registries, trajectories, logs, caches, datasets, models, build trees, binaries, maps, temporary worktrees, and all agent artifacts. Do not push.

---

## File responsibility map

- `semantic_py/openvocab_slam/p05_protocol.py`: canonical protocol parsing, exact frozen-order reconstruction, hash/identity validation, and batch-registration schema validation.
- `semantic_py/openvocab_slam/p05_noninferiority.py`: numeric input validation and deterministic paired-bootstrap calculations only; it performs no filesystem discovery.
- `config/P05_BASELINE_NONINFERIORITY_V2.json`: tracked canonical data defining all 90 blocks and their 180-condition execution order.
- `docs/design/amendments/2026-08-24-p05-baseline-noninferiority-v2.md`: approved protocol-change record and scope boundary.
- `tools/build_p05_oracle.py`: isolated detached-worktree build and atomic ignored build manifest.
- `tools/audit_p05_baseline_access.py`: parse `strace` file events and reject forbidden semantic inputs.
- `tools/run_p05_baseline_noninferiority.py`: pre-register identities, run audits, execute/resume the frozen sequential matrix, and reject ambiguous attempts.
- `tools/verify_baseline_noninferiority.py`: independently locate expected runs, revalidate artifacts and telemetry, measure ATE/pose fraction, compute statistics, and atomically publish the ignored report.
- `tools/run_orb_tum.py`: preserve existing studies while accepting an optional immutable formal identity in oracle and candidate manifests.
- `tests/python/test_p05_protocol.py`: protocol order, schema, hash, and batch-registration tests.
- `tests/python/test_p05_noninferiority.py`: bootstrap reproducibility, pairing, domain, and boundary tests.
- `tests/python/test_build_p05_oracle.py`: builder command construction and manifest identity tests without compiling ORB-SLAM2.
- `tests/python/test_p05_access_audit.py`: allowed/forbidden syscall parsing tests.
- `tests/python/test_run_p05_baseline_noninferiority.py`: runner registration, order, resume, duplicate, and stale-producer tests.
- `tests/python/test_verify_baseline_noninferiority.py`: synthetic run-tree hard-gate and atomic-report tests.

### Task 1: Freeze and validate the approved protocol

**Files:**
- Create: `semantic_py/openvocab_slam/p05_protocol.py`
- Create: `config/P05_BASELINE_NONINFERIORITY_V2.json`
- Create: `docs/design/amendments/2026-08-24-p05-baseline-noninferiority-v2.md`
- Create: `tests/python/test_p05_protocol.py`

**Interfaces:**
- Consumes: `config/EXPERIMENT_MANIFEST.yaml` as JSON and the constants copied in Global Constraints.
- Produces: `expected_blocks() -> list[dict[str, object]]`, `load_protocol(path: Path, experiment_path: Path) -> dict[str, object]`, `sha256_file(path: Path) -> str`, and `validate_batch_registration(registration: Mapping[str, object], protocol: Mapping[str, object], protocol_sha256: str) -> None`.

- [ ] **Step 1: Write failing order and schema tests**

```python
import json
import tempfile
import unittest
from pathlib import Path

from semantic_py.openvocab_slam.p05_protocol import expected_blocks, load_protocol

ROOT = Path(__file__).resolve().parents[2]


class P05ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_tracked_protocol_is_exact_and_balanced(self) -> None:
        protocol = load_protocol(
            ROOT / "config/P05_BASELINE_NONINFERIORITY_V2.json",
            ROOT / "config/EXPERIMENT_MANIFEST.yaml",
        )
        self.assertEqual(protocol["blocks"], expected_blocks())
        self.assertEqual(len(protocol["blocks"]), 90)
        self.assertEqual(sum(len(block["execution_order"]) for block in protocol["blocks"]), 180)
        for sequence_id in protocol["sequence_ids"]:
            first = [block["execution_order"][0] for block in protocol["blocks"]
                     if block["sequence_id"] == sequence_id]
            self.assertEqual(sorted(first.count(name) for name in ("oracle", "candidate")), [7, 8])

    def test_wrong_margin_is_rejected(self) -> None:
        source = json.loads((ROOT / "config/P05_BASELINE_NONINFERIORITY_V2.json").read_text())
        source["statistics"]["pose_delta_lower_margin"] = -0.09
        path = self.temp_path / "protocol.json"
        path.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pose margin"):
            load_protocol(path, ROOT / "config/EXPERIMENT_MANIFEST.yaml")
```

Add table-driven mutations for these exact fields and expected error fragments:

```python
def test_frozen_fields_reject_mutation(self) -> None:
    cases = (
        (("study_id",), "wrong", "study"),
        (("oracle", "producer_commit"), "0" * 40, "oracle"),
        (("candidate", "producer_policy"), "CURRENT_HEAD", "candidate policy"),
        (("statistics", "resamples"), 99999, "resample"),
        (("statistics", "seed"), 23011, "bootstrap seed"),
        (("statistics", "quantile_method"), "nearest", "quantile"),
    )
    for keys, replacement, message in cases:
        with self.subTest(keys=keys):
            value = json.loads((ROOT / "config/P05_BASELINE_NONINFERIORITY_V2.json").read_text())
            target = value
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = replacement
            path = self.temp_path / ("-".join(keys) + ".json")
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_protocol(path, ROOT / "config/EXPERIMENT_MANIFEST.yaml")
```

Add individual tests for a changed repetition range, duplicate block, changed execution order, and dataset-order mismatch because those mutations touch arrays rather than scalar paths.

- [ ] **Step 2: Run the focused test and confirm the missing module failure**

Run: `python3 -m unittest tests.python.test_p05_protocol -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_py.openvocab_slam.p05_protocol'`.

- [ ] **Step 3: Implement deterministic order reconstruction and strict loading**

```python
SEQUENCE_IDS = (
    "fr1_desk", "fr1_room", "fr3_sitting_xyz",
    "fr3_sitting_halfsphere", "fr3_walking_xyz",
    "fr3_walking_halfsphere",
)
REPETITION_IDS = tuple(range(23011, 23026))
IMPLEMENTATIONS = ("oracle", "candidate")


def expected_blocks() -> list[dict[str, object]]:
    rng = random.Random(23010)
    blocks: list[dict[str, object]] = []
    for sequence_id in SEQUENCE_IDS:
        extra_first = rng.choice(IMPLEMENTATIONS)
        other = "candidate" if extra_first == "oracle" else "oracle"
        first_positions = [extra_first] * 8 + [other] * 7
        rng.shuffle(first_positions)
        for repetition_id, first in zip(REPETITION_IDS, first_positions, strict=True):
            second = "candidate" if first == "oracle" else "oracle"
            blocks.append({
                "block_id": f"{sequence_id}-rep-{repetition_id}",
                "sequence_id": sequence_id,
                "repetition_id": repetition_id,
                "execution_order": [first, second],
            })
    return blocks
```

In `load_protocol`, parse both files as JSON objects, reject any key values that differ from the constants, require `blocks == expected_blocks()`, require the protocol dataset IDs to exactly match the experiment manifest in order, and require the exact metric/statistics objects below:

```python
EXPECTED_STATISTICS = {
    "algorithm": "paired_bootstrap",
    "generator": "PCG64",
    "seed": 23010,
    "resamples": 100000,
    "generator_scope": "reinitialize_per_sequence",
    "shared_index_matrix_for_metrics": True,
    "quantile_method": "linear",
    "confidence_level_one_sided": 0.95,
    "pose_delta_lower_margin": -0.10,
    "ate_geometric_ratio_upper_margin": 1.25,
}
EXPECTED_METRICS = {
    "pose_delta": "candidate_valid_pose_fraction_minus_oracle",
    "ate_log_ratio": "log_candidate_ate_over_oracle_ate",
    "trajectory_alignment": "SE3",
    "scale_alignment": False,
    "timestamp_association_max_seconds": 0.02,
}
```

Implement `sha256_file` with 1 MiB streaming blocks. Implement `validate_batch_registration` to require schema version 1, protocol hash, exact study ID, oracle commit, a 40-character lowercase candidate commit, 64-character lowercase executable hashes, oracle build-manifest hash, candidate `HEAD` resolution policy, and state `REGISTERED`.

- [ ] **Step 4: Add the canonical tracked protocol and amendment**

Create canonical JSON with `indent=2`, sorted keys, and a trailing newline. It must contain schema version 1, the exact study/oracle/candidate identities, `sequence_ids`, `repetition_ids`, `metrics`, `statistics`, `experiment_manifest: "config/EXPERIMENT_MANIFEST.yaml"`, and the complete value returned by `expected_blocks()`.

The amendment must state: approval token `APPROVE_P05_V2_SPEC`; old five-repeat gate and failure reason; new 15-pair margins and bootstrap settings; preserved-invalid old results; P05-only scope; unchanged P08 primary study; no result-driven reruns; and Git exclusions.

- [ ] **Step 5: Run protocol tests and repository config checks**

Run: `python3 -m unittest tests.python.test_p05_protocol -v`

Expected: all protocol tests PASS.

Run: `python3 -m json.tool config/P05_BASELINE_NONINFERIORITY_V2.json >/dev/null`

Expected: exit 0.

- [ ] **Step 6: Commit the independently reviewable protocol layer**

```bash
git add config/P05_BASELINE_NONINFERIORITY_V2.json docs/design/amendments/2026-08-24-p05-baseline-noninferiority-v2.md semantic_py/openvocab_slam/p05_protocol.py tests/python/test_p05_protocol.py
git diff --cached --check
git commit -m "feat: freeze P05 noninferiority protocol"
```

### Task 2: Implement paired noninferiority statistics

**Files:**
- Create: `semantic_py/openvocab_slam/p05_noninferiority.py`
- Create: `tests/python/test_p05_noninferiority.py`

**Interfaces:**
- Consumes: 15 oracle and 15 candidate mappings with `repetition_id: int`, `valid_pose_fraction: float`, and `ate_rmse_m: float`; statistics mapping validated by Task 1.
- Produces: `bootstrap_indices(pair_count: int, resamples: int, seed: int) -> numpy.ndarray`, `evaluate_sequence(oracle_rows: Sequence[Mapping[str, object]], candidate_rows: Sequence[Mapping[str, object]], statistics_config: Mapping[str, object]) -> dict[str, object]`, and `evaluate_study(rows_by_sequence: Mapping[str, tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]], sequence_ids: Sequence[str], statistics_config: Mapping[str, object]) -> dict[str, object]`.

- [ ] **Step 1: Write failing pairing, reproducibility, and boundary tests**

```python
def rows(pose_delta: float, ate_ratio: float):
    oracle = [{"repetition_id": value, "valid_pose_fraction": 0.8,
               "ate_rmse_m": 0.1} for value in range(23011, 23026)]
    candidate = [{"repetition_id": value, "valid_pose_fraction": 0.8 + pose_delta,
                  "ate_rmse_m": 0.1 * ate_ratio} for value in range(23011, 23026)]
    return oracle, candidate


def test_exact_boundaries_pass(self) -> None:
    oracle, candidate = rows(-0.10, 1.25)
    result = evaluate_sequence(oracle, candidate, self.statistics)
    self.assertTrue(result["pose_pass"])
    self.assertTrue(result["ate_pass"])


def test_immediately_outside_boundaries_fails(self) -> None:
    oracle, candidate = rows(-0.100001, 1.250001)
    result = evaluate_sequence(oracle, candidate, self.statistics)
    self.assertFalse(result["pose_pass"])
    self.assertFalse(result["ate_pass"])
```

Also assert identical results across two invocations, exact repetition pairing, rejection of 14 pairs, duplicates, mismatched repetitions, pose values outside `[0,1]`, zero/negative/nonfinite ATE, and that a mocked `bootstrap_indices` return is used for both metric arrays.

- [ ] **Step 2: Run the focused test and confirm the missing module failure**

Run: `python3 -m unittest tests.python.test_p05_noninferiority -v`

Expected: FAIL with `ModuleNotFoundError` for `p05_noninferiority`.

- [ ] **Step 3: Implement validation and the shared-index bootstrap**

```python
def bootstrap_indices(pair_count: int, resamples: int, seed: int) -> np.ndarray:
    if pair_count != 15 or resamples != 100000 or seed != 23010:
        raise ValueError("bootstrap configuration differs from frozen protocol")
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.integers(0, pair_count, size=(resamples, pair_count), dtype=np.int16)


def evaluate_sequence(oracle_rows, candidate_rows, statistics_config):
    if dict(statistics_config) != EXPECTED_STATISTICS:
        raise ValueError("statistics configuration differs from frozen protocol")
    oracle = _validated_by_repetition(oracle_rows, "oracle")
    candidate = _validated_by_repetition(candidate_rows, "candidate")
    if tuple(oracle) != tuple(candidate):
        raise ValueError("oracle and candidate repetition IDs are not paired")
    repetitions = tuple(oracle)
    pose = np.asarray([candidate[key][0] - oracle[key][0] for key in repetitions])
    ate_log = np.asarray([math.log(candidate[key][1] / oracle[key][1]) for key in repetitions])
    indices = bootstrap_indices(15, 100000, 23010)
    pose_means = pose[indices].mean(axis=1)
    ate_log_means = ate_log[indices].mean(axis=1)
    pose_lower = float(np.quantile(pose_means, 0.05, method="linear"))
    ate_upper = float(math.exp(np.quantile(ate_log_means, 0.95, method="linear")))
    return {
        "valid": pose_lower >= -0.10 and ate_upper <= 1.25,
        "pose_pass": pose_lower >= -0.10,
        "ate_pass": ate_upper <= 1.25,
        "pose_estimate": float(pose.mean()),
        "pose_lower_95": pose_lower,
        "pose_two_sided_95": [
            float(np.quantile(pose_means, 0.025, method="linear")),
            float(np.quantile(pose_means, 0.975, method="linear")),
        ],
        "ate_geometric_ratio_estimate": float(math.exp(ate_log.mean())),
        "ate_ratio_upper_95": ate_upper,
        "ate_ratio_two_sided_95": [
            float(math.exp(np.quantile(ate_log_means, 0.025, method="linear"))),
            float(math.exp(np.quantile(ate_log_means, 0.975, method="linear"))),
        ],
        "paired_values": [
            {"repetition_id": key,
             "oracle_valid_pose_fraction": oracle[key][0],
             "candidate_valid_pose_fraction": candidate[key][0],
             "pose_delta": candidate[key][0] - oracle[key][0],
             "oracle_ate_rmse_m": oracle[key][1],
             "candidate_ate_rmse_m": candidate[key][1],
             "ate_log_ratio": math.log(candidate[key][1] / oracle[key][1])}
            for key in repetitions
        ],
        "margins": {"pose_delta_lower": -0.10, "ate_ratio_upper": 1.25},
        "bootstrap": {"generator": "PCG64", "seed": 23010,
                      "resamples": 100000, "quantile_method": "linear",
                      "numpy_version": np.__version__},
        "unpaired_summaries": _unpaired_summaries(oracle, candidate),
    }


def evaluate_study(rows_by_sequence, sequence_ids, statistics_config):
    if tuple(rows_by_sequence) != tuple(sequence_ids):
        raise ValueError("study sequences differ from frozen protocol")
    sequences = {
        sequence_id: evaluate_sequence(*rows_by_sequence[sequence_id], statistics_config)
        for sequence_id in sequence_ids
    }
    return {"valid": all(value["valid"] for value in sequences.values()),
            "sequences": sequences}
```

Implement summaries with one shared numeric helper and direct inclusive comparisons without rounding:

```python
def _summary(values: Sequence[float]) -> dict[str, float]:
    return {"mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
            "minimum": float(min(values)), "maximum": float(max(values))}


def _unpaired_summaries(oracle, candidate) -> dict[str, object]:
    return {
        "oracle": {
            "valid_pose_fraction": _summary([value[0] for value in oracle.values()]),
            "ate_rmse_m": _summary([value[1] for value in oracle.values()]),
        },
        "candidate": {
            "valid_pose_fraction": _summary([value[0] for value in candidate.values()]),
            "ate_rmse_m": _summary([value[1] for value in candidate.values()]),
        },
    }
```

- [ ] **Step 4: Run focused and existing equivalence tests**

Run: `python3 -m unittest tests.python.test_p05_noninferiority tests.python.test_baseline_equivalence -v`

Expected: all tests PASS and the historical verifier behavior remains unchanged.

- [ ] **Step 5: Commit the pure statistics layer**

```bash
git add semantic_py/openvocab_slam/p05_noninferiority.py tests/python/test_p05_noninferiority.py
git diff --cached --check
git commit -m "feat: add paired baseline noninferiority statistics"
```

### Task 3: Add immutable formal identities to existing run producers

**Files:**
- Modify: `tools/run_orb_tum.py:351-738`
- Modify: `tests/python/test_run_orb_tum.py`

**Interfaces:**
- Consumes: optional `formal_identity: Mapping[str, object] | None` containing `study_id`, `block_id`, `implementation`, `protocol_manifest_sha256`, and either `build_manifest_sha256` for oracle or `candidate_registration_commit` for candidate.
- Produces: backward-compatible `run_baseline_condition(..., formal_identity: Mapping[str, object] | None = None) -> RunResult` and `run_ov_condition(..., formal_identity: Mapping[str, object] | None = None) -> RunResult`; formal identity becomes part of resume matching and the immutable run manifest.

- [ ] **Step 1: Write failing formal-identity and stale-resume tests**

```python
def test_formal_identity_is_persisted_and_selects_resume(self) -> None:
    first_identity = {
        "study_id": "ovorb2_p05_baseline_noninferiority_v2",
        "block_id": "fr1_desk-rep-23011",
        "implementation": "candidate",
        "protocol_manifest_sha256": "a" * 64,
        "candidate_registration_commit": "b" * 40,
    }
    first = run_ov_condition(self.condition, formal_identity=first_identity, **self.arguments)
    resumed = run_ov_condition(self.condition, formal_identity=first_identity, **self.arguments)
    changed = dict(first_identity, protocol_manifest_sha256="c" * 64)
    replacement = run_ov_condition(self.condition, formal_identity=changed, **self.arguments)
    self.assertEqual(first.run_dir, resumed.run_dir)
    self.assertNotEqual(first.run_dir, replacement.run_dir)
    manifest = json.loads((first.run_dir / "run_manifest.json").read_text())
    self.assertEqual(manifest["formal_identity"], first_identity)
```

Add equivalent legacy-oracle coverage and assert calls omitting `formal_identity` keep schema, run directory, and resume behavior used by existing tests.

- [ ] **Step 2: Run the focused tests and observe the signature failure**

Run: `python3 -m unittest tests.python.test_run_orb_tum -v`

Expected: FAIL with `unexpected keyword argument 'formal_identity'`.

- [ ] **Step 3: Implement exact validation and resume binding**

```python
def _validated_formal_identity(value: Mapping[str, object] | None) -> dict[str, object] | None:
    if value is None:
        return None
    copied = dict(value)
    required = {"study_id", "block_id", "implementation", "protocol_manifest_sha256"}
    if not required <= copied.keys():
        raise ValueError("formal identity is incomplete")
    if copied["study_id"] != "ovorb2_p05_baseline_noninferiority_v2":
        raise ValueError("formal identity has wrong study")
    if copied["implementation"] not in {"oracle", "candidate"}:
        raise ValueError("formal identity has wrong implementation")
    if not _is_sha256(copied["protocol_manifest_sha256"]):
        raise ValueError("formal identity has invalid protocol hash")
    json.dumps(copied, sort_keys=True, allow_nan=False)
    return copied
```

Validate before creating a run directory. Add the validated value to both `base_manifest` and candidate `registration_identity`; pass it into `_completed_attempt` and `_completed_ov_attempt` comparisons. Extend accepted study names only with exact `ovorb2_p05_baseline_noninferiority_v2`; do not broaden them to arbitrary strings.

- [ ] **Step 4: Run producer tests and compile the module**

Run: `python3 -m unittest tests.python.test_run_orb_tum -v`

Expected: all tests PASS.

Run: `python3 -m py_compile tools/run_orb_tum.py`

Expected: exit 0.

- [ ] **Step 5: Commit the manifest/resume extension**

```bash
git add tools/run_orb_tum.py tests/python/test_run_orb_tum.py
git diff --cached --check
git commit -m "feat: bind P05 runs to frozen formal identities"
```

### Task 4: Build and identify the legacy oracle reproducibly

**Files:**
- Create: `tools/build_p05_oracle.py`
- Create: `tests/python/test_build_p05_oracle.py`

**Interfaces:**
- Consumes: repository path, explicit ignored build root, exact oracle commit, `ORB_SLAM2_BUILD_JOBS`, and tracked `build.sh` from the detached source.
- Produces: `oracle_commands(repository: Path, build_root: Path, commit: str, jobs: int) -> list[list[str]]`, `build_oracle(repository: Path, build_root: Path, commit: str, jobs: int, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> dict[str, object]`, and atomic `<build-root>/oracle_build_manifest.json` containing source/build/tool identities and executable SHA-256.

- [ ] **Step 1: Write failing builder tests using a recording runner**

```python
class RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append([str(value) for value in command])
        return subprocess.CompletedProcess(command, 0, stdout="recorded\n", stderr="")


def test_worktree_is_detached_at_frozen_commit(self) -> None:
    runner = RecordingRunner()
    commands = oracle_commands(Path("/repo"), Path("/ignored/oracle"), ORACLE_COMMIT, 2)
    self.assertEqual(commands[0], ["git", "-C", "/repo", "worktree", "add", "--detach",
                                   "/ignored/oracle/source", ORACLE_COMMIT])
    self.assertEqual(commands[1], ["git", "-C", "/ignored/oracle/source",
                                   "status", "--porcelain", "--untracked-files=no"])
    self.assertIn(["bash", "/ignored/oracle/source/build.sh"], commands)
```

Also test rejection of a build root inside the detached source, existing nonempty source directory, dirty status, wrong resolved commit, missing executable, and wrong final executable hash shape. Test atomic publication by asserting no manifest exists when a simulated build command raises.

- [ ] **Step 2: Run the focused test and confirm the missing module failure**

Run: `python3 -m unittest tests.python.test_build_p05_oracle -v`

Expected: FAIL with `ModuleNotFoundError` for `tools.build_p05_oracle`.

- [ ] **Step 3: Implement detached build, version capture, and atomic manifest**

```python
ORACLE_COMMIT = "58014b7c1f2b73427b67b4e80a8cf334127f48ea"


def oracle_commands(repository: Path, build_root: Path, commit: str, jobs: int):
    source = build_root / "source"
    return [
        ["git", "-C", str(repository), "worktree", "add", "--detach", str(source), commit],
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        ["bash", str(source / "build.sh")],
    ]
```

Run the build with environment `ORB_SLAM2_BUILD_JOBS=<jobs>`. Capture `cmake --version`, `${CXX:-c++} --version`, `pkg-config --modversion opencv4`, `pkg-config --modversion eigen3`, and `openssl version`; reject failed identity probes instead of recording unknown values. Require executable `Examples/RGB-D/rgbd_tum` and atomically write schema version, UTC time, source commit, repository path, build root, build script hash, configure arguments (`Release`, viewer off, testing on), job count, tool versions, executable absolute path/hash/size, and worktree clean state. Fsync the file and parent directory.

- [ ] **Step 4: Run builder tests and CLI help**

Run: `python3 -m unittest tests.python.test_build_p05_oracle -v`

Expected: all tests PASS.

Run: `python3 tools/build_p05_oracle.py --help >/dev/null`

Expected: exit 0.

- [ ] **Step 5: Commit the reproducible builder**

```bash
git add tools/build_p05_oracle.py tests/python/test_build_p05_oracle.py
git diff --cached --check
git commit -m "feat: add reproducible P05 oracle builder"
```

### Task 5: Implement file-access auditing and the formal matrix runner

**Files:**
- Create: `tools/audit_p05_baseline_access.py`
- Create: `tools/run_p05_baseline_noninferiority.py`
- Create: `tests/python/test_p05_access_audit.py`
- Create: `tests/python/test_run_p05_baseline_noninferiority.py`

**Interfaces:**
- Consumes: Task 1 protocol/batch validators, Task 3 run functions, Task 4 oracle manifest, candidate executable, frozen data manifests, ignored run root, and ignored registry.
- Produces: `audit_trace(trace_paths: Sequence[Path], forbidden_roots: Sequence[Path], forbidden_files: Sequence[Path], cwd: Path = Path("/")) -> dict[str, object]`; `register_batch(protocol: Mapping[str, object], protocol_path: Path, repository_root: Path, oracle_build_manifest_path: Path, candidate_executable: Path, registration_path: Path, registry_path: Path) -> dict[str, object]`; `expected_conditions(protocol: Mapping[str, object]) -> list[dict[str, object]]`; `validate_resume(run_root: Path, registration: Mapping[str, object], protocol: Mapping[str, object]) -> dict[str, Path]`; and CLI modes `--validate-only`, `--audit-only`, `--register-only`, and default sequential execution. Audit CLI accepts repeated `--forbidden-root` and `--forbidden-file` arguments.

- [ ] **Step 1: Write failing access parser tests**

```python
def test_forbidden_cache_open_is_rejected(self) -> None:
    trace = self.root / "trace.123"
    trace.write_text('openat(AT_FDCWD, "/repo/cache/semantic/v1/index", O_RDONLY) = 3\n')
    with self.assertRaisesRegex(ValueError, "forbidden file access"):
        audit_trace([trace], [Path("/repo/cache/semantic"), Path("/repo/cache/dynamic")], [])


def test_unrelated_dataset_and_library_access_is_allowed(self) -> None:
    trace = self.root / "trace.124"
    trace.write_text('openat(AT_FDCWD, "/repo/data/tum/rgb.png", O_RDONLY) = 3\n')
    result = audit_trace([trace], [Path("/repo/cache/semantic")], [Path("/repo/config/PROMPTS.yaml")])
    self.assertEqual(result["forbidden_accesses"], [])
```

Test quoted escaped paths, unfinished/resumed syscalls, multiple `-ff` files, prompt config, semantic-model config, dynamic identity config, relative paths resolved from the registered working directory, and failed syscall attempts (attempted access still fails).

- [ ] **Step 2: Write failing registration/order/resume tests**

```python
def test_expected_conditions_preserve_all_frozen_within_block_orders(self) -> None:
    conditions = expected_conditions(self.protocol)
    self.assertEqual(len(conditions), 180)
    self.assertEqual(conditions[:2], [
        {"block_id": self.protocol["blocks"][0]["block_id"],
         "sequence_id": self.protocol["blocks"][0]["sequence_id"],
         "repetition_id": self.protocol["blocks"][0]["repetition_id"],
         "implementation": self.protocol["blocks"][0]["execution_order"][0]},
        {"block_id": self.protocol["blocks"][0]["block_id"],
         "sequence_id": self.protocol["blocks"][0]["sequence_id"],
         "repetition_id": self.protocol["blocks"][0]["repetition_id"],
         "implementation": self.protocol["blocks"][0]["execution_order"][1]},
    ])


def test_duplicate_valid_expected_attempt_fails_resume(self) -> None:
    self.make_attempt("attempt-001", producer=self.candidate_commit, valid=True)
    self.make_attempt("attempt-002", producer=self.candidate_commit, valid=True)
    with self.assertRaisesRegex(ValueError, "duplicate valid attempts"):
        validate_resume(self.run_root, self.registration, self.protocol)
```

Also test wrong study, block, producer, executable hash, and protocol hash; a missing attempt is returned as pending; a stale-producer attempt is ignored; and a completed expected run is reused. `register_batch` must reject dirty tracked state, detached/wrong candidate `HEAD`, unvalidated oracle manifest, changed executable hashes, an existing different registration, or registration after any formal attempt exists.

- [ ] **Step 3: Run both focused suites and confirm missing-module failures**

Run: `python3 -m unittest tests.python.test_p05_access_audit tests.python.test_run_p05_baseline_noninferiority -v`

Expected: FAIL with missing audit/runner modules.

- [ ] **Step 4: Implement strict trace parsing and atomic audit reports**

```python
FILE_CALL = re.compile(r'^(?:\[pid\s+\d+\]\s+)?(?:open|openat|openat2|stat|lstat|access|readlink|newfstatat)\([^\"]*\"((?:\\.|[^\"])*)\"')


def audit_trace(trace_paths, forbidden_roots, forbidden_files, cwd=Path("/")):
    forbidden_root_values = tuple(path.resolve() for path in forbidden_roots)
    forbidden_file_values = {path.resolve() for path in forbidden_files}
    accesses = []
    for trace_path in trace_paths:
        for line_number, line in enumerate(trace_path.read_text(errors="strict").splitlines(), 1):
            match = FILE_CALL.search(line)
            if match is None:
                continue
            decoded = bytes(match.group(1), "utf-8").decode("unicode_escape")
            path = Path(decoded)
            resolved = (path if path.is_absolute() else cwd / path).resolve(strict=False)
            if resolved in forbidden_file_values or any(
                    resolved == root or root in resolved.parents for root in forbidden_root_values):
                accesses.append({"trace": str(trace_path), "line": line_number,
                                 "path": str(resolved), "syscall": line})
    if accesses:
        raise ValueError(f"forbidden file access: {accesses}")
    return {"schema_version": 1, "valid": True, "forbidden_accesses": [],
            "trace_files": [str(path) for path in trace_paths]}
```

Publish each ignored audit report atomically with trace hashes, sequence, candidate producer/executable, protocol hash, exact forbidden inputs, command, and UTC completion time.

- [ ] **Step 5: Implement immutable registration and sequential execution**

```python
def expected_conditions(protocol):
    result = []
    for block in protocol["blocks"]:
        for implementation in block["execution_order"]:
            result.append({
                "block_id": block["block_id"],
                "sequence_id": block["sequence_id"],
                "repetition_id": block["repetition_id"],
                "implementation": implementation,
            })
    if len(result) != 180:
        raise ValueError("formal matrix must contain exactly 180 conditions")
    return result
```

`register_batch` resolves clean candidate `HEAD`, hashes both executables and the oracle manifest, verifies the oracle commit/hash, captures compatibility/vocabulary/settings/association/dataset/source-tree/experiment/protocol identities, writes `batch_registration.json` atomically, and appends the same complete record once to the ignored registry. It refuses replacement with differing content.

For each condition, construct the exact formal identity and call `run_baseline_condition` for oracle or `run_ov_condition(mode="baseline")` for candidate. Use separate roots `<run-root>/oracle` and `<run-root>/candidate`, pass no semantic cache values, run synchronously, and stop immediately on an invalid result. Before executing, require six passing access-audit reports bound to the registered candidate. Resume only after scanning all expected condition roots; fail on duplicate expected valid attempts instead of selecting one.

- [ ] **Step 6: Run focused suites plus legacy producer tests**

Run: `python3 -m unittest tests.python.test_p05_access_audit tests.python.test_run_p05_baseline_noninferiority tests.python.test_run_orb_tum -v`

Expected: all tests PASS.

Run: `python3 tools/run_p05_baseline_noninferiority.py --help >/dev/null`

Expected: exit 0.

- [ ] **Step 7: Commit audit and runner tools**

```bash
git add tools/audit_p05_baseline_access.py tools/run_p05_baseline_noninferiority.py tests/python/test_p05_access_audit.py tests/python/test_run_p05_baseline_noninferiority.py
git diff --cached --check
git commit -m "feat: add P05 formal matrix runner and access audit"
```

### Task 6: Implement the independent hard-gate and statistical verifier

**Files:**
- Create: `tools/verify_baseline_noninferiority.py`
- Create: `tests/python/test_verify_baseline_noninferiority.py`

**Interfaces:**
- Consumes: Task 1 protocol/batch validators, Task 2 `evaluate_study`, legacy JSONL oracle telemetry, candidate CSV telemetry, run manifests/artifacts, six access reports, experiment/data manifests, source tree, and ground-truth trajectories.
- Produces: `build_report(protocol_path: Path, experiment_path: Path, registration_path: Path, oracle_root: Path, candidate_root: Path, audit_root: Path, data_root: Path, data_manifest_root: Path, repository_root: Path) -> dict[str, object]` and an atomic CLI output whose exit code is zero only for a complete pass.

- [ ] **Step 1: Write failing synthetic-tree hard-gate tests**

```python
def test_complete_synthetic_study_passes_and_records_tool_identity(self) -> None:
    self.make_complete_study(pose_delta=0.0, ate_ratio=1.0)
    report = build_report(**self.paths)
    self.assertTrue(report["valid"])
    self.assertEqual(report["deterministic_gates"]["expected_run_count"], 180)
    self.assertEqual(report["statistics"]["resamples"], 100000)
    self.assertEqual(len(report["sequences"]), 6)
    self.assertRegex(report["verifier_sha256"], r"^[0-9a-f]{64}$")


def test_candidate_semantic_access_blocks_statistics(self) -> None:
    self.make_complete_study(pose_delta=0.0, ate_ratio=1.0)
    self.mutate_candidate_telemetry("fr1_desk", 23011, "semantic_accessed", "1")
    with self.assertRaisesRegex(ValueError, "baseline accessed semantic state"):
        build_report(**self.paths)
```

The fixture must create parseable three-pose trajectories, ground truth, complete telemetry for every expected frame, matching artifact hashes, manifests for all 180 runs, a valid registration, and six audit reports. Add separate mutations for missing/duplicate valid attempt, wrong block/study/producer/executable/protocol/data/source identity, candidate cache identity/root, raw/used mismatch, nonzero removal/cache/policy values, missing keyframe trajectory, corrupt hash, incomplete telemetry, invalid ATE, and failed audit.

- [ ] **Step 2: Write failing atomic publication and statistical failure tests**

```python
def test_cli_does_not_publish_partial_report_on_validation_failure(self) -> None:
    self.make_complete_study(pose_delta=0.0, ate_ratio=1.0)
    self.remove_expected_attempt("fr1_room", 23012, "oracle")
    exit_code = main(self.cli_arguments())
    self.assertEqual(exit_code, 1)
    self.assertFalse(self.output_path.exists())
    self.assertFalse(self.output_path.with_name(f".{self.output_path.name}.partial").exists())


def test_noninferiority_failure_is_complete_but_returns_failure(self) -> None:
    self.make_complete_study(pose_delta=-0.2, ate_ratio=1.5)
    report = build_report(**self.paths)
    self.assertFalse(report["valid"])
    self.assertFalse(report["sequences"]["fr1_desk"]["pose_pass"])
    self.assertFalse(report["sequences"]["fr1_desk"]["ate_pass"])
```

- [ ] **Step 3: Run the focused suite and confirm the missing module failure**

Run: `python3 -m unittest tests.python.test_verify_baseline_noninferiority -v`

Expected: FAIL with missing verifier module.

- [ ] **Step 4: Implement exact run selection and artifact revalidation**

```python
def completed_expected_attempt(condition_root, expected):
    valid = []
    for attempt in sorted(condition_root.glob("attempt-*")):
        manifest_path = attempt / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest.get("formal_identity")
        if (manifest.get("state") == "COMPLETED" and manifest.get("valid") is True and
                identity == expected["formal_identity"] and
                manifest.get("producer_commit") == expected["producer_commit"] and
                manifest.get("executable", {}).get("sha256") == expected["executable_sha256"]):
            valid.append(attempt)
    if len(valid) != 1:
        raise ValueError(f"expected exactly one valid attempt, got {len(valid)}: {condition_root}")
    return valid[0]
```

Independently hash and parse camera/keyframe trajectories and telemetry, compare every manifest artifact hash, verify complete frame counts, enforce all candidate baseline invariants including zero timing, and measure ATE using `parse_tum_trajectory`/`compute_ate_rmse` from the historical verifier. Never use the old five-run acceptance function. Bind every hard-gate identity listed in the spec and require six unique audit reports.

- [ ] **Step 5: Compute the frozen study and publish only a complete report**

Group measured rows by sequence and implementation, call `evaluate_study`, and set `valid` to the conjunction of every hard gate and all 12 metric gates. Record protocol/experiment/dataset/source/registration/audit/tool hashes, candidate/oracle commits and executables, NumPy metadata, all 90 pairs, all descriptive shifts, UTC generation time, and failure reasons. Atomic write uses a sibling `.partial`, file fsync, `os.replace`, and parent-directory fsync. Validation exceptions remove only that exact partial path and return 1; a complete statistical failure report is preserved and also returns 1.

- [ ] **Step 6: Run verifier, statistics, and historical tests**

Run: `python3 -m unittest tests.python.test_verify_baseline_noninferiority tests.python.test_p05_noninferiority tests.python.test_baseline_equivalence -v`

Expected: all tests PASS.

Run: `python3 tools/verify_baseline_noninferiority.py --help >/dev/null`

Expected: exit 0.

- [ ] **Step 7: Commit the verifier**

```bash
git add tools/verify_baseline_noninferiority.py tests/python/test_verify_baseline_noninferiority.py
git diff --cached --check
git commit -m "feat: verify P05 baseline noninferiority"
```

### Task 7: Verify implementation and freeze the candidate registration

**Files:**
- Modify only if a verification failure requires a product/test correction: files introduced or modified in Tasks 1-6.
- Generate ignored only: `artifacts/p05-v2/oracle/**`, `build/**`, `runs/p05-baseline-noninferiority-v2/batch_registration.json`, and registry entries.

**Interfaces:**
- Consumes: all tracked implementation tasks and the existing build/test system.
- Produces: a clean final candidate commit, candidate Release executable, oracle build manifest/executable, and immutable ignored batch registration; no generated file is staged.

- [ ] **Step 1: Run the full Python suite from a clean candidate tree**

Run: `python3 -m unittest discover -s tests/python -p 'test_*.py' -v`

Expected: all tests PASS.

- [ ] **Step 2: Build and run all C++ tests**

Run: `ORB_SLAM2_BUILD_JOBS=2 ./build.sh`

Expected: Release headless build succeeds and CTest reports 100% tests passed, including RGB-D API, optional/null frame invariants, cache mask provider, feature policy, g2o ABI, and telemetry tests.

- [ ] **Step 3: Run static sanity checks and inspect Git scope**

Run: `python3 -m py_compile semantic_py/openvocab_slam/p05_protocol.py semantic_py/openvocab_slam/p05_noninferiority.py tools/build_p05_oracle.py tools/audit_p05_baseline_access.py tools/run_p05_baseline_noninferiority.py tools/verify_baseline_noninferiority.py tools/run_orb_tum.py`

Expected: exit 0.

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intentional reproducibility-critical tracked changes are visible. No `runs/`, `reports/`, `artifacts/`, cache, build, environment, or agent path is staged.

- [ ] **Step 4: Stop registration if any verification command failed**

If Steps 1-3 did not all exit zero, do not build/register the formal batch. Return to the owning task, add one focused regression test named for the observed failure, run it alone to confirm failure, make the smallest source correction, rerun that focused test, then rerun Steps 1-3. Stage only the corrected reproducibility-critical source/test files.

- [ ] **Step 5: Commit verification corrections before registration**

```bash
git add semantic_py/openvocab_slam/p05_protocol.py semantic_py/openvocab_slam/p05_noninferiority.py tools/run_orb_tum.py tools/build_p05_oracle.py tools/audit_p05_baseline_access.py tools/run_p05_baseline_noninferiority.py tools/verify_baseline_noninferiority.py tests/python/test_p05_protocol.py tests/python/test_p05_noninferiority.py tests/python/test_run_orb_tum.py tests/python/test_build_p05_oracle.py tests/python/test_p05_access_audit.py tests/python/test_run_p05_baseline_noninferiority.py tests/python/test_verify_baseline_noninferiority.py
git diff --cached --quiet || git commit -m "fix: harden P05 noninferiority workflow"
```

Expected: either no staged correction exists or exactly one correction commit is created. Do not change tracked code after the next registration step.

- [ ] **Step 6: Build the frozen legacy oracle in its ignored detached worktree**

Run: `python3 tools/build_p05_oracle.py --repository . --build-root artifacts/p05-v2/oracle --commit 58014b7c1f2b73427b67b4e80a8cf334127f48ea --jobs 2`

Expected: `artifacts/p05-v2/oracle/oracle_build_manifest.json` is complete, reports the exact commit, a clean detached worktree, Release/headless/tested build settings, and a 64-character executable hash.

- [ ] **Step 7: Validate and register the exact candidate/oracle batch once**

Run: `python3 tools/run_p05_baseline_noninferiority.py --protocol config/P05_BASELINE_NONINFERIORITY_V2.json --experiment-manifest config/EXPERIMENT_MANIFEST.yaml --oracle-build-manifest artifacts/p05-v2/oracle/oracle_build_manifest.json --candidate-executable Examples/RGB-D/rgbd_tum_ov --data-root data/tum/raw --data-manifests data/tum/manifests --run-root runs/p05-baseline-noninferiority-v2 --registry runs/registry.jsonl --validate-only`

Expected: exit 0 and no run is launched.

Run: `python3 tools/run_p05_baseline_noninferiority.py --protocol config/P05_BASELINE_NONINFERIORITY_V2.json --experiment-manifest config/EXPERIMENT_MANIFEST.yaml --oracle-build-manifest artifacts/p05-v2/oracle/oracle_build_manifest.json --candidate-executable Examples/RGB-D/rgbd_tum_ov --data-root data/tum/raw --data-manifests data/tum/manifests --run-root runs/p05-baseline-noninferiority-v2 --registry runs/registry.jsonl --register-only`

Expected: exactly one immutable batch registration is written, resolving candidate policy to current clean `HEAD` and pinning both executable hashes. Record `git rev-parse HEAD` in the ignored registration and make no later tracked edit.

### Task 8: Execute access gates, the 180-run study, verification, and P05 closure

**Files:**
- Generate ignored only: `runs/p05-baseline-noninferiority-v2/**`, `runs/registry.jsonl`, task/phase reports already excluded by `.gitignore`.
- Modify tracked files only if the study passes and the existing phase-closure protocol explicitly requires a reproducibility-critical state document; never commit raw results or agent artifacts.

**Interfaces:**
- Consumes: immutable batch registration from Task 7 and all frozen external data/build identities.
- Produces: six valid access audits, exactly 180 valid formal runs, one complete verifier report, and—only on pass—the existing `fr3_sitting_xyz` semantic smoke plus P05 phase-closure evidence.

- [ ] **Step 1: Run all six separately registered access probes before statistics**

Run: `python3 tools/run_p05_baseline_noninferiority.py --protocol config/P05_BASELINE_NONINFERIORITY_V2.json --experiment-manifest config/EXPERIMENT_MANIFEST.yaml --oracle-build-manifest artifacts/p05-v2/oracle/oracle_build_manifest.json --candidate-executable Examples/RGB-D/rgbd_tum_ov --data-root data/tum/raw --data-manifests data/tum/manifests --run-root runs/p05-baseline-noninferiority-v2 --registry runs/registry.jsonl --forbidden-root cache/semantic --forbidden-root cache/dynamic --forbidden-file config/PROMPTS.yaml --forbidden-file config/SEMANTIC_MODELS.json --forbidden-file config/DYNAMIC_CACHE_IDENTITY.json --audit-only`

Expected: six sequential candidate-baseline `strace -ff -e trace=file` probes complete; each ignored audit report is valid and bound to its sequence, protocol hash, candidate commit, and executable hash. Probe attempts are not under oracle/candidate statistical roots.

- [ ] **Step 2: Revalidate registration and audits without launching statistical runs**

Run: `python3 tools/run_p05_baseline_noninferiority.py --protocol config/P05_BASELINE_NONINFERIORITY_V2.json --experiment-manifest config/EXPERIMENT_MANIFEST.yaml --oracle-build-manifest artifacts/p05-v2/oracle/oracle_build_manifest.json --candidate-executable Examples/RGB-D/rgbd_tum_ov --data-root data/tum/raw --data-manifests data/tum/manifests --run-root runs/p05-baseline-noninferiority-v2 --registry runs/registry.jsonl --validate-only`

Expected: exit 0, all six audits accepted, zero or only correctly registered resumable statistical conditions discovered, and no duplicate expected valid attempts.

- [ ] **Step 3: Execute the frozen order sequentially once**

Run: `python3 tools/run_p05_baseline_noninferiority.py --protocol config/P05_BASELINE_NONINFERIORITY_V2.json --experiment-manifest config/EXPERIMENT_MANIFEST.yaml --oracle-build-manifest artifacts/p05-v2/oracle/oracle_build_manifest.json --candidate-executable Examples/RGB-D/rgbd_tum_ov --data-root data/tum/raw --data-manifests data/tum/manifests --run-root runs/p05-baseline-noninferiority-v2 --registry runs/registry.jsonl`

Expected: 180 conditions execute in the exact tracked order; every condition becomes one valid complete expected-producer attempt. On an invalid run, preserve it and use the runner's identity-safe resume only for that same frozen condition; never rerun a valid completed condition.

- [ ] **Step 4: Independently verify deterministic and statistical gates**

Run: `python3 tools/verify_baseline_noninferiority.py --protocol config/P05_BASELINE_NONINFERIORITY_V2.json --experiment-manifest config/EXPERIMENT_MANIFEST.yaml --registration runs/p05-baseline-noninferiority-v2/batch_registration.json --oracle-root runs/p05-baseline-noninferiority-v2/oracle --candidate-root runs/p05-baseline-noninferiority-v2/candidate --audit-root runs/p05-baseline-noninferiority-v2/audits --data-root data/tum/raw --data-manifests data/tum/manifests --repository . --output reports/p05_baseline_noninferiority_v2.json`

Expected on pass: exit 0; all deterministic gates pass; every sequence has `pose_lower_95 >= -0.10` and `ate_ratio_upper_95 <= 1.25`; report records 90 pairs, 100,000 resamples per sequence, all descriptive shifts, versions, and hashes.

Expected on failure: exit 1 with the complete failed report preserved when statistics were reached, or no final report when hard validation failed. Stop P05 execution, preserve evidence, do not launch semantic smoke, and request a separately approved amendment/study ID before any new performance experiment.

- [ ] **Step 5: If and only if V2 passes, run the frozen semantic-feedback smoke**

Run: `python3 tools/run_orb_tum.py --mode semantic-feedback --study smoke --sequence fr3_sitting_xyz --seed 23011 --manifest config/EXPERIMENT_MANIFEST.yaml --data-root data/tum/raw --data-manifests data/tum/manifests --output-root runs/smoke/semantic-feedback --registry runs/registry.jsonl --executable Examples/RGB-D/rgbd_tum_ov --vocabulary Vocabulary/ORBvoc.txt --dynamic-cache-root cache/dynamic/v1 --semantic-cache-root cache/semantic/v1 --prompt-config config/PROMPTS.yaml --dynamic-cache-identities config/DYNAMIC_CACHE_IDENTITY.json`

Expected: one valid complete semantic-feedback smoke with valid frozen cache identity and complete artifacts; it remains ignored.

- [ ] **Step 6: Run the repository's P05 closure checks and inspect commit scope**

Run: `python3 -m unittest discover -s tests/python -p 'test_*.py' -v`

Run: `ctest --test-dir build --output-on-failure`

Run: `git diff --check`

From `../OpenVocab-ORB-SLAM2-Agent-Pack-v1`, update ignored/live P05 report, evidence JSON, checklist, state ledger, command log, and registry entries with the V2 report and smoke hashes, then run `python3 scripts/audit_pack.py --root .` and `python3 scripts/verify_phase_closure.py --root . --phase P05`.

Expected: closure accepts the V2 study ID rather than historical equivalence reports; tests pass; `git status --short` contains no staged generated artifacts. If closure requires a tracked protocol reference, add only that reproducibility-critical reference, test it, and commit it separately as `docs: close P05 noninferiority gate`. Do not push.

---

## Completion evidence checklist

- The tracked protocol hash, immutable registration hash, oracle/candidate commits, and executable hashes form an unbroken identity chain.
- Exactly six access audits and 180 statistical runs exist; audit runs are excluded from metrics.
- Every expected condition has exactly one valid complete attempt for its registered producer; stale attempts are neither selected nor deleted.
- The verifier independently validates artifacts, telemetry, identities, pairing, and the frozen statistical algorithm.
- P05 closure occurs only after all deterministic gates, all 12 noninferiority gates, and the single semantic-feedback smoke pass.
- Git history contains only config, source, tests, design, and build instructions needed for reproduction; no agent/run/cache/build/report/environment artifact is committed or pushed.
