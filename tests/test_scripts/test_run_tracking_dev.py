import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from scripts import run_tracking_dev as cli
from scripts.prepare_tracking_evaluation import inventory_sequence
from src.services.evaluation.tracking_payload import canonical_bytes, materialize


@pytest.fixture
def prepared(tmp_path):
    training = tmp_path / "dataset/training"
    (training / "label_02").mkdir(parents=True)
    rows = []
    hashes = {}
    for sequence in cli.SEQUENCES:
        images = training / "image_02" / sequence
        images.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (24, 20), (30 + i, 50, 70)).save(images / f"{i:06d}.png")
        label = training / "label_02" / f"{sequence}.txt"
        label.write_text("0 0 Car 0 0 0 0 0 10 10 1 1 1 0 0 10 0\n")
        inventory = inventory_sequence(training, sequence)
        directory = tmp_path / "prepared" / sequence
        row = materialize(training, inventory, directory)[0]
        row.update(prepared_directory=str(directory.relative_to(tmp_path)), cadence="all_frames")
        rows.append(row)
        hashes[sequence] = {"labels": cli.file_hash(label)}
    protocol = {
        "revision": 2,
        "dataset_root": "dataset",
        "splits": {"dev": list(cli.SEQUENCES), "held_out": ["0004"]},
        "gpu_plan": rows,
        "input_hashes": hashes,
        "runs": [],
        "class_map": {"Car": "car", "Pedestrian": "person"},
    }
    path = tmp_path / "protocol.json"
    path.write_bytes(canonical_bytes(protocol))
    return tmp_path, path, protocol, rows


def fake_result(config, mapping):
    c = config["configurations"][0]
    probe = {
        "checkpoint_sha256": cli.EXPECTED_MODEL["checkpoint_sha256"],
        "environment": {"versions": {"sam3_commit": cli.EXPECTED_MODEL["sam3_commit"]}},
    }
    checkpoints = []
    for i, (_, batch) in enumerate(cli.windows(mapping["frames"], c["window_size"], c["overlap"])):
        entry = {
            "schema": "autolabel-sam3-window/v1",
            "config_id": c["id"],
            "input_sha256": cli.checkpoint_identity(mapping["frames"], c),
            "window_id": i,
            "frame_ids": [f["source_frame_index"] for f in batch],
            "model": cli.model_identity(probe),
            "predictions": [],
        }
        entry["output_sha256"] = hashlib.sha256(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        checkpoints.append(entry)
    return {
        "schema": "autolabel-sam3-benchmark/v1",
        "completed": True,
        "model_runtime": probe,
        "model_load_seconds": 2,
        "configurations": [
            {
                "id": c["id"],
                "status": "completed",
                "config": c,
                "predictions": [],
                "evaluated_source_frames": [0, 1, 2],
                "window_checkpoints": checkpoints,
                "window_predictions": [],
                "window_frame_count": sum(len(e["frame_ids"]) for e in checkpoints),
                "runtime": {"total_seconds": 3},
            }
        ],
    }


def test_dry_run_validates_without_settings_network_or_writes(prepared, monkeypatch):
    root, path, _, _ = prepared
    monkeypatch.setattr(cli, "Settings", lambda: pytest.fail("dry-run must not read credentials"))
    before = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    result = cli.run(root, path)
    assert result["frames"] == 9 and result["submitted"] is False
    assert sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file()) == before


@pytest.mark.parametrize("case", ["held_out", "archive", "source", "mapping", "gt"])
def test_rejects_unapproved_or_modified_inputs(prepared, case):
    root, _, protocol, rows = prepared
    row = rows[0]
    archive = root / row["prepared_directory"] / row["archive_name"]
    if case == "held_out":
        protocol["splits"]["held_out"].append(row["sequence"])
    elif case == "archive":
        row["archive_sha256"] = "0" * 64
    elif case == "source":
        (root / "dataset/training/image_02/0001/000000.png").write_bytes(b"changed")
    else:
        with zipfile.ZipFile(archive) as stream:
            files = {n: stream.read(n) for n in stream.namelist()}
        if case == "gt":
            files["label_02/0001.txt"] = b"GT"
        else:
            mapping = json.loads(files["frames.json"])
            mapping["frames"][1]["original_source_index"] = 0
            files["frames.json"] = canonical_bytes(mapping)
            row["frames_sha256"] = hashlib.sha256(files["frames.json"]).hexdigest()
        with zipfile.ZipFile(archive, "w") as stream:
            for name, value in files.items():
                stream.writestr(name, value)
        row["archive_sha256"] = cli.file_hash(archive)
    with pytest.raises(ValueError):
        cli.verify_payload(root, protocol, row)


def test_result_requires_complete_mapping_and_checked_overlap(prepared):
    root, _, protocol, rows = prepared
    _, config, mapping = cli.verify_payload(root, protocol, rows[0])
    result = fake_result(config, mapping)
    cli.validate_result(result, config, mapping)
    result["configurations"][0]["evaluated_source_frames"] = [0, 2]
    with pytest.raises(ValueError, match="every source"):
        cli.validate_result(result, config, mapping)
    result = fake_result(config, mapping)
    result["configurations"][0]["window_checkpoints"] = []
    with pytest.raises(ValueError, match="window checkpoints"):
        cli.validate_result(result, config, mapping)
    result = fake_result(config, mapping)
    result["configurations"][0]["predictions"] = [
        {"frame_index": 0, "label": "car", "track_id": "invented", "bbox": [0, 0, 10, 10]}
    ]
    with pytest.raises(ValueError, match="direct predictions"):
        cli.validate_result(result, config, mapping)


def test_registry_only_complete_results_and_correct_mapping(prepared):
    root, path, protocol, rows = prepared
    output = root / "live"
    cli.save_equal_or_new(output / "source-protocol.json", path.read_bytes())
    _, config, mapping = cli.verify_payload(root, protocol, rows[0])
    cli.save_equal_or_new(output / "0001/result.json", canonical_bytes(fake_result(config, mapping)))
    extended = json.loads(cli.register(root, protocol, rows, output).read_text())
    assert extended["prediction_sequences"] == ["0001"]
    assert extended["observation_cohorts"]["all_frames"] == ["0001"]
    loaded = cli.load_frozen_run(extended["runs"][0], root)
    assert loaded.source == [0, 1, 2] and loaded.records[2]["frame_index"] == 2
    result = fake_result(config, mapping)
    result["configurations"][0]["status"] = "failed"
    cli.save_state(output / "0001/result.json", result)
    with pytest.raises(ValueError, match="incomplete"):
        cli.register(root, protocol, rows, output)


def test_remote_file_inventory_paginates_and_detects_cycles():
    class Runner:
        calls = 0

        def _command(self, *args, **kwargs):
            self.calls += 1
            text = (
                "Next Page Token = next\nname,size,creationDate\na.jpg,3,now"
                if self.calls == 1
                else "name,size,creationDate\nb.jpg,3,now"
            )
            return subprocess.CompletedProcess(args, 0, stdout=text)

    assert cli.remote_names(Runner(), "owner/data") == ["a.jpg", "b.jpg"]


def test_resume_uses_reserved_identity_and_never_calls_run(prepared, monkeypatch):
    root, _, protocol, rows = prepared
    archive, _, _ = cli.verify_payload(root, protocol, rows[0])
    notebook = root / "notebook.ipynb"
    notebook.write_text("{}")

    class Runner:
        username = "owner"
        sam3_checkpoint_dataset = "owner/checkpoint"
        machine_shape = "NvidiaTeslaT4"
        work_root = root / "runs"

        def run(self, **kwargs):
            run_dir = self.work_root / kwargs["job_id"] / "kaggle/autolabel-sam3-benchmark-reserved"
            cli.save_state(run_dir / "dataset/dataset-metadata.json", {"id": "owner/data"})
            cli.save_state(
                run_dir / "kernel/kernel-metadata.json",
                {
                    "id": "owner/autolabel-sam3-benchmark-reserved-run",
                    "dataset_sources": ["owner/data", self.sam3_checkpoint_dataset],
                },
            )
            (run_dir / "kernel/main.ipynb").write_bytes(notebook.read_bytes())
            with zipfile.ZipFile(run_dir / "dataset/payload.zip", "w") as wrapper:
                wrapper.writestr("payload.zip", archive.read_bytes())
            return {"first": True}

        def _wait_for_dataset(self, reference):
            assert reference == "owner/data"

    runner = Runner()
    assert cli.execute_one(runner, archive, notebook, root / "0001") == {"first": True}
    runner.run = lambda **kwargs: pytest.fail("resume must not create another kernel")
    monkeypatch.setattr(
        cli, "_resume_reserved_run", lambda *args, **kwargs: {"resumed": kwargs == {"upload_missing": False}}
    )
    assert cli.execute_one(runner, archive, notebook, root / "0001") == {"resumed": True}
    notebook.write_text("changed")
    with pytest.raises(ValueError, match="incompatible"):
        cli.execute_one(runner, archive, notebook, root / "0001")


def test_unknown_push_is_collect_only_even_without_receipt(prepared, monkeypatch):
    root, _, protocol, rows = prepared
    archive, _, _ = cli.verify_payload(root, protocol, rows[0])
    notebook = root / "notebook.ipynb"
    notebook.write_text("{}")
    runner = cli.TrackingDevRunner(
        username="owner", work_root=root / "runs", sam3_checkpoint_dataset="owner/checkpoint"
    )

    def first(**kwargs):
        directory = runner.work_root / kwargs["job_id"] / "kaggle/autolabel-sam3-benchmark-reserved"
        cli.save_state(directory / "dataset/dataset-metadata.json", {"id": "owner/data"})
        cli.save_state(
            directory / "kernel/kernel-metadata.json",
            {
                "id": "owner/autolabel-sam3-benchmark-reserved-run",
                "dataset_sources": ["owner/data", "owner/checkpoint"],
            },
        )
        (directory / "kernel/main.ipynb").write_bytes(notebook.read_bytes())
        with zipfile.ZipFile(directory / "dataset/payload.zip", "w") as wrapper:
            wrapper.writestr("payload.zip", archive.read_bytes())
        cli.save_state(directory / "kernel-submission-started.json", {"status": "unknown"})
        raise TimeoutError("push response lost")

    monkeypatch.setattr(runner, "run", first)
    with pytest.raises(TimeoutError):
        cli.execute_one(runner, archive, notebook, root / "0001")
    monkeypatch.setattr(runner, "run", lambda **kw: pytest.fail("no second run"))
    monkeypatch.setattr(runner, "_wait_for_dataset", lambda ref: None)
    monkeypatch.setattr(cli, "_resume_reserved_run", lambda *a, **kw: pytest.fail("no inventory-driven push"))
    monkeypatch.setattr(runner, "_collect_result", lambda *a: {"collected_exact": a[2]})
    assert (
        cli.execute_one(runner, archive, notebook, root / "0001")["collected_exact"]
        == "owner/autolabel-sam3-benchmark-reserved-run"
    )


def test_push_marker_precedes_provider_and_prevents_second_attempt(tmp_path, monkeypatch):
    kernel = tmp_path / "run/kernel"
    kernel.mkdir(parents=True)
    runner = cli.TrackingDevRunner(username="owner", work_root=tmp_path)
    calls = []

    def command(self, *args, **kwargs):
        assert (kernel.parent / "kernel-submission-started.json").is_file()
        calls.append(args)
        raise TimeoutError("accepted remotely, response lost")

    monkeypatch.setattr(cli.NativeTrialRunner, "_command", command)
    with pytest.raises(TimeoutError):
        runner._command("kernels", "push", "-p", str(kernel), timeout=60)
    with pytest.raises(FileExistsError):
        runner._command("kernels", "push", "-p", str(kernel), timeout=60)
    assert len(calls) == 1


def test_live_lock_rejects_second_invocation_before_provider(prepared, monkeypatch):
    root, path, _, _ = prepared
    monkeypatch.setattr(cli, "Settings", lambda: pytest.fail("locked invocation must not read credentials"))
    with cli.live_lock(root / "state/evaluation/phase1-tracking/live-dev"):
        with pytest.raises(RuntimeError, match="live lock"):
            cli.run(root, path, live=True)


def test_bounded_payload_only_changes_deadlines(prepared):
    root, _, protocol, rows = prepared
    archive, config, _ = cli.verify_payload(root, protocol, rows[0])
    bounded = cli.bounded_payload(archive, config, root / "execution", 400)
    with zipfile.ZipFile(archive) as before, zipfile.ZipFile(bounded) as after:
        for name in before.namelist():
            if name != "benchmark-config.json":
                assert before.read(name) == after.read(name)
        changed = json.loads(after.read("benchmark-config.json"))
        assert changed["max_total_seconds"] == 400 and changed["max_config_seconds"] == 220
        assert changed["configurations"] == config["configurations"]


@pytest.mark.parametrize("case", ["slow", "failure"])
def test_partial_run_registers_success_and_slow_speed_blocks_next(prepared, monkeypatch, case):
    root, path, protocol, rows = prepared

    class Runner:
        api_token = "private-test-token"

    monkeypatch.setattr(cli, "Settings", lambda: None)
    monkeypatch.setattr(cli, "make_runner", lambda *a: Runner())
    monkeypatch.setattr(cli, "preflight", lambda *a: {"verified": True})
    calls = []

    def execute(runner, archive, notebook, directory):
        calls.append(directory.name)
        if directory.name != "0001":
            raise RuntimeError("simulated provider failure")
        _, config, mapping = cli.verify_payload(root, protocol, rows[0])
        result = fake_result(config, mapping)
        if case == "slow":
            result["configurations"][0]["runtime"]["total_seconds"] = 180
        return result

    monkeypatch.setattr(cli, "execute_one", execute)
    summary = cli.run(root, path, live=True)
    assert calls == (["0001"] if case == "slow" else ["0001", "0019"])
    assert summary["status"] == ("budget_reestimate_exceeded" if case == "slow" else "incomplete")
    extended = json.loads(Path(summary["protocol"]).read_text())
    assert extended["prediction_sequences"] == ["0001"]
    assert summary["budget_accounted_minutes"] > 0


def test_diagnosis_keeps_cadence_cohorts_separate(prepared, monkeypatch):
    from src.services.evaluation import tracking_diagnosis

    root, _, protocol, _ = prepared
    protocol["runs"] = [{"sequence": "0013"}, {"sequence": "0001"}]
    protocol["observation_cohorts"] = {"legacy_sampled": ["0013", "0020"], "all_frames": ["0001"]}
    output = root / "live/dev-protocol.json"
    cli.save_state(output, protocol)
    populations = []

    def diagnosis(path, root, directory):
        p = json.loads(path.read_text())
        populations.append([r["sequence"] for r in p["runs"]])
        return {"event_count": 0, "population_support": {}, "pooled": {}}

    monkeypatch.setattr(tracking_diagnosis, "diagnose", diagnosis)
    summary = cli.diagnose_available(root, output)
    assert populations[0] == ["0013", "0001"]
    assert {tuple(p) for p in populations[1:]} == {("0013",), ("0001",)}
    assert set(summary) == {"all_dev", "legacy_sampled", "all_frames", "note"}
    assert "not algorithm improvements" in summary["note"]
