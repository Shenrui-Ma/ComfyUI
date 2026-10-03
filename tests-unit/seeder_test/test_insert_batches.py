"""The fast scan inserts in small batches with a pause after each, so a foreground write
(an upload, an output being registered) gets the write lock while it runs."""

import os
import threading
import time
from types import SimpleNamespace

import folder_paths
import pytest

from app.assets import mode
from app.assets import scanner as scanner_module
from app.assets import seeder as seeder_module
from app.assets.seeder import ScanPhase, State, _AssetSeeder, _ScanState
from app.assets.services import ingest


@pytest.fixture
def scan_seeder(monkeypatch: pytest.MonkeyPatch) -> _AssetSeeder:
    instance = _AssetSeeder()
    instance._state = State.RUNNING
    instance._scan_state = _ScanState()
    instance._roots = ("input",)
    instance._phase = ScanPhase.FAST
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: True)
    monkeypatch.setattr(instance, "_log_scan_config", lambda roots: None)
    return instance


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _run_fast_phase(scan_seeder, monkeypatch, count, seconds_per_file):
    """Run the fast phase over ``count`` specs with insert_asset_specs replaced by one
    that takes ``seconds_per_file`` of fake time per file."""
    clock = _Clock()
    batches: list[int] = []

    def insert(batch, _tags, _progress):
        batches.append(len(batch))
        clock.now += seconds_per_file * len(batch)
        return len(batch), None

    specs = [{"tags": []} for _ in range(count)]
    monkeypatch.setattr(seeder_module, "time", SimpleNamespace(
        perf_counter=clock.perf_counter, sleep=clock.sleep, monotonic=time.monotonic, thread_time=time.thread_time
    ))
    monkeypatch.setattr(seeder_module, "insert_asset_specs", insert)
    monkeypatch.setattr(seeder_module, "sync_root_safely", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(seeder_module, "collect_paths_for_roots", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(seeder_module, "build_asset_specs", lambda *_args, **_kwargs: (specs, set(), 0))
    monkeypatch.setattr(seeder_module, "tick_watch_list", lambda _progress=None: None)
    result = scan_seeder._run_fast_phase(("input",))
    return result, batches, clock.sleeps


def test_inserts_go_in_batches_with_a_pause_between_them(scan_seeder, monkeypatch):
    monkeypatch.setattr(seeder_module, "INSERT_BATCH_SIZE", 4)

    (created, _, _), batches, sleeps = _run_fast_phase(scan_seeder, monkeypatch, 10, 0.001)

    assert created == 10
    assert batches == [4, 4, 2]
    # Fast batches get the minimum pause; nothing follows the last one.
    assert sleeps == [seeder_module.INSERT_PAUSE_SECONDS] * 2


def test_the_pause_grows_with_a_slow_batch_up_to_a_cap(scan_seeder, monkeypatch):
    monkeypatch.setattr(seeder_module, "INSERT_BATCH_SIZE", 4)

    # Batches of 1.2 s, then 20 s (as when hashing large files before the write).
    _, _, slow = _run_fast_phase(scan_seeder, monkeypatch, 8, 0.3)
    _, _, very_slow = _run_fast_phase(scan_seeder, monkeypatch, 8, 5.0)

    assert slow == [pytest.approx(seeder_module.INSERT_PAUSE_RATIO * 1.2)]
    assert very_slow == [seeder_module.INSERT_PAUSE_MAX_SECONDS]


def test_the_pause_outlasts_the_busy_handlers_longest_poll():
    # SQLite's busy handler sleeps at most 100 ms between polls; a shorter pause can fall
    # between two of them, and a waiting write never sees the lock free.
    assert seeder_module.INSERT_PAUSE_SECONDS > 0.1


# --- real threads, real WAL database ---------------------------------------------

# Per scanned file inside the write transaction: 500 files in one transaction, as before
# this change, hold the lock for over 5 s. The test's batches are small, so each still
# holds it for well under a second on a slow CI runner.
_SLOW_RECORD_SECONDS = 0.01
_TEST_BATCH_SIZE = 20
# A foreground write that gets in at a pause waits about one batch plus a pause.
_FOREGROUND_WAIT_LIMIT_SECONDS = 4.0


@pytest.fixture
def hashing_off():
    mode.init(SimpleNamespace(enable_asset_hashing=False))
    yield
    mode.init(None)


def test_output_registration_gets_in_while_the_scan_inserts(hashing_off, file_db, tmp_path, monkeypatch):
    library = tmp_path / "input"
    library.mkdir()
    for i in range(600):
        (library / f"f{i}.png").write_bytes(b"x" * (i + 1))
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(library))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    old = time.time() - 3600
    for path in library.iterdir():
        os.utime(path, (old, old))

    scanning = threading.Event()
    create_record = scanner_module.create_record

    def slow_create_record(*args, **kwargs):
        scanning.set()
        time.sleep(_SLOW_RECORD_SECONDS)
        return create_record(*args, **kwargs)

    monkeypatch.setattr(scanner_module, "create_record", slow_create_record)
    monkeypatch.setattr(seeder_module, "INSERT_BATCH_SIZE", _TEST_BATCH_SIZE)
    seeder = _AssetSeeder()
    assert seeder.start(roots=("input",), phase=ScanPhase.FAST)
    try:
        assert scanning.wait(30)
        waits = []
        for i in range(3):
            path = output / f"out{i}.png"
            path.write_bytes(b"output")
            started = time.perf_counter()
            registered = ingest.register_executed_output(str(path), job_id="job")
            waits.append(time.perf_counter() - started)
            assert registered is not None
        scan_was_running = seeder.get_status().state == State.RUNNING
    finally:
        seeder.cancel()
        assert seeder.wait(timeout=60)

    assert scan_was_running, "the scan ended before the foreground writes; nothing was contended"
    assert max(waits) < _FOREGROUND_WAIT_LIMIT_SECONDS, waits
