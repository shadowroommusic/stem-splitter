"""The splitter's own logic, with a fake separator instead of a model.

The model is slow and heavy; what needs testing here is everything around it — the job lifecycle, the
hand-off to a command or to demucs, finding the stems a backend wrote, and failing with a sentence a
human can act on. `STEM_SPLIT_COMMAND` makes that a two-line script.
"""
import json
import os
import shutil
import warnings
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Jobs are deliberately detached, so a test's own process objects are still alive at collection time
# and Python warns about it. That is the design, not a leak in the test.
warnings.filterwarnings("ignore", category=ResourceWarning)

from shadow_stem_splitter.audio import read_wav, write_wav
from shadow_stem_splitter.backends import STEMS, describe, resolve
from shadow_stem_splitter.jobs import JobStore

SRC = Path(__file__).resolve().parents[1] / "src"


def tone(seconds: float = 0.4, rate: int = 44100, frequency: float = 220.0) -> np.ndarray:
    time = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * frequency * time) * 0.5).astype(np.float32)


def fake_separator(script: Path) -> Path:
    """A stand-in backend: writes four audible stems into `{output_dir}`."""
    script.write_text(
        "\n".join(
            [
                "import sys, numpy as np, wave",
                "source, out_dir = sys.argv[1], sys.argv[2]",
                "for index, stem in enumerate(['vocals', 'drums', 'bass', 'other']):",
                "    rate = 44100",
                "    time = np.arange(rate // 2) / rate",
                "    signal = (np.sin(2 * np.pi * (110 * (index + 1)) * time) * 0.4).astype(np.float32)",
                "    with wave.open(out_dir + '/' + stem + '.wav', 'wb') as handle:",
                "        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(rate)",
                "        handle.writeframes((signal * 32767).astype('<i2').tobytes())",
                "print('done')",
            ]
        )
    )
    return script


def run_job_until_done(job_dir: Path, job_id: str, timeout: float = 30.0) -> dict:
    store = JobStore(job_dir)
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = store.status(job_id)
        if record["status"] != "running":
            return record
        time.sleep(0.2)
    raise AssertionError(f"job did not finish: {store.status(job_id)}")


class BackendTests(unittest.TestCase):
    def test_without_demucs_or_a_command_it_says_how_to_enable_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = {
                "PATH": os.environ.get("PATH", ""),
                "SHADOWROOM_HOME": tmp,
                "STEM_SPLIT_PYTHON": sys.executable,
            }
            previous = {key: os.environ.get(key) for key in ("SHADOWROOM_HOME", "STEM_SPLIT_COMMAND", "STEM_SPLIT_PYTHON")}
            os.environ.update(environment)
            os.environ.pop("STEM_SPLIT_COMMAND", None)
            try:
                described = describe()
                demucs = next(entry for entry in described["backends"] if entry["id"] == "demucs")
                # this interpreter cannot import demucs, so the detail must say what to install
                if not demucs["available"]:
                    self.assertIn("demucs", demucs["detail"])
                    with self.assertRaises(ValueError) as caught:
                        resolve("auto")
                    self.assertIn("STEM_SPLIT_COMMAND", str(caught.exception))
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_a_command_backend_is_preferred_when_it_is_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = fake_separator(Path(tmp) / "fake.py")
            previous = os.environ.get("STEM_SPLIT_COMMAND")
            os.environ["STEM_SPLIT_COMMAND"] = f"{sys.executable} {script} {{input}} {{output_dir}}"
            try:
                described = describe()
                self.assertEqual(described["preferred"], "command")
                self.assertEqual(resolve("auto")["id"], "command")
            finally:
                if previous is None:
                    os.environ.pop("STEM_SPLIT_COMMAND", None)
                else:
                    os.environ["STEM_SPLIT_COMMAND"] = previous


class JobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = write_wav(self.root / "song.wav", tone(0.5), 44100)
        self.previous = {key: os.environ.get(key) for key in ("STEM_SPLIT_COMMAND", "SHADOWROOM_HOME")}
        script = fake_separator(self.root / "fake.py")
        os.environ["STEM_SPLIT_COMMAND"] = f"{sys.executable} {script} {{input}} {{output_dir}}"
        os.environ["SHADOWROOM_HOME"] = str(self.root)

    def tearDown(self) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def test_a_job_writes_four_stems_and_reports_every_stage(self):
        store = JobStore(self.root / "jobs")
        started = store.start(self.source, out_dir=self.root / "out")
        self.assertEqual(started["status"], "running")
        finished = run_job_until_done(store.directory, started["job_id"])
        self.assertEqual(finished["status"], "ok", finished)
        self.assertEqual(sorted(finished["stems"]), sorted(STEMS))
        for name, path in finished["stems"].items():
            self.assertTrue(Path(path).is_file(), name)
            samples, rate = read_wav(path)
            self.assertEqual(rate, 44100)
            self.assertGreater(float(np.abs(samples).max()), 0.1, name)
        self.assertEqual([entry["name"] for entry in finished["stages"]], ["decode", "collect"])
        self.assertEqual(finished["backend"], "command")

    def test_a_backend_that_writes_nothing_fails_with_a_usable_message(self):
        os.environ["STEM_SPLIT_COMMAND"] = f"{sys.executable} -c pass"
        store = JobStore(self.root / "jobs")
        started = store.start(self.source, out_dir=self.root / "empty")
        finished = run_job_until_done(store.directory, started["job_id"])
        self.assertEqual(finished["status"], "failed")
        self.assertIn("did not write", finished["error"])

    def test_cancelling_stops_the_job_and_takes_the_half_written_stems_with_it(self):
        slow = self.root / "slow.py"
        slow.write_text("import time\ntime.sleep(30)\n")
        os.environ["STEM_SPLIT_COMMAND"] = f"{sys.executable} {slow}"
        store = JobStore(self.root / "jobs")
        started = store.start(self.source, out_dir=self.root / "cancelled")
        time.sleep(0.6)
        cancelled = store.cancel(started["job_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertFalse((self.root / "cancelled").exists())

    def test_a_missing_file_is_refused_before_a_job_exists(self):
        store = JobStore(self.root / "jobs")
        with self.assertRaises(FileNotFoundError):
            store.start(self.root / "nope.mp3")


class McpTests(unittest.TestCase):
    def test_the_server_lists_its_four_tools(self):
        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        completed = subprocess.run(
            [sys.executable, "mcp_server.py"],
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            cwd=str(SRC.parent),
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )
        reply = json.loads(completed.stdout.strip().splitlines()[0])
        self.assertEqual(
            [tool["name"] for tool in reply["result"]["tools"]],
            ["separate_stems", "stem_job_status", "stem_job_cancel", "stem_backends"],
        )
        self.assertEqual(completed.stderr.strip(), "")


if __name__ == "__main__":
    unittest.main()
