"""splitter-worker 三件套测试：热进程协议、缓存、失败隔离、看门狗常量。"""
import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_stem_splitter import runner
from shadow_stem_splitter.jobs import JobStore
from shadow_stem_splitter.pool import STALL_TIMEOUT_SECONDS, WorkerPool, cache_key, stems_cache_dir


def write_wav(path: Path, seconds: float = 1.0, rate: int = 44100) -> Path:
    count = int(rate * seconds)
    t = np.arange(count) / rate
    data = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((data * 32767).astype("<i2").tobytes())
    return path


class FakeWorker:
    """假 demucs worker：读一行请求，写四条 wav，回 @@DONE@@（协议测试不用真模型）。"""

    SCRIPT = r'''
import json, sys
from pathlib import Path
sys.stdout.write("@@READY@@fake\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    out = Path(req["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    for stem in ("vocals", "drums", "bass", "other"):
        (out / f"{stem}.wav").write_bytes(b"RIFF" + stem.encode() + b"\x00" * 100)
    sys.stdout.write("@@DONE@@\n"); sys.stdout.flush()
'''


class WorkerPoolTests(unittest.TestCase):
    def test_pool_reuses_one_process_across_jobs(self):
        """验收① 的等价物：两个 job 之间 worker 进程不换（spawn 只发生一次）。"""
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_worker.py"
            script.write_text(FakeWorker.SCRIPT)
            pool = WorkerPool()
            # 直接用假脚本替换 spawn（不 monkeypatch 类，简单替换 _spawn 行为）
            def fake_spawn(model):
                import subprocess
                proc = subprocess.Popen(
                    [sys.executable, str(script)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                pool._loads += 1
                pool._model = model
                pool._proc = proc
                return proc
            pool._spawn = fake_spawn  # type: ignore[method-assign]
            first = Path(tmp) / "a"
            second = Path(tmp) / "b"
            pool.separate(Path(tmp) / "in.wav", first, "htdemucs")
            pool.separate(Path(tmp) / "in2.wav", second, "htdemucs")
            self.assertEqual(pool.loads, 1, "两个 job 只允许加载一次模型")
            self.assertTrue((first / "vocals.wav").exists())
            self.assertTrue((second / "drums.wav").exists())
            pool.shutdown()

    def test_failure_marker_raises_and_resets_pool(self):
        """worker 报 @@ERROR@@ → 异常上抛，池子置空（下次重启）。"""
        error_script = (
            "import json, sys\n"
            "sys.stdout.write('@@READY@@fake\\n'); sys.stdout.flush()\n"
            "line = sys.stdin.readline()\n"
            "sys.stdout.write('@@ERROR@@' + json.dumps('OOM during inference') + '\\n'); sys.stdout.flush()\n"
            "sys.exit(1)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "err_worker.py"
            script.write_text(error_script)
            pool = WorkerPool()
            def fake_spawn(model):
                import subprocess
                proc = subprocess.Popen(
                    [sys.executable, str(script)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                pool._loads += 1
                pool._model = model
                pool._proc = proc
                return proc
            pool._spawn = fake_spawn  # type: ignore[method-assign]
            with self.assertRaises(ValueError) as caught:
                pool.separate(Path(tmp) / "x.wav", Path(tmp) / "out", "htdemucs")
            self.assertIn("OOM", str(caught.exception))
            self.assertIsNone(pool._proc, "失败后池子必须置空（下次重启新进程）")

    def test_stall_watchdog_is_30_minutes(self):
        self.assertEqual(STALL_TIMEOUT_SECONDS, 30 * 60)


class CacheTests(unittest.TestCase):
    def test_cache_key_depends_on_content_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = write_wav(Path(tmp) / "a.wav")
            b = write_wav(Path(tmp) / "b.wav", seconds=2.0)
            self.assertEqual(cache_key(a, "htdemucs"), cache_key(Path(tmp) / "a.wav", "htdemucs"))
            self.assertNotEqual(cache_key(a, "htdemucs"), cache_key(b, "htdemucs"))
            self.assertNotEqual(cache_key(a, "htdemucs"), cache_key(a, "htdemucs_ft"))

    def test_cache_dir_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = write_wav(Path(tmp) / "song.wav")
            directory = stems_cache_dir(a, "htdemucs")
            self.assertIn("htdemucs", str(directory))
            self.assertTrue(directory.name == "song", "stem 目录名 = 源文件 stem")
            self.assertEqual(directory.parent.parent.parent.name, "_stems-cache")


class QuarantineTests(unittest.TestCase):
    def test_failed_job_gets_error_txt_with_stage_timings(self):
        """任务③：必败 job → jobs/failed/<id>/error.txt 含错误原文与 stage 记录。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            # 造一个必失败 job：source 是存在但读不了的垃圾内容 → decode 阶段抛错
            #（JobStore.start 会先验存在性，所以不能用「不存在」的路径）
            garbage = Path(tmp) / "garbage.wav"
            garbage.write_bytes(b"this is not audio at all, just bytes")
            record0 = store.start(str(garbage), out_dir=str(Path(tmp) / "out"))
            job_id = record0["job_id"]
            record = runner.run_job(store.path(job_id))
            self.assertEqual(record["status"], "failed")
            failed_dir = Path(tmp) / "jobs" / "failed" / job_id
            error_file = failed_dir / "error.txt"
            self.assertTrue(error_file.is_file(), f"error.txt 必须存在：{error_file}")
            text = error_file.read_text(encoding="utf-8")
            self.assertIn("could not decode", text, "错误原文要在 error.txt 里")
            self.assertIn("stage_at_failure", text)
            # 主目录里的 job 文件仍保留 failed 状态（status 轮询读得到），failed/ 里是诊断副本
            self.assertEqual(store.status(job_id)["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class SixSourceModelTests(unittest.TestCase):
    """splitter-6s：模型决定轨数，4 路径不回退。"""

    def test_stems_for_matrix(self):
        from shadow_stem_splitter.backends import MODELS, STEMS, STEMS_6S, stems_for

        self.assertEqual(stems_for("htdemucs"), STEMS)
        self.assertEqual(stems_for("htdemucs_ft"), STEMS)
        self.assertEqual(stems_for("mdx_extra"), STEMS)
        self.assertEqual(stems_for("htdemucs_6s"), STEMS_6S)
        self.assertEqual(set(STEMS_6S) - set(STEMS), {"guitar", "piano"})
        self.assertIn("htdemucs_6s", MODELS, "6s 必须在模型列表里（显式选项）")
        self.assertNotEqual(MODELS[0], "htdemucs_6s", "默认模型仍是 4 轨 htdemucs")

    def test_unknown_model_is_rejected_readable(self):
        import tempfile

        from shadow_stem_splitter.jobs import JobStore

        with tempfile.TemporaryDirectory() as tmp:
            wav = write_wav(Path(tmp) / "x.wav")
            store = JobStore(Path(tmp) / "jobs")
            with self.assertRaises(ValueError) as caught:
                store.start(str(wav), model="htdemucs_9s")
            self.assertIn("pick one of", str(caught.exception))
            self.assertIn("htdemucs_6s", str(caught.exception))

    def test_collect_finds_six_stems_for_6s_model(self):
        """_collect 按模型找轨：6s 模型的目录里有 guitar/piano 也要被收进。"""
        import tempfile

        from shadow_stem_splitter import runner
        from shadow_stem_splitter.audio import write_wav as w

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "htdemucs_6s" / "song"
            work.mkdir(parents=True)
            for stem in ("vocals", "drums", "bass", "other", "guitar", "piano"):
                w(work / f"{stem}.wav", (0.2 * np.ones(44100)).astype(np.float32), 44100)
            out = Path(tmp) / "out"
            stems = runner._collect(Path(tmp), out, model="htdemucs_6s")
            self.assertEqual(sorted(stems), ["bass", "drums", "guitar", "other", "piano", "vocals"])
