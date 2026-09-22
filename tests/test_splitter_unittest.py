"""ADR-025 v2 组增益测试 + 既有 7 条。"""
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_stem_splitter import runner
from shadow_stem_splitter.audio import read_wav




class GroupGainTests(unittest.TestCase):
    """ADR-025 v2：四条的相对电平从头到尾只被一个标量动过。"""

    def _fake_demucs_output(self, work, originals, pre_gain):
        """模拟真实数据流写出（前置衰减过的）stem。

        真实路径是 run_job 先 write_wav（clip ±1）再 ×PRE_GAIN 交给 Demucs——所以 fake 也
        必须「先 clip 后衰减」，否则 >1 的测试峰值会在错误的位置被截，组峰值算错。
        """
        from shadow_stem_splitter.audio import write_wav as w
        work.mkdir(parents=True, exist_ok=True)
        nest = work / "htdemucs" / "track"
        nest.mkdir(parents=True, exist_ok=True)
        for name, samples in originals.items():
            clipped = np.clip(samples, -1.0, 1.0)
            w(nest / f"{name}.wav", clipped * pre_gain, 44100)
            originals[name] = clipped  # 断言也要用 clip 后的（真实世界的原曲就是 ±1 内的）

    def test_stems_sum_restores_the_original(self):
        """四轨相加 vs 原曲：相关系数 ≥0.999、RMS 差 ≤0.5 dB。"""
        rate = 44100
        t = np.arange(rate * 2) / rate
        rng = np.random.default_rng(7)
        # 人工造四条不同电平的 stem（其中 drums 峰值 >0.99，正是 ADR 要求的场景）
        vocals = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        drums = (0.95 * rng.standard_normal(t.size)).astype(np.float32)
        drums[:100] = 1.2  # 故意超 0.99
        bass = (0.2 * np.sin(2 * np.pi * 80 * t)).astype(np.float32)
        other = (0.1 * rng.standard_normal(t.size)).astype(np.float32)
        originals = {"vocals": vocals, "drums": drums, "bass": bass, "other": other}
        self._fake_demucs_output(Path(tempfile.mkdtemp()) / "dry", originals, runner.PRE_GAIN)
        original_mix = sum(originals.values())  # fake 已把 >1 的峰值 clip 到 ±1（真实原曲如此）

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            out = Path(tmp) / "out"
            self._fake_demucs_output(work, originals, runner.PRE_GAIN)
            stems = runner._collect(work, out)

            restored = None
            for stem_path in stems.values():
                samples, r = read_wav(stem_path)
                self.assertEqual(r, rate)
                restored = samples if restored is None else restored + samples
            correlation = float(np.corrcoef(restored, original_mix)[0, 1])
            rms_orig = float(np.sqrt(np.mean(original_mix**2)))
            rms_rest = float(np.sqrt(np.mean(restored**2)))
            rms_db = 20 * np.log10(rms_rest / rms_orig)
            self.assertGreaterEqual(correlation, 0.999, f"相关系数 {correlation}")
            self.assertLessEqual(abs(rms_db), 0.5, f"RMS 差 {rms_db} dB")

    def test_group_peak_is_capped_not_per_stem(self):
        """组峰值超 0.99 → 整组缩到 0.99；没有 stem 被单独拉低。"""
        rate = 44100
        t = np.arange(rate) / rate
        loud = (1.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        quiet = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "w"
            out = Path(tmp) / "o"
            self._fake_demucs_output(work, {"drums": loud, "bass": quiet}, runner.PRE_GAIN)
            stems = runner._collect(work, out)
            drums, _ = read_wav(stems["drums"])
            bass, _ = read_wav(stems["bass"])
            self.assertLessEqual(float(np.max(np.abs(drums))), 0.991, "组峰值必须 ≤0.99")
            # 相对电平：quiet/loud 的比值不变（同一个标量）；基准取 clip 后的（写盘 ±1 是真实世界）
            ratio_before = float(np.max(np.abs(np.clip(quiet, -1, 1))) / np.max(np.abs(np.clip(loud, -1, 1))))
            ratio_after = float(np.max(np.abs(bass)) / np.max(np.abs(drums)))
            self.assertAlmostEqual(ratio_before, ratio_after, places=3)

    def test_rescale_signature_stem_raises(self):
        """峰值恰好 1/1.01 的 stem 是 Demucs rescale 的指纹——必须报错，不静默交付。"""
        rate = 44100
        t = np.arange(rate) / rate
        rescaled = ((1.0 / 1.01) * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "w"
            out = Path(tmp) / "o"
            self._fake_demucs_output(work, {"vocals": rescaled}, runner.PRE_GAIN)
            with self.assertRaises(ValueError) as caught:
                runner._collect(work, out)
            self.assertIn("rescale", str(caught.exception))

    def test_no_clip_when_group_under_threshold(self):
        """组峰值低于 0.99 → 不缩（gain=1.0），样本原样。"""
        rate = 44100
        t = np.arange(rate) / rate
        gentle = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "w"
            out = Path(tmp) / "o"
            self._fake_demucs_output(work, {"bass": gentle}, runner.PRE_GAIN)
            stems = runner._collect(work, out)
            bass, _ = read_wav(stems["bass"])
            self.assertAlmostEqual(float(np.max(np.abs(bass))), 0.4, delta=0.01)
