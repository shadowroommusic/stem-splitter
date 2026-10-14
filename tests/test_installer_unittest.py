"""installer 的可测面：账目（status）、断点续传、坏资产拒收、失败文案。

真下载（torch/Demucs 车轮 + 80MB 模型）不在单测里跑——那是一次性的真机验收，
数字进总线回执；这里用本地 HTTP 服务器把协议行为（Range/校验/回退/取消）钉死。
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_stem_splitter import installer


class _RangeServer(http.server.BaseHTTPRequestHandler):
    """按请求头伺服的静态文件服务器（模拟 fbaipublicfiles 的 Range 行为）。

    类属性 `flaky` 为真时，**不带 Range 的首请求**只发一半就断流，并且只断一次——
    下载器的续传逻辑应当在下一个请求（带 Range）里把剩下的补齐。
    """

    payload = b""
    flaky = False

    def do_GET(self):  # noqa: N802 — http.server 的命名
        payload = type(self).payload
        start = 0
        header = self.headers.get("Range")
        if header and header.startswith("bytes="):
            start = int(header.split("=", 1)[1].split("-", 1)[0] or 0)
        body = payload[start:]
        short = type(self).flaky and not header
        if short:
            type(self).flaky = False  # 只断一次，重试必须能续上
            body = payload[: len(payload) // 2]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload) - start))  # 断流时谎报剩余长度
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def _start_server(payload: bytes, flaky: bool = False) -> "tuple[str, http.server.ThreadingHTTPServer]":
    _RangeServer.payload = payload
    _RangeServer.flaky = flaky
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/model.th", server


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self._env = {"SHADOWROOM_HOME": str(self.home), "TORCH_HOME": str(self.home / "torch")}
        self._previous = {key: os.environ.get(key) for key in [*self._env, "STEM_SPLIT_MODEL_URLS"]}
        os.environ.update(self._env)
        # 测试用的「模型」很小：把钉死的尺寸/哈希换成本地假件的值，验证的是同一套逻辑
        self._real = (installer.MODEL_BYTES, installer.MODEL_SHA256)
        self.payload = b"model-bytes-" * 64
        installer.MODEL_BYTES = len(self.payload)
        installer.MODEL_SHA256 = hashlib.sha256(self.payload).hexdigest()

    def tearDown(self):
        installer.MODEL_BYTES, installer.MODEL_SHA256 = self._real
        for key, value in self._previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def test_status_counts_what_is_missing(self):
        # 什么都没有：要装，预估 = 车轮 + 模型，文案明示「仅一次」
        info = installer.status(demucs_available=False)
        self.assertTrue(info["needed"])
        self.assertEqual(info["estimate_bytes"], installer.WHEELS_ESTIMATE_BYTES + installer.MODEL_BYTES)
        self.assertIn("仅一次", info["note"])
        # 模型已就位（稀疏文件顶到钉死的尺寸）：只剩车轮的账
        target = installer.model_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            handle.truncate(installer.MODEL_BYTES)
        info = installer.status(demucs_available=False)
        self.assertTrue(info["needed"])
        self.assertTrue(info["model_ready"])
        self.assertEqual(info["estimate_bytes"], installer.WHEELS_ESTIMATE_BYTES)
        # 全齐：不需要装
        info = installer.status(demucs_available=True)
        self.assertFalse(info["needed"])
        self.assertEqual(info["estimate_bytes"], 0)

    def test_download_verifies_and_lands_in_the_hub_cache(self):
        url, server = _start_server(self.payload)
        try:
            os.environ["STEM_SPLIT_MODEL_URLS"] = url
            steps: "list[str]" = []
            installer._download_model(steps.append, lambda: False)
            self.assertTrue(installer.model_path().is_file())
            self.assertEqual(installer.model_path().read_bytes(), self.payload)
            self.assertFalse(installer.model_path().with_suffix(".th.part").exists())
            self.assertTrue(any("下载分轨模型" in step for step in steps))
        finally:
            server.shutdown()

    def test_a_bad_asset_is_rejected_and_the_next_source_is_used(self):
        bad = b"corrupted" * 100
        self.assertNotEqual(hashlib.sha256(bad).hexdigest(), installer.MODEL_SHA256)
        bad_url, bad_server = _start_server(bad)
        good_url, good_server = _start_server(self.payload)
        try:
            os.environ["STEM_SPLIT_MODEL_URLS"] = f"{bad_url},{good_url}"
            installer._download_model(lambda _text: None, lambda: False)
            # 坏源被拒收后没有留下脏文件，好源的文件落位
            self.assertEqual(installer.model_path().read_bytes(), self.payload)
        finally:
            bad_server.shutdown()
            good_server.shutdown()

    def test_a_flaky_connection_resumes_and_completes(self):
        url, server = _start_server(self.payload, flaky=True)
        try:
            os.environ["STEM_SPLIT_MODEL_URLS"] = url
            installer._download_model(lambda _text: None, lambda: False)
            self.assertEqual(installer.model_path().read_bytes(), self.payload)
        finally:
            server.shutdown()

    def test_cancel_stops_the_download(self):
        url, server = _start_server(self.payload)
        try:
            os.environ["STEM_SPLIT_MODEL_URLS"] = url
            with self.assertRaises(installer.InstallerError) as caught:
                installer._download_model(lambda _text: None, lambda: True)
            self.assertIn("取消", str(caught.exception))
            # 半截留在 .part（续传的资本），正式文件没有落位
            self.assertFalse(installer.model_path().exists())
        finally:
            server.shutdown()

    def test_a_dead_source_fails_with_a_usable_sentence(self):
        # 端口 1 上的 HTTP：连接立即被拒且不会恢复——有界的重试后必须给出可读错误
        os.environ["STEM_SPLIT_MODEL_URLS"] = "http://127.0.0.1:1/model.th"
        with self.assertRaises(installer.InstallerError) as caught:
            installer._download_model(lambda _text: None, lambda: False)
        message = str(caught.exception)
        self.assertIn("STEM_SPLIT_COMMAND", message)  # 降级路径必须出现在失败文案里
        self.assertIn("重试", message)

    def test_install_job_lands_cancelled_not_failed(self):
        from shadow_stem_splitter.jobs import JobStore

        store = JobStore(self.home / "jobs")
        record = store.start_install()
        job_id = record["job_id"]
        data = store.read(job_id)
        data["cancel_requested"] = True
        store.path(job_id).write_text(json.dumps(data, ensure_ascii=False))
        result = installer.run_install_job(store.path(job_id))
        self.assertEqual(result["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
