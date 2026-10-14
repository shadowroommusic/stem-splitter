"""首次分轨的后端自举（desktop-packaging-plan.md §6.2 splitter 档 / §6.4 第 3 步）。

分轨依赖（torch + Demucs ≈ 车轮 160 MB 级）不随 app 包走：第一次真要分轨时才下载，
装进 `<workspace>/_venvs/stem-splitter` + torch hub 检查点目录。两条硬要求（#94-A.3）：

1. **UI 明示**——`status()` 给出预估字节数与一句「首次分轨需下载模型（约 XXX MB，仅一次）」，
   Studio 的分轨菜单与插件面板用它提前告知；下载过程中 job 的 `stage` 一直推进。
2. **失败必须可读、不许卡死**——每个网络动作都有超时与重试；失败抛 `InstallerError`
   （中文、带下一步），调用方（run_job）把它落成 job 的 `error`，用户重试即可；
   配了 `STEM_SPLIT_COMMAND` 的机器照旧走远端命令，完全不触发下载。

下载口令（都可环境覆写，镜像即换 URL）：
  - 车轮：pip（`PIP_INDEX_URL` 等 pip 原生变量照常生效），pins 与现役冻结集一致。
  - 模型：`STEM_SPLIT_MODEL_URLS`（逗号分隔，按序回退）；默认 fbaipublicfiles 直链，
    实测支持 HTTP Range 断点续传（206）。

模型校验沿用 demucs/torch hub 的口径：文件名 `<sig>-<sha256 前 8 位>.th`，
下载完成后整文件 sha256 必须逐字节等于钉死的值——坏资产拒收并自动重下。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from itertools import chain
from typing import Callable

from .backends import workspace

__all__ = ["MODEL_NAME", "status", "install", "run_install_job", "InstallerError"]


class InstallerError(RuntimeError):
    """下载/安装失败——消息面向用户，中文、带下一步。"""


#: htdemucs（demucs 4.0.1 remote/htdemucs.yaml → 签名 955717e8）。文件名后半段是
#: torch hub `check_hash` 的约定（sha256 前 8 位），demucs 自己也按这个名字找。
MODEL_NAME = "955717e8-8726e21a.th"
MODEL_BYTES = 84_141_911
MODEL_SHA256 = "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4"
_DEFAULT_MODEL_URLS = [
    f"https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/{MODEL_NAME}",
]

#: 与本机现役 `_venvs/stem-splitter` 冻结集一致（pip freeze 实测）。两个必须显式钉的：
#: numpy 1.26.4——不钉的话 pip 解析到 2.x，torch 2.2.2 按 numpy 1.x 编，后端初始化
#: 直接失败（实测 numpy 2.0.2 下 demucs 必挂）；soundfile——torchaudio 2.2.2 在 macOS
#: 唯一可用的读 wav 后端，而 demucs 4.0.1 不声明依赖它（真 venv 是当年手动补装的）。
PIP_PINS = ["torch==2.2.2", "torchaudio==2.2.2", "numpy==1.26.4", "soundfile==0.13.1", "demucs==4.0.1"]

#: 车轮侧预估（torch 2.2.2 mac arm64 59.7MB〔PyPI 实测〕+ 依赖树 ~25MB〔冻结集实测〕）。
#: 只用于「约 XXX MB」的提示文案，不参与任何判定。
WHEELS_ESTIMATE_BYTES = 85_000_000

_CHUNK = 1 << 20  # 1MB 一块：下载循环里查一次取消/报一次进度
_TIMEOUT = 60  # 单请求超时（秒）——网络半死时这里断开重试，绝不无限挂着
_MAX_FAILURES = 3  # 同一源连续无进展的连接失败次数——到顶就换源，不无限重试


def hub_checkpoints() -> Path:
    """demucs 找模型的地方（torch hub 约定：`TORCH_HOME/hub` 否则 `~/.cache/torch/hub`）。"""
    root = os.environ.get("TORCH_HOME")
    base = Path(root) / "hub" if root else Path.home() / ".cache" / "torch" / "hub"
    return base / "checkpoints"


def model_path() -> Path:
    return hub_checkpoints() / MODEL_NAME


def venv_python() -> Path:
    return workspace() / "_venvs" / "stem-splitter" / "bin" / "python"


def _demucs_ready() -> bool:
    """venv 里 demucs 可导入（与 backends.find_demucs 同一判据，独立实现会漂移——
    这里只是轻量自检，权威判定仍是 find_demucs）。"""
    python = venv_python()
    if not python.is_file():
        return False
    probe = subprocess.run(
        [str(python), "-c", "import demucs.separate"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return probe.returncode == 0


def status(demucs_available: "bool | None" = None) -> dict:
    """给 `stem_backends` / UI 的安装账：要不要装、多大、一句话明示。"""
    ready = _demucs_ready() if demucs_available is None else demucs_available
    model_ok = _model_file_ok()
    estimate = (0 if ready else WHEELS_ESTIMATE_BYTES + (0 if _ffmpeg_available() else _FFMPEG_ESTIMATE_BYTES)) \
        + (0 if model_ok else MODEL_BYTES)
    return {
        "needed": not (ready and model_ok),
        "venv_ready": ready,
        "model_ready": model_ok,
        "estimate_bytes": estimate,
        "estimate_mb": round(estimate / 1_000_000),
        "note": f"首次分轨需下载模型（约 {round(estimate / 1_000_000)} MB，仅一次）" if estimate else "分轨组件已就绪",
    }


def _model_file_ok() -> bool:
    path = model_path()
    return path.is_file() and path.stat().st_size == MODEL_BYTES


def _base_python() -> str:
    """建 venv 用的解释器：桌面包传的 pbs runtime 优先（3.12，torch 有 cp312 车轮），
    其次当前解释器，最后 PATH 里的 python3。"""
    runtime = os.environ.get("SHADOWROOM_PYTHON_RUNTIME")
    if runtime:
        candidate = Path(runtime) / "bin" / "python3"
        if candidate.is_file():
            return str(candidate)
    return sys.executable or shutil.which("python3") or "python3"


#: demucs 的 AudioFile 直接 shell 出 `ffmpeg`（separate.py 的 errors['ffmpeg'] 分支），
#: torchaudio 2.2.2 在 mac 上没有能读 wav 的内置后端——没有 ffmpeg，分离必挂。
#: 桌面包里它在 core wheels（imageio-ffmpeg 的静态二进制），开发机在用户 site。
_FFMPEG_SITES = [
    "_venvs/.core/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*",  # 桌面打包态（共享 env）
    "_venvs/stem-splitter/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*",  # 兜底轮子装进本 venv 后
    "Library/Python/*/lib/python/site-packages/imageio_ffmpeg/binaries/ffmpeg-*",  # 开发机用户 site
]
#: 哪都没有时兜底装进本 venv 的轮子（+21MB，静态 ffmpeg）。
_FFMPEG_PIN = "imageio-ffmpeg==0.6.0"
#: 兜底轮子的体积（PyPI 实测 21.1MB），只进「约 XXX MB」的预估。
_FFMPEG_ESTIMATE_BYTES = 21_100_000


def _ffmpeg_available() -> bool:
    """PATH 或任何已知 imageio_ffmpeg 位置上有没有 ffmpeg（只读，status 用）。"""
    if _ffmpeg_on_path():
        return True
    for pattern in _FFMPEG_SITES:
        roots = [workspace()] if pattern.startswith("_venvs") else [Path.home()]
        if any(True for _ in chain.from_iterable(root.glob(pattern) for root in roots)):
            return True
    return False


def _ffmpeg_on_path() -> bool:
    from .backends import plugin_env

    return shutil.which("ffmpeg", path=plugin_env()["PATH"]) is not None


def ensure_ffmpeg() -> bool:
    """让 demucs 找得到 ffmpeg：PATH 已有 → 直接 True；否则从已装的 imageio_ffmpeg
    里挑一个二进制软链到 `<workspace>/bin/ffmpeg`（plugin_env 把这个目录排在 PATH 最前，
    这也是现役工作区的既有形态）。False = 本机一个都找不到，调用方需把 imageio-ffmpeg
    加进 pip pins。"""
    if _ffmpeg_on_path():
        return True
    for pattern in _FFMPEG_SITES:
        roots = [workspace()] if pattern.startswith("_venvs") else [Path.home()]
        for root in roots:
            found = sorted(root.glob(pattern))
            if found:
                link = workspace() / "bin" / "ffmpeg"
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(found[-1])
                if _ffmpeg_on_path():
                    return True
    return False


def _pip_log() -> Path:
    path = workspace() / ".stem-install"
    path.mkdir(parents=True, exist_ok=True)
    return path / "pip.log"


def install(
    report: "Callable[[str], None] | None" = None,
    cancel: "Callable[[], bool] | None" = None,
) -> dict:
    """把分轨后端装齐。`report(stage_text)` 每步推进；`cancel()` 为真时尽快停下。

    幂等：已就绪的部分直接跳过（半截 venv / `.part` 模型都会被正确续上或重建）。
    """
    say = report or (lambda _text: None)
    stopped = cancel or (lambda: False)

    # ffmpeg 与 venv 状态无关都要保障（torchaudio 2.2.2 在 mac 上没有它读不了 wav）：
    # 重试路径（venv 半路装好、ffmpeg 缺）不装 venv 也要把链补上。
    ffmpeg_local = ensure_ffmpeg()
    if not _demucs_ready():
        _install_venv(say, stopped, extra_pins=[] if ffmpeg_local else [_FFMPEG_PIN])
        if not ffmpeg_local:
            ensure_ffmpeg()  # 兜底轮子已进 venv：把它暴露成 <workspace>/bin/ffmpeg
        say("校验分轨依赖")
        if not _demucs_ready():
            raise InstallerError(
                "分轨依赖装完后仍然导入不了 demucs——详情见 "
                f"{_pip_log()}；重试一次，仍失败就设置 STEM_SPLIT_COMMAND 改用远端分离。"
            )
    if not _model_file_ok():
        _download_model(say, stopped)
    return status()


def _install_venv(say: "Callable[[str], None]", stopped: "Callable[[], bool]", extra_pins: "list[str] | None" = None) -> None:
    python = venv_python()
    if stopped():
        raise InstallerError("已取消")
    say("创建分轨虚拟环境（约 85 MB 依赖，仅一次）")
    base = _base_python()
    created = subprocess.run(
        [base, "-m", "venv", str(python.parents[1])],  # <workspace>/_venvs/stem-splitter
        capture_output=True,
        text=True,
        timeout=600,
    )
    if created.returncode != 0:
        raise InstallerError(f"虚拟环境创建失败：{created.stderr.strip()[-400:] or created.stdout.strip()[-400:]}")
    if stopped():
        raise InstallerError("已取消")

    say("下载分轨依赖 torch / Demucs（约 85 MB，仅一次）")
    # pip 自带 --retries 重试；日志落盘供诊断。超时兜底 30 分钟：再慢的网络也该断了报错，
    # 而不是让 job 永远挂着（#94-A.3「不能卡死」）。
    command = [
        str(python), "-m", "pip", "install", "--quiet", "--no-input",
        "--timeout", "60", "--retries", "3", *PIP_PINS, *(extra_pins or []),
    ]
    with _pip_log().open("ab") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(command)}\n".encode())
        log.flush()
        try:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        except subprocess.TimeoutExpired:
            raise InstallerError(
                "分轨依赖下载超过 30 分钟被中止——网络恢复后再点一次分轨会接着装；"
                "或设置 STEM_SPLIT_COMMAND 改用远端分离。"
            )
    if proc.returncode != 0:
        raise InstallerError(
            "分轨依赖下载失败（pip 退出码 "
            f"{proc.returncode}，详见 {_pip_log()}）。网络恢复后再点一次分轨即可重试；"
            "也可以设置 STEM_SPLIT_COMMAND 改用远端分离。"
        )


def _model_urls() -> "list[str]":
    override = os.environ.get("STEM_SPLIT_MODEL_URLS")
    if override:
        return [piece.strip() for piece in override.split(",") if piece.strip()]
    return list(_DEFAULT_MODEL_URLS)


def _download_model(say: "Callable[[str], None]", stopped: "Callable[[], bool]") -> None:
    target = model_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    errors = []
    for url in _model_urls():
        try:
            _fetch_with_resume(url, part, say, stopped)
        except InstallerError as exc:
            errors.append(f"{url}: {exc}")
            part.unlink(missing_ok=True)
            continue
        actual = _sha256(part)
        if actual != MODEL_SHA256 or part.stat().st_size != MODEL_BYTES:
            # 坏资产拒收（sha256+大小双校验，Grabber updater 口径）：删掉换下一个源重下，
            # 绝不把校验不过的文件交出去。
            errors.append(f"{url}: sha256/size 校验不符（{part.stat().st_size}B {actual[:16]}…）")
            part.unlink(missing_ok=True)
            continue
        part.replace(target)
        say("模型校验通过")
        return
    raise InstallerError(
        "分轨模型下载失败：" + "；".join(errors[-2:])
        + "。网络恢复后再点一次分轨会续传重试；或设置 STEM_SPLIT_COMMAND 改用远端分离。"
    )


def _fetch_with_resume(url: str, part: Path, say: "Callable[[str], None]", stopped: "Callable[[], bool]") -> None:
    """支持断点续传的下载（服务端 206 实测可用；不支持 Range 的源从头重下）。"""
    import http.client

    done = part.stat().st_size if part.exists() else 0
    if done > MODEL_BYTES:
        part.unlink()  # 比目标还大只可能是脏数据
        done = 0
    last_note = -1.0
    failures = 0
    while done < MODEL_BYTES:
        if stopped():
            raise InstallerError("已取消")
        request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
                total = int(response.headers.get("Content-Range", f"?/{MODEL_BYTES}").split("/")[-1])
                if done and getattr(response, "status", 200) != 206:
                    done = 0  # 源不理会 Range：从头来，绝不能把整包追加到半截文件后面
                with part.open("ab" if done else "wb") as sink:
                    while True:
                        chunk = response.read(_CHUNK)
                        if not chunk:
                            break
                        sink.write(chunk)
                        done += len(chunk)
                        failures = 0  # 有数据进来就重置连败计数
                        if total and time.time() - last_note >= 2.0:
                            last_note = time.time()
                            say(f"下载分轨模型 {done * 100 // total}%（{done // 1_000_000}/{total // 1_000_000} MB，仅一次）")
        except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException) as exc:
            # 单次连接断了不算失败：续传重试；但连续 _MAX_FAILURES 次没有进展就放弃这个源
            # （连接被拒/断流也是「失败要可读、不许卡死」的一部分）。
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise InstallerError(f"HTTP {exc.code}") from exc
            failures += 1
            if failures >= _MAX_FAILURES:
                raise InstallerError(f"连接反复中断（{type(exc).__name__}），未下完") from exc
            time.sleep(2)
            continue
    if done < MODEL_BYTES:
        raise InstallerError("连接反复中断，未下完")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(store: "JobStore", job_id: str, record: dict) -> dict:
    store.path(job_id).write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def run_install_job(job_path: "str | Path") -> dict:
    """`stem_install_backend` 的 job 形态：与分离 job 同一套文件/状态约定，
    `stem_job_status` / `stem_job_cancel` 直接可用（cancel 落在下载循环与阶段边界上）。"""
    from .jobs import JobStore

    store = JobStore(Path(job_path).expanduser().parent)
    job_id = Path(job_path).stem
    record = store.read(job_id)

    def cancel_requested() -> bool:
        try:
            return bool(store.read(job_id).get("cancel_requested"))
        except Exception:  # noqa: BLE001 — job 文件读不了就当没取消
            return False

    try:
        if record.get("cancel_requested"):
            record.update({"status": "cancelled", "finished_at": time.time()})
            return _save(store, job_id, record)

        def report(text: str) -> None:
            record["stage"] = "install"
            record["install_step"] = text
            store.update(job_id, **record)

        report("检查分轨组件")
        result = install(report=report, cancel=cancel_requested)
        record.update({"status": "ok", "stage": "install", "finished_at": time.time(),
                       "error": None, "install": result})
        return _save(store, job_id, record)
    except InstallerError as error:
        cancelled = cancel_requested()
        record.update({"status": "cancelled" if cancelled else "failed", "error": str(error), "finished_at": time.time()})
        return _save(store, job_id, record)
    except Exception as error:  # noqa: BLE001 — 消息就是产品
        record.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "finished_at": time.time()})
        return _save(store, job_id, record)
