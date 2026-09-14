# Stem Splitter

把一首成品歌拆成人声、鼓、贝斯和其它四条分轨的 MCP 服务：本地跑 Demucs，或用一条命令指向你的 GPU
机器；任何支持 MCP 的 agent 都能调用。

An MCP server that separates a finished song into **vocals, drums, bass and other** — locally with
Demucs, or through a command you point at a GPU box.

[English](README.md) · 许可证：[AGPL-3.0](LICENSE)

## 为什么是这样设计的

分离一首歌要几分钟，而一个阻塞几分钟的 MCP 调用会被客户端杀掉。所以工作是**作业**：
`separate_stems` 写出作业文件并启动一个独立进程，立刻返回 job id；客户端用 `stem_job_status`
轮询，用 `stem_job_cancel` 中止。bridge 重启、插件重启、合上盖子都不会丢掉作业 —— 磁盘上的记录
就是事实。

模型不是本服务的依赖：插件本体只有标准库 + numpy，Demucs 由"装有它的那个解释器"运行（默认是
`<ShadowRoom>/_venvs/stem-splitter`）。

## 功能

- **四条分轨**：`vocals` / `drums` / `bass` / `other`，就是制作人和 DJ 真正会要的那四份。
- **两种后端**：本地 Demucs（`htdemucs`，或慢得多的 `htdemucs_ft` 精调版），或 `STEM_SPLIT_COMMAND`
  指向远端 GPU / 容器 / 包装脚本。
- **什么格式都能进**：WAV 直接读；mp3 / m4a / flac / aiff / ogg 走 ffmpeg，没有 ffmpeg 时用 macOS
  自带的 `afconvert`。
- **输出干净**：无论后端写出什么格式，四条分轨都会被统一写成 16-bit PCM WAV，任何播放器和 DAW 都能开。
- **错误直说**：没有后端会告诉你怎样装；后端只写了三条，会点名缺哪一条。

## 环境要求

| | |
| --- | --- |
| 系统 | macOS / Linux / Windows |
| Python | 3.9+（MCP 服务本身只要标准库 + numpy） |
| 模型 | 无自带模型：Demucs 装在自己的虚拟环境里，或用远端命令 |
| 解码器 | 非 WAV 需要 ffmpeg，或 macOS 自带的 `afconvert` |

## 安装

```json
{
  "mcpServers": {
    "stem-splitter": {
      "command": "python3",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/stem-splitter"
    }
  }
}
```

本地 Demucs（约 300MB）：

```sh
python3 -m venv ~/Documents/ShadowRoom/_venvs/stem-splitter
~/Documents/ShadowRoom/_venvs/stem-splitter/bin/pip install -r requirements-local.txt
```

首次分离会下载 `htdemucs` 权重（约 80MB）。在 M4 Mac 上实测：20 秒素材 **11.7 秒**分离完成，
四条分轨相加与原曲相关系数 0.999。

版本不是随便定的：torch 2.2.2 是按 numpy 1.x 编译的，所以锁 `numpy<2`；而缺 `soundfile`（或 ffmpeg）
时 torchaudio 连普通 WAV 都打不开，Demucs 会报 "FFmpeg is not installed"。

## 配置

| 选项 | 默认值 | 作用 |
| --- | --- | --- |
| `STEM_SPLIT_PYTHON` | `<ShadowRoom>/_venvs/stem-splitter/bin/python` | 装了 Demucs 的解释器 |
| `STEM_SPLIT_COMMAND` | – | 远端命令，支持 `{input}` / `{output_dir}` / `{model}` |
| `STEM_SPLIT_OUT_DIR` | `<ShadowRoom>/stems` | 分轨输出目录 |
| `STEM_SPLIT_JOB_DIR` | `<ShadowRoom>/stem-splitter-jobs` | 作业记录目录 |
| `SHADOWROOM_HOME` | `~/Documents/ShadowRoom` | 上面这些目录的根 |

## 工具

| 工具 | 作用 |
| --- | --- |
| `separate_stems` | 启动分离（`path`、`out_dir?`、`model?`、`quality?`、`backend?`）→ `{job_id, status, out_dir}` |
| `stem_job_status` | 某个作业的状态、阶段、分轨与错误（不给 id 就列最近的） |
| `stem_job_cancel` | 中止作业并删掉半成品分轨 |
| `stem_backends` | 这台机器有哪些后端，缺的那个怎么开 |

## 安全

除了"取消作业时删掉它自己输出目录里没写完的分轨"，本插件不删除任何文件；源文件只读。

## 许可证

[AGPL-3.0](LICENSE)。分离模型 Demucs（MIT）单独安装，不随本仓库分发。
