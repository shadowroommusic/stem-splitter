#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stem Splitter — MCP 服务入口（stdio transport）。

把 `src/` 加进 sys.path 再跑插件自己的 MCP 服务，不需要事先 pip 安装本仓库。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from shadow_stem_splitter.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
