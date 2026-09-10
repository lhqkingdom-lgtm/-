"""Create a small CP-shaped SQLite source for the generic memory smoke test."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def initialize(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing smoke source: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                parent_id TEXT,
                directory TEXT,
                title TEXT,
                version TEXT,
                time_created INTEGER,
                time_updated INTEGER
            );
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                time_created INTEGER,
                time_updated INTEGER,
                data TEXT NOT NULL
            );
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                time_created INTEGER,
                time_updated INTEGER,
                data TEXT NOT NULL
            );
            INSERT INTO session VALUES
                ('smoke-alpha', 'project-alpha', NULL,
                 'D:\\Synthetic Memory Smoke\\Alpha', 'Alpha 固件修复', '1', 1000, 1000),
                ('smoke-beta', 'project-beta', NULL,
                 'D:\\Synthetic Memory Smoke\\Beta', 'Beta 配置流程', '1', 2000, 2000);
            """
        )


def _turn(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    prefix: str,
    timestamp: int,
    user_text: str,
    assistant_text: str,
    tools: list[dict[str, object]] | None = None,
) -> None:
    user_id = f"{prefix}-user"
    assistant_id = f"{prefix}-assistant"
    connection.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        (
            user_id,
            session_id,
            timestamp,
            timestamp,
            _json({"role": "user", "time": {"created": timestamp}}),
        ),
    )
    connection.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (
            f"{prefix}-user-text",
            user_id,
            session_id,
            timestamp + 1,
            timestamp + 1,
            _json({"type": "text", "text": user_text}),
        ),
    )
    connection.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        (
            assistant_id,
            session_id,
            timestamp + 10,
            timestamp + 30,
            _json(
                {
                    "role": "assistant",
                    "parentID": user_id,
                    "finish": "stop",
                    "time": {
                        "created": timestamp + 10,
                        "completed": timestamp + 30,
                    },
                }
            ),
        ),
    )
    for index, tool in enumerate(tools or [], start=1):
        connection.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"{prefix}-tool-{index}",
                assistant_id,
                session_id,
                timestamp + 10 + index,
                timestamp + 10 + index,
                _json({"type": "tool", **tool}),
            ),
        )
    connection.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (
            f"{prefix}-assistant-text",
            assistant_id,
            session_id,
            timestamp + 29,
            timestamp + 29,
            _json({"type": "text", "text": assistant_text}),
        ),
    )


def append_turns(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    base = int(time.time() * 1000) + 1_000
    with sqlite3.connect(path) as connection:
        _turn(
            connection,
            session_id="smoke-alpha",
            prefix="alpha-preference",
            timestamp=base,
            user_text="以后无论在哪个项目，回答我时先给一句结论，再说明关键步骤。",
            assistant_text="明白，以后默认先给一句结论，再说明关键步骤。",
        )
        _turn(
            connection,
            session_id="smoke-alpha",
            prefix="alpha-engineering",
            timestamp=base + 1_000,
            user_text=(
                "请修复这次固件链接失败。若确认了原因和有效方案，请把可复用的条件、失败现象和验证结果说清楚。"
            ),
            assistant_text=(
                "已定位：切换存储布局后复用了旧对象文件，导致链接地址越界。清理对象目录并完整重编译后，"
                "链接恢复且结果为 0 个错误；该方法只适用于布局变化后仍复用旧对象的情况。"
            ),
            tools=[
                {
                    "tool": "firmware-build",
                    "state": {
                        "status": "completed",
                        "input": {"target": "Alpha"},
                        "output": "link failed: address space overflow after storage layout changed",
                    },
                },
                {
                    "tool": "firmware-build",
                    "state": {
                        "status": "completed",
                        "input": {"target": "Alpha", "clean": True},
                        "output": "full rebuild completed: 0 errors, 0 warnings",
                    },
                },
            ],
        )
        _turn(
            connection,
            session_id="smoke-beta",
            prefix="beta-workflow",
            timestamp=base + 2_000,
            user_text="仅在当前项目中，以后修改设备配置前先导出一份可回滚快照，修改后再执行配置校验。",
            assistant_text="已记录当前项目的配置修改流程：先快照，再修改，最后校验。",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("initialize", "append"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "initialize":
        initialize(args.output)
    else:
        append_turns(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
