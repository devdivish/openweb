from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

from app.schemas import ToolResult, ToolSpec
from app.store import JsonStore


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    should_run: Callable[[str], bool]
    run: Callable[[str], str]


class ToolRegistry:
    def __init__(self, store: JsonStore):
        self.store = store
        self._tools = {
            "calculator": Tool(
                spec=ToolSpec(
                    id="calculator",
                    name="Calculator",
                    description="Evaluates simple arithmetic expressions safely.",
                ),
                should_run=lambda message: bool(re.search(r"\b(calculate|math|sum|plus|minus|times|divided|=)\b", message, re.I))
                or bool(re.search(r"\d+\s*[-+*/^]\s*\d+", message)),
                run=self._calculator,
            ),
            "list_files": Tool(
                spec=ToolSpec(
                    id="list_files",
                    name="List Files",
                    description="Lists files currently indexed for retrieval.",
                ),
                should_run=lambda message: bool(re.search(r"\b(list|show|what).*\b(files|documents|uploads)\b", message, re.I)),
                run=self._list_files,
            ),
            "file_stats": Tool(
                spec=ToolSpec(
                    id="file_stats",
                    name="File Stats",
                    description="Summarizes indexed file, chunk, and character counts.",
                ),
                should_run=lambda message: bool(re.search(r"\b(stats|statistics|how many|count).*\b(files|chunks|documents)\b", message, re.I)),
                run=self._file_stats,
            ),
            "time": Tool(
                spec=ToolSpec(id="time", name="Current Time", description="Returns the current UTC server time."),
                should_run=lambda message: bool(re.search(r"\b(time|date|today|now)\b", message, re.I)),
                run=lambda _: datetime.now(UTC).isoformat(),
            ),
        }

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def tool_ids(self) -> set[str]:
        return set(self._tools)

    async def run_for_message(
        self,
        message: str,
        tool_ids: list[str] | None = None,
        force_tool_ids: list[str] | None = None,
    ) -> list[ToolResult]:
        selected = []
        requested = set(tool_ids or [])
        forced = set(force_tool_ids or [])
        for tool_id, tool in self._tools.items():
            if tool_id in forced:
                selected.append(tool)
            elif requested and tool_id in requested:
                selected.append(tool)
            elif not requested and tool.should_run(message):
                selected.append(tool)

        results = []
        for tool in selected:
            try:
                output = tool.run(message)
            except Exception as exc:
                output = f"Tool failed: {exc}"
            results.append(
                ToolResult(
                    tool_id=tool.spec.id,
                    name=tool.spec.name,
                    input=message,
                    output=output,
                )
            )
        return results

    def _list_files(self, _: str) -> str:
        files = self.store.list_files()
        if not files:
            return "No files are indexed."
        return "\n".join(
            f"- {item.filename} ({item.id}): {item.chunk_count} chunks, {item.text_chars} text chars"
            for item in files
        )

    def _file_stats(self, _: str) -> str:
        files = self.store.list_files()
        chunks = self.store.chunks()
        chars = sum(item.text_chars for item in files)
        return f"{len(files)} files indexed, {len(chunks)} chunks stored, {chars} extracted text characters."

    def _calculator(self, message: str) -> str:
        expr = extract_expression(message)
        if not expr:
            return "No arithmetic expression found."
        value = safe_eval(expr)
        return f"{expr} = {value}"


def extract_expression(message: str) -> str:
    normalized = (
        message.lower()
        .replace("plus", "+")
        .replace("minus", "-")
        .replace("times", "*")
        .replace("multiplied by", "*")
        .replace("divided by", "/")
        .replace("^", "**")
    )
    matches = re.findall(r"[-+*/().\d\s*]+", normalized)
    matches = [match.strip() for match in matches if re.search(r"\d", match) and re.search(r"[-+*/]", match)]
    return max(matches, key=len) if matches else ""


def safe_eval(expression: str) -> float | int:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](eval_node(node.operand))
        raise ValueError("unsupported expression")

    parsed = ast.parse(expression, mode="eval")
    result = eval_node(parsed)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result
