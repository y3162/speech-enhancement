from typing import Any

from .versions import v1_0_0


def parse(data: dict[str, Any]) -> v1_0_0.Pipeline:
    version = data["version"]
    match version:
        case "1.0":
            return v1_0_0.parse(data)
        case _:
            raise ValueError(f"Unsupported noise config version: {version!r}")
