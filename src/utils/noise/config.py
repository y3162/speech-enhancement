from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..database.schema import Column, Row, Table


NOISE_CONFIGS_TABLE = Table(
    name="noise_configs",
    columns=(
        Column(name="id", type=int, auto_increment=True, primary=True),
        Column(name="json", type=dict, nullable=False, unique=True),
        Column(name="split", type=str, nullable=False),
        Column(name="created_at", type=datetime, default="CURRENT_TIMESTAMP"),
        Column(name="updated_at", type=datetime, default="CURRENT_TIMESTAMP"),
    ),
)


@dataclass(frozen=True, kw_only=True)
class NoiseConfig(Row):
    json: dict[str, Any]
    split: str
