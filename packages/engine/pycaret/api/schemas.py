"""Higher-level schemas that compose `ParameterCard`s into UI-ready forms."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pycaret.api.cards import ParameterCard


@dataclass(frozen=True)
class SetupParamSchema:
    """Schema describing every parameter of `Experiment.fit(...)` / `setup(...)`.

    A React form can render directly from this: iterate `parameters`, look up
    `kind`/`default`/`choices` per field, and group by `group`.
    """

    task: str
    parameters: list[ParameterCard] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)  # ordered group names, for tab layout

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["parameters"] = [p.to_dict() for p in self.parameters]
        return d
