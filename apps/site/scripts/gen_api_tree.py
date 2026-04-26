"""Generate ``content/api-tree.json`` for the docs site.

Walks the public ``pycaret`` package using griffe (the same static
analyser mkdocstrings uses), filters out underscore-private names, and
emits a JSON tree the Next.js ``/reference/[...slug]`` route consumes
to render auto-generated API documentation.

Why griffe over inspect? Griffe parses the source statically, so
optional / heavy imports (sktime, pyod, …) don't fire. It also
preserves comments, Markdown docstrings, and signature TypedDicts
better than runtime introspection. mkdocstrings uses it under the
hood; we get a similar quality bar without committing to mkdocs.

Run from repo root::

    cd apps/site && uv run --with griffe python scripts/gen_api_tree.py

CI runs this before ``next build`` (see ``apps/site/scripts/sync-content.mjs``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import griffe


SITE_ROOT = Path(__file__).resolve().parent.parent
ENGINE_SRC = SITE_ROOT.parent.parent / "packages" / "engine"
OUTPUT_PATH = SITE_ROOT / "content" / "api-tree.json"

# The public surface we expose in the API reference. Submodules below
# this list are recursed into; everything else (internal, deprecated,
# experimental) is skipped at the top level so the reference stays
# focused on what 4.0 callers should actually use.
PUBLIC_ROOTS = [
    "pycaret.classification",
    "pycaret.regression",
    "pycaret.clustering",
    "pycaret.anomaly",
    "pycaret.time_series",
    "pycaret.tasks",
    "pycaret.core.experiment",
    "pycaret.core.results",
    "pycaret.core.errors",
    "pycaret.core.tasks",
    "pycaret.plots",
    "pycaret.plots.classification",
    "pycaret.plots.regression",
    "pycaret.plots.feature",
    "pycaret.plots.clustering",
    "pycaret.plots.anomaly",
    "pycaret.plots.time_series",
    "pycaret.plots.eda",
    "pycaret.datasets",
    "pycaret.api",
    "pycaret.logging",
    "pycaret.logging.events",
    "pycaret.logging.memory",
]


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _serialize_param(p: griffe.Parameter) -> dict[str, Any]:
    return {
        "name": p.name,
        "kind": str(p.kind) if p.kind else None,
        "annotation": _stringify(p.annotation),
        "default": _stringify(p.default) if p.default is not None else None,
    }


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:  # noqa: BLE001 — defensive
        return None


def _serialize_function(obj: griffe.Function, parent: str) -> dict[str, Any]:
    params = list(obj.parameters)
    return {
        "kind": "function",
        "name": obj.name,
        "qualname": f"{parent}.{obj.name}",
        "signature": _function_signature(obj),
        "docstring": obj.docstring.value if obj.docstring else None,
        "parameters": [_serialize_param(p) for p in params],
        "returns": _stringify(obj.returns),
        "is_async": getattr(obj, "is_async", False),
    }


def _function_signature(func: griffe.Function) -> str:
    """Format a Python-like signature string."""
    parts: list[str] = []
    for p in func.parameters:
        if p.kind and "var-positional" in str(p.kind):
            parts.append(f"*{p.name}")
        elif p.kind and "var-keyword" in str(p.kind):
            parts.append(f"**{p.name}")
        else:
            chunk = p.name
            if p.annotation is not None:
                chunk = f"{chunk}: {p.annotation}"
            if p.default is not None:
                chunk = f"{chunk} = {p.default}"
            parts.append(chunk)
    rendered = ", ".join(parts)
    ret = f" -> {func.returns}" if func.returns is not None else ""
    return f"{func.name}({rendered}){ret}"


def _serialize_class(obj: griffe.Class, parent: str) -> dict[str, Any]:
    methods: list[dict[str, Any]] = []
    attributes: list[dict[str, Any]] = []
    for member_name, member in obj.members.items():
        if not _is_public(member_name):
            continue
        if isinstance(member, griffe.Function):
            methods.append(_serialize_function(member, f"{parent}.{obj.name}"))
        elif isinstance(member, griffe.Attribute):
            attributes.append(
                {
                    "name": member.name,
                    "annotation": _stringify(member.annotation),
                    "value": _stringify(member.value)[:200]
                    if member.value is not None
                    else None,
                    "docstring": member.docstring.value if member.docstring else None,
                }
            )
    return {
        "kind": "class",
        "name": obj.name,
        "qualname": f"{parent}.{obj.name}",
        "bases": [_stringify(b) for b in obj.bases],
        "docstring": obj.docstring.value if obj.docstring else None,
        "methods": methods,
        "attributes": attributes,
    }


def _serialize_module(obj: griffe.Module, qualname: str) -> dict[str, Any]:
    classes: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    attributes: list[dict[str, Any]] = []
    for member_name, member in obj.members.items():
        if not _is_public(member_name):
            continue
        if isinstance(member, griffe.Class):
            classes.append(_serialize_class(member, qualname))
        elif isinstance(member, griffe.Function):
            functions.append(_serialize_function(member, qualname))
        elif isinstance(member, griffe.Attribute):
            attributes.append(
                {
                    "name": member.name,
                    "annotation": _stringify(member.annotation),
                    "value": (_stringify(member.value) or "")[:200] or None,
                    "docstring": member.docstring.value if member.docstring else None,
                }
            )
    return {
        "kind": "module",
        "qualname": qualname,
        "docstring": obj.docstring.value if obj.docstring else None,
        "classes": classes,
        "functions": functions,
        "attributes": attributes,
    }


def main() -> int:
    sys.path.insert(0, str(ENGINE_SRC))
    loader = griffe.GriffeLoader(search_paths=[str(ENGINE_SRC)])
    tree: dict[str, Any] = {}
    for qualname in PUBLIC_ROOTS:
        try:
            mod = loader.load(qualname)
        except Exception as exc:  # noqa: BLE001 — best-effort
            print(f"  [skip] {qualname}: {exc}", file=sys.stderr)
            continue
        if isinstance(mod, griffe.Module):
            tree[qualname] = _serialize_module(mod, qualname)
        else:
            print(f"  [skip] {qualname}: not a module", file=sys.stderr)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(tree, indent=2, default=str), encoding="utf-8")
    n_modules = len(tree)
    n_classes = sum(len(m.get("classes", [])) for m in tree.values())
    n_functions = sum(len(m.get("functions", [])) for m in tree.values())
    print(
        f"  wrote {OUTPUT_PATH.relative_to(SITE_ROOT)} "
        f"({n_modules} modules · {n_classes} classes · {n_functions} functions)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
