"""What differs between languages, in one place.

Phase 2 was written against Python and had its assumptions spread through the
parser: `self`, docstrings as the first statement, `__init__.py` for exports.
None of those hold for JavaScript. Everything downstream of the parser (ranking,
chunking, writing, verification) never cared what language it was reading, so
only this file and the walker needed to learn the difference.

Adding a language means adding a spec here and a grammar to requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Language, Node


@dataclass(frozen=True)
class LangSpec:
    name: str
    extensions: tuple[str, ...]
    language: Language

    # Node types that introduce a definition.
    func_nodes: tuple[str, ...]
    class_nodes: tuple[str, ...]
    # Nodes that wrap a definition and should be treated as part of its span:
    # a Python decorator, a JavaScript `export`.
    wrapper_nodes: tuple[str, ...] = ()
    # `self.foo()` in Python, `this.foo()` in JavaScript.
    self_words: tuple[str, ...] = ()
    params_field: str = "parameters"
    bases_field: str = "superclasses"
    # Assignments that create an instance attribute.
    attr_objects: tuple[str, ...] = ()
    module_strips: tuple[str, ...] = field(default_factory=tuple)

    def owns(self, path: Path) -> bool:
        return path.suffix in self.extensions


def _python() -> LangSpec:
    import tree_sitter_python

    return LangSpec(
        name="python",
        extensions=(".py",),
        language=Language(tree_sitter_python.language()),
        func_nodes=("function_definition",),
        class_nodes=("class_definition",),
        wrapper_nodes=("decorated_definition",),
        self_words=("self", "cls"),
        params_field="parameters",
        bases_field="superclasses",
        attr_objects=("self", "cls"),
        module_strips=("src", "lib"),
    )


def _javascript() -> LangSpec:
    import tree_sitter_javascript

    return LangSpec(
        name="javascript",
        extensions=(".js", ".jsx", ".mjs", ".cjs"),
        language=Language(tree_sitter_javascript.language()),
        # An arrow function assigned to a name is a function in every sense
        # that matters here, and modern JavaScript is full of them.
        func_nodes=("function_declaration", "method_definition",
                    "generator_function_declaration", "variable_declarator"),
        class_nodes=("class_declaration",),
        wrapper_nodes=("export_statement",),
        self_words=("this",),
        params_field="parameters",
        bases_field="class_heritage",
        attr_objects=("this",),
        module_strips=("src", "lib", "dist"),
    )


def _tsx() -> LangSpec:
    """.tsx needs its own grammar.

    Parsing JSX with the plain TypeScript grammar fails on the markup: it
    produced 422 partial parses out of 441 .tsx files in one repo.
    """
    import tree_sitter_typescript

    base = _typescript()
    return LangSpec(
        **{**base.__dict__,
           "name": "tsx",
           "extensions": (".tsx",),
           "language": Language(tree_sitter_typescript.language_tsx())}
    )


def _typescript() -> LangSpec:
    import tree_sitter_typescript

    return LangSpec(
        name="typescript",
        extensions=(".ts", ".mts"),
        language=Language(tree_sitter_typescript.language_typescript()),
        func_nodes=("function_declaration", "method_definition",
                    "generator_function_declaration", "variable_declarator",
                    "abstract_method_signature"),
        class_nodes=("class_declaration", "abstract_class_declaration",
                     "interface_declaration"),
        wrapper_nodes=("export_statement",),
        self_words=("this",),
        params_field="parameters",
        bases_field="class_heritage",
        attr_objects=("this",),
        module_strips=("src", "lib", "dist"),
    )


_BUILDERS = (_python, _javascript, _typescript, _tsx)
_CACHE: dict[str, LangSpec] = {}


def all_specs() -> list[LangSpec]:
    """Built lazily: importing three grammars costs time nobody always needs."""
    for build in _BUILDERS:
        spec = build()
        _CACHE.setdefault(spec.name, spec)
    return list(_CACHE.values())


def spec_for(path: Path) -> LangSpec | None:
    for spec in all_specs():
        if spec.owns(path):
            return spec
    return None


def extensions() -> tuple[str, ...]:
    return tuple(e for s in all_specs() for e in s.extensions)


def docstring_of(spec: LangSpec, node: Node, body: Node | None) -> str | None:
    """Python puts docs inside the body, JavaScript puts them above the code."""
    if spec.name == "python":
        if body is None:
            return None
        for child in body.named_children:
            if child.type != "expression_statement":
                return None
            inner = child.named_children[0] if child.named_children else None
            if inner is None or inner.type != "string":
                return None
            for piece in inner.named_children:
                if piece.type == "string_content":
                    return piece.text.decode("utf8", "replace").strip()
            return None
        return None

    # JSDoc: the comment immediately above the definition.
    prev = node.prev_named_sibling
    if prev is not None and prev.type == "comment":
        text = prev.text.decode("utf8", "replace")
        if text.startswith("/*"):
            lines = [ln.strip().lstrip("*").strip()
                     for ln in text.strip("/*").strip("*/").splitlines()]
            cleaned = " ".join(ln for ln in lines if ln and not ln.startswith("@"))
            return cleaned.strip() or None
    return None
