"""Phase 2 — parse each file with tree-sitter and build a call graph.

Consumes the file list from Phase 1 (`ingest.ingest`) and produces the symbol
table + caller->callee edges that Phase 3 chunks and Phase 5 reasons over.

tree-sitter 0.26 API note: build the parser with `Parser(Language(...))`.
There is no `set_language()` and no `.so` grammar to compile.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from ingest import ingest
from languages import LangSpec, docstring_of, spec_for

PY_LANGUAGE = Language(tree_sitter_python.language())

# Calls to these are noise in a call graph — they resolve to nothing in-repo.
BUILTINS = {
    "print", "len", "range", "str", "int", "float", "bool", "list", "dict", "set",
    "tuple", "isinstance", "issubclass", "getattr", "setattr", "hasattr", "delattr",
    "super", "type", "repr", "hash", "iter", "next", "enumerate", "zip", "map",
    "filter", "sorted", "reversed", "sum", "min", "max", "abs", "any", "all",
    "open", "format", "vars", "dir", "id", "callable", "property", "staticmethod",
    "classmethod", "object", "Exception", "ValueError", "TypeError", "KeyError",
    "AttributeError", "RuntimeError", "NotImplementedError", "append", "get",
    "add", "update", "join", "split", "strip", "startswith", "endswith", "items",
    "keys", "values", "pop", "copy", "extend", "insert", "remove", "format_map",
    # Container/str/IO methods. Without type inference `options.setdefault()`
    # would otherwise match any repo symbol that happens to be named setdefault.
    "setdefault", "popitem", "clear", "discard", "sort", "count", "index",
    "find", "replace", "lower", "upper", "title", "encode", "decode",
    "read", "write", "close", "seek", "flush", "splitlines", "rstrip", "lstrip",
}

DEF_NODES = ("function_definition", "class_definition", "decorated_definition")
# Same idea for JavaScript and TypeScript: a nested definition is its own
# symbol, so its calls do not belong to the function that encloses it.
JS_DEF_NODES = ("function_declaration", "class_declaration", "method_definition",
                "generator_function_declaration", "abstract_class_declaration",
                "interface_declaration")


@dataclass
class Symbol:
    """One definition in the repo: a function, a method, or a class."""

    qualname: str
    name: str
    kind: str  # function | method | class
    file: str
    start_line: int
    end_line: int
    docstring: str | None
    source: str
    owner_class: str | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)  # (simple, display)
    bases: list[str] = field(default_factory=list)  # class_definition only
    params: list[str] = field(default_factory=list)  # parameter names
    exported: bool = False  # `export` in JS/TS; Python uses __init__.py

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


def _collect_self_attrs(node: Node, acc: set[str],
                        objects: tuple[str, ...] = ("self", "cls")) -> None:
    """Names assigned as `self.x = ...`, anywhere in the file.

    Instance attributes are not functions or classes, so they never entered the
    symbol table, and documentation that mentioned one (`self._blueprints`) was
    reported as an invented name.
    """
    if node.type in ("assignment", "augmented_assignment"):
        left = node.child_by_field_name("left")
        if left is not None and left.type in ("attribute", "member_expression"):
            obj = left.child_by_field_name("object")
            attr = (left.child_by_field_name("attribute")
                    or left.child_by_field_name("property"))
            if (obj is not None and attr is not None
                    and obj.text.decode("utf8", "replace") in objects):
                acc.add(attr.text.decode("utf8", "replace"))
    for child in node.named_children:
        _collect_self_attrs(child, acc, objects)


@dataclass
class Analysis:
    root: Path
    symbols: dict[str, Symbol]
    edges: list[tuple[str, str]]  # (caller qualname, callee qualname)
    unresolved: list[tuple[str, str]]  # (caller qualname, raw call text)
    parse_errors: list[str]
    attributes: set[str] = field(default_factory=set)  # self.x names

    def fan(self, qualname: str) -> tuple[int, int]:
        """(callers, callees) — the two numbers every ranking decision uses."""
        return len(self.callers(qualname)), len(self.callees(qualname))

    def callees(self, qualname: str) -> list[str]:
        return sorted({c for caller, c in self.edges if caller == qualname})

    def callers(self, qualname: str) -> list[str]:
        return sorted({caller for caller, c in self.edges if c == qualname})


def module_qualname(path: Path, root: Path, spec: LangSpec | None = None) -> str:
    """src/flask/app.py -> flask.app ; src/cordis/index.ts -> cordis"""
    strips = spec.module_strips if spec else ("src", "lib")
    parts = list(path.relative_to(root).with_suffix("").parts)
    while parts and parts[0] in strips:
        parts = parts[1:]
    # A package entry point adds nothing to the name of what is inside it.
    if parts and parts[-1] in ("__init__", "index"):
        parts = parts[:-1]
    return ".".join(parts)


def _docstring(body: Node | None) -> str | None:
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


def _param_names(params: Node | None) -> list[str]:
    """Parameter names, including *args/**kwargs and typed/defaulted forms.

    Phase 7 needs these: docs legitimately write `directory` or `**kwargs` in
    backticks, and without this list the Critic reports them as invented names.
    """
    if params is None:
        return []
    names: list[str] = []
    for child in params.named_children:
        node = child
        # typed_parameter / default_parameter wrap the real identifier
        while node.type in (
            "typed_parameter", "default_parameter", "typed_default_parameter",
            "list_splat_pattern", "dictionary_splat_pattern",
            # JavaScript / TypeScript
            "required_parameter", "optional_parameter", "assignment_pattern",
            "rest_pattern",
        ):
            inner = node.child_by_field_name("name")
            if inner is None:
                inner = next((c for c in node.named_children
                              if c.type in ("identifier", "list_splat_pattern",
                                            "dictionary_splat_pattern")), None)
            if inner is None or inner is node:
                break
            node = inner
        if node.type == "identifier":
            names.append(node.text.decode("utf8", "replace"))
    return [n for n in names if n not in ("self", "cls")]


def _call_name(fn: Node) -> tuple[str, str] | None:
    """Return (simple_name, display_text) for the thing being called."""
    display = fn.text.decode("utf8", "replace")
    if fn.type == "identifier":
        return display, display
    if fn.type in ("attribute", "member_expression"):
        attr = (fn.child_by_field_name("attribute")
                or fn.child_by_field_name("property"))
        if attr is not None:
            return attr.text.decode("utf8", "replace"), display
    return None


def _collect_calls(node: Node, acc: list[tuple[str, str]]) -> None:
    """Gather calls in this scope, stopping at nested def/class boundaries."""
    for child in node.named_children:
        if child.type in DEF_NODES or child.type in JS_DEF_NODES:
            continue  # nested definitions are their own symbols
        # Python calls it `call`, JavaScript `call_expression`, and `new Foo()`
        # is a construction that matters just as much for a call graph.
        if child.type in ("call", "call_expression", "new_expression"):
            fn = child.child_by_field_name("function")
            if fn is None:
                fn = child.child_by_field_name("constructor")
            if fn is not None:
                named = _call_name(fn)
                if named is not None:
                    acc.append(named)
                _collect_calls(fn, acc)  # chained: a().b()
            args = child.child_by_field_name("arguments")
            if args is not None:
                _collect_calls(args, acc)  # nested: f(g(x))
            continue
        _collect_calls(child, acc)


class _FileParser:
    def __init__(self, path: Path, root: Path, spec: LangSpec):
        self.path = path
        self.root = root
        self.spec = spec
        self.parser = Parser(spec.language)
        self.rel = path.relative_to(root).as_posix()
        self.symbols: dict[str, Symbol] = {}
        self.attrs: set[str] = set()

    def run(self) -> tuple[dict[str, Symbol], str | None]:
        try:
            src_bytes = self.path.read_bytes()
        except OSError as exc:
            return {}, f"{self.rel}: {exc}"
        tree = self.parser.parse(src_bytes)
        if tree.root_node.has_error:
            # Still usable — tree-sitter recovers. Record it and keep going.
            err = f"{self.rel}: syntax errors (partial parse)"
        else:
            err = None
        self.src = src_bytes.decode("utf8", "replace")
        _collect_self_attrs(tree.root_node, self.attrs, self.spec.attr_objects)
        self._visit(tree.root_node,
                    module_qualname(self.path, self.root, self.spec), None)
        return self.symbols, err

    def _visit(self, node: Node, prefix: str, owner_class: str | None) -> None:
        spec = self.spec
        for child in node.named_children:
            if child.type in spec.wrapper_nodes:
                # A Python decorator or a JavaScript `export` wraps the real
                # definition. Keep the wrapper as the span so the decorator or
                # the `export` keyword shows up in the source we quote.
                defn = child.child_by_field_name("definition")
                if defn is None:
                    defn = next((c for c in child.named_children
                                 if c.type in spec.func_nodes + spec.class_nodes),
                                None)
                if defn is not None:
                    self._handle_def(defn, prefix, owner_class, outer=child)
                else:
                    self._visit(child, prefix, owner_class)
            elif child.type in spec.func_nodes + spec.class_nodes:
                # `const x = 1` is also a variable_declarator. Only treat it as
                # a function when a function is actually on the right.
                if (child.type == "variable_declarator"
                        and not self._declarator_is_function(child)):
                    self._visit(child, prefix, owner_class)
                    continue
                self._handle_def(child, prefix, owner_class)
            else:
                self._visit(child, prefix, owner_class)

    @staticmethod
    def _declarator_is_function(node: Node) -> bool:
        value = node.child_by_field_name("value")
        return value is not None and value.type in (
            "arrow_function", "function_expression", "function")

    def _handle_def(
        self, node: Node, prefix: str, owner_class: str | None, outer: Node | None = None
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        spec = self.spec
        name = name_node.text.decode("utf8", "replace")
        qualname = f"{prefix}.{name}" if prefix else name
        span_node = outer or node  # decorators and `export` count as part of it
        is_class = node.type in spec.class_nodes

        # `const greet = () => ...` keeps its body and parameters on the arrow,
        # not on the declarator that names it.
        defn_node = node
        if node.type == "variable_declarator":
            value = node.child_by_field_name("value")
            if value is not None:
                defn_node = value
        body = defn_node.child_by_field_name("body")

        sym = Symbol(
            qualname=qualname,
            name=name,
            kind="class" if is_class else ("method" if owner_class else "function"),
            file=self.rel,
            start_line=span_node.start_point[0] + 1,
            end_line=span_node.end_point[0] + 1,
            docstring=docstring_of(spec, span_node, body),
            source=self.src[span_node.start_byte : span_node.end_byte],
            owner_class=owner_class,
            # In JavaScript the public API is stated at the definition itself,
            # not collected in a package file.
            exported=bool(outer is not None
                          and outer.type == "export_statement"),
        )
        if is_class:
            supers = node.child_by_field_name(spec.bases_field)
            if supers is not None:
                for arg in supers.named_children:
                    if arg.type == "identifier":
                        sym.bases.append(arg.text.decode("utf8", "replace"))
                    elif arg.type in ("attribute", "member_expression"):
                        attr = (arg.child_by_field_name("attribute")
                                or arg.child_by_field_name("property"))
                        if attr is not None:
                            sym.bases.append(attr.text.decode("utf8", "replace"))
        elif body is not None:
            _collect_calls(body, sym.calls)
            sym.params = _param_names(
                defn_node.child_by_field_name(spec.params_field))

        # Same name twice in a file (conditional defs, overloads) — keep the first.
        self.symbols.setdefault(qualname, sym)

        if body is not None:
            # Inside a class, children are methods. Inside a function, nested defs
            # are plain functions again.
            self._visit(body, qualname, qualname if is_class else None)


def analyze_files(root: Path, files: list[Path]) -> Analysis:
    symbols: dict[str, Symbol] = {}
    errors: list[str] = []
    attributes: set[str] = set()

    for path in files:
        spec = spec_for(path)
        if spec is None:
            continue
        fp = _FileParser(path, root, spec)
        found, err = fp.run()
        attributes |= fp.attrs
        symbols.update(found)
        if err:
            errors.append(err)

    by_simple: dict[str, list[str]] = {}
    for qual, sym in symbols.items():
        by_simple.setdefault(sym.name, []).append(qual)

    def mro(class_qual: str) -> list[str]:
        """Class + its in-repo ancestors, breadth-first. Cycle-safe."""
        chain, queue, seen_cls = [], [class_qual], {class_qual}
        while queue:
            cur = queue.pop(0)
            chain.append(cur)
            cls_sym = symbols.get(cur)
            if cls_sym is None:
                continue
            for base in cls_sym.bases:
                for cand in by_simple.get(base, []):
                    if symbols[cand].kind == "class" and cand not in seen_cls:
                        seen_cls.add(cand)
                        queue.append(cand)
        return chain

    edges: list[tuple[str, str]] = []
    unresolved: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for qual, sym in symbols.items():
        for simple, display in sym.calls:
            target = None
            # self.foo() / super().foo() -> search the class, then its ancestors.
            # Must be exactly two segments: `self.options.get` is a call on an
            # attribute of self, not a method on self.
            on_self = display in (f"self.{simple}", f"super().{simple}", f"cls.{simple}")
            if on_self and sym.owner_class:
                for cls in mro(sym.owner_class):
                    cand = f"{cls}.{simple}"
                    if cand in symbols and cand != qual:
                        target = cand
                        break
            if target is None:
                # A bare `.get()` on some local object is not a call graph edge.
                if simple in BUILTINS:
                    continue
                matches = by_simple.get(simple, [])
                if len(matches) == 1:
                    target = matches[0]
                elif len(matches) > 1:
                    unresolved.append((qual, f"{display} (ambiguous: {len(matches)})"))
                    continue
            if target is None:
                unresolved.append((qual, display))
                continue
            if target == qual:
                continue  # ignore direct recursion in the graph
            key = (qual, target)
            if key not in seen:
                seen.add(key)
                edges.append(key)

    return Analysis(root, symbols, edges, unresolved, errors, attributes)


def analyze(url: str, subdir: str = "") -> Analysis:
    root, files = ingest(url, subdir=subdir)
    return analyze_files(root, files)


def _report(a: Analysis) -> None:
    kinds: dict[str, int] = {}
    for s in a.symbols.values():
        kinds[s.kind] = kinds.get(s.kind, 0) + 1

    print(f"\n[analyze] {len(a.symbols)} symbols  "
          + "  ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    total = len(a.edges) + len(a.unresolved)
    pct = (100 * len(a.edges) / total) if total else 0
    print(f"[analyze] {len(a.edges)} resolved edges, "
          f"{len(a.unresolved)} unresolved ({pct:.0f}% resolved)")
    for err in a.parse_errors:
        print(f"[analyze] WARN {err}")

    fan_out = sorted(a.symbols, key=lambda q: len(a.callees(q)), reverse=True)
    print("\n--- most connected functions (what to document first) ---")
    for qual in fan_out[:8]:
        outs, ins = a.callees(qual), a.callers(qual)
        if not outs:
            break
        sym = a.symbols[qual]
        print(f"\n{qual}  ({sym.file}:{sym.start_line}, "
              f"{len(outs)} calls out, {len(ins)} in)")
        for callee in outs[:6]:
            print(f"    -> {callee}")
        if len(outs) > 6:
            print(f"    -> ... {len(outs) - 6} more")

    entry = [q for q in a.symbols if not a.callers(q) and a.callees(q)]
    print(f"\n--- {len(entry)} entry points (called by nothing in-repo) ---")
    for qual in sorted(entry)[:10]:
        print(f"  {qual}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python analyze.py <github-repo-url>")
    _report(analyze(sys.argv[1]))
