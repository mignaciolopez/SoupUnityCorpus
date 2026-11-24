#!/usr/bin/env python3
"""
Construye un corpus para Soup (soup-cli) a partir de un proyecto Unity.

Recolecta:
  * Assets/**/*.cs                      -> codigo propio del proyecto
  * Packages/<pkg>/** (embebidos/locales) -> *.cs y *.md
  * Library/PackageCache/<pkg>@<ver>/**   -> *.cs y *.md (solo los packages que usa el proyecto,
                                             segun Packages/packages-lock.json o manifest.json)
  * Documentation~/ de cada package       -> *.md (Unity ignora carpetas "~", pero ahi esta la doc)

Salida:
  <out>/corpus.jsonl     {"text": ...}  -> para `task: pretrain`, `format: plaintext`
  <out>/docs_flat/*.md   docs aplanadas en UN nivel -> para `soup data forge --docs`
  <out>/manifest.json    que se incluyo (package, version, archivos, chars)

Uso:
  python prepare_unity_corpus.py /ruta/MiProyectoUnity -o ./data
  python prepare_unity_corpus.py /ruta/MiProyectoUnity -o ./data --max-chars 12000 --include-tests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

CODE_EXT = {".cs"}
DOC_EXT = {".md"}
SKIP_DIRS = {".git", "obj", "bin", "node_modules", ".vs", ".idea"}
TEST_DIR_RE = re.compile(r"(^|/)(Tests?|Editor/Tests?)(/|$)", re.IGNORECASE)
GENERATED_RE = re.compile(r"(\.g\.cs|\.designer\.cs|AssemblyInfo\.cs)$", re.IGNORECASE)
MAX_FILE_BYTES = 1_000_000


# ---------------------------------------------------------------- packages
def load_used_packages(project: Path) -> dict[str, dict]:
    """name -> {"version": str, "source": str}. Prefiere packages-lock.json (incluye transitivos)."""
    lock = project / "Packages" / "packages-lock.json"
    manifest = project / "Packages" / "manifest.json"
    pkgs: dict[str, dict] = {}
    if lock.exists():
        data = json.loads(lock.read_text(encoding="utf-8"))
        for name, info in data.get("dependencies", {}).items():
            pkgs[name] = {"version": str(info.get("version", "")), "source": info.get("source", "")}
    elif manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for name, ver in data.get("dependencies", {}).items():
            pkgs[name] = {"version": str(ver), "source": ""}
    return pkgs


def resolve_package_dir(project: Path, name: str, info: dict) -> Path | None:
    ver = info.get("version", "")
    # 1) embebido en Packages/<name> (o carpeta con package.json cuyo "name" coincide)
    embedded = project / "Packages" / name
    if embedded.is_dir():
        return embedded
    # 2) local "file:../algo"
    if ver.startswith("file:"):
        p = (project / "Packages" / ver[5:]).resolve()
        return p if p.is_dir() else None
    # 3) Library/PackageCache/<name>@<version o hash>
    cache = project / "Library" / "PackageCache"
    if cache.is_dir():
        exact = cache / f"{name}@{ver}"
        if exact.is_dir():
            return exact
        cands = sorted(cache.glob(f"{name}@*"))
        if cands:
            return cands[-1]
    # 4) buscar por package.json dentro de Packages/
    for pj in (project / "Packages").glob("*/package.json"):
        try:
            if json.loads(pj.read_text(encoding="utf-8")).get("name") == name:
                return pj.parent
        except Exception:
            pass
    return None


# ---------------------------------------------------------------- archivos
def iter_files(root: Path, exts: set[str], include_tests: bool):
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(root).as_posix()
        parts = set(rel.split("/"))
        if parts & SKIP_DIRS:
            continue
        if not include_tests and TEST_DIR_RE.search(rel):
            continue
        if GENERATED_RE.search(p.name):
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            continue
        yield p, rel


def read_text(p: Path) -> str | None:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return None


def clean(text: str, is_code: bool) -> str:
    text = text.replace("\r\n", "\n").replace("\t", "    ")
    if is_code:
        # quitar bloques de licencia largos al inicio (opcional, reduce ruido repetido)
        text = re.sub(r"\A(\s*//[^\n]*\n){8,}", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def chunk(text: str, max_chars: int) -> list[str]:
    """Corta en lineas en blanco cercanas al limite para no partir metodos a la mitad."""
    if len(text) <= max_chars:
        return [text]
    out, buf = [], []
    size = 0
    for block in re.split(r"(\n\s*\n)", text):
        if size + len(block) > max_chars and buf:
            out.append("".join(buf).strip() + "\n")
            buf, size = [], 0
        while len(block) > max_chars:            # bloque gigante: corte duro por lineas
            cut = block.rfind("\n", 0, max_chars) or max_chars
            out.append(block[:cut].strip() + "\n")
            block = block[cut:]
        buf.append(block)
        size += len(block)
    if "".join(buf).strip():
        out.append("".join(buf).strip() + "\n")
    return out


def header(origin: str, rel: str, is_code: bool, part: int, total: int) -> str:
    suffix = f" (parte {part}/{total})" if total > 1 else ""
    if is_code:
        return f"// Source: {origin}\n// File: {rel}{suffix}\n\n"
    return f"<!-- Source: {origin} | File: {rel}{suffix} -->\n\n"


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path, help="raiz del proyecto Unity")
    ap.add_argument("-o", "--out", type=Path, default=Path("data"))
    ap.add_argument("--max-chars", type=int, default=12000,
                    help="~3.5 chars/token en C#: 12000 ~= 3.4k tokens (para max_length 4096)")
    ap.add_argument("--include-tests", action="store_true")
    ap.add_argument("--skip-assets", action="store_true", help="no incluir Assets/ (solo packages)")
    ap.add_argument("--only", nargs="*", default=None, help="limitar a estos packages (prefijo de nombre)")
    args = ap.parse_args()

    project = args.project.resolve()
    if not (project / "Packages").is_dir():
        print(f"ERROR: {project} no parece un proyecto Unity (falta Packages/)", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    docs_dir = args.out / "docs_flat"
    docs_dir.mkdir(exist_ok=True)

    sources: list[tuple[str, Path]] = []
    if not args.skip_assets and (project / "Assets").is_dir():
        sources.append((f"project {project.name} (Assets)", project / "Assets"))

    missing = []
    for name, info in sorted(load_used_packages(project).items()):
        if args.only and not any(name.startswith(o) for o in args.only):
            continue
        d = resolve_package_dir(project, name, info)
        if d is None:
            # com.unity.modules.* son built-in: no tienen fuente en disco, es normal
            if not name.startswith("com.unity.modules."):
                missing.append(name)
            continue
        ver = info.get("version", "")
        sources.append((f"package {name}@{ver}" if ver else f"package {name}", d))

    seen: set[str] = set()
    report = {"sources": [], "missing_packages": missing}
    n_rows = n_docs = n_dups = 0

    with (args.out / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for origin, root in sources:
            stats = {"origin": origin, "path": str(root), "cs": 0, "md": 0, "chars": 0}
            for exts, is_code in ((CODE_EXT, True), (DOC_EXT, False)):
                for p, rel in iter_files(root, exts, args.include_tests):
                    raw = read_text(p)
                    if not raw or not raw.strip():
                        continue
                    body = clean(raw, is_code)
                    h = hashlib.sha256(body.encode()).hexdigest()
                    if h in seen:                      # mismo archivo en Packages/ y PackageCache
                        n_dups += 1
                        continue
                    seen.add(h)
                    parts = chunk(body, args.max_chars)
                    for i, part in enumerate(parts, 1):
                        f.write(json.dumps({"text": header(origin, rel, is_code, i, len(parts)) + part},
                                           ensure_ascii=False) + "\n")
                        n_rows += 1
                    stats["cs" if is_code else "md"] += 1
                    stats["chars"] += len(body)
                    if not is_code:
                        safe = re.sub(r"[^\w.@-]+", "_", f"{origin.split(' ', 1)[1]}__{rel}")
                        (docs_dir / safe).write_text(header(origin, rel, False, 1, 1) + body, encoding="utf-8")
                        n_docs += 1
            report["sources"].append(stats)

    (args.out / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    total_chars = sum(s["chars"] for s in report["sources"])
    print(f"fuentes: {len(sources)}  filas: {n_rows}  docs .md: {n_docs}  duplicados omitidos: {n_dups}")
    print(f"~{total_chars/1e6:.1f}M chars  (~{total_chars/3.5/1e6:.1f}M tokens aprox)")
    if missing:
        print(f"packages sin fuente en disco ({len(missing)}): {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else "") + "  -> abri el proyecto en Unity para poblar PackageCache")
    print(f"-> {args.out/'corpus.jsonl'}\n-> {docs_dir}/\n-> {args.out/'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
