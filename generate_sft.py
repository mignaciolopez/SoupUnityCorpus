#!/usr/bin/env python3
"""
generate_sft.py — builds an SFT dataset (ChatML) for Soup "Option A" from your Unity sources,
using a local Ollama model as the teacher.

Reads data/manifest.json (written by prepare_unity_corpus.py) to locate every source folder, then:
  * your own code  (Assets + packages matching --own-prefix, e.g. com.cyberentt): ALL .cs and .md files
  * Unity packages (--unity-packages): the N most "public-API-heavy" runtime .cs files + the main docs
  * optional general C#/Unity questions (--general N) so the 3B base keeps general skills

For each chunk it asks the teacher for grounded Q&A pairs (usage / explain / write-code / pitfalls),
validates them, and appends ChatML rows:
  data/sft_generated.jsonl      {"messages":[system,user,assistant]}   <- train on this
  data/sft_generated.prov.jsonl provenance (package, file, chunk) per row, same order

Resumable: chunks already done are recorded in data/sft_generated.state.json; re-run to continue.
Stdlib only (no pip installs). Requires Ollama running:  ollama pull qwen2.5-coder:7b

Examples (PowerShell, from D:\\Projects\\LLM\\SoupUnityCorpus, with .venv active):
  python generate_sft.py --limit 5                 # quick test: 5 chunks, then read the output
  python generate_sft.py                           # full run (hours; resumable)
  python generate_sft.py --general 300             # also add 300 general C#/Unity rows
  python generate_sft.py --dry-run                 # only list what would be processed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_UNITY_PACKAGES = [
    "com.unity.entities", "com.unity.collections", "com.unity.mathematics", "com.unity.burst",
    "com.unity.physics", "com.unity.netcode", "com.unity.transport", "com.unity.inputsystem",
    "com.unity.entities.graphics", "com.unity.addressables",
]
SYSTEM_PROMPT = ("You are an expert Unity C# assistant for the CyberEntity project. "
                 "You know its packages and the Unity packages it uses (Entities/DOTS, Netcode, "
                 "Physics, Input System, etc.). Answer precisely and use real APIs.")

SKIP_DIR_PARTS = {"Tests", "Test", "Samples~", "Tests~", "Documentation~/images", "ThirdParty"}
SKIP_DOC_NAMES = re.compile(r"(CHANGELOG|LICENSE|Third Party Notices|THIRD-PARTY|CONTRIBUTING)",
                            re.IGNORECASE)
PUBLIC_TYPE_RE = re.compile(r"\bpublic\s+(?:static\s+|sealed\s+|abstract\s+|partial\s+|readonly\s+|unsafe\s+)*"
                            r"(?:class|struct|interface|enum|record)\s+\w+")
PUBLIC_MEMBER_RE = re.compile(r"\bpublic\s+[\w<>\[\],\s]+\s+\w+\s*[\(\{=;]")
IDENT_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]{3,}\b")

KINDS_OWN = ("usage: how to use a specific class/method from this code, with a short C# example; "
             "explain: what a specific type or method does and why; "
             "write-code: a realistic task solved by writing new C# that calls this code; "
             "pitfalls: a mistake or edge case when using this code and how to avoid it")
KINDS_DOC = ("concept: explain a concept from the docs; how-to: steps or C# code to accomplish "
             "something described; comparison: when to use one option vs another; "
             "pitfalls: a caveat or limitation the docs mention")

PROMPT_TEMPLATE = """You are creating training data for a coding assistant specialised in a Unity project.

Source: {origin}
File: {rel}

<source>
{body}
</source>

Write {n} diverse question/answer pairs a Unity developer would realistically ask about the source above.
Cover these kinds: {kinds}.

Rules:
- Ground every answer in the source. Use ONLY type, method and field names that appear in it
  (plus standard C#/.NET/UnityEngine APIs). Never invent APIs.
- Questions must be self-contained: name the class/package; never say "the code above" or "this passage".
- Copy constructors, method signatures, generic parameters and enum values EXACTLY as declared in the source.
  If you are not sure how to construct or call something, explain it in prose instead of writing code.
- Every code example must compile. If you announce an example, include the full ```csharp block.
- Answers: clear prose plus C# code blocks (```csharp) when useful. 60-400 words.
- Write in {lang}.

Return ONLY JSON: {{"pairs": [{{"kind": "...", "question": "...", "answer": "..."}}]}}"""

GENERAL_TOPICS = [
    "C# async/await vs Unity coroutines", "Unity ECS systems and SystemAPI queries", "Burst-compatible code rules",
    "NativeArray and Allocator lifetimes", "IJobEntity vs IJobChunk", "Netcode for Entities ghosts and prediction",
    "Unity Physics collision filters", "Input System action maps", "ScriptableObject architecture",
    "Unity serialization rules", "C# generics and constraints", "LINQ performance in hot paths",
    "Unity profiling and GC allocations", "Addressables loading and release", "unit testing with Unity Test Framework",
    "C# structs vs classes in Unity", "Entities baking and authoring components", "Unity.Mathematics vs Vector3",
    "dependency handling of jobs (JobHandle)", "C# events and delegates in Unity", "Unity editor scripting basics",
    "fixing NullReferenceException in Unity", "object pooling", "state machines for gameplay",
]


# ----------------------------------------------------------------------------- selection
def is_skipped(rel: str) -> bool:
    return any(part in SKIP_DIR_PARTS for part in rel.split("/"))


def api_score(text: str) -> float:
    """Public API per KB: favours hand-written API files over huge generated ones."""
    raw = 5 * len(PUBLIC_TYPE_RE.findall(text)) + len(PUBLIC_MEMBER_RE.findall(text))
    return raw / max(1.0, len(text) / 1000)


def read(p: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return ""


def select_files(manifest: dict, own_prefix: list[str], unity_pkgs: list[str],
                 files_per_pkg: int, docs_per_pkg: int, include_editor: bool,
                 exclude: re.Pattern, max_unity_file_kb: int):
    """Yields (origin, root, path, rel, is_own, is_code)."""
    for src in manifest["sources"]:
        origin, root = src["origin"], Path(src["path"])
        if not root.is_dir():
            print(f"  ! missing folder, skipped: {root}", file=sys.stderr)
            continue
        name = origin.split(" ", 1)[1].split("@", 1)[0] if origin.startswith("package ") else ""
        is_project = origin.startswith("project ")
        is_own = is_project or any(name.startswith(p) for p in own_prefix)
        is_unity = any(name == p for p in unity_pkgs)
        if not (is_own or is_unity):
            continue

        cs, md = [], []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if is_skipped(rel) or exclude.search(rel):
                continue
            ext = p.suffix.lower()
            if ext == ".cs":
                if not is_own and not include_editor and "/Editor/" in f"/{rel}":
                    continue
                cs.append((p, rel))
            elif ext == ".md" and not SKIP_DOC_NAMES.search(p.name):
                md.append((p, rel))

        if not is_own:  # Unity package: keep the most API-dense files and the biggest docs
            cs = [x for x in cs if x[0].stat().st_size <= max_unity_file_kb * 1024]  # skip generated giants
            cs = sorted(cs, key=lambda x: api_score(read(x[0])), reverse=True)[:files_per_pkg]
            md = sorted(md, key=lambda x: x[0].stat().st_size, reverse=True)[:docs_per_pkg]
        for p, rel in cs:
            yield origin, root, p, rel, is_own, True
        for p, rel in md:
            yield origin, root, p, rel, is_own, False


def chunks_of(text: str, max_chars: int) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    out, buf, size = [], [], 0
    for block in re.split(r"(\n\s*\n)", text):
        if size + len(block) > max_chars and buf:
            out.append("".join(buf).strip())
            buf, size = [], 0
        while len(block) > max_chars:
            cut = block.rfind("\n", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(block[:cut].strip())
            block = block[cut:]
        buf.append(block)
        size += len(block)
    if "".join(buf).strip():
        out.append("".join(buf).strip())
    return [c for c in out if len(c) > 200]  # tiny fragments make poor questions


# ----------------------------------------------------------------------------- teacher
def ollama_chat(host: str, model: str, prompt: str, num_ctx: int, temperature: float,
                timeout: int, retries: int = 3) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"num_ctx": num_ctx, "temperature": temperature},
    }).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{host}/api/chat", data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Ollama request failed after {retries} tries: {last}")


def parse_pairs(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    pairs = data.get("pairs", data if isinstance(data, list) else [])
    return [p for p in pairs if isinstance(p, dict)]


CODE_BLOCK_RE = re.compile(r"```(?:csharp|cs|c#)?\s*\n(.*?)```", re.S | re.I)
COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
WORD_RE = re.compile(r"\b[A-Za-z_]\w*\b")
STD_NS = ("System", "Unity", "UnityEngine", "UnityEditor", "Unity.", "TMPro", "NUnit")
# members that are fine even if the source never mentions them
COMMON_MEMBERS = {
    "Temp", "TempJob", "Persistent", "None", "Log", "LogWarning", "LogError", "LogException", "Assert",
    "Instance", "Count", "Length", "Dispose", "ToString", "Equals", "GetHashCode", "Value", "IsCreated",
    "Add", "Remove", "Clear", "Contains", "Create", "Destroy", "Instantiate", "GetComponent", "AddComponent",
    "Invoke", "Empty", "Zero", "One", "Max", "Min", "Abs", "Sqrt", "Format", "Concat", "Join", "Now", "Time",
    "deltaTime", "Update", "Start", "Awake", "Run", "Schedule", "Complete", "CompleteDependency",
}


def invented_api(answer: str, source: str, is_code: bool) -> str | None:
    """Flags code in the answer that uses names the source does not have."""
    blocks = CODE_BLOCK_RE.findall(answer)
    if not blocks:
        return None
    src = COMMENT_RE.sub(" ", source) if is_code else source
    src_words = set(WORD_RE.findall(src))
    declared = set(re.findall(r"\b(?:class|struct|interface|enum|record)\s+(\w+)", src))
    for code in blocks:
        # 1) project namespaces must exist in the source
        for ns in re.findall(r"^\s*using\s+([\w.]+)\s*;", code, re.M):
            if not ns.startswith(STD_NS) and ns not in source:
                return "wrong namespace"
        # 2) Type.Member where Type is from the source -> Member must be in the source too
        for typ, mem in re.findall(r"\b([A-Z]\w*)\.([A-Za-z_]\w*)", code):
            if typ in src_words and mem not in src_words and mem not in COMMON_MEMBERS:
                return "invented member"
        # 3) variables of source-declared types: var.Member / new T { Member = .. } must exist
        vars_of = {v: t for t, v in re.findall(r"\b([A-Z]\w*)(?:<[^<>]*>)?\s+([a-z_]\w*)\s*[=;),]", code)
                   if t in declared}
        vars_of.update({v: t for v, t in re.findall(r"\bvar\s+(\w+)\s*=\s*new\s+([A-Z]\w*)", code)
                        if t in declared})
        for var, mem in re.findall(r"\b([a-z_]\w*)\.([A-Za-z_]\w*)", code):
            if var in vars_of and mem not in src_words and mem not in COMMON_MEMBERS:
                return "invented member"
        for typ, body in re.findall(r"new\s+([A-Z]\w*)\s*\{([^}]*)\}", code):
            if typ in declared:
                for mem in re.findall(r"\b(\w+)\s*=", body):
                    if mem not in src_words:
                        return "invented member"
    return None


BAD_PHRASES = ("the code above", "this passage", "the passage", "the source above", "(offline)",
               "as an ai", "i cannot", "provided code", "given code")


def valid_pair(p: dict, source: str, is_code: bool = True) -> str | None:
    """Returns a rejection reason, or None if the pair is OK."""
    q, a = str(p.get("question", "")).strip(), str(p.get("answer", "")).strip()
    if len(q) < 15 or len(a) < 80:
        return "too short"
    if len(a) > 6000:
        return "too long"
    if len(a.split()) < 20:
        return "too short"
    if a.rstrip().endswith(":") or (re.search(r"(here'?s|following|below).{0,40}(example|code|snippet)", a, re.I)
                                    and "```" not in a):
        return "missing code"
    if a.count("```") % 2:
        return "broken code block"
    low = (q + " " + a).lower()
    if any(b in low for b in BAD_PHRASES):
        return "not self-contained"
    src_idents = set(IDENT_RE.findall(source))
    if src_idents and not (set(IDENT_RE.findall(q + " " + a)) & src_idents):
        return "not grounded"  # mentions nothing that exists in the source
    spoken = {" ".join(re.findall(r"[A-Z][a-z0-9]+", t)).lower() for t in
              re.findall(r"\b(?:class|struct|interface|enum)\s+(\w+)", source)}  # DeferredLog -> "deferred log"
    if (src_idents and not (set(IDENT_RE.findall(q)) & src_idents) and not re.search(r"cyberent", q, re.I)
            and not any(sp and sp in q.lower() for sp in spoken)):
        return "vague question"  # e.g. "How do I add an item to the queue?" without naming the type
    return invented_api(a, source, is_code)


def show_rows(out: Path, prov_path: Path, n: int) -> int:
    if not out.exists():
        print(f"{out} does not exist yet")
        return 1
    rows = out.read_text(encoding="utf-8").splitlines()
    prov = prov_path.read_text(encoding="utf-8").splitlines() if prov_path.exists() else []
    idx = sorted(random.sample(range(len(rows)), min(n, len(rows))))
    for i in idx:
        msgs = json.loads(rows[i])["messages"]
        src = json.loads(prov[i]) if i < len(prov) else {}
        print("=" * 100)
        print(f"#{i}  {src.get('origin', '')} | {src.get('file', '')} | {src.get('kind', '')}")
        print("-" * 100)
        print("Q:", next(m["content"] for m in msgs if m["role"] == "user"))
        print("\nA:", next(m["content"] for m in msgs if m["role"] == "assistant"))
    print("=" * 100)
    print(f"{len(rows)} rows in {out}")
    return 0


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    ap.add_argument("--out", type=Path, default=Path("data/sft_generated.jsonl"))
    ap.add_argument("--own-prefix", nargs="+", default=["com.cyberentt"])
    ap.add_argument("--unity-packages", nargs="*", default=DEFAULT_UNITY_PACKAGES)
    ap.add_argument("--files-per-package", type=int, default=15, help="top API files per Unity package")
    ap.add_argument("--docs-per-package", type=int, default=8, help="top .md docs per Unity package")
    ap.add_argument("--include-editor", action="store_true", help="include Editor/ code of Unity packages")
    ap.add_argument("--max-unity-file-kb", type=int, default=120, help="skip bigger Unity .cs files (generated)")
    ap.add_argument("--max-chunks-per-file", type=int, default=2, help="cap per Unity file (own code: no cap)")
    ap.add_argument("--max-chunks-per-package", type=int, default=40, help="cap per Unity package")
    ap.add_argument("--exclude", default=r"(^|/)TutorialInfo/", help="regex of paths to skip")
    ap.add_argument("--pairs-own", type=int, default=6, help="Q&A pairs per chunk of your own code")
    ap.add_argument("--pairs-unity", type=int, default=3, help="Q&A pairs per chunk of Unity packages")
    ap.add_argument("--general", type=int, default=0, help="extra general C#/Unity rows (approx.)")
    ap.add_argument("--chunk-chars", type=int, default=6000)
    ap.add_argument("--lang", default="English")
    ap.add_argument("--no-system", action="store_true", help="omit the system message in rows")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--model-own", default=None, help="teacher for your own code (default: --model)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="process at most N new chunks (testing)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="print N random rows of --out and exit")
    ap.add_argument("--fresh", action="store_true", help="move existing --out/state/prov to .bak and start over")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    state_path = args.out.with_suffix(".state.json")
    prov_path = args.out.with_suffix(".prov.jsonl")
    if args.show:
        return show_rows(args.out, prov_path, args.show)
    if args.fresh:
        for f in (args.out, state_path, prov_path):
            if f.exists():
                f.replace(f.with_name(f.name + ".bak"))
        print("previous output moved to *.bak")
    done: set[str] = set(json.loads(state_path.read_text())) if state_path.exists() else set()

    # ---- build work list
    work = []
    per_pkg: dict[str, int] = {}
    for origin, root, p, rel, is_own, is_code in select_files(
            manifest, args.own_prefix, args.unity_packages, args.files_per_package,
            args.docs_per_package, args.include_editor, re.compile(args.exclude), args.max_unity_file_kb):
        chunks = chunks_of(read(p), args.chunk_chars)
        if not is_own:
            chunks = chunks[:args.max_chunks_per_file]
            room = args.max_chunks_per_package - per_pkg.get(origin, 0)
            chunks = chunks[:max(0, room)]
            per_pkg[origin] = per_pkg.get(origin, 0) + len(chunks)
        for i, body in enumerate(chunks):
            cid = hashlib.sha1(f"{origin}|{rel}|{i}|{hashlib.sha1(body.encode()).hexdigest()}".encode()).hexdigest()[:16]
            work.append({"id": cid, "origin": origin, "rel": rel, "body": body, "own": is_own, "code": is_code,
                         "n": args.pairs_own if is_own else args.pairs_unity})
    for t in range(0, max(0, args.general) // 4 + (1 if args.general % 4 else 0)):
        topic = GENERAL_TOPICS[t % len(GENERAL_TOPICS)]
        work.append({"id": f"general-{t}", "origin": "general", "rel": topic, "body": "", "own": False,
                     "code": False, "n": 4})

    todo = [w for w in work if w["id"] not in done]
    own_n = sum(w["own"] for w in todo)
    print(f"chunks: {len(work)} total, {len(done)} already done, {len(todo)} to do "
          f"({own_n} own-code, {len(todo) - own_n} Unity/general)")
    exp_rows = sum(w["n"] for w in todo)
    print(f"expected rows: ~{int(exp_rows * 0.85)} (after validation)")
    if args.dry_run:
        by = {}
        for w in todo:
            by[w["origin"]] = by.get(w["origin"], 0) + 1
        for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
            print(f"  {v:5d}  {k}")
        return 0
    if args.limit:
        todo = todo[:args.limit]

    # ---- check teacher
    try:
        with urllib.request.urlopen(f"{args.host}/api/tags", timeout=10) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception as e:
        print(f"ERROR: cannot reach Ollama at {args.host} ({e}). Start it with: ollama serve", file=sys.stderr)
        return 1
    for need in {args.model, args.model_own or args.model}:
        if not any(m == need or m.split(":")[0] == need for m in models):
            print(f"ERROR: model '{need}' not pulled. Run: ollama pull {need}", file=sys.stderr)
            return 1

    # ---- generate
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = rejected = failed = 0
    reasons: dict[str, int] = {}
    t0 = time.time()
    with args.out.open("a", encoding="utf-8") as fo, prov_path.open("a", encoding="utf-8") as fp:
        for k, w in enumerate(todo, 1):
            if w["origin"] == "general":
                prompt = (f"Write {w['n']} diverse question/answer pairs about: {w['rel']}. "
                          f"Audience: Unity developers. Answers 60-400 words with ```csharp examples "
                          f"when useful. Write in {args.lang}. Return ONLY JSON: "
                          '{"pairs": [{"kind": "...", "question": "...", "answer": "..."}]}')
            else:
                prompt = PROMPT_TEMPLATE.format(origin=w["origin"], rel=w["rel"], body=w["body"], n=w["n"],
                                                kinds=KINDS_OWN if w["code"] else KINDS_DOC, lang=args.lang)
            try:
                model = (args.model_own or args.model) if w["own"] else args.model
                raw = ollama_chat(args.host, model, prompt, args.num_ctx, args.temperature, args.timeout)
            except RuntimeError as e:
                failed += 1
                print(f"  ! {w['origin']} {w['rel']}: {e}", file=sys.stderr)
                continue  # not marked done -> retried on next run
            for pair in parse_pairs(raw):
                reason = None if w["origin"] == "general" else valid_pair(pair, w["body"], w["code"])
                if reason is None and w["origin"] == "general":
                    q, a = str(pair.get("question", "")), str(pair.get("answer", ""))
                    reason = None if len(q) >= 15 and len(a) >= 80 else "too short"
                if reason:
                    rejected += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                msgs = [] if args.no_system else [{"role": "system", "content": SYSTEM_PROMPT}]
                msgs += [{"role": "user", "content": pair["question"].strip()},
                         {"role": "assistant", "content": pair["answer"].strip()}]
                fo.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                fp.write(json.dumps({"chunk": w["id"], "origin": w["origin"], "file": w["rel"],
                                     "kind": pair.get("kind", "")}, ensure_ascii=False) + "\n")
                kept += 1
            fo.flush()
            fp.flush()
            done.add(w["id"])
            state_path.write_text(json.dumps(sorted(done)))
            el = time.time() - t0
            eta = el / k * (len(todo) - k)
            print(f"[{k}/{len(todo)}] kept {kept} rejected {rejected} failed {failed} | "
                  f"{w['origin'].split('@')[0]} {w['rel'][:50]} | {el/k:.0f}s/chunk ETA {eta/3600:.1f} h",
                  flush=True)

    print(f"\ndone: kept {kept}, rejected {rejected} {reasons}, failed {failed}")
    print(f"-> {args.out}\n-> {prov_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
