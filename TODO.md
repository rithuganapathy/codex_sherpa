# What is left

Honest state of the project, written 17 August 2026.

## How complete is it

**About 90%.** The tool is finished and works. What remains is publishing it and
a handful of things that were considered and deliberately not built.

| Area | State |
|---|---|
| The 12-phase build plan | 100% — every phase done |
| Works on real repositories | 100% — 8 repos, 3 languages, most recent run 8/8 |
| Tests | ~182 offline checks, all passing |
| Documentation for a stranger | 90% — README rewritten, screenshot missing |
| Published | 0% — never pushed anywhere |

The 10% gap is almost entirely "nobody else can see it".

---

## Must do to call it finished

### 1. Screenshot (5 minutes)

The README references `docs/screenshot.png`, which does not exist yet, so the
image is currently broken on GitHub.

```bash
streamlit run app.py
```

Load a repo, take a screenshot of the browser, save it as `docs/screenshot.png`.
Nothing else needs changing: the README already points at it.

### 2. Push to GitHub (5 minutes)

Steps are at the bottom of this file.

### 3. Decide about a licence (2 minutes)

There is none, which means the code is **all rights reserved** by default:
people may read it on GitHub but may not legally copy, run or build on it. That
is a valid choice for a portfolio piece. If you want people to be able to use
it, add an MIT licence.

---

## Worth doing next, if the project continues

Ordered by value for the effort.

### Blast radius (half a day)

Pick a function, get everything that transitively depends on it. Pure graph
traversal, no model, and it answers the question people actually ask before
changing unfamiliar code: what will this break.

### Stale-docs auditor (1 day)

Point the Critic at a repo's *existing* README and docstrings and report which
claims no longer match the source. The machinery is already built; only the
input changes. This is arguably a stronger idea than the generator itself, and
there is already a real example: flask documents `send_from_directory` as using
`send_file`, while the code delegates to werkzeug.

### Diff mode (1-2 days)

Compare two commits or a PR branch: which functions appeared, which call edges
changed, which documentation is now stale. This is the version a team would
keep using, because docs rotting is the real problem.

### Churn x centrality (1 day)

Combine how often a function changes (git history) with how many things depend
on it. The overlap is where bugs live. GitPython is already a dependency.
Nobody ships this, and the insight is genuinely non-obvious.

### Smaller things

- **Model bake-off.** Run the same repo through different local models and
  compare verification scores. Weak as a feature, strong as a talking point,
  because it turns the Critic into a hallucination benchmark.
- **More languages.** Go, Rust and Java are a grammar plus a `LangSpec` each.
  `languages.py` was built for this.
- **Dependency alternatives for npm.** The feature reads PyPI only, so a
  JavaScript project gets nothing useful from it.
- **Auto-detect the main package.** `--subdir` is manual. A repo whose largest
  folder is a bundled web app could be detected rather than configured.

---

## Known limits, all deliberate

These are documented in the app and in every generated document. They are not
bugs and they are not on the list to fix.

- Only the top-ranked functions are described, not the whole repository
- Roughly half of all calls go to third-party code and cannot be verified
- Anything decided at runtime is invisible to a syntax tree
- Only two kinds of claim are machine-checked: that A calls B, and that a
  symbol lives where the text says
- **Grouping is not checked**, so a function can be described under the wrong
  component and still pass. This has happened
- Retrieval is the weak link in the Ask tab. A confidently verified answer to
  the wrong question is the failure mode to watch

## Known rough edges

Small, real, unfixed.

- `transformers` logs `ModuleNotFoundError: No module named 'torchvision'` while
  probing an optional image processor. Noisy, harmless
- The first question after a restart takes about 40 seconds, because the
  embedding model loads on first use. Every question after is about 20
- A phase-flow step occasionally gets a weak title when the ranked symbol has no
  useful purpose text, e.g. "Is null session"
- Ollama updates itself and restarts, which interrupts a run for a minute or two

---

## How to push this to GitHub

Nothing has been pushed. All 28 commits are local.

### 1. Create an empty repository on GitHub

Go to <https://github.com/new>. Give it a name, for example `codex-sherpa`.

**Do not** tick "Add a README", "Add .gitignore" or "Choose a licence". The repo
must be empty, or the first push will be rejected for having unrelated history.

### 2. Point this folder at it

```bash
cd D:\project\codex_sherpa
git remote add origin https://github.com/<your-username>/codex-sherpa.git
```

Check it took:

```bash
git remote -v
```

### 3. Push

```bash
git push -u origin master
```

GitHub will ask you to sign in. Use a browser sign-in if offered, or a personal
access token as the password — GitHub stopped accepting account passwords for
git in 2021.

`-u` remembers the destination, so later pushes are just `git push`.

### 4. Check it

Open the repo page. You should see the README with the flowers screenshot, the
two example documents under `docs/`, and 28 commits.

### If the branch name is wrong

GitHub defaults to `main`, this repo is on `master`. Either push as-is, or
rename first:

```bash
git branch -M main
git push -u origin main
```

### What will not be pushed

`.gitignore` keeps these out, correctly:

- `venv/` — 2GB of dependencies, rebuilt from `requirements.txt`
- `repos/` — cloned repositories
- `out/` — generated documents, except the two copied into `docs/`
- `.chroma/`, `.cache/` — search index and PyPI cache
