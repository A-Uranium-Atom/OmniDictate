---
trigger: glob
globs: **/*.py
---

# Python Language Profile

## 1. Tech Stack & Tooling
* **Package Manager:** Standardize on `uv` for dependency management. Ensure changes are updated in `pyproject.toml` rather than just `requirements.txt`.
* **Linting & Formatting:** Use `Ruff` for combined linting and formatting. Conform to PEP 8 strictly.
* **Testing Ecosystem:** Use functional `pytest` styles.

## 2. Self-Documenting Code & Typing (Pythonic Practices)
Python's dynamic nature requires strict adherence to self-documenting structures to remain maintainable.
* **Explicit Type Hints:** Include full type annotations for all function/method signatures and complex variables. Do not use `Any` unless absolutely unavoidable.
  * *Bad:* `def process(user, data):`
  * *Good:* `def process_user_analytics(user: User, data: list[dict[str, int]]) -> bool:`
* **Use Enums and Literals:** Prevent invalid states by using `Enum` or `Literal` instead of passing arbitrary strings or integers.
  * *Bad:* `def set_mode(mode: str):` (Is it "fast", "slow", "1"?)
  * *Good:* `def set_mode(mode: Literal["fast", "accurate"]) -> None:`
* **Structured Data:** Use `@dataclass` or `Pydantic` models instead of generic dictionaries when passing complex domain data. This makes the expected keys obvious.
  * *Bad:* `def calculate_tax(item: dict) -> float:`
  * *Good:* `def calculate_tax(item: ProductData) -> float:`
* **Docstrings:** Require Google-style docstrings for all *public* functions, classes, and modules. Keep inline comments minimal and reserved for explaining non-obvious logic (e.g., `# using bitwise shift for performance as requested by CTranslate2 bound`).

## 3. Modularity & Complexity Limits
* **Immutability:** Avoid mutable default arguments at all costs (e.g., `def func(items: list = [])` is strictly forbidden. Use `None`).
* **Path Management:** Use `pathlib` for all file system operations; strictly avoid `os.path`.
* **File Size:** Prefer making new files for a distinct purpose. Prefer under 500 lines per file.
* **Function Size:** Keep functions focused on a single purpose, ideally under 50 lines. Keep cyclomatic complexity under 10.
* *Exception:* If a function or file needs to be long to maintain explicit clarity or avoid fragmented logic, do so.

## 4. I/O & Execution Boundaries
* Maintain a clear boundary for I/O-bound operations. 
* Strictly avoid mixing synchronous (`time.sleep`, standard `requests`) and asynchronous (`asyncio`, `aiohttp`) I/O in the same scope.

## 5. Verification & Commit Requirements
* **Test Suite:** Always run the `pytest` suite after major changes.
* **Coverage:** Maintain a minimum 80% code coverage for new modules.
* **Pre-Commit Check:** Run `uv run ruff check . && uv run pytest` before suggesting any commit or marking a task complete. Do not mark complete if tests fail.
* **The Karpathy Move:** If tests fail autonomously, feed the full stack trace back into your context to diagnose deeply before guessing at a fix.

## 6. Security & Unsafe Execution Boundaries
Never use unsafe dynamic execution or insecure serialization. If you need to parse data or run commands, you must use the strict, safe alternatives provided below:
* **No `eval()` or `exec()`:** These are strictly forbidden. 
  * *Alternative:* If you need to evaluate a string into a Python dictionary, list, or primitive, use `ast.literal_eval()` or other such alternatives.
* **No `os.system` or `shell=True`:** Never run subprocesses with `shell=True` as it opens the door to shell injection.
  * *Alternative:* Use `subprocess.run()` with `shell=False` and pass arguments as a structured list (e.g., `["ls", "-l"]`).
* **No Unsafe Serialization (`pickle`):** Do not use `pickle` or `dill` to load data from untrusted sources or file paths, as they can execute arbitrary code upon deserialization.
  * *Alternative:* Use `json` for standard data structures, or `Pydantic` / `@dataclass` for parsing complex structured data.