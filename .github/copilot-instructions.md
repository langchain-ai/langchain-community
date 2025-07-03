### 1. Avoid Breaking Changes (Stable Public Interfaces)

* Carefully preserve **function signatures**, argument positions, and names for any exported/public methods.
* Be cautious when **renaming**, **removing**, or **reordering** arguments — even small changes can break downstream consumers.
* Use keyword-only arguments or clearly mark experimental features to isolate unstable APIs.

Bad:

```python
def get_user(id, verbose=False):  # Changed from `user_id`
```

Good:

```python
def get_user(user_id: str, verbose: bool = False):  # Maintains stable interface
```

🧠 *Ask yourself:* “Would this change break someone's code if they used it last week?”

---

### 2. Simplify Code and Use Clear Variable Names

* Prefer descriptive, **self-explanatory variable names**. Avoid overly short or cryptic identifiers.
* Break up overly long or deeply nested functions for **readability and maintainability**.
* Avoid unnecessary abstraction or premature optimization.

Bad:

```python
def p(u, d):
    return [x for x in u if x not in d]
```

Good:

```python
def filter_unknown_users(users: List[str], known_users: Set[str]) -> List[str]:
    return [user for user in users if user not in known_users]
```

---

### 3. Ensure Unit Tests Cover New and Updated Functionality

* Every new feature or bugfix should be **covered by a unit test**.
* Test edge cases and failure conditions.
* Use `pytest`, `unittest`, or the project’s existing framework consistently.

Checklist:

* [ ] Does the test suite fail if your new logic is broken?
* [ ] Are all expected behaviors exercised (happy path, invalid input, etc)?
* [ ] Do tests use fixtures or mocks where needed?

---

### 4. Look for Suspicious or Risky Code

* Watch out for:

  * Use of `eval()`, `exec()`, or `pickle` on user-controlled input.
  * Silent failure modes (`except: pass`).
  * Unreachable code or commented-out blocks.
  * Race conditions or resource leaks (file handles, sockets, threads).

Bad:

```python
def load_config(path):
    with open(path) as f:
        return eval(f.read())  # ⚠️ Never eval config
```

Good:

```python
import json

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
```

---

### 5. Use Google-Style Docstrings (with Args section)

* All public functions should include a **Google-style docstring**.
* Include an `Args:` section where relevant.
* Types should NOT be written in the docstring — use type hints instead.

Bad:

```python
def send_email(to, msg):
    """Send an email to a recipient."""
```

Good:

```python
def send_email(to: str, msg: str) -> None:
    """
    Sends an email to a recipient.

    Args:
        to: The email address of the recipient.
        msg: The message body.
    """
```

📌 *Tip:* Keep descriptions concise but clear. Only document return values if non-obvious.

---

### 6. Propose Better Designs When Applicable

* If there's a **cleaner**, **more scalable**, or **simpler** design, highlight it.
* Suggest improvements, even if they require some refactoring — especially if the new code would:

  * Reduce duplication
  * Make unit testing easier
  * Improve separation of concerns
  * Add clarity without adding complexity

Instead of:

```python
def save(data, db_conn):
    # manually serializes fields
```

You might suggest:

```python
# Suggest using dataclasses or Pydantic for automatic serialization and validation
```
