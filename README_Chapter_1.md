# Chapter 1: Getting Started in Python 3

**Notebook:** [`Chapter_1_Getting_Started_in_Python3.ipynb`](Chapter_1_Getting_Started_in_Python3.ipynb)  
**Source:** *Mastering Machine Learning with Python in Six Steps* (2nd ed.) — Manohar Swamynathan, Step 1  
**Code listings:** 1-1 through 1-53 (53 examples)

---

## Who This Chapter Is For

This chapter assumes **no prior Python experience**. By the end, you will be able to write small programs, work with Python’s core data types, control program flow, define functions, read and write files, and handle basic errors — everything you need before opening a machine learning library.

---

## What You Will Learn

- Why Python 3 is the standard for data science and machine learning
- How to install and run Python (Anaconda recommended)
- Python syntax rules: indentation, identifiers, keywords, comments
- All core built-in types and when to use each one
- Operators, loops, and conditional logic
- Lists, tuples, sets, and dictionaries in depth
- How to write reusable functions and import modules
- File input/output and exception handling

---

## Topics Covered (Complete Chapter Outline)

### 1. The Best Things in Life Are Free

Python is open source. You can download, use, and share it without licensing fees — a major reason it dominates data science.

**The Zen of Python** (`import this`) is a set of guiding principles for writing clear, readable Python code. The notebook runs this so you can see the philosophy behind the language (e.g., “Readability counts”, “Simple is better than complex”).

---

### 2. The Rising Star — Why Python for Machine Learning?

Python is popular for ML because:

- **Readable syntax** — easier to learn and maintain than many alternatives
- **Rich ecosystem** — NumPy, pandas, scikit-learn, TensorFlow, and hundreds of other libraries
- **Community and jobs** — widely used in industry and research
- **Integration** — works well with databases, web APIs, notebooks, and cloud platforms

---

### 3. Python 2 vs Python 3

Always use **Python 3** for new work. Python 2 reached end-of-life in 2020. This course and all notebooks use Python 3 syntax (`print()` as a function, `/` true division, Unicode strings by default, etc.).

---

### 4. Setting Up Python

#### Anaconda (recommended for beginners)

Anaconda bundles Python with Jupyter, NumPy, pandas, matplotlib, and scikit-learn pre-installed.

| Platform | What to do |
|----------|------------|
| **Windows** | Download Anaconda installer → run → open Anaconda Prompt or Jupyter |
| **macOS (OSX)** | Same installer; use Terminal or Jupyter |
| **Linux** | Install via installer or package manager |
| **Official python.org** | Alternative if you prefer a minimal install |

#### Running Python

You can run code in:

- **Jupyter Notebook / JupyterLab** — interactive cells (used in this course)
- **Python REPL** — type commands in a terminal
- **Script files** — save as `.py` and run with `python my_script.py`

---

### 5. Key Concepts

#### 5.1 Python Identifiers

Names for variables, functions, and modules. Rules:

- Start with a letter or underscore; then letters, digits, or underscores
- Case-sensitive (`age` ≠ `Age`)
- Cannot use reserved keywords

#### 5.2 Keywords (Reserved Words)

Words like `if`, `else`, `for`, `while`, `def`, `class`, `import`, `True`, `False`, `None` have special meaning. The notebook lists them with `keyword.kwlist`.

#### 5.3 My First Python Program

```python
print("Hello, World!")
```

`print()` sends output to the screen. In Jupyter, the last expression in a cell can also display a value.

#### 5.4 Code Blocks — Indentation and Suites

Python uses **indentation** (spaces, typically 4) instead of braces `{}` to define blocks. Every block after `if`, `else`, `for`, `while`, `def`, etc. must be indented consistently. Mixed tabs and spaces cause `IndentationError`.

Listings 1-1 and 1-2 show correct vs incorrect indentation.

#### 5.5 Basic Object Types

Everything in Python is an object. Core types:

| Type | Example | Notes |
|------|---------|-------|
| **int** | `42` | Whole numbers |
| **float** | `3.14` | Decimals; also `inf`, `nan` |
| **bool** | `True`, `False` | Logical values |
| **complex** | `2+8j` | Real + imaginary (`j` not `i`) |
| **str** | `'hello'`, `"world"` | Text; immutable |
| **list** | `[1, 2, 3]` | Ordered, mutable sequence |
| **tuple** | `(1, 2, 3)` | Ordered, **immutable** sequence |
| **set** | `{1, 2, 3}` | Unordered, unique elements |
| **dict** | `{'a': 1}` | Key–value mappings |
| **file** | `open('data.csv')` | Handle for reading/writing files |

#### When to Use List, Tuple, Set, or Dictionary?

| Structure | Use when… |
|-----------|-----------|
| **List** | Order matters; you need to add, remove, or change items |
| **Tuple** | Fixed record (e.g., coordinates); use as dict keys; return multiple values from a function |
| **Set** | You need unique values or fast membership tests |
| **Dictionary** | Look up values by a meaningful key (name → age, id → record) |

#### 5.6 Comments in Python

- `#` for single-line comments
- Triple quotes `""" ... """` for multi-line docstrings and block comments

Comments explain *why*; the code should explain *what*.

#### 5.7 Multiline Statements

Long expressions can continue inside `()`, `[]`, or `{}` without a backslash. Backslash `\` at end of line also continues a statement.

#### 5.8 Multiple Statements on a Single Line

Use semicolon `;` to put short statements on one line (use sparingly for readability).

---

### 6. Basic Operators

#### 6.1 Arithmetic Operators

`+`, `-`, `*`, `/` (float division), `//` (floor division), `%` (modulus), `**` (exponent).

#### 6.2 Comparison (Relational) Operators

`==`, `!=`, `<`, `>`, `<=`, `>=` — return `True` or `False`.

#### 6.3 Assignment Operators

`=`, `+=`, `-=`, `*=`, `/=`, `%=`, `**=`, `//=` — update a variable in place.

#### 6.4 Bitwise Operators

`&`, `|`, `^`, `~`, `<<`, `>>` — operate on individual bits of integers. Used in low-level programming and some ML optimizations; less common day-to-day.

#### 6.5 Logical Operators

`and`, `or`, `not` — combine boolean conditions.

#### 6.6 Membership Operators

`in`, `not in` — test whether a value appears in a sequence or collection.

#### 6.7 Identity Operators

`is`, `is not` — test whether two names refer to the **same object** in memory (not the same as `==` value equality).

---

### 7. Control Structures

#### Selection (if / elif / else)

- **if** — run a block when a condition is true
- **if–else** — choose between two paths
- **nested if** — decisions inside decisions

#### Iteration

- **for loop** — repeat over each item in a sequence (`for x in items:`)
- **while loop** — repeat while a condition is true (watch for infinite loops)
- **while–else** — `else` runs when the loop finishes without `break`

#### Loop control

- **break** — exit the loop immediately
- **continue** — skip to the next iteration
- **pass** — placeholder that does nothing (empty block)

---

### 8. Lists

Ordered, mutable collections. Common operations (Listings 1-20 through 1-24):

| Operation | Example |
|-----------|---------|
| Length | `len(my_list)` |
| Append | `my_list.append(x)` |
| Extend | `my_list.extend(other)` |
| Insert | `my_list.insert(i, x)` |
| Remove | `my_list.remove(x)` |
| Pop | `my_list.pop()` |
| Index / count | `my_list.index(x)`, `my_list.count(x)` |
| Sort / reverse | `my_list.sort()`, `my_list.reverse()` |
| Slicing | `my_list[1:4]`, `my_list[::-1]` |
| List comprehension | `[x*2 for x in range(5)]` |

---

### 9. Tuples

Like lists but **immutable** — cannot change elements after creation. Useful for fixed records, function returns, and dictionary keys. Support indexing, slicing, and unpacking: `a, b = (1, 2)`.

---

### 10. Sets

Unordered collections of **unique** elements. Operations include `add`, `remove`, `union` (`|`), `intersection` (`&`), `difference` (`-`), and `symmetric_difference` (`^`).

---

### 11. Dictionaries

Key–value maps. Operations include:

- Create, access, update, delete keys
- `keys()`, `values()`, `items()`
- `get(key, default)` — safe lookup
- `in` — test key membership
- Dictionary comprehension

---

### 12. User-Defined Functions

#### Defining functions

```python
def greet(name):
    return f"Hello, {name}"
```

#### Scope of variables

- **Local** — inside the function
- **Global** — module level (use `global` keyword to modify from inside a function)

#### Default arguments

```python
def power(x, n=2):
    return x ** n
```

#### Variable-length arguments

- `*args` — extra positional arguments as a tuple
- `**kwargs` — extra keyword arguments as a dictionary

These patterns appear constantly in ML library APIs.

---

### 13. Modules

Reuse code across files:

```python
import math
from math import sqrt
import numpy as np  # common alias
```

- `dir(module)` — list names defined in a module
- `__name__ == "__main__"` — script vs import guard

---

### 14. File Input/Output

Typical workflow: **open → read/write → close** (or use `with` for automatic close).

| Mode | Meaning |
|------|---------|
| `'r'` | Read (default) |
| `'w'` | Write (overwrites) |
| `'a'` | Append |
| `'rb'` / `'wb'` | Binary read/write |

The `with open(...) as f:` pattern is preferred — the file closes even if an error occurs.

---

### 15. Exception Handling

Programs fail gracefully with **try / except / else / finally**:

```python
try:
    result = 10 / x
except ZeroDivisionError:
    print("Cannot divide by zero")
finally:
    print("Cleanup always runs")
```

Common built-in exceptions: `ValueError`, `TypeError`, `IndexError`, `KeyError`, `FileNotFoundError`, `ZeroDivisionError`.

---

## Code Listings Summary

| Listings | Topic |
|----------|-------|
| 1-1 – 1-2 | Indentation (correct vs incorrect) |
| 1-3 – 1-6 | Basic types, comments, multiline code |
| 1-7 – 1-13 | All operator types |
| 1-14 – 1-19 | if, for, while loops |
| 1-20 – 1-24 | List operations |
| 1-25 – 1-29 | Tuple operations |
| 1-30 – 1-37 | Set operations |
| 1-38 – 1-42 | Dictionary operations |
| 1-43 – 1-48 | Functions (*args, **kwargs) |
| 1-49 – 1-50 | Modules |
| 1-51 – 1-52 | File I/O |
| 1-53 | Exception handling |

---

## How to Use the Notebook

1. Open `Chapter_1_Getting_Started_in_Python3.ipynb` in Jupyter or VS Code.
2. Run cells **top to bottom** — later cells may depend on earlier ones.
3. Read the markdown before each listing; it explains the concept and what to look for in the output.
4. **Experiment:** change numbers, break indentation on purpose, add your own `print()` statements.
5. If a cell fails, read the error message — Python errors usually name the line and problem type.

---

## Key Takeaways

- Python uses **indentation** for structure; consistency is mandatory.
- Choose the **right collection type** (list, tuple, set, dict) for your data.
- **Functions** and **modules** keep code organized as projects grow.
- **Exceptions** prevent crashes and give meaningful feedback.
- This chapter is the foundation for NumPy, pandas, and scikit-learn in Chapter 2.

---

## Next Step

Continue to **[Chapter 2: Introduction to Machine Learning](README_Chapter_2.md)** — ML concepts, AI history, and the NumPy / pandas / matplotlib toolkit.
