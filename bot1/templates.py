"""Starter file contents used by the offline (no-AI) engine."""
from __future__ import annotations

import posixpath
import re


def _class_name(path: str) -> str:
    base = posixpath.splitext(posixpath.basename(path))[0]
    return "".join(p.capitalize() for p in re.split(r"[\W_]+", base) if p) or "Main"


def render(path: str, instruction: str = "") -> str:
    ext = posixpath.splitext(path)[1].lower()
    base = posixpath.splitext(posixpath.basename(path))[0]
    low = instruction.lower()
    title = base.replace("_", " ").replace("-", " ").title()
    cls = _class_name(path)

    if ext == ".py" and "calculator" in low + base.lower():
        return ('"""Simple command-line calculator."""\n\n\n'
                "def calculate(a: float, op: str, b: float) -> float:\n"
                "    if op == '+':\n        return a + b\n"
                "    if op == '-':\n        return a - b\n"
                "    if op == '*':\n        return a * b\n"
                "    if op == '/':\n"
                "        if b == 0:\n            raise ZeroDivisionError('Cannot divide by zero')\n"
                "        return a / b\n"
                "    raise ValueError(f'Unknown operator: {op}')\n\n\n"
                "def main() -> None:\n"
                "    a = float(input('First number: '))\n"
                "    op = input('Operator (+ - * /): ').strip()\n"
                "    b = float(input('Second number: '))\n"
                "    print('Result:', calculate(a, op, b))\n\n\n"
                "if __name__ == '__main__':\n    main()\n")
    if ext == ".py":
        return f'"""{title}."""\n\n\ndef main() -> None:\n    print("Hello from {base}")\n\n\nif __name__ == "__main__":\n    main()\n'
    if ext in (".html", ".htm"):
        return ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                f"  <title>{title}</title>\n</head>\n<body>\n  <h1>{title}</h1>\n  <p>Welcome!</p>\n</body>\n</html>\n")
    if ext == ".css":
        return "* {\n  box-sizing: border-box;\n}\n\nbody {\n  margin: 0;\n  font-family: Arial, sans-serif;\n}\n"
    if ext == ".java":
        return (f"public class {cls} {{\n\n    public static void main(String[] args) {{\n"
                f"        System.out.println(\"Hello from {cls}\");\n    }}\n}}\n")
    if ext == ".js":
        return f"'use strict';\n\nfunction main() {{\n  console.log('Hello from {base}');\n}}\n\nmain();\n"
    if ext == ".md":
        return f"# {title}\n\nDescribe this document here.\n"
    if ext == ".json":
        return "{\n}\n"
    if ext == ".c":
        return '#include <stdio.h>\n\nint main(void) {\n    printf("Hello, world!\\n");\n    return 0;\n}\n'
    if ext == ".cpp":
        return '#include <iostream>\n\nint main() {\n    std::cout << "Hello, world!" << std::endl;\n    return 0;\n}\n'
    if ext == ".sql":
        return "-- " + title + "\n"
    if ext in (".yml", ".yaml"):
        return "# " + title + "\n"
    return f"{title}\n"
