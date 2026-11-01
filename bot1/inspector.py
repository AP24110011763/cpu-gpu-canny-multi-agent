"""Inspect the repository: languages, frameworks, dependencies, structure."""
from __future__ import annotations

import json
import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

LANGS = {
    ".py": "Python", ".java": "Java", ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".c": "C", ".cpp": "C++", ".h": "C/C++ header", ".cs": "C#", ".go": "Go", ".rs": "Rust",
    ".rb": "Ruby", ".php": "PHP", ".kt": "Kotlin", ".swift": "Swift", ".sql": "SQL",
    ".sh": "Shell", ".md": "Markdown", ".json": "JSON", ".yml": "YAML", ".yaml": "YAML", ".xml": "XML",
}
MANIFESTS = ["requirements.txt", "package.json", "pom.xml", "build.gradle", "pyproject.toml",
             "go.mod", "Cargo.toml", "composer.json", "Gemfile"]
FRAMEWORKS = {
    "flask": "Flask", "django": "Django", "fastapi": "FastAPI", "pytest": "pytest", "pygithub": "PyGithub",
    "react": "React", "next": "Next.js", "vue": "Vue", "express": "Express", "@angular/core": "Angular",
    "jest": "Jest", "spring-boot": "Spring Boot", "spring": "Spring", "junit": "JUnit",
    "hibernate": "Hibernate", "tailwindcss": "Tailwind CSS", "bootstrap": "Bootstrap",
}


@dataclass
class RepoInfo:
    languages: Dict[str, int] = field(default_factory=dict)
    frameworks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    manifests: List[str] = field(default_factory=list)
    summary: str = ""


def _deps(name: str, text: str) -> List[str]:
    try:
        if name == "requirements.txt":
            return [m.group(0) for l in text.splitlines()
                    if (m := re.match(r"^[A-Za-z0-9_.\-]+", l.strip()))]
        if name in ("package.json", "composer.json"):
            d = json.loads(text)
            return sorted({*d.get("dependencies", {}), *d.get("devDependencies", {}), *d.get("require", {})})
        if name == "pom.xml":
            return sorted(set(re.findall(r"<artifactId>([^<]+)</artifactId>", text)))
        if name == "build.gradle":
            return sorted(set(re.findall(r"(?:implementation|api|compile|testImplementation)\s*\(?['\"]([^'\"]+)['\"]", text)))
        if name in ("go.mod", "Cargo.toml", "Gemfile", "pyproject.toml"):
            return []
    except (ValueError, TypeError):
        pass
    return []


def inspect_repo(repo, files: List[str]) -> RepoInfo:
    info = RepoInfo()
    counts = Counter(LANGS[e] for f in files if (e := posixpath.splitext(f)[1].lower()) in LANGS)
    info.languages = dict(counts.most_common())
    found = [f for f in files if posixpath.basename(f) in MANIFESTS][:6]
    info.manifests = found
    deps: List[str] = []
    for f in found:
        got = repo.read(f)
        if got:
            deps += _deps(posixpath.basename(f), got[0])
    info.dependencies = sorted(set(deps))[:60]
    lower = [d.lower() for d in info.dependencies]
    fw = {FRAMEWORKS[k] for k in FRAMEWORKS if any(k == d or d.startswith(k) for d in lower)}
    info.frameworks = sorted(fw)

    top = sorted({f.split("/")[0] + ("/" if "/" in f else "") for f in files})[:40]
    lines = [
        f"Languages: {', '.join(f'{k} ({v} files)' for k, v in info.languages.items()) or 'none yet'}",
        f"Frameworks: {', '.join(info.frameworks) or 'none detected'}",
        f"Dependencies: {', '.join(info.dependencies[:30]) or 'none detected'}",
        f"Top-level entries: {', '.join(top) or '(empty repository)'}",
        "Files:\n" + "\n".join(files[:150]) + (f"\n... and {len(files) - 150} more" if len(files) > 150 else ""),
    ]
    info.summary = "\n".join(lines)[:8000]
    return info
