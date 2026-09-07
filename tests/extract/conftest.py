from pathlib import Path


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    lines = [",".join(header)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
