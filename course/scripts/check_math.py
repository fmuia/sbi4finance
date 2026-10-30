"""Check that the course writes its math in the two forms GitHub and the VS Code preview both render, and convert it.

    python course/scripts/check_math.py          # report math in any other form; exit 1 if there is any
    python course/scripts/check_math.py --fix    # rewrite it in place

The two forms:

- inline math between a dollar and a backtick, $`\\sigma\\sqrt{T}`$;
- display math in a fenced block whose language is `math`.

GitHub runs Markdown over math written as $...$ or $$...$$ before MathJax sees it. It drops the backslash of
\\, \\; \\! \\{ \\} \\% \\| and \\\\ (and a bare % then hides the rest of the formula), and it leaves an inline formula
unrendered when its * or _ pair up as emphasis, or when it sits in italic or bold text. Inside the code span of
$`...`$ and inside a fence GitHub passes the TeX through unchanged. VS Code's built-in preview (and Cursor's
"Markdown: Open Preview to the Side") renders both forms with KaTeX from VS Code 1.99 on.

GitHub also escapes < and > twice in inline math, so inline math writes them \\lt and \\gt. The script only changes
delimiters and these two characters; the TeX is otherwise left as written.
"""

import argparse
import re
import sys
from pathlib import Path

COURSE = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")


def is_word(char):
    return re.fullmatch(r"[A-Za-z0-9_]", char) is not None


def can_open(line, i):
    """VS Code's rule for an opening $: not after a $, a backslash or a word character."""
    before = line[i - 1] if i else ""
    return before not in ("$", "\\") and not is_word(before)


def can_close(line, i):
    """VS Code's rule for a closing $: not before a $ or a word character."""
    after = line[i + 1] if i + 1 < len(line) else ""
    return after != "$" and not is_word(after)


def closing_dollar(line, start):
    """The first unescaped $ at or after start, or -1."""
    j = line.find("$", start)
    while j != -1:
        k = j - 1
        while k >= 0 and line[k] == "\\":
            k -= 1
        if (j - k) % 2 == 1:
            return j
        j = line.find("$", j + 1)
    return -1


def safe_inline(tex):
    """Write < and > as \\lt and \\gt, spaced so that a following letter does not join the command name."""
    return re.sub(r" ?([<>]) ?", lambda m: r" \lt " if m[1] == "<" else r" \gt ", tex)


def scan_line(line):
    """Split a line of Markdown into ('text', s), ('code', s) for a code span, ('bare', tex) for $tex$ and
    ('ticked', tex) for $`tex`$."""
    pieces, text, i = [], [], 0
    while i < len(line):
        char = line[i]
        if char == "\\" and i + 1 < len(line):
            text.append(line[i:i + 2])
            i += 2
            continue
        if char == "`":
            run = len(line[i:]) - len(line[i:].lstrip("`"))
            end = line.find("`" * run, i + run)
            while end != -1 and line[end + run:end + run + 1] == "`":
                end = line.find("`" * run, end + run + 1)
            if end == -1:
                text.append(line[i:i + run])
                i += run
            else:
                pieces += [("text", "".join(text)), ("code", line[i:end + run])]
                text, i = [], end + run
            continue
        if char == "$" and line[i + 1:i + 2] == "`":
            run = len(line[i + 1:]) - len(line[i + 1:].lstrip("`"))
            end = line.find("`" * run + "$", i + 1 + run)
            if end != -1:
                pieces += [("text", "".join(text)), ("ticked", line[i + 1 + run:end])]
                text, i = [], end + run + 1
                continue
        if char == "$" and can_open(line, i):
            j = closing_dollar(line, i + 1)
            if j > i + 1 and can_close(line, j):
                pieces += [("text", "".join(text)), ("bare", line[i + 1:j])]
                text, i = [], j + 1
                continue
        text.append(char)
        i += 1
    pieces.append(("text", "".join(text)))
    return [p for p in pieces if p != ("text", "")]


def convert(source):
    """Return the converted text and a list of (line number, problem) for everything that was not in a safe form."""
    lines = source.split("\n")
    out, problems = [], []
    fence = None
    display = None
    for number, line in enumerate(lines, 1):
        if fence:
            out.append(line)
            if line.strip().startswith(fence) and line.strip().strip(fence[0]) == "":
                fence = None
            continue
        if display is not None:
            if line.strip() == "$$":
                out.append(display + "```")
                display = None
            else:
                out.append(line)
            continue
        opening = FENCE.match(line)
        if opening:
            fence = opening[2]
            out.append(line)
            continue
        if line.strip() == "$$":
            problems.append((number, "display math between $$ lines"))
            display = line[: len(line) - len(line.lstrip())]
            out.append(display + "```math")
            continue
        parts = scan_line(line)
        if any(kind == "text" and "$$" in value for kind, value in parts):
            problems.append((number, "$$ inside a line"))
        pieces = []
        for kind, value in parts:
            if kind in ("text", "code"):
                pieces.append(value)
                continue
            tex = safe_inline(value)
            if kind == "bare":
                problems.append((number, f"inline math between bare dollars: ${value}$"))
            elif tex != value:
                problems.append((number, f"< or > in inline math: $`{value}`$"))
            pieces.append(f"$`{tex}`$")
        out.append("".join(pieces))
    if display is not None:
        problems.append((len(lines), "unclosed $$ block"))
    return "\n".join(out), problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="*", type=Path, help="Markdown files (default: course/*.md)")
    parser.add_argument("--fix", action="store_true", help="rewrite the files in place")
    args = parser.parse_args()
    files = args.files or sorted(COURSE.glob("*.md"))
    total = 0
    for path in files:
        source = path.read_text()
        converted, problems = convert(source)
        total += len(problems)
        if args.fix and converted != source:
            path.write_text(converted)
            print(f"fixed {path.name}: {len(problems)} formulas")
        elif not args.fix:
            for number, problem in problems:
                print(f"{path.name}:{number}: {problem[:120]}")
    if not args.fix:
        print(f"{total} formulas not in a form GitHub and VS Code both render" if total else "ok: all math is safe")
        sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
