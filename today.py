"""Build dark_mode.svg and light_mode.svg: a neofetch-style card with live GitHub stats.

Run by .github/workflows/build.yml every day. Locally:
    ACCESS_TOKEN=$(gh auth token) python today.py

Only the standard library is used, so the workflow needs no pip install.
"""
import datetime as dt
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).parent

# --------------------------------------------------------------------------------------
# Edit this part. Values wrapped in "[...]" are placeholders for you to fill in.
# --------------------------------------------------------------------------------------
USER_NAME = os.environ.get("USER_NAME", "suryar113")
HEADER = "suryar113"

# Set to your birthday, e.g. dt.date(2007, 3, 14), to make "Uptime" your age.
# Left as None, Uptime counts from the day your GitHub account was created.
BIRTHDAY = None

# Each entry is a line on the card: None is a blank line, "- Title" is a section rule,
# and a (key, value) pair is a dotted row. {uptime} is filled in when the card is built.
INFO = [
    ("OS", "macOS"),
    ("Uptime", "{uptime}"),
    ("Host", "Charlotte, NC"),
    None,
    ("Languages.Programming", "Java, TypeScript, JavaScript, HTML, CSS"),
    ("Languages.Real", "English"),
    None,
    ("Hobbies.Software", "Competitive programming, Web dev"),
    ("Hobbies.Hardware", "Cars"),
    None,
    "- Contact",
    ("Website", "me.skience.io"),
    ("GitHub", "github.com/" + USER_NAME),
    ("Email", "surya.raghvan@gmail.com"),
    ("LinkedIn", "suryaraghavan"),
    None,
    "- GitHub Stats",
    # The three stats rows are appended automatically.
]

# Commit and line counts include every branch of every repo you can push to, forks included,
# and count each commit once. A repo where you authored more than this many commits (for
# example an auto-committing "update log" repo) is too big to read commit by commit, so its
# commits are counted from the repo's own total and its lines of code are left out.
BULK_REPO_COMMITS = 20_000

# --------------------------------------------------------------------------------------
# Layout and colours (GitHub's own palette, so the card matches the page around it)
# --------------------------------------------------------------------------------------
FONT_SIZE = 16
CHAR_W = 9.64  # width of one monospace cell at 16px (0.6em, plus a hair of slack)
LINE_H = 20
PAD = 15
GAP = 30
MIN_ART_SCALE = 0.53  # smallest art font, as a fraction of FONT_SIZE
MAX_ART_W = 460  # widest the art may get, in px; wide art is shrunk and centred vertically
WIDTH_CHARS = 64  # right-hand column width in characters

THEMES = {
    "dark_mode.svg": dict(bg="#161b22", text="#c9d1d9", key="#ffa657", value="#a5d6ff",
                          add="#3fb950", dele="#f85149", cc="#616e7f"),
    "light_mode.svg": dict(bg="#f6f8fa", text="#24292f", key="#953800", value="#0a3069",
                           add="#1a7f37", dele="#cf222e", cc="#c2cfde"),
}

# --------------------------------------------------------------------------------------
# GitHub API
# --------------------------------------------------------------------------------------
TOKEN = os.environ.get("ACCESS_TOKEN") or os.environ.get("GITHUB_TOKEN")
CACHE = ROOT / "cache" / "loc.json"
calls = 0


def gql(query, **variables):
    global calls
    calls += 1
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Authorization": f"bearer {TOKEN}", "User-Agent": "profile-readme-builder"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.load(resp)
    if body.get("errors"):
        raise RuntimeError(json.dumps(body["errors"], indent=2))
    return body["data"]


USER_Q = """
query($login: String!) {
  user(login: $login) {
    id
    createdAt
    contributed: repositories(ownerAffiliations: [OWNER, COLLABORATOR, ORGANIZATION_MEMBER]) { totalCount }
  }
}"""

REPOS_Q = """
query($login: String!, $after: String, $aff: [RepositoryAffiliation]) {
  user(login: $login) {
    repositories(first: 100, after: $after, ownerAffiliations: $aff) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        nameWithOwner
        refs(refPrefix: "refs/heads/", first: 100) { nodes { name target { oid } } }
      }
    }
  }
}"""

HISTORY_Q = """
query($owner: String!, $name: String!, $ref: String!, $author: ID!, $after: String) {
  repository(owner: $owner, name: $name) {
    ref(qualifiedName: $ref) { target { ... on Commit {
      history(first: 100, after: $after, author: {id: $author}) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes { oid additions deletions }
      }
    } } }
  }
}"""


def all_repos(affiliations):
    repos, after = [], None
    while True:
        page = gql(REPOS_Q, login=USER_NAME, after=after, aff=affiliations)["user"]["repositories"]
        repos += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return page["totalCount"], repos
        after = page["pageInfo"]["endCursor"]


def authored_commits(name_with_owner, branches, user_id):
    """({sha256(oid): (additions, deletions)}, 0) for commits you authored on any branch of a repo.

    For a repo with more than BULK_REPO_COMMITS of them (see the note above) it returns
    (None, total) instead, where total is the commit count of its busiest branch.
    """
    owner, name = name_with_owner.split("/")
    found, bulk = {}, 0
    for branch in branches:
        after = None
        while True:
            repo = gql(HISTORY_Q, owner=owner, name=name, ref="refs/heads/" + branch,
                       author=user_id, after=after)["repository"]
            history = repo["ref"]["target"]["history"]
            if history["totalCount"] > BULK_REPO_COMMITS:
                bulk = max(bulk, history["totalCount"])
                break
            for node in history["nodes"]:
                found[hashlib.sha256(node["oid"].encode()).hexdigest()] = (node["additions"], node["deletions"])
            if not history["pageInfo"]["hasNextPage"]:
                break
            after = history["pageInfo"]["endCursor"]
    return (None, bulk) if bulk else (found, 0)


def lines_of_code(user_id):
    """Commits and lines you authored, counted once each even if they appear on several
    branches or in both a fork and its parent. Lines of code leave out bulk repos.

    A repo is re-read only when one of its branches has moved since the last run. The cache
    holds hashes, not names or SHAs, so nothing about private repos lands in this public repo.
    """
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    fresh, seen, bulk = {}, {}, 0
    _, repos = all_repos(["OWNER", "COLLABORATOR", "ORGANIZATION_MEMBER"])
    for repo in repos:
        branches = {b["name"]: b["target"]["oid"] for b in repo["refs"]["nodes"] if b["target"]}
        if not branches:
            continue  # empty repo
        key = hashlib.sha256(repo["nameWithOwner"].encode()).hexdigest()
        sig = hashlib.sha256(json.dumps(branches, sort_keys=True).encode()).hexdigest()
        if cache.get(key, {}).get("sig") != sig or "bulk" not in cache[key]:
            found, count = authored_commits(repo["nameWithOwner"], list(branches), user_id)
            cache[key] = {"sig": sig, "bulk": count,
                          "commits": {k: list(v) for k, v in (found or {}).items()}}
        fresh[key] = cache[key]
        bulk += cache[key]["bulk"]
        for oid, (adds, dels) in cache[key]["commits"].items():
            seen[oid] = (adds, dels)
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(fresh, sort_keys=True, separators=(",", ":")) + "\n")
    return len(seen) + bulk, sum(a for a, _ in seen.values()), sum(d for _, d in seen.values())


def get_stats():
    user = gql(USER_Q, login=USER_NAME)["user"]
    repo_count, _ = all_repos(["OWNER"])
    commits, adds, dels = lines_of_code(user["id"])
    created = dt.datetime.fromisoformat(user["createdAt"].replace("Z", "+00:00")).date()
    return {
        "since": BIRTHDAY or created,
        "repos": repo_count,
        "contributed": user["contributed"]["totalCount"],
        "commits": commits,
        "loc_add": adds,
        "loc_del": dels,
    }


# --------------------------------------------------------------------------------------
# Card layout
# --------------------------------------------------------------------------------------
def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def uptime(start, today=None):
    today = today or dt.date.today()
    years, months, days = today.year - start.year, today.month - start.month, today.day - start.day
    if days < 0:
        months -= 1
        last_month_end = today.replace(day=1) - dt.timedelta(days=1)
        days += last_month_end.day
    if months < 0:
        years -= 1
        months += 12
    return f"{plural(years, 'year')}, {plural(months, 'month')}, {plural(days, 'day')}"


def entry(key, value, width):
    """A dotted 'key: .... value' cell padded to exactly `width` characters.

    `value` is a string or a list of (text, css class) runs.
    """
    runs = [(value, "value")] if isinstance(value, str) else value
    head = [(". ", "cc"), (key, "key"), (":", None)]
    used = sum(len(t) for t, _ in head + runs)
    gap = max(1, width - used)
    dots = " " * gap if gap < 3 else " " + "." * (gap - 2) + " "
    return head + [(dots, "cc")] + runs


def rule(title):
    return [(title + " ", None), ("-" * (WIDTH_CHARS - len(title) - 1), None)]


def card_lines(stats):
    n = lambda v: f"{v:,}"
    lines = [rule(HEADER)]
    for item in INFO:
        if item is None:
            lines.append([])
        elif isinstance(item, str):
            lines.append(rule(item))
        else:
            key, value = item
            lines.append(entry(key, value.format(uptime=uptime(stats["since"])), WIDTH_CHARS))

    lines.append(
        entry("Repos", [(n(stats["repos"]), "value"), (" {", None), ("Contributed", "key"),
                        (": ", None), (n(stats["contributed"]), "value"), ("}", None)], WIDTH_CHARS)
    )
    lines.append(entry("Commits", n(stats["commits"]), WIDTH_CHARS))
    lines.append(
        entry("Lines of Code on GitHub",
              [(n(stats["loc_add"] - stats["loc_del"]), "value"), (" ( ", None),
               (n(stats["loc_add"]) + "++", "addColor"), (", ", None),
               (n(stats["loc_del"]) + "--", "delColor"), (" )", None)],
              WIDTH_CHARS)
    )
    return lines


ART_CLASSES = {"k": "key", "v": "value", "c": "cc", "a": "addColor", "d": "delColor"}


def art_runs(text, codes):
    """Split a line of art into (text, css class) runs using its line in ascii_colors.txt."""
    runs = []
    for ch, code in zip(text, codes.ljust(len(text))):
        cls = ART_CLASSES.get(code)
        if runs and runs[-1][1] == cls:
            runs[-1][0] += ch
        else:
            runs.append([ch, cls])
    return runs


def render(art, lines, colors, art_colors=None):
    # The art is scaled to the card's height, unless that would make it wider than MAX_ART_W.
    # The card is at least as tall as the info column,
    # and grows when the art has many rows so its characters never shrink below ~8px.
    rows = max(len(lines), round(len(art) * MIN_ART_SCALE))
    height = PAD * 2 + rows * LINE_H
    art_cols = max((len(l) for l in art), default=0)
    scale = min(rows / max(1, len(art)), MAX_ART_W / max(1, art_cols * CHAR_W))
    top = (rows - len(lines)) // 2  # centre the info column beside taller art
    art_top = (rows - len(art) * scale) * LINE_H / 2  # and the art beside a taller info column
    info_x = PAD + art_cols * CHAR_W * scale + GAP
    width = round(info_x + WIDTH_CHARS * CHAR_W + PAD)
    y = lambda i: PAD + LINE_H * (i + 1 + top) - 5
    art_y = lambda i: PAD + art_top + LINE_H * scale * (i + 1) - 5 * scale

    def run(text, cls):
        text = escape(text)
        return f'<tspan class="{cls}">{text}</tspan>' if cls else text

    codes = art_colors or [""] * len(art)
    art_svg = "\n".join(
        f'<tspan x="{PAD}" y="{art_y(i):.1f}">{"".join(run(t, c) for t, c in art_runs(l, codes[i] if i < len(codes) else ""))}</tspan>'
        for i, l in enumerate(art)
    )
    info_svg = "\n".join(
        f'<tspan x="{info_x:.1f}" y="{y(i)}">{"".join(run(t, c) for t, c in line)}</tspan>'
        for i, line in enumerate(lines) if line
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" height="{height}px" font-size="{FONT_SIZE}px"
     font-family="ConsolasFallback,Consolas,'SF Mono',Menlo,'DejaVu Sans Mono',monospace">
<style>
@font-face {{
  font-family: ConsolasFallback;
  src: local('Consolas');
  font-display: swap;
  size-adjust: 109%;
}}
.key {{ fill: {colors['key']}; }}
.value {{ fill: {colors['value']}; }}
.addColor {{ fill: {colors['add']}; }}
.delColor {{ fill: {colors['dele']}; }}
.cc {{ fill: {colors['cc']}; }}
text, tspan {{ white-space: pre; }}
</style>
<rect width="{width}px" height="{height}px" fill="{colors['bg']}" rx="15"/>
<text fill="{colors['text']}" font-size="{FONT_SIZE * scale:.2f}px" xml:space="preserve">
{art_svg}
</text>
<text fill="{colors['text']}" xml:space="preserve">
{info_svg}
</text>
</svg>
"""


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("Set ACCESS_TOKEN (a GitHub token) first, e.g. ACCESS_TOKEN=$(gh auth token) python today.py")
    stats = get_stats()
    load = lambda f: (ROOT / f).read_text().rstrip("\n").split("\n")
    art = {"dark_mode.svg": load("ascii.txt")}
    # Photos get a second, tone-inverted copy for the light card; a monogram reuses the dark one.
    art["light_mode.svg"] = load("ascii_light.txt") if (ROOT / "ascii_light.txt").exists() else art["dark_mode.svg"]
    # Optional per-character colours for the art, one code per character (see ART_CLASSES).
    art_colors = load("ascii_colors.txt") if (ROOT / "ascii_colors.txt").exists() else None
    lines = card_lines(stats)
    readme = (ROOT / "README.md").read_text()
    for name, colors in THEMES.items():
        svg = render(art[name], lines, colors, art_colors)
        (ROOT / name).write_text(svg)
        # Browsers cache the card for a few minutes. Tagging its URL with a hash of its content
        # makes every changed card a new URL, so the profile shows it straight away.
        version = hashlib.sha256(svg.encode()).hexdigest()[:10]
        readme = re.sub(re.escape(name) + r"(\?v=[0-9a-f]+)?", f"{name}?v={version}", readme)
    (ROOT / "README.md").write_text(readme)
    print(json.dumps({k: str(v) for k, v in stats.items()}, indent=2))
    print(f"{calls} GraphQL calls")
