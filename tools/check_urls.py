"""This file checks every web link in the project and confirms that each cited paper title matches
the real page. It keeps the references honest. It is a maintenance tool and is not part of an
area.

More technical notes:
Verify every URL cited in the toolkit's docs/code: HTTP status must be 200 (after redirects).

For arXiv links we ALSO fetch the paper title and check it against `tools/references.json` (a map arxiv-id -> distinctive title
words that I expect). A mismatch means a wrong citation. Output: tools/url_check_report.json + exit code 1 on any failure.
Usage: python tools/check_urls.py [--no-title-check]
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL_RE = re.compile(r"https?://[^\s)>\]\"'`,]+")
SKIP_PARTS = (".venv", ".hf_cache", ".bin", "node_modules", "outputs", "results", "cache", "data", "__pycache__", ".git", ".pytest_cache")
IGNORE = ("localhost", "127.0.0.1", "HOST", "example.com", "<", "{", "pypi.org/simple", "download.pytorch.org", "abetlen.github.io/llama-cpp-python/whl",
          "api.github.com", "github.com/ggml-org/llama.cpp/releases/download", "huggingface.co/api", "www.python.org/ftp")


def collect():
    """Find every web link in the repository's text files and return them with the files where
    they appear.
    """
    urls: dict = {}
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".md", ".py", ".sh", ".ps1", ".yaml", ".txt", ".json"} and not any(s in p.parts for s in SKIP_PARTS):
            if p.name in ("references.json", "url_check_report.json"):
                continue
            for u in URL_RE.findall(p.read_text(errors="ignore")):
                u = u.rstrip(".;:*_")
                if any(i in u for i in IGNORE):
                    continue
                urls.setdefault(u, []).append(str(p.relative_to(ROOT)))
    return urls


def fetch(url, tries=4):
    """Download a web page with curl, trying again a few times if the server is busy, and
    return the status code and the page text.
    """
    for t in range(tries):
        r = subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0 (X11; Linux x86_64) toolkit-url-check", "-o", "/tmp/_u.html", "-w", "%{http_code}", "--max-time", "40", url],
                           capture_output=True, text=True)
        code = r.stdout.strip()
        if code in ("429", "500", "502", "503", "000") and t < tries - 1:
            time.sleep(4 * (t + 1)); continue
        return code, Path("/tmp/_u.html").read_text(errors="ignore") if Path("/tmp/_u.html").exists() else ""
    return code, ""


def arxiv_title(html):
    """Pull the paper title out of an arXiv web page."""
    m = re.search(r'<meta name="citation_title" content="([^"]+)"', html) or re.search(r"<title>(.*?)</title>", html, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def main():
    """Check that every link works and that every cited paper's title matches the list in
    references.json, then save a report.
    """
    ap = argparse.ArgumentParser(); ap.add_argument("--no-title-check", action="store_true"); a = ap.parse_args()
    refs = json.load(open(ROOT / "tools" / "references.json"))
    urls = collect()
    report, bad = {}, 0
    for u in sorted(urls):
        md = re.match(r"https?://doi\.org/(10\.\d{4,9}/\S+)$", u)
        if md:  # publishers often block bots (403), so verify DOIs through Crossref metadata instead (title must match references.json)
            doi = md.group(1)
            code, js = fetch("https://api.crossref.org/works/" + doi)
            try:
                title = json.loads(js)["message"]["title"][0]
            except Exception:
                title = ""
            want = refs.get("doi:" + doi)
            ok = code == "200" and want is not None and all(w.lower() in title.lower() for w in want)
            report[u] = {"status": code, "via": "crossref", "title": title, "title_check": "ok" if ok else f"MISMATCH/UNEXPECTED (want {want})", "files": sorted(set(urls[u]))}
            bad += 0 if ok else 1
            print("OK " if ok else "BAD", code, u, title[:70], flush=True)
            continue
        code, html = fetch(u)
        entry = {"status": code, "files": sorted(set(urls[u]))}
        m = re.match(r"https?://arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", u)
        if m and code == "200" and not a.no_title_check:
            aid = m.group(1)
            if "/pdf/" in u:
                _, html = fetch(f"https://arxiv.org/abs/{aid}")
            title = arxiv_title(html)
            entry["title"] = title
            want = refs.get(aid)
            if want is None:
                entry["title_check"] = "NO EXPECTATION IN references.json"; bad += 1
            elif not all(w.lower() in title.lower() for w in want):
                entry["title_check"] = f"MISMATCH expected words {want}"; bad += 1
            else:
                entry["title_check"] = "ok"
        if code != "200":
            bad += 1
        report[u] = entry
        flag = "OK " if code == "200" and entry.get("title_check", "ok") == "ok" else "BAD"
        print(flag, code, u, entry.get("title", "")[:70], entry.get("title_check", "") if flag == "BAD" else "", flush=True)
        time.sleep(0.4)
    (ROOT / "tools" / "url_check_report.json").write_text(json.dumps(report, indent=1))
    print(f"\n{len(urls)} urls, {bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
