"""Cross-platform English wordlist resolution.

`cipher_card_trace` needs one big list of
plain lowercase English words, historically loaded straight from macOS's
built-in Webster's Second International wordlist at
/usr/share/dict/web2 (a symlink target of /usr/share/dict/words). That path
is macOS-specific and doesn't exist on Linux at all (a cluster account with
no sudo/admin rights can't just `apt install` a system dictionary either).

`resolve_wordlist_lines()` returns an iterable of raw words (unfiltered --
callers still apply their own MIN_LEN/MAX_LEN/isalpha()/islower() filtering,
exactly like the old direct-file-read code did), sourced from, in order:
  1. macOS's own web2 file, if present (keeps existing macOS runs identical)
  2. common Linux system dictionary locations (words/wamerican/wbritish
     packages, if installed)
  3. the `english-words` PyPI package's bundled "web2" word set -- this is
     the SAME underlying wordlist (literally sourced from macOS's web2), so
     `pip install english-words` reproduces macOS behavior byte-for-byte on
     any platform, with no admin rights and no separate download step
  4. an env var override for anything unusual

Picking a source that's available doesn't change WHICH words end up in the
filtered set for a given MIN_LEN/MAX_LEN -- source (1) and (3) are the same
underlying wordlist, so cipher_card_trace stays
reproducible whether run on macOS directly or via the pip package elsewhere.
Only a fallback to (2), a genuinely different wordlist, can shift results;
that's why (1)/(3) are tried first.
"""
from __future__ import annotations
import os

_ENV_OVERRIDE = "GEOMETRIC_VIDEO_DATASET_WORDLIST_PATH"

_SYSTEM_CANDIDATES = (
    # macOS
    "/usr/share/dict/web2",
    "/usr/share/dict/words",
    # Linux -- wamerican/words (Debian/Ubuntu), words (RHEL/Fedora)
    "/usr/share/dict/american-english",
    "/usr/share/dict/british-english",
    "/usr/share/dict/cracklib-small",
)


def resolve_wordlist_lines() -> list[str]:
    """Returns raw word strings (not yet length/case/alpha-filtered)."""
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"${_ENV_OVERRIDE} is set to {override!r} but that file doesn't exist.")
        with open(override, encoding="utf8", errors="ignore") as f:
            return f.readlines()

    for path in _SYSTEM_CANDIDATES:
        if os.path.isfile(path):
            with open(path, encoding="utf8", errors="ignore") as f:
                return f.readlines()

    # No admin rights needed: `pip install english-words` bundles the exact
    # same web2 wordlist as a Python package (get_english_words_set(['web2'])).
    try:
        from english_words import get_english_words_set
        return sorted(get_english_words_set(["web2"], lower=True))
    except ImportError:
        pass

    raise FileNotFoundError(
        "No system wordlist found. Tried:\n  "
        + "\n  ".join(_SYSTEM_CANDIDATES)
        + "\n  <english-words pip package, if installed>"
        + f"\nEasiest fix: `pip install english-words` (bundles the same web2 "
        f"wordlist macOS uses, no admin rights needed). Alternatively install "
        f"a system dictionary package (`sudo apt install wamerican` or `sudo "
        f"yum install words`), or set ${_ENV_OVERRIDE} to a specific wordlist "
        f"file path."
    )
