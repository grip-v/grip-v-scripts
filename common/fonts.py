"""Cross-platform bold sans-serif font resolution.

Several datasets render text (labels, grid coordinates, letters) via PIL's
ImageFont.truetype and need one concrete .ttf path. That path is NOT
portable: macOS ships Arial under /System/Library/Fonts/Supplemental/, but
Linux clusters (no macOS system fonts at all) typically have DejaVu Sans
Bold or Liberation Sans Bold instead, usually from the `fonts-dejavu-core`
or `liberation-fonts` system packages (or a fontconfig-managed location).

`resolve_font_path()` tries a list of known-good candidates in order and
returns the first that exists on disk, so the same config.py works
unmodified on a macOS dev machine and a Linux SLURM cluster. If nothing is
found, it raises with an actionable message rather than silently falling
back to some unstyled default -- every dataset that renders text is relying
on a genuine BOLD font for legibility/contrast, not whatever PIL's built-in
placeholder happens to look like.
"""
from __future__ import annotations
import os

# Ordered by preference: exact look (Arial Bold) first, then the closest
# common substitutes actually present on most Linux distros/clusters.
_CANDIDATES = (
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
    # Linux -- fonts-dejavu-core (Debian/Ubuntu), dejavu-sans-fonts (RHEL/Fedora)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/DejaVuSans-Bold.ttf",
    # Linux -- liberation-fonts (metric-compatible Arial substitute)
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    # conda/mamba environments commonly vendor DejaVu via matplotlib's data dir
    os.path.expanduser("~/.conda/envs/*/fonts/ttf/DejaVuSans-Bold.ttf"),
)

# Env var escape hatch: point this at any .ttf directly (e.g. a cluster
# module-provided font, or a path not covered above) without editing code.
_ENV_OVERRIDE = "GEOMETRIC_VIDEO_DATASET_FONT_PATH"


def resolve_font_path() -> str:
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(
                f"${_ENV_OVERRIDE} is set to {override!r} but that file doesn't exist."
            )
        return override

    for path in _CANDIDATES:
        if "*" in path:
            import glob
            matches = glob.glob(path)
            if matches:
                return matches[0]
            continue
        if os.path.isfile(path):
            return path

    # Last resort, no admin/system-package access required: matplotlib ships
    # the full DejaVu family (DejaVuSans-Bold.ttf included) as package data,
    # so `pip install matplotlib` alone guarantees a bold sans .ttf exists,
    # on any platform, inside any venv/conda env -- including a cluster
    # account with no sudo. Only used if matplotlib happens to be installed;
    # never added to requirements.txt as a forced dependency just for this.
    try:
        import matplotlib
        mpl_font = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")
        if os.path.isfile(mpl_font):
            return mpl_font
    except ImportError:
        pass

    raise FileNotFoundError(
        "No bold sans-serif .ttf font found. Tried:\n  "
        + "\n  ".join(_CANDIDATES)
        + "\n  <matplotlib's bundled DejaVuSans-Bold.ttf, if matplotlib is installed>"
        + f"\nEasiest fix: `pip install matplotlib` (bundles DejaVu Sans Bold, no "
        f"admin rights needed). Alternatively install a system font package "
        f"(`sudo apt install fonts-dejavu-core` or `sudo yum install "
        f"liberation-fonts`), or set ${_ENV_OVERRIDE} to a specific .ttf path."
    )
