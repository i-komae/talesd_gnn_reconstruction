from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "TALE-SD GNN Reconstruction"
author = "Ichiro Komae"
release = "0.1.0"
html_title = "TALE-SD GNN Reconstruction 0.1.0 documentation"
html_short_title = "TALE-SD GNN"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "ja"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = False
