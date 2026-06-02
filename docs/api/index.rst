TALE-SD GNN API Documentation
=============================

This Sphinx tree documents how to use the command-line and Python APIs for
HDF5 graph export, model training, prediction, diagnostics, and feature
analysis.

日本語版と英語版を同じSphinxプロジェクト内に置いています。

Build
-----

.. code-block:: bash

   uv sync --dev
   .venv/bin/sphinx-build -b html docs/api docs/api/_build/html

Open ``docs/api/_build/html/index.html`` after the build.

.. toctree::
   :maxdepth: 2
   :caption: 日本語

   ja/index

.. toctree::
   :maxdepth: 2
   :caption: English

   en/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference
