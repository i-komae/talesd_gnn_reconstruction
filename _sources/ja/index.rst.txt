日本語 API 利用ガイド
========================

この文書は、HDF5作成、学習、推論、診断を行う時に、どのコマンドとPython APIを使うかをまとめます。
単にコマンドを列挙するのではなく、実行した時にどのソースコードへ入り、どのデータ構造が作られ、
どの出力が後段で使われるかを追えるようにしています。

コードを読む場合は、まず :doc:`code_map` で全体の流れを確認し、その後に
:doc:`code_walkthrough` でモデル、message passing、loss の実装を読むのが自然です。

.. toctree::
   :maxdepth: 2

   cli
   hetero_workflow
   hetero_model
   code_map
   code_walkthrough
   python_api
   server_workflow
