"""S05 tournament annotation tooling (development only; not part of `resize()`).

`bracket`, `store`, `candidates`, `export` and `source_review` are plain Python/SQLite and need no UI
package. Only `app` imports Gradio (requirements-annotation.txt). `dev_candidates` is a diagnostic S05
generator, not the S04 production generator.
"""
