"""Kuku Problem 1 - Microdrama Promo Discovery, Stitching, Ranking & Quality Auditing.

Five-stage hybrid pipeline:
  1. parser.py               deterministic multi-track parser + wall-clock timeline slicer   (code)
  2. candidate_generator.py  continuous / intra-episode / cross-episode candidate builder    (code + signals)
  3. evaluator.py            blind cold-viewer scorer: keyword scorer or a model (LLMAdapter)
  4. ranker.py               score formula + Post first / Post next / Later / Hold (code)
  5. quality_auditor.py      failure-mode detector, naive-vs-guarded comparisons             (code)
"""

__version__ = "1.0.0"
