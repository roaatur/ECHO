latest: 
- new_analysis_summary.csv — Feature correlation table. All 65 features ranked by how well they predict difficulty on DETRAC. Top: fast_keypoints_half (ρ=0.495).

- new_full_eval_results.pdf — DETRAC main result. LSTM 78.9% vs MedSample 79.0% — tied. Features work but don't beat probing on static cameras.

- new_seed_validation_results.pdf — 80% ceiling proof. 3 architectures × 5 seeds all hit ~80%. No model breaks through — the limit is the features, not the model.

- viss_full_evaluation_results.pdf — VisDrone main result (strongest). GRU 77% vs MedSample 50.4% — +27 point gap. On moving cameras, features crush probing.

- viss_trend_analysis_results.pdf — Negative result. Feature slopes don't predict difficulty. Only 2/113 features where slope beats raw. Intra-sequence transitions remain unsolved.
