# Experiment 2: irrelevant-context retrieval robustness

## Question

Does making the differential coefficient more granular improve exact retrieval when a query must recover one relevant fact from an increasing number of irrelevant facts?

## Synthetic NIAH task

Each example is an integer-token prompt containing `n` distractor facts and one needle fact. A fact is `[type, key, value, SEP]`; distractor and needle use distinct type tokens. The suffix is `[QUERY, needle_key, ANSWER]`. The model receives the prompt and is scored only on predicting `needle_value` from the last suffix position. Distractor keys are guaranteed not to equal the queried key. Values are uniform random. This rules out position-only and value-frequency shortcuts.

## Controlled comparison

- Architecture: the Experiment 1 transformer (6 layers, 256 width, 8 heads, 704 FFN), retaining its 8,192-token vocabulary so every Experiment 2 run can be fine-tuned from its matching Experiment 1 checkpoint. The synthetic task uses only a reserved subset of these IDs.
- Training: answer-only cross entropy; 8,000 updates, batch 128; each update samples 4--30 distractors and cycles early/middle/late needle placement. All variants will use the same generated batches per seed.
- Evaluation: held-out deterministic sets, 512 prompts per `(distractor count, needle depth)` cell. Distractor counts are 0, 4, 8, 16, 30, 60, 120, 240, and 480, corresponding to 7 through 1,927 input tokens. The first five are in or near the training range; the last four test length extrapolation. For nonzero distractors, needle fact placement is stratified into early, middle, and late thirds.
- Primary outcome: exact-match answer accuracy, macro-averaged across the three depth buckets at every distractor count. The robustness summary is the equal-weight mean accuracy over the four extrapolation counts (60--480 distractors), plus the accuracy drop from 30 to 480 distractors.
- Secondary outcomes: per-depth accuracy, answer cross-entropy, and accuracy drop from 0 to 30 distractors. Report three training seeds and mean ± standard deviation; do paired seed-wise comparisons against Standard Attention.

## Pilot gate

Before the full matrix, run only Standard Attention for 1,500 updates. The pilot checks CUDA execution, checkpoint/result writing, and calibrates the evaluation range. It achieved 100% at 120 distractors but degraded at 240 and 480 distractors, motivating the extrapolation grid above. It is not used for a model comparison.
