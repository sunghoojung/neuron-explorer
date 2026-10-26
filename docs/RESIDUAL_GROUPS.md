# Residual groups workspace

The Residual groups workspace compares prompt-only residual states across user-defined conversation groups, trains small SOMs on training rows, and optionally scores held-out candidate continuations. Open it from the explorer's **residuals** tab at `/#residuals`.

The starting dataset is a scripted fictional example for checking the interface. It is not experiment data. Starting a capture sends only the edited dataset to the local explorer runtime and saves artifacts under `research_outputs/residual_groups/<model-namespace>/<opaque-id>/`.

## Dataset format and matching

The request body contains a `dataset` object with a name and samples:

```json
{
  "name": "example",
  "samples": [
    {
      "id": "case-01-neutral",
      "pair_id": "case-01",
      "family_id": "family-01",
      "split": "train",
      "group": "neutral",
      "messages": [
        {"role": "user", "content": "Answer from the evidence in this fictional case."}
      ],
      "target": " the supported answer",
      "foil": " the alternative answer"
    }
  ]
}
```

Each sample needs a unique ID, pair ID, family ID, split (`train`, `validation`, or `test`), group, and one or more role-tagged messages. A family stays within one split. A pair stays in one family and split, and may contain at most one sample per group. The same `pair_id` is the unit used to form matched group differences.

The local validator caps a set at 64 samples and 8 groups. Prompts must render to at most 512 tokens. Captures can select at most 28 model layers, any subset of the five supported points, a 1-16 token window, and at most 256 MiB of raw arrays. The default point set is `resid_pre`, `resid_mid`, `resid_post`, `attn_write`, and `mlp_write`; default layers are 6, 13, and 20 when those layers exist.

Capture uses the model adapter's chat template with an assistant generation prefix and no answer continuation. The saved states therefore represent prompt-only context. They are not teacher-forced answer-token activations. Candidate scoring later uses `target` and `foil`; these fields are optional for capture and SOM fitting.

## Capture summaries

For a selected layer and point, the summary reports:

- Per-group sample count, mean last-token L2 norm, mean RMS norm, and mean distance from the group centroid.
- For every matched pair of groups, the number of complete pairs, mean L2 norm of the paired vector difference, and cosine between group centroids.
- A token strip for each sample. Token positions are relative to that prompt's final token; token IDs and token text are not aligned between different prompts.

All measurements are descriptive geometry. They do not establish that a response is truthful, that groups are behaviorally different in a reliable population, or that a particular condition caused a change.

## SOM fitting

`states` mode trains the SOM on final-token states from training samples only. `paired_delta` mode trains on one training vector per complete pair, computed as `group_a - group_b`. Pair and family split rules prevent those examples from crossing between training and held-out splits.

Normalization, when selected, divides each training vector by its own norm. No held-out statistic is used to center, scale, or fit prototypes. The completed fit saves prototypes, the orthonormal prototype span, and train-derived comparison bases. Every split is assigned afterward to the frozen prototypes. The map reports occupancy and quantization error as geometry, not as a class label or scientific result.

The projection basis is the global span of the SOM prototypes. It is not a cell-routed or per-prompt SOM gate. Candidate probes are available only for `paired_delta` fits, so their candidate subspace is trained on paired group differences. `states` fits can still be inspected geometrically but cannot start a candidate probe.

## Candidate probes

Probes use at most 8 held-out rows, selected deterministically from complete pairs in the chosen `validation` or `test` split. The selected two group rows must have identical target and foil strings; prompt messages may differ. Both continuations are scored against each row's prompt. Projection is available only at `resid_pre`, `resid_mid`, or `resid_post` on a Llama-family model.

The methods are baseline, zero-strength SOM, SOM projection removal, global-mean removal, centered-PCA removal, and a same-rank random control. Positive UI strength from 0 to 1 maps to a negative projection coefficient in the scoring helper, where -1 removes the full component. A zero-strength run is checked against baseline with a tolerance of 1e-6 and reported explicitly. Rank-zero controls are recorded as identity controls; they are not replaced with a nonzero basis.

Scores use mean token log-probability for each continuation, reducing continuation-length confounding. The reported target-minus-foil margin is a candidate preference measure only. It is not semantic accuracy, truthfulness, or evidence of successful steering. Candidate scoring is a bounded pilot and does not include inference statistics or a held-out selection procedure.

## API and saved artifacts

The router prefix is `/api/residuals`. It provides configuration and scripted demo data, dataset validation, asynchronous capture/fit/probe jobs, job progress and cancellation, saved capture/fit lists, selected-capture summaries and export, fit maps and cell details, and probe results. Request bodies are limited to 1 MiB and compressed bodies are rejected. IDs are opaque and the API does not accept caller-supplied paths.

New completed outputs carry SHA-256 receipts. Reads check the exact completed-job artifact set, file hashes, metadata-to-array shapes, model and dataset provenance, and SOM assignments before returning results. Earlier version-1 outputs without job receipts can still be opened after structural checks, but are labeled `structure_only` rather than `verified`. Captures save the normalized dataset, a token-map sidecar, a manifest, and an NPZ containing `vectors`, `token_ids`, `positions`, and `valid`. Fits save prototypes, basis and control arrays plus a result manifest. Probe reports contain only bounded aggregate scores and sample identifiers, not prompt or continuation text.

The CPU-only service tests use a fake runtime and do not load weights. A separate local MPS smoke run checked capture, paired-delta fitting, probe plumbing, and array export on the scripted example. Those checks establish integration and artifact consistency only; they do not validate a research hypothesis.
