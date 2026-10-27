# Taxonomy activation atlas

Open `/#atlas/groups` in the existing explorer. The atlas uses the original
navigation and visual style. Choose a layer, readout, seed, family split, and
script condition. Click a hexagon to inspect its group composition and supplied
responses. Group share highlights matching examples and shows matching/total
counts; empty or unrelated cells remain neutral. The behavior tab displays a
separate measured preference experiment when its summary is installed.

These are teacher-forced activation maps. Group labels describe supplied
stimulus conditions, not discovered circuits or spontaneous model behavior.
Legitimate correction, preference accommodation, and warm empathy controls must
not be treated as negative examples. Evaluation counts refer to sequences or
pairs, with the independent world-family counts shown above the map.

## Installing local artifacts

The read-only API discovers application-owned directories under
`research_outputs/atlases/<atlas-id>/`. Artifacts are ignored by Git. An
installation contains `atlas.json` and `receipt.json`; HTTP requests never
accept caller-supplied filesystem paths. Atlas IDs use letters, digits,
underscores, and hyphens, with a maximum of 64 characters.

The receipt has this shape:

```json
{
  "schema": "taxonomy-som-atlas-receipt-v1",
  "id": "example-atlas",
  "atlas_sha256": "SHA256_OF_EXACT_ATLAS_BYTES",
  "atlas_bytes": 12345
}
```

The atlas schema is `taxonomy-som-atlas-v1`. It contains model/revision metadata,
group definitions, coverage, maps, linked examples, and scope notes. Maps use
actual MiniSom Euclidean hexagon coordinates, not logical row/column indices.
Supported readouts are `answer_mean`, `shared_suffix`, and `paired_delta`, at
layers 6, 13, and 20 with seeds 0, 1, and 2. Each map has 36 cells. Every point
records its group, family, split, script condition, pair ID, and cell assignment.
Paired-difference points link to both supplied script examples.

The importer must calculate receipts after adding derived display metadata.
Keep completed source runs unchanged. Receipt validation proves file integrity
and structural compatibility; it does not validate semantic labels or establish
scientific success. The API rejects changed files and symlinked installations.
It exposes no model execution or fitting action.

`GET /api/atlases` lists verified installations and rejected IDs.
`GET /api/atlases/<atlas-id>` returns a verified atlas, bounded to 32 MiB.
The optional `metadata.behavior_summary` holds a distinct preference-study
table, its endpoint, denominators, conclusion, and caveats.

## Prompt-routed transfer screen

The same explorer also accepts `sycophancy-som-transfer-atlas-v1` artifacts.
Open `/#atlas/groups` for prompt-state and descriptive output maps, or
`/#atlas/steering` for the teacher-forced likelihood screen. The run selector
uses each artifact's display title and ID so runs with the same model name stay
distinct.

Transfer atlases use a row-major 4 by 4 hexagonal map at layers 3, 6, 9, 13,
16, 20, 23, and 26. `prompt_routing` maps use the pre-answer prompt state and
are the only map family marked route-eligible. `output_contrast` maps show
answer-mean or shared-suffix contrasts for description only. The layer,
readout, split, and cohort controls are populated from the installed map
descriptors. The cell edge marks the frozen training routing gate; source
cohort counts remain separate from synthetic group counts.

The strict transfer schema contains bounded cell coordinates and aggregate
counts for train, validation, and held-out splits. Per-cell synthetic groups,
existing-source collections, and routing eligibility are validated as
partitions of the corresponding totals. `coverage` reports synthetic pair,
row, and family counts separately from existing-source component and pair
counts. The likelihood summary includes a validation layer/strength grid and
held-out arm, group, control, and source-collection aggregates. Its metric is
teacher-forced likelihood, so it is a score surrogate rather than evidence of
free-generation behavior or causal efficacy. The bounded `metadata.not_established`
list is shown under that heading in the results view.

The transfer atlas has no prompt examples, generated responses, or raw point
records. Logical codebook and BMU references are checked as identifiers only;
the API does not resolve or serve their local audit artifacts. Both atlas
formats use the existing `taxonomy-som-atlas-receipt-v1` envelope, which binds
the exact `atlas.json` bytes by SHA-256 and byte length. The transfer validator
checks the new schema without changing validation of installed six by six
taxonomy atlases.
