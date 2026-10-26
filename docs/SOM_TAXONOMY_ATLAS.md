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
