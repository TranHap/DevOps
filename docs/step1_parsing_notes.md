# Step 1 - Log parsing: notes for the report

## Setup
- Parser: Drain via `drain3` (v0.9.x). Params: depth=4, sim_th=0.5, max_children=100
  (the standard benchmark setting, Zhu et al. 2019; also what Wu et al. used).
- Drain runs on the **whole dataset before the train/test split**, like the paper did.
  -> threat to validity. To limit it, the MCV vocabulary is built **from the train split only**
  (templates seen only in test go to an "unknown" column).
- Preprocessing (in the spirit of Qin et al., SANER'25): before Drain, mask variables with regexes
  (`src/parse.py`: COMMON_REGEX, DATASET_REGEX, NUMBER_REGEX):
  - common: block id, IP(:port), 0x-hex, file paths, standalone numbers
  - BGL: Wu et al.'s refined regexes + 2 of ours (iDo chip ids `FF:F2:9F:...`, node locations `R02-M1-N0-C:J12-U11`)
  - merge adjacent wildcards and runs of wildcard tokens (variable-length lists, e.g. `delete blk_1 blk_2 ...`)
- Messages that are all wildcards after masking (997 BGL lines) go to one fixed template, EventId 0.
  Without this, drain3 scores them as similarity 0 to every cluster and makes **one cluster per line**
  (first run: 1,444 BGL templates, 997 of them `<*>` of size 1).
- BGL: 34,470 lines have an empty Content (e.g. `... RAS KERNEL FATAL`). They are kept as `<EMPTY>`, not
  dropped, because they can carry anomaly labels.

## Parsing accuracy (Loghub-2k, corrected ground truth from logparser)  -> `results/parsing_accuracy.csv`
| Dataset | Preprocessing | GA | #templates (pred / truth) |
|---|---|---|---|
| HDFS | no  | 0.998 | 17 / 14 |
| HDFS | yes | **1.000** | 14 / 14 |
| BGL  | no  | 0.963 | 108 / 120 |
| BGL  | yes | **0.970** | 106 / 120 |

Lessons worth reporting:
- Preprocessing helps, but **how wildcards are written back matters**. Our first version re-spaced wildcards
  (`<*>:` -> `<*> :`). That changes the token count Drain partitions on, and BGL GA fell to 0.918, *below* no
  preprocessing. Found by leave-one-out ablation of each regex.
- Most of BGL's remaining errors are Drain over-merging short messages that differ in one key word
  (`program interrupt: illegal instruction` vs `privileged instruction`; `Connection reset by peer` vs
  `Link has been severed`). Tuning depth/sim_th moved GA only between 0.969 and 0.973, so we kept the
  defaults (no overfitting to the 2k sample).

## Full datasets
| Dataset | Lines | Templates | Parse time | Notes |
|---|---|---|---|---|
| HDFS | 11,175,629 | 46 | 511 s | 575,061 sessions (block_id), 16,838 anomalous (2.93%): matches Loghub labels. Paper: MCV = 46 dims on HDFS |
| BGL  | 4,747,963 | 382 (first run: 1,444) | 308 s | 54 templates seen <5 times, 22 seen once; 348,460 alert lines (7.34%): matches Loghub |

Machine: Windows 11, CPU only, 32 GB RAM, Python 3.12.

## Error analysis (BGL-2k)
- GA on anomalous lines 0.909 vs normal lines 0.974. All 13 misgrouped anomalous lines come from 2 GT
  templates that Drain merges into one (`ciod: Error reading message prefix ... Link has been severed` /
  `... Connection reset by peer`). Both are anomalies, so the merge loses the error *type* but should barely
  hurt binary detection.
- The 34,470 empty-Content lines all map to one template (`<EMPTY>`, EventId 194); none is anomalous.
  The 997 all-wildcard lines (EventId 0) are all normal too.

## Freeze
EventIds depend on regexes, Drain params and line order: re-parsing with any change renumbers them.
Frozen as git tag `step1-final`; parsed outputs are verified by `manifests/step1_parsed.sha256`
(`cd data && sha256sum -c ../manifests/step1_parsed.sha256`).
