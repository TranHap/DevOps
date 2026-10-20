# Local explanations: RF + MCV (without step 3a), seed 0

Test set of seed 0: 172519 sessions; TP 4998, FN 9, FP 12. Cases: 2 TP, every FN, up to 5 FP.
SHAP values are for P(anomaly); base value (mean prediction of the forest) = 0.0293; decision: P ≥ 0.5.
'normal median' = median count of the template in normal training sessions; 'in x% of normal' = share of normal training sessions that contain it.

## 1. TP: blk_8517222510013069799 (random true positive (seeded))

- Label anomaly; P(anomaly) = 1.0000 → predicted anomaly. Figure: `local/waterfall_00_TP_blk_8517222510013069799.png`
- Sequence (20 events, consecutive repeats collapsed): T2 → T1×3 → T3 → T4 → T3 → T4 → T5×3 → T3 → T4 → T32×3 → T17 → T33 → T17×2

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T33: Unexpected error trying to delete bloc… | 1 | 0 | 0.037% | +0.9322 |
| T17: Deleting block <*> file <*> | 3 | 3 | 81.8% | +0.0274 |
| T32: BLOCK* NameSystem.delete: <*> is <*> t… | 3 | 3 | 81.5% | +0.0095 |
| T10: <*> Served block <*> to <*> | 0 | 0 | 24.0% | +0.0057 |
| T35: <*>:Got exception while serving <*> to… | 0 | 0 | 22.6% | +0.0056 |

- Reading: correctly flagged — T33 occurs (1×) although only 0.037% of normal sessions contain it (towards anomaly, +0.932); T17 occurs 3× (= normal median) (towards anomaly, +0.027); T32 occurs 3× (= normal median) (towards anomaly, +0.009).

## 2. TP: blk_-5375761801379702192 (true positive with the lowest score)

- Label anomaly; P(anomaly) = 0.5000 → predicted anomaly. Figure: `local/waterfall_01_TP_blk_-5375761801379702192.png`
- Sequence (230 events, consecutive repeats collapsed): T1 → T3 → T4 → T1 → T3 → T4 → T1 → T3 → T4 → T10 → T2 → T5×3 → T10×202 → T1 → T12 → T9 → T6 → T7 → T5×2 → T17 → T32×3 → T17×3

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T9: BLOCK* ask <*> to replicate <*> to dat… | 1 | 0 | 0.32% | +0.2783 |
| T5: BLOCK* NameSystem.addStoredBlock: bloc… | 5 | 3 | 100.0% | +0.2416 |
| T10: <*> Served block <*> to <*> | 203 | 0 | 24.0% | -0.2251 |
| T17: Deleting block <*> file <*> | 4 | 3 | 81.8% | +0.0896 |
| T12: <*> Starting thread to transfer block … | 1 | 0 | 0.32% | +0.0770 |

- Reading: correctly flagged — T9 occurs (1×) although only 0.32% of normal sessions contain it (towards anomaly, +0.278); T5 occurs 5× (normal median 3) (towards anomaly, +0.242); T10 occurs 203× (normal median 0) (towards normal, -0.225).

## 3. FN: blk_8133939776207978471 (false negative)

- Label anomaly; P(anomaly) = 0.2798 → predicted normal. Figure: `local/waterfall_02_FN_blk_8133939776207978471.png`
- Sequence (33 events, consecutive repeats collapsed): T2 → T1×3 → T3 → T4 → T3 → T4 → T3 → T4 → T5×2 → T13 → T9 → T12 → T1 → T5 → T7 → T6 → T5 → T17 → T13 → T35×2 → T10 → T13×2 → T32×3 → T17×3

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T9: BLOCK* ask <*> to replicate <*> to dat… | 1 | 0 | 0.32% | +0.3016 |
| T13: Verification succeeded for <*> | 4 | 0 | 14.4% | -0.2265 |
| T17: Deleting block <*> file <*> | 4 | 3 | 81.8% | +0.1147 |
| T5: BLOCK* NameSystem.addStoredBlock: bloc… | 4 | 3 | 100.0% | +0.1064 |
| T12: <*> Starting thread to transfer block … | 1 | 0 | 0.32% | +0.0772 |

- Reading: missed — T9 occurs (1×) although only 0.32% of normal sessions contain it (towards anomaly, +0.302); T13 occurs 4× (normal median 0) (towards normal, -0.227); T17 occurs 4× (normal median 3) (towards anomaly, +0.115).

## 4. FN: blk_5248431706625489737 (false negative)

- Label anomaly; P(anomaly) = 0.2000 → predicted normal. Figure: `local/waterfall_03_FN_blk_5248431706625489737.png`
- Sequence (38 events, consecutive repeats collapsed): T1 → T2 → T1×2 → T3 → T4 → T3 → T4 → T3 → T4 → T5×3 → T9 → T12 → T1 → T6 → T7 → T5×2 → T17 → T9 → T12 → T1 → T6 → T7 → T5 → T21 → T5 → T17 → T9 → T12 → T1 → T7 → T6 → T5×2 → T17

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T21: BLOCK* NameSystem.addStoredBlock: Redu… | 1 | 0 | 0.00051% | +0.3893 |
| T7: <*>:Transmitted block <*> to <*> | 3 | 0 | 0.32% | -0.1690 |
| T17: Deleting block <*> file <*> | 3 | 3 | 81.8% | -0.0849 |
| T5: BLOCK* NameSystem.addStoredBlock: bloc… | 9 | 3 | 100.0% | -0.0807 |
| T12: <*> Starting thread to transfer block … | 3 | 0 | 0.32% | +0.0680 |

- Reading: missed — T21 occurs (1×) although only 0.00051% of normal sessions contain it (towards anomaly, +0.389); T7 occurs (3×) although only 0.32% of normal sessions contain it (towards normal, -0.169); T17 occurs 3× (= normal median) (towards normal, -0.085).

## 5. FN: blk_-2891794341254261063 (false negative)

- Label anomaly; P(anomaly) = 0.2000 → predicted normal. Figure: `local/waterfall_04_FN_blk_-2891794341254261063.png`
- Sequence (284 events, consecutive repeats collapsed): T9 → T1 → T3 → T4 → T10 → T1 → T3 → T4 → T1 → T3 → T4 → T2 → T5×2 → T8 → T6 → T1 → T10×16 → T6 → T1 → T10×22 → T7 → T10×11 → T5×2 → T10×89 → T9 → T10×7 → T8×2 → T9 → T6 → T1 → T10×7 → T7 → T10×11 → T6 → T1 → T10×11 → T5×2 → T10×22 → T9 → T7 → T10×2 → T1 → T8 → T10 → T7 → T1 → T10×3 → T8 → T9 → T6 → T5 → T6 → T1 → T6 → T1 → T6 → T1×2 → T5×3 → T7 → T6×2 → T5×2 → T17×2 → T5 → T17 → T32×10 → T17×10

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T9: BLOCK* ask <*> to replicate <*> to dat… | 5 | 0 | 0.32% | +0.1975 |
| T5: BLOCK* NameSystem.addStoredBlock: bloc… | 13 | 3 | 100.0% | +0.1511 |
| T1: Receiving block <*> src: <*> dest: <*> | 13 | 3 | 100.0% | +0.1205 |
| T32: BLOCK* NameSystem.delete: <*> is <*> t… | 10 | 3 | 81.5% | -0.1163 |
| T10: <*> Served block <*> to <*> | 203 | 0 | 24.0% | -0.0596 |

- Reading: missed — T9 occurs (5×) although only 0.32% of normal sessions contain it (towards anomaly, +0.197); T5 occurs 13× (normal median 3) (towards anomaly, +0.151); T1 occurs 13× (normal median 3) (towards anomaly, +0.120).

## 6. FN: blk_-5914787548301061613 (false negative)

- Label anomaly; P(anomaly) = 0.2000 → predicted normal. Figure: `local/waterfall_05_FN_blk_-5914787548301061613.png`
- Sequence (38 events, consecutive repeats collapsed): T1×3 → T2 → T3 → T4 → T3 → T4 → T3 → T4 → T5×3 → T9 → T12 → T1 → T6 → T7 → T21 → T5×2 → T17 → T9 → T12 → T1 → T7 → T6 → T5×2 → T17 → T9 → T12 → T1 → T6 → T7 → T5×2 → T17

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T21: BLOCK* NameSystem.addStoredBlock: Redu… | 1 | 0 | 0.00051% | +0.3893 |
| T7: <*>:Transmitted block <*> to <*> | 3 | 0 | 0.32% | -0.1690 |
| T17: Deleting block <*> file <*> | 3 | 3 | 81.8% | -0.0849 |
| T5: BLOCK* NameSystem.addStoredBlock: bloc… | 9 | 3 | 100.0% | -0.0807 |
| T12: <*> Starting thread to transfer block … | 3 | 0 | 0.32% | +0.0680 |

- Reading: missed — T21 occurs (1×) although only 0.00051% of normal sessions contain it (towards anomaly, +0.389); T7 occurs (3×) although only 0.32% of normal sessions contain it (towards normal, -0.169); T17 occurs 3× (= normal median) (towards normal, -0.085).

## 7. FN: blk_8557172101111875221 (false negative)

- Label anomaly; P(anomaly) = 0.0113 → predicted normal. Figure: `local/waterfall_06_FN_blk_8557172101111875221.png`
- Sequence (32 events, consecutive repeats collapsed): T2 → T1×3 → T5×3 → T3 → T4 → T3 → T4 → T3 → T4 → T10×2 → T35 → T10×2 → T35 → T10×2 → T35 → T10×2 → T35 → T32×3 → T17×3 → T33

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T33: Unexpected error trying to delete bloc… | 1 | 0 | 0.037% | +0.4927 |
| T10: <*> Served block <*> to <*> | 8 | 0 | 24.0% | -0.2713 |
| T35: <*>:Got exception while serving <*> to… | 4 | 0 | 22.6% | -0.2263 |
| T7: <*>:Transmitted block <*> to <*> | 0 | 0 | 0.32% | -0.0058 |
| T17: Deleting block <*> file <*> | 3 | 3 | 81.8% | +0.0054 |

- Reading: missed — T33 occurs (1×) although only 0.037% of normal sessions contain it (towards anomaly, +0.493); T10 occurs 8× (normal median 0) (towards normal, -0.271); T35 occurs 4× (normal median 0) (towards normal, -0.226).

## 8. FN: blk_8232988695166315630 (false negative)

- Label anomaly; P(anomaly) = 0.0113 → predicted normal. Figure: `local/waterfall_07_FN_blk_8232988695166315630.png`
- Sequence (32 events, consecutive repeats collapsed): T2 → T1×3 → T5×3 → T3 → T4 → T3 → T4 → T3 → T4 → T35 → T10×3 → T35 → T10×3 → T35 → T10 → T35 → T10 → T32×3 → T17×3 → T33

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T33: Unexpected error trying to delete bloc… | 1 | 0 | 0.037% | +0.4927 |
| T10: <*> Served block <*> to <*> | 8 | 0 | 24.0% | -0.2713 |
| T35: <*>:Got exception while serving <*> to… | 4 | 0 | 22.6% | -0.2263 |
| T7: <*>:Transmitted block <*> to <*> | 0 | 0 | 0.32% | -0.0058 |
| T17: Deleting block <*> file <*> | 3 | 3 | 81.8% | +0.0054 |

- Reading: missed — T33 occurs (1×) although only 0.037% of normal sessions contain it (towards anomaly, +0.493); T10 occurs 8× (normal median 0) (towards normal, -0.271); T35 occurs 4× (normal median 0) (towards normal, -0.226).

## 9. FN: blk_7540335091384503130 (false negative)

- Label anomaly; P(anomaly) = 0.0106 → predicted normal. Figure: `local/waterfall_08_FN_blk_7540335091384503130.png`
- Sequence (34 events, consecutive repeats collapsed): T2 → T1×3 → T5×3 → T3 → T4 → T3 → T4 → T3 → T4 → T10 → T35 → T10×2 → T35 → T10×3 → T35 → T10×3 → T35 → T10 → T32×3 → T17×3 → T33

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T33: Unexpected error trying to delete bloc… | 1 | 0 | 0.037% | +0.4913 |
| T10: <*> Served block <*> to <*> | 10 | 0 | 24.0% | -0.2732 |
| T35: <*>:Got exception while serving <*> to… | 4 | 0 | 22.6% | -0.2238 |
| T7: <*>:Transmitted block <*> to <*> | 0 | 0 | 0.32% | -0.0058 |
| T17: Deleting block <*> file <*> | 3 | 3 | 81.8% | +0.0054 |

- Reading: missed — T33 occurs (1×) although only 0.037% of normal sessions contain it (towards anomaly, +0.491); T10 occurs 10× (normal median 0) (towards normal, -0.273); T35 occurs 4× (normal median 0) (towards normal, -0.224).

## 10. FN: blk_7900397200560512220 (false negative)

- Label anomaly; P(anomaly) = 0.0106 → predicted normal. Figure: `local/waterfall_09_FN_blk_7900397200560512220.png`
- Sequence (34 events, consecutive repeats collapsed): T2 → T1×3 → T5×3 → T3 → T4 → T3 → T4 → T3 → T4 → T35 → T10×4 → T35 → T10×2 → T35 → T10×3 → T35 → T10 → T32×3 → T17×3 → T33

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T33: Unexpected error trying to delete bloc… | 1 | 0 | 0.037% | +0.4913 |
| T10: <*> Served block <*> to <*> | 10 | 0 | 24.0% | -0.2732 |
| T35: <*>:Got exception while serving <*> to… | 4 | 0 | 22.6% | -0.2238 |
| T7: <*>:Transmitted block <*> to <*> | 0 | 0 | 0.32% | -0.0058 |
| T17: Deleting block <*> file <*> | 3 | 3 | 81.8% | +0.0054 |

- Reading: missed — T33 occurs (1×) although only 0.037% of normal sessions contain it (towards anomaly, +0.491); T10 occurs 10× (normal median 0) (towards normal, -0.273); T35 occurs 4× (normal median 0) (towards normal, -0.224).

## 11. FN: blk_-2074647664485597823 (false negative)

- Label anomaly; P(anomaly) = 0.0000 → predicted normal. Figure: `local/waterfall_10_FN_blk_-2074647664485597823.png`
- Sequence (46 events, consecutive repeats collapsed): T1×3 → T2 → T4 → T5 → T3 → T1 → T3 → T22 → T1 → T25 → T45 → T3 → T4 → T25 → T45 → T3 → T4 → T5×2 → T3 → T12 → T9 → T1 → T7 → T6 → T5×2 → T17 → T12 → T9 → T1 → T7 → T6 → T5×2 → T17 → T9 → T12 → T1 → T6 → T7 → T5×2 → T17

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T7: <*>:Transmitted block <*> to <*> | 3 | 0 | 0.32% | -0.2237 |
| T3: PacketResponder <*> for block <*> <*> | 5 | 3 | 100.0% | +0.1232 |
| T1: Receiving block <*> src: <*> dest: <*> | 8 | 3 | 100.0% | +0.0871 |
| T17: Deleting block <*> file <*> | 3 | 3 | 81.8% | -0.0739 |
| T12: <*> Starting thread to transfer block … | 3 | 0 | 0.32% | +0.0720 |

- Reading: missed — T7 occurs (3×) although only 0.32% of normal sessions contain it (towards normal, -0.224); T3 occurs 5× (normal median 3) (towards anomaly, +0.123); T1 occurs 8× (normal median 3) (towards anomaly, +0.087).

## 12. FP: blk_-8289665945161423952 (false positive (highest scores first))

- Label normal; P(anomaly) = 0.9872 → predicted anomaly. Figure: `local/waterfall_11_FP_blk_-8289665945161423952.png`
- Sequence (30 events, consecutive repeats collapsed): T2 → T1×3 → T5×2 → T3 → T4 → T3 → T4 → T3 → T4 → T12 → T9 → T1 → T5 → T6 → T7 → T5 → T17 → T35 → T10 → T35 → T13 → T32×3 → T17×3

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T9: BLOCK* ask <*> to replicate <*> to dat… | 1 | 0 | 0.32% | +0.3058 |
| T17: Deleting block <*> file <*> | 4 | 3 | 81.8% | +0.1709 |
| T5: BLOCK* NameSystem.addStoredBlock: bloc… | 4 | 3 | 100.0% | +0.1692 |
| T7: <*>:Transmitted block <*> to <*> | 1 | 0 | 0.32% | +0.1021 |
| T12: <*> Starting thread to transfer block … | 1 | 0 | 0.32% | +0.0851 |

- Reading: wrongly flagged — T9 occurs (1×) although only 0.32% of normal sessions contain it (towards anomaly, +0.306); T17 occurs 4× (normal median 3) (towards anomaly, +0.171); T5 occurs 4× (normal median 3) (towards anomaly, +0.169).

## 13. FP: blk_6937712548866091907 (false positive (highest scores first))

- Label normal; P(anomaly) = 0.9872 → predicted anomaly. Figure: `local/waterfall_12_FP_blk_6937712548866091907.png`
- Sequence (30 events, consecutive repeats collapsed): T2 → T1×3 → T3 → T4 → T3 → T4 → T5×2 → T3 → T4 → T9 → T12 → T1 → T5 → T6 → T7 → T5 → T17 → T35 → T10 → T35 → T13 → T32×3 → T17×3

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T9: BLOCK* ask <*> to replicate <*> to dat… | 1 | 0 | 0.32% | +0.3058 |
| T17: Deleting block <*> file <*> | 4 | 3 | 81.8% | +0.1709 |
| T5: BLOCK* NameSystem.addStoredBlock: bloc… | 4 | 3 | 100.0% | +0.1692 |
| T7: <*>:Transmitted block <*> to <*> | 1 | 0 | 0.32% | +0.1021 |
| T12: <*> Starting thread to transfer block … | 1 | 0 | 0.32% | +0.0851 |

- Reading: wrongly flagged — T9 occurs (1×) although only 0.32% of normal sessions contain it (towards anomaly, +0.306); T17 occurs 4× (normal median 3) (towards anomaly, +0.171); T5 occurs 4× (normal median 3) (towards anomaly, +0.169).

## 14. FP: blk_-8356640976087556977 (false positive (highest scores first))

- Label normal; P(anomaly) = 0.9872 → predicted anomaly. Figure: `local/waterfall_13_FP_blk_-8356640976087556977.png`
- Sequence (30 events, consecutive repeats collapsed): T2 → T1×3 → T5×2 → T3 → T4 → T3 → T4 → T3 → T4 → T9 → T12 → T1 → T6 → T5 → T7 → T5 → T17 → T35 → T10 → T35 → T13 → T32×3 → T17×3

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T9: BLOCK* ask <*> to replicate <*> to dat… | 1 | 0 | 0.32% | +0.3058 |
| T17: Deleting block <*> file <*> | 4 | 3 | 81.8% | +0.1709 |
| T5: BLOCK* NameSystem.addStoredBlock: bloc… | 4 | 3 | 100.0% | +0.1692 |
| T7: <*>:Transmitted block <*> to <*> | 1 | 0 | 0.32% | +0.1021 |
| T12: <*> Starting thread to transfer block … | 1 | 0 | 0.32% | +0.0851 |

- Reading: wrongly flagged — T9 occurs (1×) although only 0.32% of normal sessions contain it (towards anomaly, +0.306); T17 occurs 4× (normal median 3) (towards anomaly, +0.171); T5 occurs 4× (normal median 3) (towards anomaly, +0.169).

## 15. FP: blk_3628884547512717347 (false positive (highest scores first))

- Label normal; P(anomaly) = 0.9417 → predicted anomaly. Figure: `local/waterfall_14_FP_blk_3628884547512717347.png`
- Sequence (31 events, consecutive repeats collapsed): T2 → T1×3 → T3 → T4 → T3 → T4 → T3 → T4 → T5×2 → T9 → T12 → T1 → T5×2 → T7 → T6 → T17 → T13×2 → T35 → T10 → T35 → T32×3 → T17×3

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T9: BLOCK* ask <*> to replicate <*> to dat… | 1 | 0 | 0.32% | +0.3048 |
| T17: Deleting block <*> file <*> | 4 | 3 | 81.8% | +0.1663 |
| T5: BLOCK* NameSystem.addStoredBlock: bloc… | 4 | 3 | 100.0% | +0.1645 |
| T7: <*>:Transmitted block <*> to <*> | 1 | 0 | 0.32% | +0.0937 |
| T12: <*> Starting thread to transfer block … | 1 | 0 | 0.32% | +0.0857 |

- Reading: wrongly flagged — T9 occurs (1×) although only 0.32% of normal sessions contain it (towards anomaly, +0.305); T17 occurs 4× (normal median 3) (towards anomaly, +0.166); T5 occurs 4× (normal median 3) (towards anomaly, +0.165).

## 16. FP: blk_8557779918081539564 (false positive (highest scores first))

- Label normal; P(anomaly) = 0.9417 → predicted anomaly. Figure: `local/waterfall_15_FP_blk_8557779918081539564.png`
- Sequence (31 events, consecutive repeats collapsed): T2 → T1×3 → T5×2 → T3 → T4 → T3 → T4 → T3 → T4 → T9 → T12 → T1 → T5×2 → T6 → T7 → T17 → T35 → T10 → T35 → T13×2 → T32×3 → T17×3

| template | count here | normal median | share of normal sessions with it | SHAP |
|---|---|---|---|---|
| T9: BLOCK* ask <*> to replicate <*> to dat… | 1 | 0 | 0.32% | +0.3048 |
| T17: Deleting block <*> file <*> | 4 | 3 | 81.8% | +0.1663 |
| T5: BLOCK* NameSystem.addStoredBlock: bloc… | 4 | 3 | 100.0% | +0.1645 |
| T7: <*>:Transmitted block <*> to <*> | 1 | 0 | 0.32% | +0.0937 |
| T12: <*> Starting thread to transfer block … | 1 | 0 | 0.32% | +0.0857 |

- Reading: wrongly flagged — T9 occurs (1×) although only 0.32% of normal sessions contain it (towards anomaly, +0.305); T17 occurs 4× (normal median 3) (towards anomaly, +0.166); T5 occurs 4× (normal median 3) (towards anomaly, +0.165).
