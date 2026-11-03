# AX7-SOC Synthetic Chip-Test Dataset

Synthetic wafer sort, retest, final test and burn-in data for the **AI-Optimized Yield Advisor**.
It follows the data model in the technical design document (Section 5) and contains known,
injected yield problems so each AI engine can be developed and scored against ground truth.

All names (product, fab, tools, lots) are fictional. Regenerate or scale with:

```
python generate.py --lots 40 --wafers 25 --seed 2026 --out data
```

## At a glance

| Item | Value |
|---|---|
| Product / program | AX7-SOC, `AX7_CP1_R4.2`, 4-site parallel probe at 85 C |
| Lots / wafers | 40 lots x 25 wafers = 1,000 wafers (sorted 19 Aug to 15 Oct 2026) |
| Die per wafer | 500 (26 x 26 grid, round wafer) |
| Sort rows | 501,440 (500,000 first insertion + 1,440 contact retests) |
| Final test rows | 446,715 die that passed sort, joined by ECID |
| Mean yield | 89.3% final (89.2% first pass); clean wafers about 90.7% |
| Burn-in fails | 700 |

## Files

| File | Grain | Description |
|---|---|---|
| `sort_die_results.csv` | die x insertion | Wafer sort results: bins, test time, all parametric values |
| `wafer_summary.csv` | wafer | Gross/good die, first-pass and final yield, sort start |
| `lots.csv` | lot | Product, tester, probe card, program revision, dates |
| `lot_context_mes.csv` | wafer x process step | MES genealogy: equipment, chamber, reticle, recipe, timestamps |
| `wat_pcm.csv` | wafer x WAT site | E-test: Vt_n, Vt_p, Idsat, poly Rs, contact chain (5 sites/wafer) |
| `test_limits.csv` | test | Execution order, limits, units, bin mapping, test time |
| `bin_definitions.csv` | soft bin | Soft bin names and hard bin mapping |
| `final_test_burnin.csv` | die | FT_HOT result, FT Fmax, assembly lot, burn-in fail label |
| `ground_truth/wafer_labels.csv` | wafer | Injected pattern labels and causes, plus the hidden drivers |
| `ground_truth/die_labels.csv` | die | Defect source, latent-outlier flag, GDBN flag, neighbour fail count |
| `ground_truth/injected_events.csv` | event | Every injected problem, its scope and which engine should find it |
| `dataset_stats.json` | - | Summary counts and final fail Pareto |
| `generate.py` | - | Generator (numpy + pandas) |

Keep `ground_truth/` out of model inputs; use it only for evaluation.

## Join keys

- Wafer: `lot_id` + `wafer_id`
- Die: `lot_id` + `wafer_id` + `die_x` + `die_y`
- Final test to sort: `ecid` in `final_test_burnin.csv` also carries the die key
- Final disposition: when a die has `insertion = 2`, that row supersedes insertion 1

## sort_die_results.csv columns

| Column | Unit | Notes |
|---|---|---|
| lot_id, wafer_id, die_x, die_y | - | Die key; (0,0) is top-left, notch down |
| site | 1-4 | Probe site, `(x % 2) + 2*(y % 2) + 1` |
| stage, insertion | - | `CP1`; 1 = first pass, 2 = retest |
| tester_id, probe_card_id, program_rev, temp_c | - | Test context |
| hard_bin, soft_bin, pass | - | 1/1/1 = good die |
| test_time_ms | ms | Sum of executed tests (stop on first fail) |
| CONTACT_RES | ohm | Fail > 2.0, SB30 |
| IDDQ | uA | Fail outside 40-65, SB12 |
| LEAKAGE | nA | Fail > 60, SB21 |
| SCAN_FAIL_COUNT | count | Fail > 0, SB03 |
| MBIST_FAIL | flag | Fail = 1, SB09 |
| PLL_LOCK | us | Fail > 25, SB15 |
| VMIN | V | Fail > 0.68, SB07 |
| FMAX | MHz | Fail < 2250, SB05 |
| RING_OSC | MHz | Fail < 1120, SB06 |
| ADC_INL | LSB | Fail > 1.5, SB18 |
| ts | - | Die test timestamp |

Tests run in the order listed above. After the first failing test the remaining test values are
blank, as on a real stop-on-fail tester.

## Injected problems (ground truth)

| ID | What was injected | Where | Engine it tests |
|---|---|---|---|
| EXCURSION_ETCH | Edge-ring loss on wafers etched in **ETCH-07 chamber C** after its PM on 30 Sep 2026 | 15 wafers, yield about 66% | Wafer map AI, root cause, excursion |
| CMP_CENTER | Center pattern on about 55% of **CMP-03** wafers | Lots L262210 to L262216 | Wafer map AI, root cause |
| VT_SHIFT | WAT **Vt_n +18 mV** raises Vmin and lowers Fmax; SB07 and SB05 rise | Lots L262226 to L262231, yield about 85% vs 90% | Excursion, WAT-to-sort correlation, forecast |
| PROBE_SITE3 | **PC-4412 site 3** contact resistance climbs with card touchdowns; about 74% of SB30 fails pass on retest | PC-4412 lots, later ones worst | Root cause (probe vs process), retest advice |
| SCRATCH | Scratch lines from wafer handler **ROB-02** | About 6% of ROB-02 wafers | Wafer map AI |
| LOCAL_CLUSTER | Chuck-particle clusters | About 3% of wafers | Wafer map AI, GDBN |
| RETICLE | Repeating fail at position (1,2) of each 3x3 die field, reticle **RT-118B** | Lots L262218 to L262220 | Wafer map AI (open-set class) |
| LATENT_OUTLIERS | 0.3% of die with IDDQ 8 to 11 uA high but inside spec | `is_latent_outlier` | DPAT / NNR: about 35% burn-in fail vs 0.05% |
| GDBN | Passing die with 5 or more failing neighbours | `is_gdbn` | GDBN screen: about 2.7% burn-in fail |
| REDUNDANT_TEST | RING_OSC tracks FMAX (r = 0.996, about 13 unique fails); ADC_INL almost never fails | T4120, T6000 | Test time optimizer |

Background behaviour: random defects follow a gamma-mixed Poisson (negative binomial) model,
edge die have extra loss, IDDQ has a radial trend (useful for NNR), and each lot and wafer has a
small random offset.

## Suggested exercises

1. Train a wafer map classifier on `sort_die_results.csv` (insertion 1, `pass` as the map) and
   score it against `wafer_labels.pattern_labels`.
2. Run commonality analysis of wafer yield against `lot_context_mes.csv` and confirm ETCH-07/C and CMP-03.
3. Chart lot yield in sort order with EWMA and find the onset of the Vt shift; confirm with `wat_pcm.csv`.
4. Compare site yield and CONTACT_RES by probe card; quantify retest recovery from insertion 2.
5. Compute per-wafer DPAT limits on IDDQ, flag outliers, and measure precision and recall
   against `burnin_fail` in `final_test_burnin.csv`.
6. Count unique fails per test and estimate the escape bound for removing RING_OSC (rule of three).
