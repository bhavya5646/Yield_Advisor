# AX7-SOC Yield Advisor
Runs engines from the technical design on the synthetic dataset in `data/` and builds a self-contained `dashboard.html`.
Requires Python 3.9+ with pandas, numpy, scipy.  Run `python run.py` (or `python analyze.py [data_dir]` and open dashboard.html).
Engines: lot EWMA/Shewhart excursion detector, wafer-map pattern classifier (radial features), Fisher commonality root cause,
WAT-to-yield correlation, probe card/site and retest recovery, DPAT + GDBN screens scored against burn-in, test-time optimizer (rule of three), ranked Advisor inbox.
Swap `data/` for your own files with the same schema (see DATASET_README.md). Ground-truth label files are intentionally not included.
