# Synchrotron Micro-CT Morphometry

Python pipeline for 3D synchrotron micro-CT morphometric analysis, spatial Z-axis profiling, and robust inferential statistics.

## Features
- **Data Preprocessing:** Automated parsing, Z-axis anatomical polarity alignment, and boundary artifact trimming.
- **Spatial Profiling:** Rolling-average low-pass filtering for Z-coordinate morphometric tracks.
- **Statistical Analysis:** Model diagnostics (Shapiro-Wilk, Levene), Welch's t-tests, Dunnett's post-hoc tests, and Cohen's $d$ effect size calculations.
- **Automated Visualization:** Standardized academic plotting engine for global, individual, and comparative metric distributions.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
