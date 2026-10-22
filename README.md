# Synchrotron Micro-CT Morphometry

Python pipeline for 3D synchrotron micro-CT morphometric analysis, spatial Z-axis profiling, and robust inferential statistics.

## Features
- **Data Preprocessing:** Automated parsing, Z-axis anatomical polarity alignment, and boundary artifact trimming[cite: 2].
- **Spatial Profiling:** Rolling-average low-pass filtering for Z-coordinate morphometric tracks[cite: 2].
- **Statistical Analysis:** Model diagnostics (Shapiro-Wilk, Levene), Welch's t-tests, Dunnett's post-hoc tests, and Cohen's $d$ effect size calculations[cite: 2].
- **Automated Visualization:** Standardized academic plotting engine for global, individual, and comparative metric distributions[cite: 2].

## Requirements
- Python 3.x
- NumPy[cite: 2]
- Pandas[cite: 2]
- Matplotlib[cite: 2]
- Seaborn[cite: 2]
- SciPy[cite: 2]
