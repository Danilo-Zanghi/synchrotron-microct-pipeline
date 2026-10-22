import os
import glob
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

"""
=================================================================
MASTER PIPELINE: MORPHOMETRIC ANALYSIS & STATISTICAL REPORTING
=================================================================
"""

# Suppress non-critical pandas and seaborn warnings to maintain a clean terminal output.
# This ensures the researcher only visualizes critical statistical outputs and execution errors.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- A4 PRINT OPTIMIZATION ---
# Standardizes global matplotlib parameters for high-resolution academic printing.
plt.rcParams.update({
    'xtick.labelsize': 14,      # X-axis tick label size
    'ytick.labelsize': 14,      # Y-axis tick label size
    'axes.labelsize': 16,       # Axis title size
    'axes.titlesize': 18,       # Main title size
    'legend.fontsize': 12,      # Legend text size
    'legend.title_fontsize': 14 # Legend title size
})

# =================================================================
# GLOBAL HELPER FUNCTIONS
# =================================================================

def calculate_cohen_d(target_data, control_data):
    """
    Calculates Cohen's d using pooled standard deviation.
    
    STATISTICAL RATIONALE:
    When dealing with small sample sizes (n=2), p-values can be artificially inflated 
    by natural biological variance. Cohen's d measures the absolute 'Effect Size' 
    (the magnitude of the biological insult or recovery), providing a robust metric 
    that complements the p-value.
    
    Formula: (Mean1 - Mean2) / Pooled_SD
    """
    n_t, n_c = len(target_data), len(control_data)
    
    # Calculate pooled variance to weight the standard deviations of both groups equally
    sd_pooled = math.sqrt(((n_t - 1) * np.var(target_data, ddof=1) + 
                           (n_c - 1) * np.var(control_data, ddof=1)) / (n_t + n_c - 2))
    
    # Return effect size, avoiding ZeroDivisionError if variances are completely flat
    return (np.mean(target_data) - np.mean(control_data)) / sd_pooled if sd_pooled != 0 else 0


# =================================================================
# PHASE 0: CONFIGURATION AND MAPPING
# =================================================================

# Group Mapping: Links raw file abbreviations to full descriptive academic names.
group_map = {
    "CTRL": "CTRL",
    "CUP1": "CUP1",
    "CUP2": "CUP2",
    "FUS1": "FUS1",
    "FUS2": "FUS2"
}

"""
VISUAL RATIONALE FOR COLOR MAPPING:
Assigns medium-light [0], base [1], and dark [2] shades to each group.
The [0] index (used for individual subjects) is slightly darkened to ensure 
high contrast against the standard deviation shading (alpha=0.15) in spatial plots.
"""
color_map = {
    "CTRL":   ["#5DADE2", "#3498DB", "#1B4F72"], # Blue palette (Baseline)
    "CUP1":   ["#F5B041", "#F39C12", "#873600"], # Orange palette (Onset)
    "CUP2":   ["#EC7063", "#E74C3C", "#7B241C"], # Red palette (Peak Damage)
    "FUS1":   ["#58D68D", "#27AE60", "#145A32"], # Green palette (Early Recovery)
    "FUS2":   ["#AF7AC5", "#8E44AD", "#4A235A"]  # Purple palette (Full Recovery)
}

# Metric Mapping: Translates internal dictionary keys to publishable Y-axis labels.
metric_map = {
    "Mean":      "Mean Thickness (mm)",
    "Median":    "Median Thickness (mm)",
    "Mode":      "Mode (mm)",
    "Std_Dev":   "Standard Deviation (mm)",
    "CV_Pct":    "CV (%)",
    "IQR":       "Interquartile Range (mm)",
    "Skew":      "Skewness",
    "Kurt":      "Kurtosis",
    "P10":       "10th Percentile (mm)", # Represents small-caliber nerve fibers
    "P90":       "90th Percentile (mm)"  # Represents giant/large-caliber nerve fibers
}

# Directory initialization for safe file exporting
output_dir = "Advanced_Statistical_Reports"
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

# Extract sequential lists for strict plot ordering and master color assignments
group_order = list(color_map.keys())
group_colors = [v[1] for v in color_map.values()]


# =================================================================
# PHASE 1: DATA LOADING AND PRE-PROCESSING
# =================================================================
print("="*70)
print("--- PHASE 1: DATA LOADING AND PRE-PROCESSING ---")
print("="*70)

# Fetch all local CSV files in the working directory
all_csv_files = glob.glob("*.csv")

# Filter files matching valid group keys (safeguard against unrelated CSVs)
files = [f for f in all_csv_files if any(k in f for k in group_map.keys())]

stats_list, raw_list = [], []

for f in files:
    df = pd.read_csv(f, sep=';')
    filename = os.path.basename(f)
    sample_id = filename.split('-')[0]
    
    """
    ANATOMICAL RATIONALE (CB35 POLARITY CORRECTION):
    Aligns physically inverted acquisitions (caudal-rostral) to the standard 
    biological orientation (rostral-caudal starting from the Genu).
    """
    if "CB35" in sample_id:
        df['Z-Coordinate (mm)'] = df['Z-Coordinate (mm)'].max() - df['Z-Coordinate (mm)']
        print(f"[PROCESS] Sample {sample_id}: Z-polarity mathematically inverted for anatomical alignment.")
    
    """
    TECHNICAL RATIONALE (DATA TRIMMING):
    Segmentation ROI smoothing often produces noisy voxel layers at the extreme boundaries. 
    A standard 150 microns offset is trimmed from both caudal and rostral ends.
    """
    z_offset = 0.15 
    z_min, z_max = df['Z-Coordinate (mm)'].min(), df['Z-Coordinate (mm)'].max()
    
    # Filter dataset keeping only the reliable morphological zone
    df = df[(df['Z-Coordinate (mm)'] >= (z_min + z_offset)) & 
            (df['Z-Coordinate (mm)'] <= (z_max - z_offset))]
    
    print(f"[INFO] Sample {sample_id} loaded. Reliable Z-range: {df['Z-Coordinate (mm)'].min():.2f} - {df['Z-Coordinate (mm)'].max():.2f} mm")
    
    # Identify experimental group based on filename keywords
    group = next((v for k, v in group_map.items() if k in filename), "Unknown")
    thickness, z_coord = df["Thickness (mm)"], df["Z-Coordinate (mm)"]
    
    # Calculate and store aggregated statistics per sample
    sample_data = {
        "Sample_ID": sample_id, 
        "Group":     group,
        "Mean":      thickness.mean(), 
        "Median":    thickness.median(), 
        # Mode is binned to 3 microns to avoid continuous data scattering anomalies
        "Mode":      (thickness / 0.003).round().mode()[0] * 0.003, 
        "Std_Dev":   thickness.std(),
        "CV_Pct":    (thickness.std() / thickness.mean()) * 100, 
        "IQR":       thickness.quantile(0.75) - thickness.quantile(0.25),
        "Skew":      thickness.skew(), 
        "Kurt":      thickness.kurtosis(),
        "P10":       thickness.quantile(0.1), # Baseline tissue structure
        "P90":       thickness.quantile(0.9)  # Major structural axons
    }
    stats_list.append(sample_data)
    
    # Archive raw spatial points for Z-mapping and density (violin) plots
    raw_list.append(pd.DataFrame({"Thickness": thickness, "Z_Coord": z_coord, "Sample_ID": sample_id, "Group": group}))

# Compile master dataframes
df_stats = pd.DataFrame(stats_list)
df_raw = pd.concat(raw_list, ignore_index=True)

# Enforce categorical logical ordering to ensure plots render sequentially from CTRL to FUS2
for d in [df_stats, df_raw]: 
    d['Group'] = pd.Categorical(d['Group'], categories=group_order, ordered=True)


# =================================================================
# PHASE 2: SMART PLOTTER DEFINITION & GENERATION
# =================================================================
print("\n" + "="*70)
print("--- PHASE 2: GENERATING METRIC PLOTS ---")
print("="*70)

def format_save_standard(ax, title, ylabel, filename, is_skew=False, rotate=False):
    """
    Handles final plot aesthetics, grid mapping, limits, and high-res file export.
    """
    fmt = '%.2f' if is_skew else '%.3f' 
    
    # Removes default visible titles to ensure a clean export for the manuscript
    ax.set_title('\n') 
    
    # Ensures X and Y axis labels are bold and free of underscores
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xlabel("Samples" if rotate else "Groups", fontweight='bold')
    
    # Generates background grid for improved readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Standardizes ticker layout
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
    
    plt.xticks(rotation=45 if rotate else 30)
    plt.subplots_adjust(left=0.18, right=0.85, top=0.85, bottom=0.25)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def smart_plot_save(df, x, y, hue, palette, title, ylabel, filename, is_skew, p_val=None, 
                    order=None, figsize=None, rotate=False, custom_legend=None):
    """
    Universal Plotting Engine.
    Dynamically manages Group Summaries, Individual Sample breakdowns, and Dual Comparisons 
    with automated statistical bracket generation.
    """
    final_size = figsize if figsize else ((6, 7) if p_val is not None else (10, 6))
    plt.figure(figsize=final_size)
    
    # Main architecture: Barplot (Mean) + Stripplot overlay (Raw dispersion)
    ax = sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette, alpha=0.9, errorbar='sd', legend=False, order=order)
    sns.stripplot(data=df, x=x, y=y, color="black", jitter=False, order=order)
    
    # Inferential Statistics Overlay (Brackets & Asterisks for Dual Comparisons)
    if p_val is not None and order is not None:
        y_vals = df[y].values
        if len(y_vals) > 0:
            y_min, y_max = min(y_vals), max(y_vals)
            delta = (y_max - y_min) if (y_max - y_min) > 0 else 0.1
            
            # Dynamic bracket positioning relative to the maximum bar height
            h_base = y_max + (delta * 0.15)
            h_tip = delta * 0.05
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else f"ns (p={p_val:.3f})"
            
            # Draw the bracket line and significance text
            ax.plot([0, 0, 1, 1], [h_base, h_base + h_tip, h_base + h_tip, h_base], lw=1.5, c='black')
            ax.text(0.5, h_base + h_tip * 1.5, sig, ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Adjust visual limits to accommodate the newly drawn bracket
            ax.set_ylim(y_min - (delta * 0.3), h_base + (delta * 0.8))
            ax.set_xticks([0, 1])
            ax.set_xticklabels(order)

    # Attach custom legends for individual plots to clarify group associations
    if custom_legend:
        plt.legend(handles=custom_legend, title="Groups", loc='upper left', bbox_to_anchor=(1, 1))

    # Trigger standardized export protocol
    format_save_standard(ax, title, ylabel, filename, is_skew, rotate=rotate)

# --- 2.1: Generate Global Metric Charts ---
df_sorted = df_stats.sort_values(["Group", "Sample_ID"])

# Create a master legend explicitly linking Sample A/Sample B to their respective structural shades
legend_elements = [Patch(facecolor=color_map[grp][i], label=f"{grp} (Sample {'A' if i==0 else 'B'})") for grp in group_order for i in (0, 2)]

for key, label in metric_map.items():
    is_skew_kurt = "Skew" in key or "Kurt" in key
    
    # A. Global Group Plot (Mean ± SD per cohort)
    smart_plot_save(df_stats, "Group", key, "Group", group_colors, f"Groups: {label}", 
                    label, os.path.join(output_dir, f"G_{key}.png"), is_skew_kurt)
    
    # B. Detailed Individual Plot (Shows intrinsic n=2 variance)
    palette_indiv = [color_map[g][0 if i % 2 == 0 else 2] for i, g in enumerate(df_sorted["Group"])]
    smart_plot_save(df_sorted, "Sample_ID", key, "Sample_ID", palette_indiv, f"Individual Analysis: {label}", 
                    label, os.path.join(output_dir, f"I_{key}.png"), is_skew_kurt, figsize=(16, 8), rotate=True, 
                    custom_legend=legend_elements)

# --- 2.2: Raw Distribution Topology (Violin Plots) ---
"""
BIOLOGICAL RATIONALE: 
Violin plots reveal the underlying morphological distribution.
This is essential to prove that standard 'Means' are not hiding bimodal cellular populations.
"""
plt.figure(figsize=(18, 8))
sns.violinplot(data=df_raw, x="Sample_ID", y="Thickness", hue="Sample_ID", order=df_sorted["Sample_ID"],
               palette=palette_indiv, inner="box", dodge=False, legend=False)
plt.gca().set_ylim(bottom=0)

# Attach dedicated legend for the Violin Plot matrix
plt.legend(handles=legend_elements, title="Groups", loc='upper left', bbox_to_anchor=(1.02, 1))

format_save_standard(plt.gca(), "Raw Thickness Distribution per Sample", "Thickness (mm)", 
                     os.path.join(output_dir, "I_Raw_Distribution.png"), rotate=True)

print("[SUCCESS] Generated global and individual distribution plots.")


# =================================================================
# PHASE 3: SPATIAL PROFILES (Z-AXIS MAPPING) - SMOOTHED
# =================================================================
print("\n" + "="*70)
print("--- PHASE 3: SPATIAL PROFILES (ENHANCED & SMOOTHED) ---")
print("="*70)

BIN_SIZE = 0.004    # Base acquisition step size (4 microns)
SMOOTH_WINDOW = 12  # ~50 microns Rolling Average window

"""
RATIONALE FOR LOW-PASS FILTER (SMOOTH_WINDOW = 12):
High-res synchrotron data contains microscopic high-frequency noise (segmentation artifacts).
A 12-bin rolling average acts as a digital discrete low-pass filter: it removes pixel-level 
zig-zags (noise) while preserving 100% of the macroscopic topography (anatomical signal), 
such as the Genu slope, preventing architectural distortion.
"""

profile_data_list = []

# Bin and average spatial data to create continuous structural coordinates
for sample_id in df_stats['Sample_ID'].unique():
    raw_sample_df = df_raw[df_raw['Sample_ID'] == sample_id].copy()
    
    # Normalize Z coordinates to start exactly at 0.0 mm (Rostral pole)
    raw_sample_df['Z_Relative'] = raw_sample_df['Z_Coord'] - raw_sample_df['Z_Coord'].min()
    raw_sample_df['Bin'] = (raw_sample_df['Z_Relative'] / BIN_SIZE).round() * BIN_SIZE
    
    # Aggregate thickness per Z-step
    binned_avg = raw_sample_df.groupby('Bin')['Thickness'].mean()
    binned_avg.name = sample_id
    profile_data_list.append(binned_avg)

# Unify all profiles into a continuous topological matrix
df_profiles = pd.concat(profile_data_list, axis=1).sort_index()

def plot_spatial_enhanced(title, filename, target_groups=None, show_individuals=False, show_error=True):
    """
    Renders highly customized spatial profile tracks along the Z-axis.
    """
    fig, ax = plt.subplots(figsize=(18, 9))
    groups_to_plot = target_groups if target_groups else group_order
    spatial_legend = [] 
    
    df_sorted_colors = df_stats.sort_values(["Group", "Sample_ID"]).reset_index(drop=True)
    color_idx_map = {row.Sample_ID: (0 if i % 2 == 0 else 2) for i, row in df_sorted_colors.iterrows()}

    global_max_x = 0 # Tracks the end of the valid data to aggressively trim empty X-axis white space

    for grp in groups_to_plot:
        grp_sample_ids = df_stats[df_stats['Group'] == grp]['Sample_ID'].unique()
        grp_profile_data = df_profiles[grp_sample_ids].dropna(how='all')
        shades = color_map[grp]
        
        # Apply scientific low-pass filter (Rolling Average)
        if SMOOTH_WINDOW > 1:
            grp_profile_data = grp_profile_data.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()

        # Update the maximum valid X coordinate for tight layout adjustments
        current_max_x = grp_profile_data.index.max()
        if pd.notna(current_max_x) and current_max_x > global_max_x:
            global_max_x = current_max_x

        # Render individual sample dashed lines (showing intra-group variance)
        if show_individuals:
            for s_id in grp_sample_ids:
                if s_id in grp_profile_data.columns:
                    line_color = shades[color_idx_map.get(s_id, 0)] 
                    ax.plot(grp_profile_data.index, grp_profile_data[s_id], color=line_color, alpha=0.6, linewidth=1.5, linestyle='--')
                    spatial_legend.append(Patch(facecolor=line_color, label=f"  - {s_id}"))

        # Calculate central tendency and spatial dispersion metrics
        group_mean = grp_profile_data.mean(axis=1)
        group_std = grp_profile_data.std(axis=1)
        
        # Render the principal solid Mean line
        ax.plot(group_mean.index, group_mean.values, color=shades[1], lw=4)
        spatial_legend.append(Line2D([0], [0], color=shades[1], lw=4, label=f"{grp} (MEAN)"))

        # Render the standard deviation error halo (represents spatial biological variance)
        if show_error:
            ax.fill_between(group_mean.index, group_mean - group_std, group_mean + group_std, color=shades[1], alpha=0.15, linewidth=0)

    # Apply core aesthetics and axis titles
    ax.set_title('\n')
    
    # Ensures X and Y axis labels are clear and bold
    ax.set_xlabel("Distance from Genu (mm)", fontweight='bold')
    ax.set_ylabel("Thickness (mm)", fontweight='bold')
    
    ax.set_xlim(0, global_max_x)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
    # Standardize legend title
    ax.legend(handles=spatial_legend, title="Groups", loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Restore standard ticks and generate background grid
    plt.xticks(rotation=90) 
    plt.yticks()            
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.subplots_adjust(left=0.10, right=0.75, top=0.95, bottom=0.18)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight') 
    plt.close()

# --- Execution: Render all Spatial Plot Combinations ---

# A. Macro Chart (All 5 groups overlay for total perspective)
plot_spatial_enhanced(title="Z-Profile: All Groups Summary (± SD)", filename="10_Z_All_Groups_Summary.png", 
                      target_groups=group_order, show_individuals=False, show_error=True)

# B. Micro Charts (Detailed individual tracks per group to spot outliers)
for grp in group_order:
    plot_spatial_enhanced(title=f"Z-Profile: {grp} (Individual Samples & Mean)", filename=f"11_Z_Individual_{grp}.png", 
                          target_groups=[grp], show_individuals=True, show_error=True)

# C. Narrative Trajectories: Damage, Early & Late Dynamics
plot_spatial_enhanced(title="Z-Profile: Evolution of Damage", filename="12_Z_Damage_Trajectory.png", 
                      target_groups=["CTRL", "CUP1", "CUP2"], show_individuals=False, show_error=True)

plot_spatial_enhanced(title="Z-Profile: Early Demyelination & Treatment", filename="13_Z_Early_Dynamics.png", 
                      target_groups=["CTRL", "CUP1", "FUS1"], show_individuals=False, show_error=True)

plot_spatial_enhanced(title="Z-Profile: Peak Damage & Full Recovery", filename="14_Z_Late_Dynamics.png", 
                      target_groups=["CTRL", "CUP2", "FUS2"], show_individuals=False, show_error=True)

print("[SUCCESS] Phase 3 completed. Generated general, individual, and triad spatial profiles.")

# =================================================================
# PHASE 4: SCIENTIFIC REPORTING EXPORT
# =================================================================

print("\n" + "="*70)
print("--- PHASE 4: EXPORTING SCIENTIFIC REPORTS ---")
print("="*70)

target_metrics = ["Mean", "Median", "Mode", "Std_Dev", "CV_Pct", "IQR", "Skew", "Kurt", "P10", "P90"]

# Output A: Raw sample data dictionary (For reproducibility audits)
df_master_samples = df_stats.sort_values(["Group", "Sample_ID"])
df_master_samples.to_csv(os.path.join(output_dir, "Master_Sample_Detail.csv"), index=False, sep=';', decimal=',')

# Output B: Aggregated Thesis Summary (Standard 'Mean ± SD' format ready for Word copy-paste)
group_summary = df_stats.groupby('Group', observed=True)[target_metrics].agg(['mean', 'std', 'count']).reset_index()
group_summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in group_summary.columns.values]

thesis_table = pd.DataFrame({'Group': group_summary['Group'], 'N_Subjects': group_summary['Mean_count']})
for m in target_metrics:
    thesis_table[m] = group_summary[f"{m}_mean"].map('{:.3f}'.format) + " ± " + group_summary[f"{m}_std"].map('{:.3f}'.format)

thesis_table.to_csv(os.path.join(output_dir, "Thesis_Summary_Table.csv"), index=False, sep=';', decimal=',')

# Compile and export comprehensive Excel Workbook (Crucial for PhD panel inquiries)
with pd.ExcelWriter(os.path.join(output_dir, "Complete_Results_Database.xlsx")) as writer:
    thesis_table.to_excel(writer, sheet_name='Thesis_Group_Summary', index=False)
    df_master_samples.to_excel(writer, sheet_name='Individual_Sample_Detail', index=False)
    group_summary.to_excel(writer, sheet_name='Technical_Summary_Data', index=False)

print("[SUCCESS] Exported master tables and Excel databanks.")


# =================================================================
# PHASE 5: MASTER STATISTICAL WORKFLOW (10 PAIRS MATRIX)
# =================================================================

print("\n" + "="*70)
print("--- PHASE 5: MASTER STATISTICAL WORKFLOW ---")
print("="*70)

final_stats_results = []
metrics_to_test = list(metric_map.keys())

"""
Absolute comparison matrix guaranteeing valid lookup brackets in Phase 7.
Tests every possible biologically relevant interaction across the experimental protocol.
"""
all_comparison_pairs = [
    (group_map["CTRL"], group_map["CUP1"]), # 1. Baseline vs Initial Damage
    (group_map["CTRL"], group_map["CUP2"]), # 2. Baseline vs Max Damage
    (group_map["CTRL"], group_map["FUS1"]), # 3. Baseline vs Treatment 1wk
    (group_map["CTRL"], group_map["FUS2"]), # 4. Baseline vs Treatment 3wk
    (group_map["CUP1"], group_map["CUP2"]), # 5. Damage Progression
    (group_map["CUP1"], group_map["FUS1"]), # 6. Initial Damage vs Initial Treatment
    (group_map["CUP1"], group_map["FUS2"]), # 7. Initial Damage vs Final Treatment
    (group_map["CUP2"], group_map["FUS1"]), # 8. Max Damage vs Initial Treatment Recovery
    (group_map["CUP2"], group_map["FUS2"]), # 9. Max Damage vs Final Treatment (Golden Comparison)
    (group_map["FUS1"], group_map["FUS2"])  # 10. Recovery Evolution
]

for metric_key in metrics_to_test:
    
    # A. Model Diagnostics
    # Shapiro-Wilk: Tests if residuals follow a Gaussian normal distribution.
    group_averages = df_stats.groupby('Group', observed=True)[metric_key].transform('mean')
    _, p_shapiro = stats.shapiro(df_stats[metric_key] - group_averages)
    
    # Levene: Tests for homoscedasticity (equal variances across groups).
    group_data_arrays = [df_stats[df_stats['Group'] == g][metric_key].values for g in group_order]
    _, p_levene = stats.levene(*group_data_arrays)
    _, p_anova = stats.f_oneway(*group_data_arrays)
    
    # B. Conservative Post-Hoc Dunnett 
    # Strict validation mapping all treated conditions exclusively against the healthy baseline.
    ctrl_label = group_map["CTRL"]
    ctrl_values = df_stats[df_stats['Group'] == ctrl_label][metric_key].values
    treated_names = [g for g in group_order if g != ctrl_label]
    treated_values_list = [df_stats[df_stats['Group'] == g][metric_key].values for g in treated_names]
    
    dunnett_res = stats.dunnett(*treated_values_list, control=ctrl_values)
    dunnett_p_map = dict(zip(treated_names, dunnett_res.pvalue))

    # C. Granular Pairwise Analysis (Welch T-Test & Cohen's d Effect Size)
    for g1, g2 in all_comparison_pairs:
        vals1 = df_stats[df_stats['Group'] == g1][metric_key].values
        vals2 = df_stats[df_stats['Group'] == g2][metric_key].values
        
        """
        WELCH'S T-TEST RATIONALE:
        Standard Student's T-Test assumes equal variances. Since cuprizone induces 
        highly variable morphological destruction, Welch's test (equal_var=False) is mandatory.
        """
        _, p_welch = stats.ttest_ind(vals1, vals2, equal_var=False)
        cohen_d_val = calculate_cohen_d(vals2, vals1) 
        
        # Link Dunnett score only if CTRL is involved in the targeted pair
        p_dunnett = dunnett_p_map.get(g2, np.nan) if g1 == ctrl_label else np.nan
        
        # Simple macro percentage shift for rapid textual thesis reference
        mean1, mean2 = np.mean(vals1), np.mean(vals2)
        variation_pct = ((mean2 - mean1) / mean1) * 100 if mean1 != 0 else 0

        # Append to master statistical ledger
        final_stats_results.append({
            "Metric": metric_key, "Comparison": f"{g1} vs {g2}",
            "P_Shapiro_Res": p_shapiro, "P_Levene": p_levene, "P_ANOVA": p_anova,
            "P_Welch": p_welch, "P_Dunnett": p_dunnett, "Cohen_d": cohen_d_val, "Var_Pct": variation_pct
        })


# =================================================================
# PHASE 6: FINAL EXPORT AND TERMINAL REPORTING
# =================================================================
print("\n" + "="*70)
print("--- PHASE 6: EXPORTING MASTER STATISTICAL REPORT ---")
print("="*70)

df_final_report = pd.DataFrame(final_stats_results)
df_final_report.to_excel(os.path.join(output_dir, "Master_Statistical_Analysis.xlsx"), index=False)
print(f"[SUCCESS] Master statistical matrices evaluated and saved. Total pairwise records: {len(final_stats_results)}")


# =================================================================
# PHASE 7: STRATEGIC THESIS PLOTS (Definitive Brackets & Aesthetics)
# =================================================================
print("\n" + "="*70)
print("--- PHASE 7: GENERATING DEFINITIVE THESIS PLOTS ---")
print("="*70)

"""
Strategic configuration matrix mapping. 
Filenames are sequentially numbered (20 to 30) to enforce strict chronological sorting 
in the output folder, aligning perfectly with the thesis Chapter 4 narrative flow.
"""
thesis_plot_configs = [
    # ---------------------------------------------------------
    # 1. Baseline Integrity (Healthy control validation)
    # ---------------------------------------------------------
    {'metric': 'Mean', 'groups': ['CTRL'], 'comparisons': [], 
     'title': 'Baseline Validation (Internal Coherence)', 'filename': '20A_Validation_CTRL_Mean.png'},
    {'metric': 'P10', 'groups': ['CTRL'], 'comparisons': [], 
     'title': 'Baseline Validation: Small Fibers (P10)', 'filename': '20B_Validation_CTRL_P10.png'},
    {'metric': 'P90', 'groups': ['CTRL'], 'comparisons': [], 
     'title': 'Baseline Validation: Large Fibers (P90)', 'filename': '20C_Validation_CTRL_P90.png'},
    {'metric': 'CV_Pct', 'groups': ['CTRL'], 'comparisons': [], 
     'title': 'Baseline Validation: Precision (CV%)', 'filename': '20D_Validation_CTRL_CV.png'},
    
    # ---------------------------------------------------------
    # 2. Degeneration Narrative (Damage dynamics)
    # ---------------------------------------------------------
    {'metric': 'Mean', 'groups': ['CTRL', 'CUP1', 'CUP2'], 
     'comparisons': [('CTRL', 'CUP1'), ('CUP1', 'CUP2'), ('CTRL', 'CUP2')], 
     'title': 'Evolution of Damage (Mean)', 'filename': '21_Damage_Mean.png'},
    {'metric': 'P10', 'groups': ['CTRL', 'CUP1', 'CUP2'], 
     'comparisons': [('CTRL', 'CUP1'), ('CUP1', 'CUP2'), ('CTRL', 'CUP2')], 
     'title': 'Evolution of Damage: Small Fibers (P10)', 'filename': '22_Damage_P10.png'},
    {'metric': 'P90', 'groups': ['CTRL', 'CUP1', 'CUP2'], 
     'comparisons': [('CTRL', 'CUP1'), ('CUP1', 'CUP2'), ('CTRL', 'CUP2')], 
     'title': 'Evolution of Damage: Large Fibers (P90)', 'filename': '23_Damage_P90.png'},
    {'metric': 'CV_Pct', 'groups': ['CTRL', 'CUP1', 'CUP2'], 'comparisons': [], 
     'title': 'Evolution of Damage: Precision (CV%)', 'filename': '24_Damage_CV.png'},

    # ---------------------------------------------------------
    # 3. Early Dynamics: Baseline vs Onset vs Early Treatment
    # ---------------------------------------------------------
    {'metric': 'Mean', 'groups': ['CTRL', 'CUP1', 'FUS1'], 
     'comparisons': [('CTRL', 'CUP1'), ('CUP1', 'FUS1'), ('CTRL', 'FUS1')], 
     'title': 'Early Dynamics (Mean)', 'filename': '25_Early_Mean.png'},
    {'metric': 'P10', 'groups': ['CTRL', 'CUP1', 'FUS1'], 
     'comparisons': [('CTRL', 'CUP1'), ('CUP1', 'FUS1'), ('CTRL', 'FUS1')], 
     'title': 'Early Dynamics: Small Fibers (P10)', 'filename': '26_Early_P10.png'},
    {'metric': 'P90', 'groups': ['CTRL', 'CUP1', 'FUS1'], 
     'comparisons': [('CTRL', 'CUP1'), ('CUP1', 'FUS1'), ('CTRL', 'FUS1')], 
     'title': 'Early Dynamics: Large Fibers (P90)', 'filename': '27_Early_P90.png'},
    {'metric': 'CV_Pct', 'groups': ['CTRL', 'CUP1', 'FUS1'], 'comparisons': [], 
     'title': 'Early Dynamics: Precision (CV%)', 'filename': '28_Early_CV.png'},
    
    # ---------------------------------------------------------
    # 4. Late Dynamics: Baseline vs Peak Damage vs Full Recovery
    # ---------------------------------------------------------
    {'metric': 'Mean', 'groups': ['CTRL', 'CUP2', 'FUS2'], 
     'comparisons': [('CTRL', 'CUP2'), ('CUP2', 'FUS2'), ('CTRL', 'FUS2')], 
     'title': 'Late Dynamics (Mean)', 'filename': '29_Late_Mean.png'},
    {'metric': 'P10', 'groups': ['CTRL', 'CUP2', 'FUS2'], 
     'comparisons': [('CTRL', 'CUP2'), ('CUP2', 'FUS2'), ('CTRL', 'FUS2')], 
     'title': 'Late Dynamics: Small Fibers (P10)', 'filename': '30_Late_P10.png'},
    {'metric': 'P90', 'groups': ['CTRL', 'CUP2', 'FUS2'], 
     'comparisons': [('CTRL', 'CUP2'), ('CUP2', 'FUS2'), ('CTRL', 'FUS2')], 
     'title': 'Late Dynamics: Large Fibers (P90)', 'filename': '31_Late_P90.png'},
    {'metric': 'CV_Pct', 'groups': ['CTRL', 'CUP2', 'FUS2'], 'comparisons': [], 
     'title': 'Late Dynamics: Precision (CV%)', 'filename': '32_Late_CV.png'},
]

def generate_strategic_thesis_plots(data_df, raw_df, stats_report_df, metric, short_groups, comparisons, title, filename):
    """
    Engine rendering specifically formatted, thesis-ready plots encompassing effect sizes 
    and automated layout scaling.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    full_groups = [group_map[g] for g in short_groups]
    
    # ---------------------------------------------------------
    # SPECIAL CASE: VALIDATION PLOT (INDIVIDUAL COMPARISON)
    # ---------------------------------------------------------
    if len(short_groups) == 1:
        grp = full_groups[0]
        
        if metric == 'Mean':
            # Base protocol: uses raw data points to calculate p-value and error bars
            plot_data = raw_df[raw_df['Group'] == grp].copy()
            samples = sorted(plot_data['Sample_ID'].unique())
            palette_indiv = {samples[0]: color_map[grp][0], samples[1]: color_map[grp][2]}
            
            sns.barplot(x='Sample_ID', y='Thickness', data=plot_data, palette=palette_indiv, hue='Sample_ID',
                        width=0.4, errorbar='sd', capsize=0.05, err_kws={'linewidth': 1.5}, edgecolor='black', alpha=1.0, ax=ax)
            sns.stripplot(x='Sample_ID', y='Thickness', data=plot_data.groupby('Sample_ID').sample(n=500, random_state=42),
                          color='black', alpha=0.3, size=3, jitter=0.15, legend=False, ax=ax)

            v1 = plot_data[plot_data['Sample_ID'] == samples[0]]['Thickness'].values
            v2 = plot_data[plot_data['Sample_ID'] == samples[1]]['Thickness'].values
            _, p_val = stats.ttest_ind(v1, v2, equal_var=False)
            c_d = calculate_cohen_d(v2, v1)
            
            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            y_max_err = plot_data.groupby('Sample_ID')['Thickness'].mean().max() + plot_data.groupby('Sample_ID')['Thickness'].std().max()
            curr_y = y_max_err + (y_max_err * 0.05)
            
            ax.plot([0, 0, 1, 1], [curr_y, curr_y + (y_max_err*0.02), curr_y + (y_max_err*0.02), curr_y], lw=1.5, c='black')
            ax.text(0.5, curr_y + (y_max_err*0.02), f"{stars}\np={p_val:.3f}\n(|d|={abs(c_d):.2f})", 
                     ha='center', va='bottom', fontsize=16, fontweight='bold')
            ax.set_ylim(0, curr_y * 1.3)
            
        else:
            # NEW BLOCK: Targets P10, P90, CV% (omits error bars, extracts fixed scalar values)
            plot_data = data_df[data_df['Group'] == grp].copy()
            samples = sorted(plot_data['Sample_ID'].unique())
            palette_indiv = {samples[0]: color_map[grp][0], samples[1]: color_map[grp][2]}
            
            sns.barplot(x='Sample_ID', y=metric, data=plot_data, palette=palette_indiv, hue='Sample_ID',
                        width=0.4, edgecolor='black', alpha=1.0, ax=ax)
            
            curr_y = plot_data[metric].max()
            ax.set_ylim(0, curr_y * 1.25)
            
            # Auto-inserts percentage formatting if the targeted metric is the CV% chart
            if metric == 'CV_Pct':
                for p in ax.patches:
                    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold', fontsize=16)

    # ---------------------------------------------------------
    # STANDARD CASE: MULTI-GROUP TRIADS
    # ---------------------------------------------------------
    else:
        # Pre-process targeted subsets enforcing category constraints
        plot_data = data_df[data_df['Group'].isin(full_groups)].copy()
        plot_data['Group'] = pd.Categorical(plot_data['Group'], categories=full_groups, ordered=True)
        plot_data = plot_data.sort_values('Group')
        
        # Establish accurate structural shade allocations
        group_palette = {grp: color_map[grp][1] for grp in full_groups}
        sample_shade_map = {s_id: color_map[grp][0 if i % 2 == 0 else 2] for grp in full_groups 
                           for i, s_id in enumerate(sorted(plot_data[plot_data['Group'] == grp]['Sample_ID'].unique()))}

        # Sub-branch for CV Percentages (requires unique text annotation style instead of error bars)
        if metric == 'CV_Pct':
            sns.barplot(x='Group', y=metric, data=plot_data, hue='Group', palette=group_palette, width=0.4, legend=False, edgecolor='black', ax=ax)
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold')
            curr_y = plot_data[metric].max()
            
        # Primary rendering for conventional thickness metrics
        else:
            sns.barplot(x='Group', y=metric, data=plot_data, hue='Group', palette=group_palette, width=0.4, 
                        errorbar='sd', capsize=0.05, err_kws={'linewidth': 1.5}, legend=False, edgecolor='black', alpha=1.0, ax=ax)
            # Overlay points identifying individual sample contributions
            sns.stripplot(x='Group', y=metric, data=plot_data, hue='Sample_ID', palette=sample_shade_map, 
                          size=9, jitter=0.1, edgecolor='black', linewidth=1, legend=False, ax=ax)

            # Safe ceiling calculation for bracket stacking geometry
            y_max_err = plot_data.groupby('Group', observed=True)[metric].mean().max() + plot_data.groupby('Group', observed=True)[metric].std().max()
            if pd.isna(y_max_err): y_max_err = plot_data[metric].max()
            
            y_step = y_max_err * 0.09
            curr_y = y_max_err + (y_max_err * 0.05)
            
            # Layering multiple inferential brackets cleanly
            if comparisons:
                # Sort spatial width to push wider comparisons to the top level, avoiding overlap intersections
                sorted_comps = sorted(comparisons, key=lambda x: abs(short_groups.index(x[0]) - short_groups.index(x[1])))
                
                for g1_s, g2_s in sorted_comps:
                    g1_f, g2_f = group_map[g1_s], group_map[g2_s]
                    
                    # Fetch corresponding scores from Phase 5 matrix to guarantee data consistency
                    p_row = stats_report_df[(stats_report_df['Metric'] == metric) & 
                            ((stats_report_df['Comparison'] == f"{g1_f} vs {g2_f}") | 
                             (stats_report_df['Comparison'] == f"{g2_f} vs {g1_f}"))]
                    
                    if not p_row.empty:
                        p_val = p_row['P_Welch'].values[0]
                        c_d = p_row['Cohen_d'].values[0]
                        stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
                        
                        try:
                            # Plot coordinates logic connecting the two centers of the targeted bars
                            x1, x2 = full_groups.index(g1_f), full_groups.index(g2_f)
                            ax.plot([x1, x1, x2, x2], [curr_y, curr_y + y_step/2, curr_y + y_step/2, curr_y], lw=1.5, c='black')
                            ax.text((x1+x2)/2, curr_y + y_step/2, f"{stars}\np={p_val:.3f}\n(|d|={abs(c_d):.1f})", 
                                     ha='center', va='bottom', fontweight='bold', fontsize=14)
                            curr_y += y_step * 2.5 # Assign vertical buffer for the successive bracket iteration
                        except ValueError: 
                            continue # Safeguard execution if a requested group config mismatches available groups

        # Final layout execution
        ax.set_xticks(range(len(short_groups)))
        ax.set_xticklabels(short_groups)
        ax.set_ylim(0, curr_y * 1.15)

    # =========================================================
    # STRICT DECIMAL FORMATTING & BACKGROUND GRID ALIGNMENT
    # =========================================================
    
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15)) 
    y_fmt = '%.2f' if metric == 'CV_Pct' else '%.3f'
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))
    
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    ax.set_title('\n')
    
    # Y-axis: Dictionary mapping for clean and inherently bold nomenclature
    ax.set_ylabel(metric_map.get(metric, f"{metric}"), fontweight='bold')
    
    # X-axis: Dynamic text assignment filtering out programming underscores
    ax.set_xlabel("Samples" if len(short_groups) == 1 else "Groups", fontweight='bold')
    
    plt.subplots_adjust(left=0.16, right=0.95, top=0.92, bottom=0.14)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  > [INFO] Exported plot: {filename}")

# Command triggering the batch rendering strictly defined in the thesis logic matrix
for config in thesis_plot_configs:
    generate_strategic_thesis_plots(df_stats, df_raw, df_final_report, config['metric'], 
                                    config['groups'], config['comparisons'], config['title'], config['filename'])

print("\n" + "="*70)
print("[SUCCESS] SCRIPT EXECUTION COMPLETE: Pipeline and Reporting Terminated Successfully.")
print("="*70)

# =================================================================
# PHASE 8: EXPERIMENTAL PROTOCOL TIMELINE (GANTT CHART)
# =================================================================

print("\n" + "="*70)
print("--- PHASE 8: GENERATING PROTOCOL GANTT CHART ---")
print("="*70)

def generate_protocol_gantt(filename="00_Protocol_Timeline.png"):
    fig, ax = plt.subplots(figsize=(15, 6))
    
    gantt_order = ["CTRL", "CUP1", "FUS1", "CUP2", "FUS2"]
    y_pos = np.arange(len(gantt_order))
    
    for i, grp in enumerate(gantt_order):
        base_color = color_map[grp][1]
        is_acute = "1" in grp
        is_fus = "FUS" in grp
        
        # Temporal offset: Acute protocol groups commence on day 7, Chronic groups on day 0
        start_day = 7 if is_acute else 0
        
        """
        IN VIVO DURATION LOGIC: 
        Acute cohort: 21 days cuprizone + 5 days FUS = 26 days total. (Sacrifice on absolute day 33).
        Chronic cohort: 21 days cuprizone + 19 days FUS (5-2, 5-2, 5) = 40 days total. (Sacrifice on absolute day 40).
        """
        duration = 26 if is_acute else 40 
        sac_day = start_day + duration
        
        # A. Main Protocol Bar (In Vivo timeframe)
        ax.barh(i, duration, left=start_day, color=base_color, alpha=0.8, edgecolor='black', height=0.5, zorder=3)
            
        # B. tFUS Administration Overlay
        if is_fus:
            if is_acute: 
                # FUS1: 5 consecutive days starting from its 21st day (absolute day 28)
                ax.barh(i, 5, left=28, color='none', edgecolor='black', hatch='////', height=0.5, linewidth=1.5, zorder=4)
            else:        
                # FUS2: 3 alternating cycles (5 on, 2 off). The final cycle concludes on absolute day 40.
                ax.barh(i, 5, left=21, color='none', edgecolor='black', hatch='////', height=0.5, linewidth=1.5, zorder=4)
                ax.barh(i, 5, left=28, color='none', edgecolor='black', hatch='////', height=0.5, linewidth=1.5, zorder=4)
                ax.barh(i, 5, left=35, color='none', edgecolor='black', hatch='////', height=0.5, linewidth=1.5, zorder=4)
                
        # C. Sacrifice & Perfusion Marker
        ax.plot(sac_day, i, marker='X', color='black', markersize=14, markeredgecolor='white', zorder=5)

        # D. Post-Sacrifice Ex Vivo Processing (Identical sequence for all subjects)
        fix_days = 2
        deh_days = 1
        emb_days = 1
        
        # Fixation phase (48h)
        ax.barh(i, fix_days, left=sac_day, color='#D5D8DC', edgecolor='black', hatch='...', height=0.3, zorder=3)
        # Dehydration and tissue clearing phase
        ax.barh(i, deh_days, left=sac_day + fix_days, color='#ABB2B9', edgecolor='black', hatch='xxx', height=0.3, zorder=3)
        # Paraffin embedding phase
        ax.barh(i, emb_days, left=sac_day + fix_days + deh_days, color='#566573', edgecolor='black', hatch='\\\\\\', height=0.3, zorder=3)

    # Axis Formatting and Inversion
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gantt_order, fontweight='bold')
    ax.invert_yaxis() 
    
    ax.set_title('\n') 
    ax.set_xlabel('Timeline (Days)', fontweight='bold')
    
    # X-Axis limits: From 0 to 44 (Chronic timeline extends to day 44: 40 in vivo + 2 + 1 + 1 ex vivo)
    max_day = 44
    ax.set_xticks(np.arange(0, max_day + 1, 1))
    ax.set_xticklabels([str(x) for x in np.arange(0, max_day + 1, 1)], fontsize=9, rotation=90)
    
    # Standardized English Legend with Bi-Color Grouping Patches
    p_acute_cup = Patch(facecolor=color_map["CUP1"][1], edgecolor='black')
    p_acute_fus = Patch(facecolor=color_map["FUS1"][1], edgecolor='black')
    
    p_chronic_cup = Patch(facecolor=color_map["CUP2"][1], edgecolor='black')
    p_chronic_fus = Patch(facecolor=color_map["FUS2"][1], edgecolor='black')

    legend_handles = [
        Patch(facecolor=color_map["CTRL"][1], edgecolor='black'),
        (p_acute_cup, p_acute_fus),
        (p_chronic_cup, p_chronic_fus),
        Patch(facecolor='none', edgecolor='black', hatch='////'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=12, linestyle='None'),
        Patch(facecolor='#D5D8DC', edgecolor='black', hatch='...'),
        Patch(facecolor='#ABB2B9', edgecolor='black', hatch='xxx'),
        Patch(facecolor='#566573', edgecolor='black', hatch='\\\\\\')
    ]
    
    legend_labels = [
        'Healthy Baseline (CTRL)',
        'Acute Protocol (CUP1 / FUS1)',
        'Chronic Protocol (CUP2 / FUS2)',
        'tFUS Administration',
        'Sacrifice & Perfusion',
        'Sample Fixation (48h)',
        'Dehydration & Clearing',
        'Paraffin Embedding'
    ]
    
    ax.legend(handles=legend_handles, labels=legend_labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, 
              loc='upper left', bbox_to_anchor=(1.01, 1), framealpha=1, title="Experimental Flow", title_fontsize='13')
    
    ax.grid(axis='x', linestyle=':', alpha=0.6, color='gray')
    ax.set_axisbelow(True)
    
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  > [INFO] Exported Gantt Chart to: {out_path}")

# Trigger module execution
generate_protocol_gantt()