import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle


# ============================================================
# EST-style Figure 5 v8
# Comprehensive all-domain exploration and generalization boundary
# ============================================================


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'axes.linewidth': 0.8,
        'axes.labelsize': 10.0,
        'axes.titlesize': 11.0,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.2,
        'figure.dpi': 320,
        'savefig.dpi': 600,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here, *here.parents, Path.cwd(), *Path.cwd().parents]
    for c in candidates:
        if (c / 'src').exists() and (c / 'outputs').exists():
            return c
        if c.name == 'fluoride_coagulation_ml_project' and (c / 'outputs').exists():
            return c
    raise FileNotFoundError('Could not locate project root containing src/ and outputs/.')


def ensure_dirs(project_root: Path):
    fig_dir = project_root / 'outputs' / 'jece_final_package' / 'figures'
    table_dir = project_root / 'outputs' / 'jece_final_package' / 'tables'
    report_dir = project_root / 'outputs' / 'jece_final_package' / 'reports'
    log_dir = project_root / 'outputs' / 'jece_final_package' / 'logs'
    for d in (fig_dir, table_dir, report_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir, report_dir, log_dir


def read_csv(root: Path, relative_path: str) -> pd.DataFrame:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    return pd.read_csv(path)


def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12.5, fontweight='bold', ha='left', va='bottom')


def short_strategy(name: str) -> str:
    mapping = {
        'GroupKFold_FullyGrouped': 'GKF',
        'LOLO_LeaveOneLiteratureOut': 'LOLO',
        'RepeatedOrdinary_GroupedSplit': 'ROS',
        'Balanced_5Seed_Current31F': 'BAL-5',
        'RepeatedBalanced_Extended': 'RBE',
    }
    return mapping.get(name, name)


def short_method(name: str) -> str:
    mapping = {
        'CatBoost': 'CAT',
        'XGBoost': 'XGB',
        'CB-XGB': 'CB-XGB',
        'KRR_Poly': 'KRR-poly',
        'KRR_CatBoost_Weighted_2Model': 'KRR-CAT',
        'KRR_XGBoost_Weighted_2Model': 'KRR-XGB',
        'GPR_Matern': 'GPR',
        'PLS': 'PLS',
    }
    return mapping.get(name, name)


def exploration_module(ax, x, y, w, h, number, title, items, face, edge):
    """Restrained journal-style module: square body, thin border, compact header."""
    body = Rectangle(
        (x, y), w, h,
        facecolor='white', edgecolor=edge, linewidth=0.9,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(body)
    header_h = 0.31 * h
    header = Rectangle(
        (x, y + h - header_h), w, header_h,
        facecolor=face, edgecolor=edge, linewidth=0.9,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(header)
    ax.text(x + 0.018, y + h - header_h / 2, str(number),
            transform=ax.transAxes, ha='left', va='center',
            fontsize=11.8, fontweight='bold', color='#40505B')
    ax.text(x + 0.052, y + h - header_h / 2, title,
            transform=ax.transAxes, ha='left', va='center',
            fontsize=12.4, fontweight='bold', color='#24313A')
    ax.text(x + 0.020, y + h - header_h - 0.040, items,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10.4, color='#4D5963', linespacing=1.16)


def build_figure(project_root: Path):
    # -------------------- source tables --------------------
    summary = read_csv(project_root, 'outputs/tables/31n_grouped_validation_summary.csv')
    corr = read_csv(project_root, 'outputs/tables/31n_grouped_validation_residual_correlation_by_strategy.csv')
    shift = read_csv(project_root, 'outputs/tables/30_seed_level_numeric_distribution_shift.csv')
    catcov = read_csv(project_root, 'outputs/tables/30_seed_level_categorical_coverage.csv')
    paper_metrics = read_csv(project_root, 'outputs/tables/31n_grouped_validation_paper_metrics.csv')
    predictions = read_csv(project_root, 'outputs/tables/31n_grouped_validation_predictions.csv')
    hierarchy = read_csv(project_root, 'outputs/tables/32v3_metric_hierarchy_and_claim_boundaries.csv')

    # -------------------- palette --------------------
    ink = '#25313A'
    muted = '#63707B'
    grid = '#D9E0E6'
    blue = '#4E7DA7'
    teal = '#3F8D9B'
    gold = '#D5A241'
    sage = '#9FB6A0'
    pale_blue = '#E6EFF5'
    pale_teal = '#E1F0F1'
    pale_gold = '#F5E9CC'
    pale_sage = '#E7EEE7'
    light_gray = '#F5F7F8'
    missing_gray = '#EEF1F3'

    corr_cmap = LinearSegmentedColormap.from_list(
        'corr_est', ['#2F9DB7', '#ECF4F3', '#FFD66C']
    )
    r2_cmap = LinearSegmentedColormap.from_list(
        'r2_est', ['#56A4B7', '#F1F5F5', '#F3D48B']
    )
    smd_cmap = LinearSegmentedColormap.from_list(
        'smd_est', ['#F8FAFB', '#BFD0E0', '#466E98']
    )
    unseen_cmap = LinearSegmentedColormap.from_list(
        'unseen_est', ['#FBFAF4', '#E7D8A6', '#C99633']
    )

    # -------------------- layout --------------------
    fig = plt.figure(figsize=(13.3, 9.2), facecolor='white')
    outer = gridspec.GridSpec(
        3, 3, figure=fig,
        height_ratios=[0.98, 3.50, 1.92],
        width_ratios=[1.28, 1.28, 0.92],
        wspace=0.46, hspace=0.43,
        left=0.060, right=0.985, top=0.975, bottom=0.078,
    )

    # ========================================================
    # a. scope of all-domain grouped exploration
    # ========================================================
    ax_a = fig.add_subplot(outer[0, :])
    ax_a.axis('off')
    add_panel_label(ax_a, 'a', x=-0.02, y=1.00)
    module_y = 0.075
    module_h = 0.845
    widths = [0.225, 0.225, 0.225, 0.245]
    xs = [0.01, 0.255, 0.500, 0.745]

    exploration_module(
        ax_a, xs[0], module_y, widths[0], module_h, 1,
        'Validation designs',
        'GKF | LOLO | ROS\nBAL-5 | RBE\n5-27 source-isolated splits',
        pale_blue, '#9CB5C7'
    )
    exploration_module(
        ax_a, xs[1], module_y, widths[1], module_h, 2,
        'Model families',
        'CAT | XGB | KRR\nGPR | PLS | hybrid ensembles\nSingle-model and fusion screening',
        pale_teal, '#92B9BD'
    )
    exploration_module(
        ax_a, xs[2], module_y, widths[2], module_h, 3,
        'Model refinement',
        'Mechanism features | tuning\nWeighting | ablation | stacking\nResidual learning',
        pale_gold, '#D3B67A'
    )
    exploration_module(
        ax_a, xs[3], module_y, widths[3], module_h, 4,
        'Robustness audits',
        'Numeric shift | unseen categories\nPaper error | residual correlation\nSource support | claim limits',
        pale_sage, '#A9BBA9'
    )

    for x1, x2 in [(0.237, 0.252), (0.482, 0.497), (0.727, 0.742)]:
        ax_a.annotate('', xy=(x2, 0.50), xytext=(x1, 0.50),
                      xycoords=ax_a.transAxes, textcoords=ax_a.transAxes,
                      arrowprops=dict(arrowstyle='-|>', lw=1.0, color='#8F9AA3'))

    # ========================================================
    # b. residual correlation matrix in reference style
    # ========================================================
    ax_b = fig.add_subplot(outer[1, :2])
    add_panel_label(ax_b, 'b')

    methods = [
        'CatBoost', 'XGBoost', 'CB-XGB', 'KRR_Poly',
        'KRR_CatBoost_Weighted_2Model', 'KRR_XGBoost_Weighted_2Model',
        'GPR_Matern', 'PLS'
    ]
    corr_use = corr[
        (corr['Strategy'] == 'RepeatedBalanced_Extended') &
        (corr['Method'].isin(methods))
    ].copy().set_index('Method')
    corr_mat = corr_use.reindex(index=methods, columns=methods).astype(float)

    n = len(methods)
    ax_b.set_xlim(-0.5, n - 0.5)
    ax_b.set_ylim(n - 0.5, -0.5)
    ax_b.set_xticks(np.arange(n))
    ax_b.set_yticks(np.arange(n))
    ax_b.set_xticklabels([short_method(m) for m in methods], rotation=34, ha='right', rotation_mode='anchor')
    ax_b.set_yticklabels([short_method(m) for m in methods])
    ax_b.set_title('Residual correlation among grouped models', pad=11)

    norm_corr = TwoSlopeNorm(vmin=0.80, vcenter=0.90, vmax=1.00)
    for i in range(n):
        for j in range(n):
            value = corr_mat.iloc[i, j]
            if not np.isfinite(value):
                continue
            if i >= j:
                face = corr_cmap(norm_corr(value))
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor=face, edgecolor='white', linewidth=0.7)
                ax_b.add_patch(rect)
                txt_color = 'white' if value < -0.45 else ink
                ax_b.text(j, i + 0.08, f'{value:.2f}', ha='center', va='center',
                          fontsize=8.0, color=txt_color,
                          fontweight=('bold' if i == j else 'normal'))
            else:
                scaled = np.clip((value - 0.80) / 0.20, 0, 1)
                size = 45 + 520 * scaled ** 1.8
                color = corr_cmap(norm_corr(value))
                ax_b.scatter(j, i, s=size, marker='h', c=[color],
                             edgecolors='white', linewidths=0.8, zorder=3)

    # light grid and border
    for k in range(n + 1):
        ax_b.plot([-0.5, n - 0.5], [k - 0.5, k - 0.5], color='white', lw=0.55, zorder=4)
        ax_b.plot([k - 0.5, k - 0.5], [-0.5, n - 0.5], color='white', lw=0.55, zorder=4)
    ax_b.spines[:].set_visible(False)

    sm = mpl.cm.ScalarMappable(cmap=corr_cmap, norm=norm_corr)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_b, fraction=0.020, pad=0.014)
    cbar.ax.set_title('r', fontsize=8.8, pad=5)
    cbar.ax.tick_params(labelsize=7.6)


    # ========================================================
    # c. performance landscape across strategies and models
    # ========================================================
    ax_c = fig.add_subplot(outer[1, 2])
    add_panel_label(ax_c, 'c', x=-0.10, y=1.12)

    strategies = [
        'GroupKFold_FullyGrouped',
        'LOLO_LeaveOneLiteratureOut',
        'RepeatedOrdinary_GroupedSplit',
        'Balanced_5Seed_Current31F',
        'RepeatedBalanced_Extended',
    ]
    perf = summary[
        summary['Strategy'].isin(strategies) &
        summary['Method'].isin(methods)
    ][['Strategy', 'Method', 'Pooled_R2']].copy()
    perf_mat = perf.pivot(index='Method', columns='Strategy', values='Pooled_R2').reindex(index=methods, columns=strategies)

    norm_r2 = TwoSlopeNorm(vmin=-0.20, vcenter=0.0, vmax=0.26)
    im = ax_c.imshow(perf_mat.values, cmap=r2_cmap, norm=norm_r2, aspect='auto')
    ax_c.set_xticks(np.arange(len(strategies)))
    ax_c.set_yticks(np.arange(len(methods)))
    split_counts = {
        'GroupKFold_FullyGrouped': 'GKF\n5',
        'LOLO_LeaveOneLiteratureOut': 'LOLO\n27',
        'RepeatedOrdinary_GroupedSplit': 'ROS\n15',
        'Balanced_5Seed_Current31F': 'BAL-5\n5',
        'RepeatedBalanced_Extended': 'RBE\n15',
    }
    ax_c.set_xticklabels([split_counts[s] for s in strategies])
    ax_c.set_yticklabels([short_method(m) for m in methods])
    ax_c.set_title('Pooled $R^2$ across grouped designs', pad=13)

    for i in range(len(methods)):
        for j in range(len(strategies)):
            value = perf_mat.iloc[i, j]
            if pd.isna(value):
                ax_c.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor=missing_gray, edgecolor='white', linewidth=0.6))
                continue
            txt_color = 'white' if value < -0.11 or value > 0.20 else ink
            ax_c.text(j, i, f'{value:.2f}', ha='center', va='center',
                      fontsize=7.1, color=txt_color)

    # highlight audited baseline and exploratory best cell
    baseline_i = methods.index('KRR_CatBoost_Weighted_2Model')
    baseline_j = strategies.index('RepeatedBalanced_Extended')
    ax_c.add_patch(Rectangle((baseline_j - 0.5, baseline_i - 0.5), 1, 1,
                             fill=False, edgecolor=gold, linewidth=1.5))
    ax_c.spines[:].set_visible(False)

    sm_r2 = mpl.cm.ScalarMappable(cmap=r2_cmap, norm=norm_r2)
    sm_r2.set_array([])
    cbar_c = fig.colorbar(sm_r2, ax=ax_c, fraction=0.045, pad=0.022)
    cbar_c.set_label(r'Pooled $R^2$', fontsize=8.3)
    cbar_c.ax.tick_params(labelsize=7.2)

    # ========================================================
    # d. combined numeric + categorical shift audit
    # ========================================================
    ax_d = fig.add_subplot(outer[2, :2])
    add_panel_label(ax_d, 'd')

    seeds = [7, 21, 42, 66, 88]
    numeric_map = {
        '氟去除率_pct': 'Target',
        '初始氟_mg_L': 'Initial F',
        '剂量_mM': 'Dose',
        '进水pH': 'pH',
    }
    shift_use = shift[(shift['Comparison'] == 'Test_vs_Train') & shift['Original_Column'].isin(numeric_map.keys())].copy()
    shift_use['Metric'] = shift_use['Original_Column'].map(numeric_map)
    numeric_mat = shift_use.pivot_table(index='Seed', columns='Metric', values='Abs_SMD', aggfunc='mean').reindex(index=seeds, columns=list(numeric_map.values()))

    cat_map = {'Coagulant': 'Coagulant', 'Interfering_ion': 'Ion'}
    cat_use = catcov[(catcov['Comparison'] == 'Test_vs_Train') & catcov['Variable_Role'].isin(cat_map.keys())].copy()
    cat_use['Metric'] = cat_use['Variable_Role'].map(cat_map)
    cat_mat = cat_use.pivot_table(index='Seed', columns='Metric', values='Unseen_sample_fraction', aggfunc='mean').reindex(index=seeds, columns=['Coagulant', 'Ion'])

    combined_labels = ['Target', 'Initial F', 'Dose', 'pH', 'Coagulant', 'Ion']
    ax_d.set_xlim(-0.5, 5.5)
    ax_d.set_ylim(4.5, -0.5)
    ax_d.set_xticks(np.arange(6))
    ax_d.set_yticks(np.arange(5))
    ax_d.set_xticklabels(combined_labels, rotation=20, ha='right', rotation_mode='anchor')
    ax_d.set_yticklabels([f'Seed {s}' for s in seeds])
    ax_d.set_title('Train-test shift audit', pad=18)

    num_norm = Normalize(vmin=0.0, vmax=max(1.2, np.nanmax(numeric_mat.values)))
    cat_norm = Normalize(vmin=0.0, vmax=1.0)
    for i in range(5):
        for j in range(4):
            value = numeric_mat.iloc[i, j]
            face = smd_cmap(num_norm(value))
            ax_d.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor=face, edgecolor='white', linewidth=0.7))
            txt_color = 'white' if value > 0.72 else ink
            ax_d.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=7.3, color=txt_color)
        for j2 in range(2):
            value = cat_mat.iloc[i, j2]
            face = unseen_cmap(cat_norm(value))
            j = j2 + 4
            ax_d.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor=face, edgecolor='white', linewidth=0.7))
            txt_color = 'white' if value > 0.68 else ink
            ax_d.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=7.3, color=txt_color)

    ax_d.axvline(3.5, color='#9EA8B0', lw=1.1)
    ax_d.text(0.29, 1.018, 'Numeric |SMD|', transform=ax_d.transAxes,
              ha='center', va='bottom', fontsize=7.7, color=muted)
    ax_d.text(0.83, 1.018, 'Unseen fraction', transform=ax_d.transAxes,
              ha='center', va='bottom', fontsize=7.7, color=muted)
    ax_d.spines[:].set_visible(False)

    # ========================================================
    # e. literature-level error landscape
    # ========================================================
    ax_e = fig.add_subplot(outer[2, 2])
    add_panel_label(ax_e, 'e', x=-0.11, y=1.12)

    paper_use = paper_metrics[
        (paper_metrics['Strategy'] == 'RepeatedBalanced_Extended') &
        (paper_metrics['Method'] == 'KRR_CatBoost_Weighted_2Model')
    ].copy()
    paper_use['Paper_short'] = paper_use['文献编号'].str.replace('Paper_', 'P', regex=False)
    paper_use['abs_bias'] = paper_use['Paper_Bias'].abs()

    # Standardize paper-level RMSE by the pooled observed-target standard deviation.
    # This is dimensionless and avoids arbitrary visual rescaling.
    pred_ref = predictions[
        (predictions['Strategy'] == 'RepeatedBalanced_Extended') &
        (predictions['Method'] == 'KRR_CatBoost_Weighted_2Model')
    ].copy()
    if 'Data_Index' in pred_ref.columns:
        pred_ref = pred_ref.drop_duplicates(subset=['Data_Index'])
    observed_sd = float(pd.to_numeric(pred_ref['Observed'], errors='coerce').dropna().std(ddof=1))
    if not np.isfinite(observed_sd) or observed_sd <= 0:
        raise ValueError('Observed-target standard deviation is invalid; standardized RMSE cannot be computed.')
    paper_use['Standardized_RMSE'] = paper_use['Paper_RMSE'] / observed_sd

    size = 30 + 230 * (paper_use['n_rows'] / paper_use['n_rows'].max())
    sc = ax_e.scatter(
        paper_use['Standardized_RMSE'], paper_use['Observed_mean'],
        s=size, c=paper_use['Paper_Bias'], cmap='coolwarm',
        vmin=-35, vmax=35, alpha=0.78,
        edgecolor='white', linewidth=0.55,
    )
    ax_e.set_xlabel(r'Standardized RMSE ($\mathrm{RMSE}/\sigma_{obs}$)')
    ax_e.set_ylabel('Mean observed removal (%)')
    ax_e.set_title('Paper-level standardized error', pad=13)
    ax_e.set_xlim(0, 3.0)
    ax_e.set_xticks(np.arange(0, 3.1, 0.5))
    ax_e.grid(color=grid, lw=0.65, linestyle='--', alpha=0.85)
    ax_e.set_axisbelow(True)
    ax_e.spines[['top', 'right']].set_visible(False)

    label_rows = paper_use.nlargest(4, 'Paper_RMSE').reset_index(drop=True)
    label_offsets = [(7, 7), (7, -7), (7, 10), (7, -11)]
    for idx, row in label_rows.iterrows():
        dx, dy = label_offsets[idx]
        ax_e.annotate(
            row['Paper_short'],
            (row['Standardized_RMSE'], row['Observed_mean']),
            xytext=(dx, dy), textcoords='offset points',
            fontsize=7.4, color=ink,
            arrowprops=dict(arrowstyle='-', lw=0.55, color='#88939C'),
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.82),
        )

    cb_e = fig.colorbar(sc, ax=ax_e, fraction=0.050, pad=0.025)
    cb_e.set_label('Prediction bias (percentage points)', fontsize=8.2)
    cb_e.ax.tick_params(labelsize=7.2)
    ax_e.text(0.02, 0.03, 'Bubble size = literature sample count',
              transform=ax_e.transAxes, ha='left', va='bottom',
              fontsize=7.2, color=muted)

    # -------------------- export --------------------
    fig_dir, table_dir, report_dir, log_dir = ensure_dirs(project_root)
    png_path = fig_dir / 'Figure_5_all_domain_exploration_and_generalization_boundary_v9.png'
    pdf_path = fig_dir / 'Figure_5_all_domain_exploration_and_generalization_boundary_v9.pdf'
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    perf.to_csv(
        table_dir / 'Figure_5_grouped_strategy_model_performance_v8.csv',
        index=False, encoding='utf-8-sig'
    )
    corr_mat.to_csv(
        table_dir / 'Figure_5_rbe_residual_correlation_matrix_v8.csv',
        encoding='utf-8-sig'
    )
    paper_use[[
        '文献编号', 'Paper_RMSE', 'Standardized_RMSE', 'Observed_mean',
        'Paper_Bias', 'n_rows'
    ]].to_csv(
        table_dir / 'Figure_5_paper_standardized_rmse_v8.csv',
        index=False, encoding='utf-8-sig'
    )

    random_r2 = float(hierarchy.loc[hierarchy['level'] == 'random_split_highest', 'r2'].iloc[0])
    audited_r2 = float(hierarchy.loc[hierarchy['level'] == 'grouped_audit_baseline', 'r2'].iloc[0])
    audit_text = (
        'Figure 5 v8 audit\n'
        f'Random-split milestone R2: {random_r2:.6f}\n'
        f'Audited all-domain grouped R2: {audited_r2:.6f}\n'
        f'Performance matrix rows: {len(perf)}\n'
        f'Residual-correlation matrix shape: {corr_mat.shape}\n'
        f'Literature groups in error landscape: {len(paper_use)}\n'
        f'Observed-target SD used for standardized RMSE: {observed_sd:.6f}\n'
        f'Max standardized paper RMSE: {paper_use["Standardized_RMSE"].max():.6f}\n'
    )
    (report_dir / '34b_est_figure5_grouped_validation_boundary_v8_audit.txt').write_text(
        audit_text, encoding='utf-8'
    )
    (log_dir / '34b_est_figure5_grouped_validation_boundary_v8.log').write_text(
        audit_text, encoding='utf-8'
    )

    return png_path, pdf_path


if __name__ == '__main__':
    t0 = time.time()
    configure_matplotlib()
    root = find_project_root()
    png, pdf = build_figure(root)
    elapsed = time.time() - t0
    print('=' * 92)
    print('Figure 5 v9 completed successfully')
    print(f'Project root : {root}')
    print(f'PNG saved to : {png}')
    print(f'PDF saved to : {pdf}')
    print(f'Total runtime: {elapsed:.2f} s')
    print('=' * 92)
