# Code

Python scripts written as a numbered analysis sequence (01 → 34). Numbers reflect the order in
which analyses were added, not the order required to reproduce a single result.

## Primary scripts for the reported results

| Manuscript result | Script |
|---|---|
| Rebuild the model-ready matrix from the compiled workbook | `29a_rebuild_model_ready_v1_from_final_excel.py` |
| Random-split model comparison and weighted ensemble (R² = 0.882) | `19_selected_feature_fusion.py` |
| Grouped (source-isolated) validation, exploratory and audited designs (R² = 0.250 / 0.194) | `31n_krr_grouped_validation_robustness_fixed2.py` |
| Applicability-domain analysis (R² = −0.160) | `32a2_corrected_applicability_domain_analysis_v3_fixed.py` |
| Target-free selective screening policy (R² = 0.748 at 1.47% coverage) | `32c2_targeted_exhaustive_selective_ad_policy_fixed.py` |
| Threshold sensitivity and anti-cherry-picking audit | `32h_threshold_sensitivity_and_anti_cherry_picking_audit.py` |
| SHAP and partial-dependence figure (Fig. 8) | `34e_est_figure8_shap_pdp_journal_layout_v8.py` |
| Paper-level extrapolation and source diagnostics | `30_grouped_validation_diagnosis.py` |
| Main figure set | `32d_*`, `32i2_*`, `34a_*`–`34e_*` |

The remaining scripts document intermediate screening steps (model tuning, feature ablation,
alternative grouped designs, residual diagnostics). They are included so that the sequence of
decisions can be audited; several are superseded by a later `_fixed` / `_v2` variant.

## Running the scripts

- Scripts resolve the project root with `find_project_root()` and write to `outputs/`.
  Set the root to this repository, or run from a checkout that contains `data/` and `outputs/`.
- Several scripts import helper modules from the same folder (for example
  `31n_*` imports `31m2_*`), so keep the folder on `sys.path`.
- Seeds are fixed inside each script. Library versions are pinned in `requirements.txt`.

## Data expected on input

`data/model_ready_v1_with_source.csv` is already the output of `29a_*`; the downstream scripts read
that file, so they can be run without rebuilding the matrix.
