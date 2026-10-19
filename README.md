# Heterogeneous multi-literature data-driven modelling of coagulation defluoridation

Data, analysis code and result tables accompanying the manuscript:

> Zhang, R., Tao, H., Yu, X., Tian, Y., Yin, Y., Huang, Y., Xu, H., Li, D., Guan, Z. (2026).
> Heterogeneous multi-literature data-driven predictive modeling and applicability boundary
> assessment for coagulation defluoridation. *Environmental Modelling & Software* (submitted).

## What this repository contains

| Path | Content |
|---|---|
| `data/model_ready_v1_with_source.csv` | Model-ready matrix: 1,011 records × 28 fields (13 harmonized variables, 1 row index, 10 raw values, 2 derived audit fields, 1 literature identifier, 1 dose-unit flag) |
| `data/dataset_final_with_sources.xlsx` | Compiled dataset with the literature-source mapping worksheet |
| `data/source_mapping.csv` | Literature ID → full reference, record count, matching evidence |
| `data/DATA_DICTIONARY.md` | Field definitions, units and categories |
| `code/` | Python analysis scripts, as used for the reported runs |
| `results/tables/` | Result tables behind the reported metrics |
| `figures/Fig1.png` … `figures/Fig8.png` | Main figures as printed in the manuscript |

## Dataset

The dataset contains 1,011 coagulation–defluoridation records compiled from 27 literature-derived
groups, corresponding to 22 independent publications. Groups rather than publications were used as
the highest isolation level so that separate experimental batches remain distinguishable; six of the
groups are separate batches reported in Deng et al. (2024, *ES&T* 58, 14918–14928).

No personal or confidential data are included. Original articles and their supplementary files are
not redistributed because of publisher copyright. Every record keeps its literature identifier
(`文献编号`), which maps to the reference list through `data/source_mapping.csv`.

## Environment

- Python 3.12.0, tested on Windows 11 (x86-64, Intel Core i7-12700H, 16 GB RAM); no GPU required
- pandas 3.0.2, NumPy 2.4.4, scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0,
  CatBoost 1.2.10, SHAP 0.51.0, Matplotlib 3.10.8, Optuna 4.8.0

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the reported results

| Manuscript result | Script | Output |
|---|---|---|
| Model-ready matrix (1,011 × 27) | `code/29a_rebuild_model_ready_v1_from_final_excel.py` | `data/model_ready_v1_with_source.csv` |
| Random-split weighted ensemble, R² = 0.882 | `code/19_selected_feature_fusion.py` | `results/tables/19_selected_fusion_summary.csv` |
| Grouped validation, R² = 0.194 and 0.250 | `code/31n_krr_grouped_validation_robustness_fixed2.py` | `results/tables/31n_grouped_validation_*.csv` |
| Applicability domain, R² = −0.160 | `code/32a2_corrected_applicability_domain_analysis_v3_fixed.py` | `results/tables/32a2_*.csv` |
| Selective screening, R² = 0.748 at 1.47% coverage | `code/32c2_targeted_exhaustive_selective_ad_policy_fixed.py` | `results/tables/32c2_*.csv` |
| Threshold sensitivity audit | `code/32h_threshold_sensitivity_and_anti_cherry_picking_audit.py` | `results/tables/32h_*.csv` |
| SHAP and partial dependence (Fig. 8) | `code/34e_est_figure8_shap_pdp_journal_layout_v8.py` | `figures/Fig8.png` |

The scripts form a numbered analysis sequence and several of them import helper modules from the
same folder, so run them with `code/` on `sys.path`. Paths are resolved through
`find_project_root()`; intermediate output is written to `outputs/`.

Results are stochastic. Seeds are fixed in the scripts, but library versions different from those
listed above can shift the third decimal of the reported metrics.

## License

- Code: MIT, see `LICENSE`
- Data and result tables: CC BY 4.0, see `LICENSE-DATA`

## Citation

If you use these data or code, please cite the article above and the archived release:

> Zhang, R., et al. (2026). Fluoride coagulation–defluoridation modelling workflow (v1.0.0).
> Zenodo. https://doi.org/10.5281/zenodo.[XXXXXXX]

## Contact

Zeyu Guan — zyguan@wtu.edu.cn
