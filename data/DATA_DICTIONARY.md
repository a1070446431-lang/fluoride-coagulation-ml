# Data dictionary

## Files

| File | Rows | Description |
|---|---|---|
| `model_ready_v1_with_source.csv` | 1,011 | Model-ready matrix used for all reported fits (UTF-8, comma-separated) |
| `dataset_final_with_sources.xlsx` | 1,011 | Compiled dataset; `Sheet1` holds the two-row header (group row + field row), worksheet `文献编号-来源对照` holds the source mapping |
| `source_mapping.csv` | 27 | Literature ID, record count, short and full reference, matching evidence and confidence |

## Columns of `model_ready_v1_with_source.csv`

### Harmonized variables used by the models (13)

| # | Column | Meaning | Unit / values |
|---|---|---|---|
| 1 | `氟形态` | Fluorine species | `氟离子` (free F⁻), `氟离子+络合氟` (free + complexed), `络合氟` (complexed) |
| 2 | `游离氟占比` | Fraction of fluoride present as free F⁻ | 0–1 |
| 3 | `初始氟_mg_L` | Initial fluoride concentration | mg L⁻¹ |
| 4 | `干扰离子` | Coexisting-ion system | 25 categories, e.g. `无`, `SO₄²⁻`, `腐殖酸`, `Ca²⁺/Mg²⁺/HCO₃⁻/SO₄²⁻` |
| 5 | `干扰离子浓度_mmol_L` | Total concentration of coexisting ions | mmol L⁻¹ |
| 6 | `水温_C` | Water temperature | °C |
| 7 | `混凝剂` | Coagulant | 53 categories, e.g. `聚合氯化铝`, `硫酸铝`, `氯化钙`, `水合氧化锆` |
| 8 | `剂量_mM` | Coagulant dose, harmonized field | mmol L⁻¹ (see Notes) |
| 9 | `进水pH` | Influent pH | – |
| 10 | `停留时间_min` | Contact / residence time | min |
| 11 | `出水pH` | Effluent pH | – |
| 12 | `氟去除率_pct` | Fluoride removal efficiency (target) | % |
| 13 | `出水氟_mg_L` | Effluent fluoride concentration | mg L⁻¹ |

### Audit and provenance fields (15)

| # | Column | Meaning |
|---|---|---|
| 14 | `原始行号` | Row index in the compiled workbook |
| 15–24 | `*_原始` | Values as reported in the source before harmonization (free-fluoride fraction, initial F, interfering-ion concentration, temperature, dose, influent pH, residence time, effluent pH, removal, effluent F) |
| 25 | `由浓度反算去除率_pct` | Removal recalculated from initial and effluent concentrations |
| 26 | `去除率差值_pct` | Reported removal minus recalculated removal (mass-balance audit) |
| 27 | `文献编号` | Literature group ID, `Paper_001` … `Paper_032`; maps to `source_mapping.csv` |
| 28 | `剂量_原始单位` | Unit in which the dose was reported: `mM` (960 records), `mg/L` (39), `mL/L` (12) |

## Grouping

`文献编号` is the highest isolation level used in grouped validation. There are 27 groups from
22 independent publications: six groups (`Paper_012`–`Paper_017`) are separate experimental batches
reported in Deng et al. (2024), and five groups in total are additional to the publication count.
Records are never split within a group.

## Notes

- **Dose units.** `剂量_mM` is the harmonized dose field used by every model; the unit each record
  was reported in is given by `剂量_原始单位`. Where the source reported a molar concentration it was
  converted to mmol L⁻¹. For `Paper_006`, `Paper_008` and `Paper_022` (51 records, 5.0%) the source
  reported only a mass or volume dose (mg L⁻¹ or mL L⁻¹) of a commercial or composite product whose
  active content was not specified; the numeric value was carried over unchanged and these records
  are marked `mg/L` or `mL/L` in `剂量_原始单位`. They were modelled on the original scale together
  with coagulant category rather than converted on an assumed active-content basis.
- **`Paper_019`** (2 records) could not be matched to a publication from the available metadata;
  the rows are retained as a separate group and flagged in `source_mapping.csv`.
- **Zero effluent fluoride.** Where a source reported 0 mg L⁻¹ effluent, removal is 100%; these
  values are kept as reported.
- **Language.** Column names are in Chinese because the compiled workbook is the working file of
  the project; the table above gives the English equivalent of every field.
