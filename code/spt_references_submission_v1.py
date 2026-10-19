#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add a verified SPT-style reference system to the latest manuscript.

The script is intended to be copied into ``fluoride_coagulation_ml_project/src``
and run from PyCharm.  It preserves the source manuscript, adds numeric
square-bracket citations (e.g. [1]) to methodologically relevant paragraphs,
and appends a numbered reference list with clickable DOI/URL hyperlinks.

Only text/citation packaging is changed.  Existing figures, tables, results,
and scientific values are not recalculated or overwritten.
"""

from __future__ import annotations

import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor


RUN_START = time.perf_counter()
INPUT_NAME = "多源异质数据驱动的混凝除氟预测建模与适用边界评估_.docx"
OUTPUT_NAME = "Manuscript_SPT_submission_ready_references_v1.docx"


# The order follows first appearance in the manuscript.  Metadata was taken
# from Crossref, publisher/PubMed records, or the attached source articles.
# DOI strings are kept as DOI URLs in the output so that they remain clickable.
REFERENCES = [
    {
        "authors": "E. Shaji, K.V. Sarath, M. Santosh, P.K. Krishnaprasad, B.K. Arya, M.S. Babu",
        "title": "Fluoride contamination in groundwater: A global review of the status, processes, challenges, and remedial measures",
        "journal": "Geoscience Frontiers",
        "year": "2024",
        "volume": "15",
        "pages": "101734",
        "doi": "10.1016/j.gsf.2023.101734",
    },
    {
        "authors": "A. Bhatnagar, E. Kumar, M. Sillanpää",
        "title": "Fluoride removal from water by adsorption—A review",
        "journal": "Chemical Engineering Journal",
        "year": "2011",
        "volume": "171",
        "pages": "811–840",
        "doi": "10.1016/j.cej.2011.05.028",
    },
    {
        "authors": "D.T. Moussa, M.H. El-Naas, M. Nasser, M.J. Al-Marri",
        "title": "A comprehensive review of electrocoagulation for water treatment: Potentials and challenges",
        "journal": "Journal of Environmental Management",
        "year": "2017",
        "volume": "186",
        "pages": "24–41",
        "doi": "10.1016/j.jenvman.2016.10.032",
    },
    {
        "authors": "J.N. Hakizimana, B. Gourich, M. Chafi, Y. Stiriba, C. Vial, P. Drogui, J. Naja",
        "title": "Electrocoagulation process in water treatment: A review of electrocoagulation modeling approaches",
        "journal": "Desalination",
        "year": "2017",
        "volume": "404",
        "pages": "1–21",
        "doi": "10.1016/j.desal.2016.10.011",
    },
    {
        "authors": "K. Govindan, M. Raja, S. Uma Maheshwari, M. Noel, Y. Oren",
        "title": "Comparison and understanding of fluoride removal mechanism in Ca2+, Mg2+ and Al3+ ion assisted electrocoagulation process using Fe and Al electrodes",
        "journal": "Journal of Environmental Chemical Engineering",
        "year": "2015",
        "volume": "3",
        "pages": "1784–1793",
        "doi": "10.1016/j.jece.2015.06.014",
    },
    {
        "authors": "J.U. Halpegama, K.Y. Heenkenda, Z. Wu, K.G.N. Nanayakkara, R.M.G. Rajapakse, A. Bandara, A.C. Herath, X. Chen, R. Weerasooriya",
        "title": "Concurrent removal of hardness and fluoride in water by monopolar electrocoagulation",
        "journal": "Journal of Environmental Chemical Engineering",
        "year": "2021",
        "volume": "9",
        "pages": "106105",
        "doi": "10.1016/j.jece.2021.106105",
    },
    {
        "authors": "Z. Deng, A. Xu, X. Wei, X. Zhang, B. Pan",
        "title": "Formation of aluminum–fluoride complexes compromises defluoridation efficiency of aluminum-based coagulation",
        "journal": "Environmental Science & Technology",
        "year": "2024",
        "volume": "58",
        "pages": "14918–14928",
        "doi": "10.1021/acs.est.4c04722",
    },
    {
        "authors": "Z. Deng, A. Xu, X. Zhang, K. Zhang, B. Pan",
        "title": "Effects of chelators on fluoride removal by AlCl3 and Al13 coagulation: Performance and mechanism",
        "journal": "Environmental Science & Technology",
        "year": "2025",
        "volume": "59",
        "pages": "3297–3308",
        "doi": "10.1021/acs.est.4c11302",
    },
    {
        "authors": "A.A. El Mahjoub, N. Najid, F.Z. Falil, H.A. Rashwan, Y. Stiriba, F. Audonnet, C. Vial, M.C. Necibi, B. Gourich",
        "title": "Optimizing and predicting electrocoagulation performance for simultaneous removal of fluoride, silica, and hardness from real brackish groundwater: Integrating response surface methodology with interpretable machine learning",
        "journal": "Journal of Environmental Chemical Engineering",
        "year": "2025",
        "volume": "13",
        "pages": "120176",
        "doi": "10.1016/j.jece.2025.120176",
    },
    {
        "authors": "A.A. El Mahjoub, N. Najid, F.Z. Falil, F. Audonnet, C. Vial, H.A. Rashwan, Y. Stiriba, M.C. Necibi, B. Gourich",
        "title": "Pilot-scale demonstration for simultaneous removal of fluoride, silica and hardness from real brackish groundwater using electrocoagulation process: Performance evaluation and machine learning prediction",
        "journal": "Desalination",
        "year": "2026",
        "volume": "619",
        "pages": "119510",
        "doi": "10.1016/j.desal.2025.119510",
    },
    {
        "authors": "H. Goyal, P. Mondal",
        "title": "Process modelling and multi-objective optimization for sustainable treatment of arsenic and fluoride contaminated groundwater through semi-pilot electrocoagulation: An integrated response surface methodology and machine learning approach",
        "journal": "Groundwater for Sustainable Development",
        "year": "2026",
        "volume": "33",
        "pages": "101598",
        "doi": "10.1016/j.gsd.2026.101598",
    },
    {
        "authors": "A. Thakur, Dharmendra",
        "title": "Ex-situ remediation of arsenic-fluorinated water using electrocoagulation: Neural network multi-objective modelling and metaheuristic optimisation",
        "journal": "Journal of Applied Electrochemistry",
        "year": "2025",
        "volume": "55",
        "pages": "1439–1454",
        "doi": "10.1007/s10800-024-02250-5",
    },
    {
        "authors": "I. García, L.F. Castañeda, J.L. Nava, O. Coreño",
        "title": "Continuous electrocoagulation–flocculation–sedimentation process to remove arsenic, fluoride, and hydrated silica from drinking water",
        "journal": "Journal of Water Process Engineering",
        "year": "2025",
        "volume": "69",
        "pages": "106571",
        "doi": "10.1016/j.jwpe.2024.106571",
    },
    {
        "authors": "Y. Gan, L. Qu, Z. Cang, X. Ding, J. Luo, Z. Li, Z. Wang, C. Ding, Y. Cui, B. Xu, B. Wu",
        "title": "Simultaneous deep purification of fluoride and trivalent arsenic by a bimetallic composite xerogel coagulant",
        "journal": "Separation and Purification Technology",
        "year": "2025",
        "volume": "362",
        "pages": "131779",
        "doi": "10.1016/j.seppur.2025.131779",
    },
    {
        "authors": "H. Ullah, S. Khan, X. Zhu, B. Chen, Z. Rao, N. Wu, A.M. Idris",
        "title": "Machine learning-aided biochar design for the adsorptive removal of emerging inorganic pollutants in water",
        "journal": "Separation and Purification Technology",
        "year": "2025",
        "volume": "362",
        "pages": "131421",
        "doi": "10.1016/j.seppur.2025.131421",
    },
    {
        "authors": "J.A. Cordero Solano, J. Hafner, M.S. McLachlan, H. Singer, K. Fenner",
        "title": "Predicting micropollutant removal in wastewater treatment based on molecular structure: Benchmark data and models",
        "journal": "Environmental Science & Technology",
        "year": "2025",
        "volume": "59",
        "pages": "22020–22028",
        "doi": "10.1021/acs.est.5c09314",
    },
    {
        "authors": "L. Grinsztajn, E. Oyallon, G. Varoquaux",
        "title": "Why do tree-based models still outperform deep learning on typical tabular data?",
        "journal": "Advances in Neural Information Processing Systems",
        "year": "2022",
        "volume": "35",
        "pages": "507–520",
        "doi": "10.52202/068431-0037",
    },
    {
        "authors": "L. Breiman",
        "title": "Random forests",
        "journal": "Machine Learning",
        "year": "2001",
        "volume": "45",
        "pages": "5–32",
        "doi": "10.1023/A:1010933404324",
    },
    {
        "authors": "P. Geurts, D. Ernst, L. Wehenkel",
        "title": "Extremely randomized trees",
        "journal": "Machine Learning",
        "year": "2006",
        "volume": "63",
        "pages": "3–42",
        "doi": "10.1007/s10994-006-6226-1",
    },
    {
        "authors": "J.H. Friedman",
        "title": "Greedy function approximation: A gradient boosting machine",
        "journal": "The Annals of Statistics",
        "year": "2001",
        "volume": "29",
        "pages": "1189–1232",
        "doi": "10.1214/aos/1013203451",
    },
    {
        "authors": "T. Chen, C. Guestrin",
        "title": "XGBoost: A scalable tree boosting system",
        "journal": "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",
        "year": "2016",
        "volume": "",
        "pages": "785–794",
        "doi": "10.1145/2939672.2939785",
    },
    {
        "authors": "G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu",
        "title": "LightGBM: A highly efficient gradient boosting decision tree",
        "journal": "Advances in Neural Information Processing Systems",
        "year": "2017",
        "volume": "30",
        "pages": "3146–3154",
        "url": "https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
    },
    {
        "authors": "L. Prokhorenkova, G. Gusev, A. Vorobev, A.V. Dorogush, A. Gulin",
        "title": "CatBoost: Unbiased boosting with categorical features",
        "journal": "Advances in Neural Information Processing Systems",
        "year": "2018",
        "volume": "31",
        "pages": "6639–6649",
        "url": "https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html",
    },
    {
        "authors": "C.E. Rasmussen, C.K.I. Williams",
        "title": "Gaussian Processes for Machine Learning",
        "journal": "MIT Press",
        "year": "2006",
        "volume": "",
        "pages": "",
        "url": "https://gaussianprocess.org/gpml/",
    },
    {
        "authors": "S. Varma, R. Simon",
        "title": "Bias in error estimation when using cross-validation for model selection",
        "journal": "BMC Bioinformatics",
        "year": "2006",
        "volume": "7",
        "pages": "91",
        "doi": "10.1186/1471-2105-7-91",
    },
    {
        "authors": "R. Kohavi",
        "title": "A study of cross-validation and bootstrap for accuracy estimation and model selection",
        "journal": "Proceedings of the 14th International Joint Conference on Artificial Intelligence",
        "year": "1995",
        "volume": "2",
        "pages": "1137–1143",
        "url": "https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf",
    },
    {
        "authors": "D.R. Roberts, V. Bahn, S. Ciuti, M.S. Boyce, J. Elith, S. Guillera-Arroita, S. Hauenstein, J.J. Lahoz-Monfort, B. Schröder, W. Thuiller, D.I. Dormann",
        "title": "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure",
        "journal": "Ecography",
        "year": "2017",
        "volume": "40",
        "pages": "913–929",
        "doi": "10.1111/ecog.02881",
    },
    {
        "authors": "G.C. Cawley, N.L.C. Talbot",
        "title": "On over-fitting in model selection and subsequent selection bias in performance evaluation",
        "journal": "Journal of Machine Learning Research",
        "year": "2010",
        "volume": "11",
        "pages": "2079–2107",
        "url": "https://jmlr.org/papers/v11/cawley10a.html",
    },
    {
        "authors": "J. Jaworska, N. Nikolova-Jeliazkova, T. Aldenberg",
        "title": "QSAR applicability domain estimation by projection of the training set descriptor space: A review",
        "journal": "Alternatives to Laboratory Animals",
        "year": "2005",
        "volume": "33",
        "pages": "445–459",
        "doi": "10.1177/026119290503300508",
    },
    {
        "authors": "F. Sahigara, K. Mansouri, D. Ballabio, A. Mauri, V. Consonni, R. Todeschini",
        "title": "Comparison of different approaches to define the applicability domain of QSAR models",
        "journal": "Molecules",
        "year": "2012",
        "volume": "17",
        "pages": "4791–4810",
        "doi": "10.3390/molecules17054791",
    },
    {
        "authors": "S.M. Lundberg, S.-I. Lee",
        "title": "A unified approach to interpreting model predictions",
        "journal": "Advances in Neural Information Processing Systems",
        "year": "2017",
        "volume": "30",
        "pages": "4765–4774",
        "url": "https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html",
    },
    {
        "authors": "A. Goldstein, A. Kapelner, J. Bleich, E. Pitkin",
        "title": "Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation",
        "journal": "Journal of Computational and Graphical Statistics",
        "year": "2015",
        "volume": "24",
        "pages": "44–65",
        "doi": "10.1080/10618600.2014.907095",
    },
    {
        "authors": "K. Aas, M. Jullum, A. Løland",
        "title": "Explaining individual predictions when features are dependent: More accurate approximations to Shapley values",
        "journal": "Artificial Intelligence",
        "year": "2021",
        "volume": "298",
        "pages": "103502",
        "doi": "10.1016/j.artint.2021.103502",
    },
    {
        "authors": "L.F. Castañeda, I. García, J.L. Nava, O. Coreño",
        "title": "Concurrent arsenic, fluoride, and hydrated silica removal from deep well water by electrocoagulation: Comparison of sacrificial anodes (Al, Fe, and Al–Fe)",
        "journal": "Journal of Environmental Management",
        "year": "2024",
        "volume": "365",
        "pages": "121597",
        "doi": "10.1016/j.jenvman.2024.121597",
    },
    {
        "authors": "N. Ben Grich, A. Attour, M. Le Page Mostefa, S. Guesmi, M. Tlili, F. Lapicque",
        "title": "Fluoride removal from water by electrocoagulation: Effect of the type of water and the experimental parameters",
        "journal": "Electrochimica Acta",
        "year": "2019",
        "volume": "316",
        "pages": "257–265",
        "doi": "10.1016/j.electacta.2019.05.130",
    },
    {
        "authors": "V.F. Mena, A. Betancor-Abreu, S. González, S. Delgado, R.M. Souto, J.J. Santana",
        "title": "Fluoride removal from natural volcanic underground water by an electrocoagulation process: Parametric and cost evaluations",
        "journal": "Journal of Environmental Management",
        "year": "2019",
        "volume": "246",
        "pages": "472–483",
        "doi": "10.1016/j.jenvman.2019.05.147",
    },
    {
        "authors": "L.S. Thakur, P. Goyal, P. Mondal",
        "title": "Simultaneous removal of arsenic and fluoride from synthetic and real groundwater by electrocoagulation process: Parametric and cost evaluation",
        "journal": "Journal of Environmental Management",
        "year": "2017",
        "volume": "190",
        "pages": "102–112",
        "doi": "10.1016/j.jenvman.2016.12.053",
    },
    {
        "authors": "S. Vasudevan, B.S. Kannan, J. Lakshmi, S. Mohanraj, G. Sozhan",
        "title": "Effects of alternating and direct current in electrocoagulation process on the removal of fluoride from water",
        "journal": "Journal of Chemical Technology & Biotechnology",
        "year": "2011",
        "volume": "86",
        "pages": "428–436",
        "doi": "10.1002/jctb.2534",
    },
    {
        "authors": "M. Ingelsson, N. Yasri, E.P.L. Roberts",
        "title": "Electrode passivation, faradaic efficiency, and performance enhancement strategies in electrocoagulation—A review",
        "journal": "Water Research",
        "year": "2020",
        "volume": "187",
        "pages": "116433",
        "doi": "10.1016/j.watres.2020.116433",
    },
    {
        "authors": "C.M. Khor, M.E. Liao, A. Iddya, S. Ma, F. Yang, Y.H. Liu, Y.Z. Bootwala, G.G. Jang, M.S. Goorsky, E.M.V. Hoek, C. Tsouris, M. Mothersbaugh, M.C. Hatzell, D. Jassby",
        "title": "Physical and electrochemical characterization of aluminum electrodes during electrocoagulation",
        "journal": "ACS ES&T Water",
        "year": "2024",
        "volume": "4",
        "pages": "44–56",
        "doi": "10.1021/acsestwater.3c00263",
    },
    {
        "authors": "W. Den, C.-J. Wang",
        "title": "Removal of silica from brackish water by electrocoagulation pretreatment to prevent fouling of reverse osmosis membranes",
        "journal": "Separation and Purification Technology",
        "year": "2008",
        "volume": "59",
        "pages": "318–325",
        "doi": "10.1016/j.seppur.2007.07.025",
    },
]


# Each rule attaches a citation to a paragraph that makes the corresponding
# claim.  Matching is by stable text rather than paragraph number so that the
# script remains robust if the manuscript is edited before rerunning.
CITATION_RULES = [
    ("混凝工艺是一种高效", [1, 2, 3, 4, 5, 6]),
    ("机器学习作为一种变革性工具", [7, 8, 9, 10, 11, 12, 13, 14]),
    ("已有基于文献数据的环境污染物去除预测研究绝大多数", [15, 16]),
    ("应对上述多文献混凝除氟数据", [3, 4, 17]),
    ("候选算法覆盖", [17, 18, 19, 20, 21, 22, 23, 24]),
    ("融合权重只根据", [25, 26, 27, 28]),
    ("random split baseline", [25, 26, 27, 28]),
    ("适用域只依据", [29, 30]),
    ("SHAP 用于量化", [31, 32, 33]),
    ("不同文献组对总体分布", [34, 35, 36, 37, 38, 39, 40, 41]),
    ("全域泛化受限后", [29, 30]),
    ("二维和三维 PDP", [31, 32]),
    ("实际使用流程中", [3, 4, 12, 13, 26]),
    ("现有电絮凝研究多", [9, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23]),
    ("这一分析思路也适用于", [15, 25, 26, 29, 31]),
]


def project_root() -> Path:
    """Locate the project root when run from src/ or from PyCharm."""
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd()]
    for c in candidates:
        if (c / "outputs").exists() or (c / "data").exists() or (c / "src").exists():
            return c
    return Path.cwd()


def locate_input(root: Path) -> Path:
    explicit = [
        root / INPUT_NAME,
        Path.cwd() / INPUT_NAME,
        root / "最新正文.docx",
        root / "main_manuscript.docx",
    ]
    for p in explicit:
        if p.is_file():
            return p

    hits: list[Path] = []
    for p in root.rglob("*.docx"):
        low = str(p).lower()
        name = p.name.lower()
        if any(part in low for part in ("outputs", "supplementary", "qa_", "reference_ready")):
            continue
        if "多源异质" in p.name or "混凝除氟" in p.name or "manuscript" in name:
            hits.append(p)
    if hits:
        return max(hits, key=lambda x: x.stat().st_mtime)
    raise FileNotFoundError(
        f"Could not locate the latest manuscript. Copy {INPUT_NAME} to the project root "
        "or set INPUT_NAME in this script."
    )


def is_heading_or_caption(paragraph) -> bool:
    text = paragraph.text.strip()
    style = (paragraph.style.name or "").lower() if paragraph.style else ""
    if "caption" in style or text.startswith(("图", "Figure", "Table", "Supplementary Figure")):
        return True
    return style.startswith("heading") or text in {"References", "参考文献"}


def replace_text_everywhere(doc: Document, old: str, new: str) -> int:
    count = 0
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if old in run.text:
                count += run.text.count(old)
                run.text = run.text.replace(old, new)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        if old in run.text:
                            count += run.text.count(old)
                            run.text = run.text.replace(old, new)
    return count


def compress_numbers(numbers: Iterable[int]) -> str:
    nums = sorted(set(int(n) for n in numbers))
    if not nums:
        return ""
    chunks: list[str] = []
    start = previous = nums[0]
    for n in nums[1:]:
        if n == previous + 1:
            previous = n
            continue
        chunks.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = n
    chunks.append(str(start) if start == previous else f"{start}-{previous}")
    return "[" + ",".join(chunks) + "]"


def existing_citation_numbers(text: str) -> set[int]:
    found: set[int] = set()
    for m in re.finditer(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]", text):
        for n in re.findall(r"\d+", m.group(1)):
            found.add(int(n))
    return found


def append_citation(paragraph, numbers: list[int]) -> bool:
    if is_heading_or_caption(paragraph):
        return False
    numbers = [n for n in numbers if 1 <= n <= len(REFERENCES)]
    numbers = [n for n in numbers if n not in existing_citation_numbers(paragraph.text)]
    if not numbers:
        return False
    citation = compress_numbers(numbers)
    runs = paragraph.runs
    if runs:
        last = runs[-1]
        text = last.text or ""
        m = re.search(r"([。！？.!?])([\s]*)$", text)
        if m:
            body, punct = text[:m.start()], text[m.start():]
            last.text = body
            paragraph.add_run(f" {citation}")
            paragraph.add_run(punct)
            return True
    paragraph.add_run(f" {citation}")
    return True


def add_hyperlink(paragraph, label: str, url: str) -> None:
    part = paragraph.part
    rid = part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rid)
    new_run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.append(color)
    rpr.append(underline)
    new_run.append(rpr)
    text = OxmlElement("w:t")
    text.text = label
    new_run.append(text)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def ensure_reference_style(doc: Document):
    try:
        style = doc.styles["SPT Reference"]
    except KeyError:
        style = doc.styles.add_style("SPT Reference", WD_STYLE_TYPE.PARAGRAPH)
    style.font.name = "Times New Roman"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    style.font.size = Pt(9.5)
    return style


def format_reference(paragraph, idx: int, ref: dict) -> None:
    paragraph.style = "SPT Reference"
    paragraph.paragraph_format.left_indent = Inches(0.0)
    paragraph.paragraph_format.first_line_indent = Inches(-0.25)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.0
    prefix = paragraph.add_run(f"[{idx}] {ref['authors']}. {ref['title']}. ")
    prefix.font.name = "Times New Roman"
    prefix._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    details = ref["journal"]
    if ref.get("year"):
        details += f" {ref['year']}"
    if ref.get("volume"):
        details += f";{ref['volume']}"
    if ref.get("pages"):
        details += f":{ref['pages']}"
    paragraph.add_run(details + ". ")
    target = ref.get("doi") or ref.get("url")
    if target:
        url = target if target.startswith("http") else f"https://doi.org/{target}"
        label = target if target.startswith("http") else f"https://doi.org/{target}"
        add_hyperlink(paragraph, label, url)


def remove_trailing_empty_paragraphs(doc: Document) -> None:
    """Prevent an inherited empty terminal paragraph from creating a blank page."""
    while doc.paragraphs:
        paragraph = doc.paragraphs[-1]
        if paragraph.text.strip() or paragraph.runs:
            break
        element = paragraph._element
        element.getparent().remove(element)


def append_references(doc: Document) -> None:
    # Keep the reference list visually separate from the conclusions without
    # modifying any existing body content.
    remove_trailing_empty_paragraphs(doc)
    heading = doc.add_paragraph()
    heading.style = "Heading 1" if "Heading 1" in [s.name for s in doc.styles] else doc.styles["Normal"]
    heading.paragraph_format.space_before = Pt(6)
    heading.paragraph_format.space_after = Pt(8)
    run = heading.add_run("References")
    run.bold = True
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(13)
    ensure_reference_style(doc)
    for idx, ref in enumerate(REFERENCES, start=1):
        p = doc.add_paragraph()
        format_reference(p, idx, ref)


def main() -> int:
    root = project_root()
    source = locate_input(root)
    out_dir = root / "outputs" / "spt_submission_package" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / OUTPUT_NAME

    doc = Document(str(source))
    # The uploaded manuscript contains one legacy [16] placeholder in the
    # introduction.  Remove it before inserting the audited numbering below.
    legacy_removed = replace_text_everywhere(doc, "[16]", "")

    matched: dict[str, int] = {}
    cited_numbers: set[int] = set()
    for trigger, numbers in CITATION_RULES:
        attached = False
        for paragraph in doc.paragraphs:
            if trigger in paragraph.text and not is_heading_or_caption(paragraph):
                if append_citation(paragraph, numbers):
                    attached = True
                    cited_numbers.update(numbers)
                break
        matched[trigger] = int(attached)

    missing = set(range(1, len(REFERENCES) + 1)) - cited_numbers
    if missing:
        raise RuntimeError(f"Reference numbers not cited in the manuscript: {sorted(missing)}")

    append_references(doc)
    doc.core_properties.subject = "SPT submission-ready manuscript with numeric references"
    doc.save(str(output))

    print("PASS: SPT numeric reference system created.")
    print(f"Input preserved: {source}")
    print(f"Output: {output}")
    print(f"References: {len(REFERENCES)}; cited numbers: {len(cited_numbers)}; legacy markers removed: {legacy_removed}.")
    print(f"Citation rules matched: {sum(matched.values())}/{len(CITATION_RULES)}.")
    print("In-text style: non-superscripted square brackets [1]; DOI/URL strings are clickable hyperlinks.")
    print(f"Total runtime: {time.perf_counter() - RUN_START:.2f} s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Total runtime: {time.perf_counter() - RUN_START:.2f} s", file=sys.stderr)
        raise
