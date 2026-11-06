# Literature-derived gene reference lists

Retrieval date: 2026-11-06. All lists are published knowledge (paper supplementary tables / official resource APIs). No patient-level data was downloaded or used. Sample-level sheets in the supplements (Sanchez-Vega Table S1/S4, Bailey Table S5/S6) were deleted or ignored without being used.

## 1. pathways_sanchezvega2018.csv  (335 rows, 334 unique genes)
- Source: Sanchez-Vega F. et al. "Oncogenic Signaling Pathways in The Cancer Genome Atlas." Cell 173(2):321-337 (2018). doi:10.1016/j.cell.2018.03.035. **Table S3 "Curated pathway templates"** (mmc3.xlsx, Elsevier CDN: https://ars.els-cdn.com/content/image/1-s2.0-S0092867418303593-mmc3.xlsx).
- Columns: pathway, gene, role (OG/TSG from the "OG/TSG" column; NA/blank/Unknown mapped to `unknown`), source_url.
- Genes per pathway: Cell cycle 15, Hippo 38, Myc 13, Notch 71, Nrf2 3, PI3K 29, RTK-RAS 85, TGF-Beta 7, p53 6, Wnt 68.
- Completeness: complete per Table S3. Note that Table S3 contains the **extended** templates (334 genes). The "~187 genes" commonly quoted is the subset with an OG/TSG assignment used in the main figures. Here, 242 unique genes have OG/TSG labels, and 92 rows are `unknown` (mostly Notch and RTK-RAS extended members). For the core set, filter `role != unknown`. SCRIB appears in both Hippo and RTK-RAS.
- Cross-check: the PathwayMapper npm package (pathway-mapper@2.3.0, src/data/pathways.json) has the 10 TCGA pathway diagrams (165 nodes, including family nodes like "FZDs", "WNT ligands"). Table S3 is a superset of it.

## 2. drivers_bailey2018.csv  (299 rows)
- Source: Bailey M.H. et al. "Comprehensive Characterization of Cancer Driver Genes and Mutations." Cell 173(2):371-385 (2018). doi:10.1016/j.cell.2018.02.060. **Table S1 "Final gene consensus list"** (mmc1.xlsx: https://ars.els-cdn.com/content/image/1-s2.0-S009286741830237X-mmc1.xlsx).
- Table S1 has 739 gene x cancer pairs. These are collapsed to 299 unique genes.
- `role` comes from the column "Tumor suppressor or oncogene prediction (by 20/20+)" (paper's own 20/20+ labels). It is taken by majority vote across a gene's cohorts. `oncogene` and `tsg` are kept as is. "possible oncogene", "possible tsg" and empty values become `possible`. Counts: oncogene 70, tsg 86, possible 143. `role_detail_2020plus` keeps the raw label counts.
- `n_cancer_types_or_cohorts_if_available` = number of distinct cohorts (including PANCAN) in which the gene was called. `cohorts` lists them.

## 3. hotspots_chang2016.csv  (1165 rows, 247 genes)
- Source: cancerhotspots.org public API (https://www.cancerhotspots.org/api/hotspots/single). It holds the current release, v2 = Chang et al. 2018 Cancer Discov (24,592 tumors), which extends Chang et al. 2016 Nat Biotech 34:155-163 (doi:10.1038/nbt.3391).
- It is the full list: 1110 single-residue hotspots and 55 in-frame indel hotspots, sorted by tumor_count (descending). `residue` format is e.g. V600, R132, G12, or an indel range.
- Gap: this is v2, not exactly the 2016 v1 list (470 hotspots). v2 is a superset in spirit, though q-values were recomputed. The 3D hotspots (3dhotspots.org) are not included.

## 4. flags_shyr2014.csv  (100 rows)
- Source: Shyr C. et al. "FLAGS, frequently mutated genes in public exomes." BMC Med Genomics 7:64 (2014). doi:10.1186/s12920-014-0064-y. The list is the FLAGS column of supplementary file 12920_2014_64_MOESM4_ESM.txt (Additional file 1 / Table S1, datasets used). Retrieved from the PMC Open Access bucket: https://pmc-oa-opendata.s3.amazonaws.com/PMC4267152.1/12920_2014_64_MOESM4_ESM.txt
- Rank = row order (TTN, MUC16, OBSCN, AHNAK2, SYNE1, ...). This matches the paper's ranking with the most frequently mutated genes first.
- Gap: the "entire ranked list" (Table S4, MOESM3) is labeled as a PDF, but the OA copy of that file is a figure. So only the top 100 is provided here (as requested).

## 5. mmr_pole_genes.csv  (6 rows)
- MLH1, MSH2, MSH6, PMS2 (MMR → MSI-H) and POLE, POLD1 (proofreading → ultramutation), with hotspot notes.
- POLE exonuclease hotspots: P286R, V411L, S297F, A456P, S459F. Sources: TCGA UCEC 2013 (doi:10.1038/nature12113), TCGA CRC 2012 (doi:10.1038/nature11252), Rayner et al. 2016 Nat Rev Cancer (doi:10.1038/nrc.2016.51), Palles et al. 2013 Nat Genet (doi:10.1038/ng.2503, POLD1).
- cancerhotspots.org also lists POLE P286 (n=27), POLE V411 (n=18), MSH2 E580 and PMS2 K651.
- This file is hand-curated from the literature, not from a single table.

## Overlap with competition gene columns (train.csv header only; 4384 gene columns)
| list | genes | in train | 
|---|---|---|
| pathways (all) | 334 | 184 (Cell cycle 15/15, Hippo 14/38, Myc 6/13, Notch 42/71, Nrf2 2/3, PI3K 19/29, RTK-RAS 44/85, TGF-B 6/7, p53 4/6, Wnt 32/68) |
| Bailey drivers | 299 | 157 |
| hotspots (genes) | 247 | 137 (top-100 hotspots: 20/29 genes) |
| FLAGS top100 | 100 | 30 |
| MMR/POLE | 6 | 5 (MSH6 missing) |

- Train columns use **current HGNC symbols** (e.g., AARS1, H1-2, WWTR1). Old symbols in the 2014–2018 lists (HIST1H1C→H1-2, TAZ→WWTR1, GPR98→ADGRV1, H3F3A→H3-3A, TCEB1→ELOC, NOV→CCN3, PAK7→PAK5, FAM46D→TENT5D) need mapping before matching. Only a few of the mapped symbols are present, though.
- Many famous genes are **truly absent** from the columns (not a naming issue): KRAS, NRAS, KEAP1, ARID1A, SMAD4, STK11, TERT, FOXA1, KMT2C, BAP1, PBRM1, SETD2, GNAQ, GNA11, CIC, FUBP1, SF3B1, DNMT3A, FLT3, MSH6, TTN, MUC16. The column set looks like a curated subset, so features built from these lists only work through the genes that are present.
