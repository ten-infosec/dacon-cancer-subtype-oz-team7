# 참고문헌 (원문 확인한 것만)

## 변이 기반 암종 분류 (우리 조건과 같은 입력)
- Yuan Y. et al. **DeepGene: an advanced cancer type classifier based on deep learning and somatic point mutations.** BMC Bioinformatics 17(Suppl 17):476, 2016. — 12종, 유전자별 0/1, 정확도 65.5% (SVM 52.7%)
- Hasan M.A., Lonardi S. **mClass: Cancer type classification with somatic point mutation data.** RECOMB-CG 2018, LNBI 11183. — 12종, NMI 특성 선택 + 로지스틱 회귀, 정확도 67.5%
- Soh K.P. et al. **Predicting cancer type from tumour DNA signatures.** Genome Medicine 9:104, 2017. — 28종, 점돌연변이만 49.4% / +복제수 88.4%, 선형 SVM-RFE
- Lee K. et al. **CPEM: Accurate cancer type classification based on somatic alterations using an ensemble of a random forest and a deep neural network.** Scientific Reports 9:16927, 2019. — 31종, 변이율·스펙트럼·시그니처 추가로 46.9→84%
- Jiao W. et al. **A deep learning system accurately classifies primary and metastatic cancers using passenger mutation patterns.** Nature Communications 11:728, 2020. — 운전자 유전자만 F1 0.33, passenger 패턴이 핵심
- Sanjaya P. et al. **Mutation-Attention (MuAt).** 2023. — 변이 단위 트랜스포머, TCGA 엑솜 64%

## 이 대회 데이터를 쓴 논문
- 성다훈, 임유진. **암 유전체 데이터 처리 및 다중 분류 모델에 따른 성능 비교 연구.** 한국정보처리학회 ACK 2024, pp.723–726. — 원본 CatBoost 0.337, PCA+노이즈 증강 후 0.80 (재현 정보 미보고)

## 특성 설계 근거 (유전자 목록 → refs/)
- Sanchez-Vega F. et al. Oncogenic Signaling Pathways in TCGA. Cell 173:321, 2018. (Table S3)
- Bailey M.H. et al. Comprehensive Characterization of Cancer Driver Genes and Mutations. Cell 173:371, 2018. (Table S1)
- Chang M.T. et al. Identifying recurrent mutations in cancer reveals widespread lineage diversity. Nat Biotech 34:155, 2016. (cancerhotspots.org)
- Vogelstein B. et al. Cancer Genome Landscapes. Science 339:1546, 2013. (20/20 rule)
- Shyr C. et al. FLAGS, frequently mutated genes in public exomes. BMC Med Genomics 7:64, 2014.
- Campbell B.B. et al. Comprehensive Analysis of Hypermutation in Human Cancer. Cell 171:1042, 2017.

## 임상 연결
- Moon I. et al. Machine learning for genetics-based classification and treatment response prediction in cancer of unknown primary. Nature Medicine 29:2057, 2023. (OncoNPC)
- Chakravarty D. et al. OncoKB: A Precision Oncology Knowledge Base. JCO Precis Oncol 2017.
- Li M.M. et al. Standards and Guidelines for the Interpretation and Reporting of Sequence Variants in Cancer (AMP/ASCO/CAP). J Mol Diagn 19:4, 2017.

## 데이터 정의
- Broad GDAC Firehose — KIPAN = KICH+KIRC+KIRP, GBMLGG = GBM+LGG, STES = STAD+ESCA
