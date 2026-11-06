import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 유전자 변이 표기 파싱 및 도메인 피처 생성
# ==========================================
# 아미노산 1문자 표기와 자주 사용되는 변이 토큰을 인식하기 위한 정규식입니다.
AA_CODES = list("ACDEFGHIKLMNPQRSTVWY")
AA_CHANGE_RE = re.compile(r"(?<![A-Z])([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY*])", re.I)
FS_RE = re.compile(r"fs", re.I)
INDEL_RE = re.compile(r"del|ins|dup", re.I)
WT_TOKENS = {"WT", "WILDTYPE", "WILD TYPE", "NORMAL"}
MISSING_TOKENS = {"", "NA", "N/A", "NAN", "NONE", "NULL", "<NA>"}

PATHWAY_GENES = {
    "rtk_mapk": {"EGFR", "ERBB2", "ERBB3", "ERBB4", "FGFR1", "FGFR2", "FGFR3", "FGFR4", "MET", "ALK", "RET", "NTRK1", "KRAS", "NRAS", "HRAS", "BRAF", "MAP2K1"},
    "pi3k_akt_mtor": {"PIK3CA", "PIK3CB", "PIK3R1", "PIK3R2", "AKT1", "AKT2", "AKT3", "PTEN", "MTOR", "TSC1", "TSC2", "RICTOR"},
    "dna_repair": {"BRCA1", "BRCA2", "PALB2", "RAD51", "ATR", "ATM", "CHEK1", "CHEK2", "FANCA", "FANCC", "FANCD2", "MLH1", "MSH2", "MSH6", "PMS2"},
    "chromatin": {"ARID1A", "ARID1B", "PBRM1", "SMARCA4", "SMARCB1", "KMT2A", "KMT2B", "KMT2C", "KMT2D", "SETD2", "EZH2"},
    "wnt": {"APC", "CTNNB1", "AXIN1", "AXIN2", "RNF43", "AMER1", "TCF7L2"},
    "tgf_beta": {"TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4", "ACVR1B"},
    "cell_cycle_p53": {"TP53", "RB1", "CDKN2A", "CCND1", "CDK4", "CDK6", "MDM2", "CCNE1", "STK11"}
}

def extract_myeongseok_domain_features(df, gene_columns):
    """
    단백질 변이 표기를 파싱하고 오믹스 경로별 요약 피처를 생성
    (과적합을 유발하는 400개 아미노산 전이 고차원 피처는 제외하고 핵심 카운트 및 경로 점수 위주 채택)
    """
    # 행별로 변이 존재 여부와 변이 유형별 개수를 계산합니다.
    n_rows = len(df)
    raw = df.loc[:, gene_columns]
    presence = np.zeros((n_rows, len(gene_columns)), dtype=np.uint8)
    missing_cells = np.zeros(n_rows, dtype=np.int32)
    
    counts = {
        "missense_count": np.zeros(n_rows, dtype=np.int32),
        "stop_gain_count": np.zeros(n_rows, dtype=np.int32),
        "frameshift_count": np.zeros(n_rows, dtype=np.int32),
        "indel_marker_count": np.zeros(n_rows, dtype=np.int32),
        "same_residue_count": np.zeros(n_rows, dtype=np.int32),
        "unparsed_mutation_cell_count": np.zeros(n_rows, dtype=np.int32)
    }

    # 각 유전자 열을 순회하며 결측, 정상형, 변이 표기를 구분합니다.
    for col_index, gene in enumerate(gene_columns):
        values = raw[gene].astype("string").str.strip()
        normalized = values.str.upper()
        missing = normalized.isna() | normalized.isin(MISSING_TOKENS)
        missing_cells += missing.to_numpy(dtype=np.int32)
        mutated = values.notna() & ~missing & ~normalized.isin(WT_TOKENS)
        row_positions = np.flatnonzero(mutated.to_numpy())
        presence[row_positions, col_index] = 1

        for row_pos in row_positions:
            event_found = False
            seen_tokens = set()
            for token in re.split(r"[\s,;|]+", str(values.iloc[row_pos]).upper()):
                token = token.strip()
                if not token or token in seen_tokens:
                    continue
                seen_tokens.add(token)
                
                fs_hits = len(FS_RE.findall(token))
                if fs_hits:
                    counts["frameshift_count"][row_pos] += fs_hits
                    event_found = True
                    continue
                    
                changes = AA_CHANGE_RE.findall(token)
                if changes:
                    event_found = True
                    for ref_aa, _position, alt_aa in set(changes):
                        if alt_aa == "*":
                            counts["stop_gain_count"][row_pos] += 1
                        elif ref_aa == alt_aa:
                            counts["same_residue_count"][row_pos] += 1
                        else:
                            counts["missense_count"][row_pos] += 1
                    continue
                    
                if INDEL_RE.search(token):
                    counts["indel_marker_count"][row_pos] += 1
                    event_found = True
                    
                if not event_found:
                    counts["unparsed_mutation_cell_count"][row_pos] += 1

    feature_dict = {}
    feature_dict["mutated_gene_count"] = presence.sum(axis=1)
    feature_dict["missing_gene_cell_count"] = missing_cells
    feature_dict["missing_gene_cell_fraction"] = missing_cells / max(len(gene_columns), 1)
    
    for name, arr in counts.items():
        feature_dict[name] = arr
        
    feature_dict["truncating_marker_count"] = counts["stop_gain_count"] + counts["frameshift_count"]
    feature_dict["missense_per_mutated_gene"] = counts["missense_count"] / np.maximum(presence.sum(axis=1), 1)

    # 생물학적 경로별 변이 부담량
    available_gene_set = set(gene_columns)
    gene_index = {gene: i for i, gene in enumerate(gene_columns)}
    for pathway, panel in PATHWAY_GENES.items():
        available = [g for g in panel if g in available_gene_set]
        indices = [gene_index[g] for g in available]
        hits = presence[:, indices].sum(axis=1) if indices else np.zeros(n_rows, dtype=np.int32)
        feature_dict[f"pathway__{pathway}__mutated_count"] = hits
        feature_dict[f"pathway__{pathway}__fraction"] = hits / max(len(available), 1)

    # 생성한 배열을 원래 데이터와 같은 인덱스를 가진 피처 테이블로 변환합니다.
    df_out = pd.DataFrame(feature_dict, index=df.index).astype(np.float32)
    return presence, df_out

# ==========================================
# 2. 다중 오믹스 요약 피처(NMF 및 궤적 순위)
# ==========================================
def extract_multiomics_features(presence_matrix, train_mask):
    """
    NMF 기반 분자 패턴과 PCA 기반 상대 순위 피처를 생성
    """
    # 학습 데이터에만 NMF와 PCA를 적합해 테스트 데이터 정보가 학습에 사용되지 않도록 합니다.
    nmf = NMF(n_components=4, init='nndsvda', random_state=42)
    nmf.fit(presence_matrix[train_mask])
    all_nmf = nmf.transform(presence_matrix)
    
    df_nmf = pd.DataFrame(
        all_nmf, 
        columns=['NMF1_INE', 'NMF2_NE', 'NMF3_OXPHOS', 'NMF4_MYC']
    )
    
    pca = PCA(n_components=1, random_state=42)
    pca.fit(presence_matrix[train_mask])
    all_rank = pca.transform(presence_matrix)
    
    scaler = MinMaxScaler()
    scaler.fit(all_rank[train_mask])
    scaled_rank = scaler.transform(all_rank).flatten()
    df_nmf['Pseudostage_Rank'] = scaled_rank
    
    df_nmf['TypeIV_Infiltrative_Score'] = df_nmf['NMF3_OXPHOS'] * 0.5 + df_nmf['NMF4_MYC'] * 0.5 + df_nmf['Pseudostage_Rank'] * 0.3
    df_nmf['Immune_Exclusion_Index'] = (df_nmf['NMF4_MYC'] + 1e-4) / (df_nmf['NMF1_INE'] + 1e-4)
    
    return df_nmf

# ==========================================
# 3. 다층 텍스트 토큰화 및 TF-IDF
# ==========================================
def create_text_documents(df, gene_columns):
    """
    유전자 변이를 텍스트 문서로 확장 (Word/Char TF-IDF 입력용)
    """
    # 한 행의 유전자 변이 정보를 토큰 문서로 구성합니다.
    docs = []
    for idx, row in df[gene_columns].iterrows():
        tokens = []
        for gene in gene_columns:
            val = str(row[gene]).strip().upper()
            if val and val not in WT_TOKENS and val not in MISSING_TOKENS:
                tokens.append(f"G={gene}")
                tokens.append(f"GT={gene}:{val}")
                for sub in re.split(r"[\s,;|]+", val):
                    if sub:
                        tokens.append(f"E={gene}:{sub}")
        docs.append(" ".join(tokens))
    return docs

# ==========================================
# 4. 계층 관계를 고려한 레이블 후처리(KIPAN/GBMLGG)
# ==========================================
def apply_hierarchy_post_processing(prob_matrix, classes, margin_threshold=0.15):
    """
    상위·하위 코호트 레이블 간 예측 확률 차이가 작을 때 조건부로 보정
    """
    # 우선 가장 높은 확률의 레이블을 선택한 뒤, 지정한 계층 관계만 보정합니다.
    preds = np.argmax(prob_matrix, axis=1)
    final_preds = []
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    kipan_i = class_to_idx.get('KIPAN', None)
    kirc_i = class_to_idx.get('KIRC', None)
    gbmlgg_i = class_to_idx.get('GBMLGG', None)
    lgg_i = class_to_idx.get('LGG', None)
    
    for row_i, p_idx in enumerate(preds):
        probs = prob_matrix[row_i]
        top_class = classes[p_idx]
        
        # KIPAN과 KIRC의 확률 차이가 작을 때 하위 레이블을 우선
        if top_class in ['KIPAN', 'KIRC'] and (kipan_i is not None and kirc_i is not None):
            if abs(probs[kipan_i] - probs[kirc_i]) < margin_threshold:
                if probs[kirc_i] >= probs[kipan_i] * 0.85:
                    p_idx = kirc_i
        # GBMLGG와 LGG의 확률 차이가 작을 때 하위 레이블을 우선
        elif top_class in ['GBMLGG', 'LGG'] and (gbmlgg_i is not None and lgg_i is not None):
            if abs(probs[gbmlgg_i] - probs[lgg_i]) < margin_threshold:
                if probs[lgg_i] >= probs[gbmlgg_i] * 0.85:
                    p_idx = lgg_i
                    
        final_preds.append(classes[p_idx])
        
    return np.array(final_preds)

# ==========================================
# 5. 전체 파이프라인 실행
# ==========================================
def run_myeongseok_v3_pipeline(train_csv="./train.csv", test_csv="./test.csv", sub_csv="./sample_submission.csv"):
    print("Running the classification pipeline...")
    
    # 입력 파일이 없으면 파이프라인 구조를 확인할 수 있도록 예시 데이터를 생성합니다.
    if not os.path.exists(train_csv):
        print("Simulation mode: creating mock data for demonstration...")
        np.random.seed(42)
        n_tr, n_te = 600, 200
        gene_cols = [f"GENE_{i:04d}" for i in range(50)]
        c_labels = ['ACC', 'BLCA', 'BRCA', 'GBMLGG', 'KIPAN', 'KIRC', 'LGG', 'SKCM', 'STES', 'THCA', 'UCEC', 'DLBC']
        
        tr_data = {g: np.random.choice(['WT', 'P.V600E', 'R175H', 'FS', 'DEL', ''], size=n_tr, p=[0.85, 0.03, 0.03, 0.03, 0.03, 0.03]) for g in gene_cols}
        tr_data['ID'] = [f"TRAIN_{i:04d}" for i in range(n_tr)]
        tr_data['SUBCLASS'] = np.random.choice(c_labels, size=n_tr)
        train_df = pd.DataFrame(tr_data)
        
        te_data = {g: np.random.choice(['WT', 'P.V600E', 'R175H', 'FS', 'DEL', ''], size=n_te, p=[0.85, 0.03, 0.03, 0.03, 0.03, 0.03]) for g in gene_cols}
        te_data['ID'] = [f"TEST_{i:04d}" for i in range(n_te)]
        test_df = pd.DataFrame(te_data)
        
        sub_df = pd.DataFrame({'ID': test_df['ID'], 'SUBCLASS': 'LGG'})
    else:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        sub_df = pd.read_csv(sub_csv)

    TARGET_COL = 'SUBCLASS'
    ID_COL = 'ID'
    # ID와 정답 열은 모델 입력에서 제외하고 유전자 열만 사용합니다.
    gene_cols = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]
    
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[TARGET_COL])
    classes = le.classes_
    
    # 동일한 전처리를 위해 학습·테스트의 유전자 열을 이어 붙입니다.
    combined_df = pd.concat([train_df[gene_cols], test_df[gene_cols]], ignore_index=True)
    n_train = len(train_df)
    train_mask = np.zeros(len(combined_df), dtype=bool)
    train_mask[:n_train] = True
    
    print("1. Extracting mutation and pathway features...")
    presence_matrix, myeongseok_feats = extract_myeongseok_domain_features(combined_df, gene_cols)
    
    print("2. Computing NMF and relative-rank features...")
    multiomics_feats = extract_multiomics_features(presence_matrix, train_mask)
    
    scaler = StandardScaler()
    tab_all = pd.concat([myeongseok_feats, multiomics_feats], axis=1)
    scaler.fit(tab_all[train_mask])                      # FIX: 평균·표준편차를 train으로만 계산
    tab_features_scaled = scaler.transform(tab_all)
    
    X_train_tab = tab_features_scaled[:n_train]
    X_test_tab = tab_features_scaled[n_train:]
    
    print("3. Constructing the TF-IDF text matrix...")
    text_docs = create_text_documents(combined_df, gene_cols)
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2)
    tfidf.fit(text_docs[:n_train])                      # FIX: train 문서로만 어휘·IDF 학습 (test로 fit 금지)
    X_text_all = tfidf.transform(text_docs)
    X_train_text = X_text_all[:n_train]
    X_test_text = X_text_all[n_train:]
    
    print("4. Training base models with 5-fold cross-validation...")
    import hashlib
    groups = train_df[gene_cols].fillna("WT").apply(lambda r: hashlib.md5("|".join(r.values).encode()).hexdigest(), axis=1).values
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)   # FIX: 동일 프로필은 같은 fold
    
    oof_xgb = np.zeros((n_train, len(classes)))
    oof_svc = np.zeros((n_train, len(classes)))
    oof_et = np.zeros((n_train, len(classes)))
    
    test_xgb = np.zeros((len(test_df), len(classes)))
    test_svc = np.zeros((len(test_df), len(classes)))
    test_et = np.zeros((len(test_df), len(classes)))
    
    # 각 폴드에서 기본 모델을 학습하고 검증 예측과 테스트 예측을 누적합니다.
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tab, y_train, groups)):
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='mlogloss',
            random_state=42
        )
        xgb_model.fit(X_train_tab[train_idx], y_train[train_idx])
        oof_xgb[val_idx] = xgb_model.predict_proba(X_train_tab[val_idx])
        test_xgb += xgb_model.predict_proba(X_test_tab) / 5.0
        
        svc = LinearSVC(C=0.5, class_weight='balanced', random_state=42)
        svc.fit(X_train_text[train_idx], y_train[train_idx])
        svc_dec_val = svc.decision_function(X_train_text[val_idx])
        svc_dec_test = svc.decision_function(X_test_text)
        oof_svc[val_idx] = np.exp(svc_dec_val) / np.sum(np.exp(svc_dec_val), axis=1, keepdims=True)
        test_svc += (np.exp(svc_dec_test) / np.sum(np.exp(svc_dec_test), axis=1, keepdims=True)) / 5.0
        
        et = ExtraTreesClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42)
        et.fit(X_train_tab[train_idx], y_train[train_idx])
        oof_et[val_idx] = et.predict_proba(X_train_tab[val_idx])
        test_et += et.predict_proba(X_test_tab) / 5.0
        
    print("5. Combining base-model outputs with logistic regression...")
    # 기본 모델의 검증 예측을 메타 모델의 입력으로 사용합니다.
    meta_train = np.hstack([oof_xgb, oof_svc, oof_et])
    meta_test = np.hstack([test_xgb, test_svc, test_et])
    
    meta_model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=42)
    meta_model.fit(meta_train, y_train)
    
    oof_meta_probs = cross_val_predict(LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=42),
                                       meta_train, y_train, cv=StratifiedGroupKFold(5, shuffle=True, random_state=7), groups=groups, method='predict_proba')  # FIX
    test_meta_probs = meta_model.predict_proba(meta_test)
    
    raw_oof_preds = classes[np.argmax(oof_meta_probs, axis=1)]
    raw_oof_f1 = f1_score(train_df[TARGET_COL], raw_oof_preds, average='macro')
    print(f"   - Stacking Raw OOF Macro F1: {raw_oof_f1:.4f}")
    
    print("6. Applying hierarchy-based post-processing...")
    post_oof_preds = apply_hierarchy_post_processing(oof_meta_probs, classes)
    post_oof_f1 = f1_score(train_df[TARGET_COL], post_oof_preds, average='macro')
    print(f"   - Post-processed OOF Macro F1: {post_oof_f1:.4f}")
    
    # 메타 모델의 테스트 확률에도 같은 후처리 규칙을 적용합니다.
    final_test_preds = apply_hierarchy_post_processing(test_meta_probs, classes)
    sub_df['SUBCLASS'] = final_test_preds
    
    from sklearn.metrics import classification_report
    rep = pd.DataFrame(classification_report(train_df[TARGET_COL], post_oof_preds, output_dict=True, zero_division=0)).T
    print(rep.loc[classes, ['f1-score','support']].sort_values('f1-score').round(3).to_string())
    print("test 예측 분포 %:", (pd.Series(final_test_preds).value_counts(normalize=True)*100).round(1).head(6).to_dict())
    out_file = "./1106_myeongseok_v3_fixed.csv"
    sub_df.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f" Predictions exported to: {out_file}")

if __name__ == "__main__":
    run_myeongseok_v3_pipeline("/home/claude/data_raw/train.csv","/home/claude/data_raw/test.csv","/home/claude/data_raw/sample_submission.csv")
