# build_patient_text.py

import json
import re
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TEXT_DIR, TARGET_COL, SPLIT_STRATEGY, SPLITS


# =====================================================
# CONFIGURACIÓN SEMÁNTICA DE VARIABLES
# =====================================================

FORCE_NUMERIC_COLS = [
    "Edad al ingreso",
    "Frecuencia cardiaca",
    "Presión diastólica cardíaca",
    "Presión sistólica cardíaca",
    "Peso",
    "PESO (Kg)",
    "Talla",
    "TALLA (m)",
    "IMC",
    "FEVI",
    "Índice de comorbilidad de Charlson",
    "Potasio",
    "Pro-BNP",
    "Creatinina",
    "Glóbulos rojos",
    "Leucocitos",
    "Hemoglobina",
    "Volumen corpuscular medio",
    "Hemoglobina corpuscular media",
    "Concentración de hemoglobina corpuscular media",
    "RDW",
    "Volumen plaquetario",
    "Neutrófilos",
    "Linfocitos",
    "Monocitos",
    "Eosinófilos",
    "Basófilos",
    "MLR",
    "NLR",
    "PLR",
    "SII",
    "AISI",
    "SIRI",
    "BUN",
    "MPR",
    "NPR",
    "MNR",
    "RPR",
    "Estancia hospitalaria",
]

SPECIAL_BINARY_MAPPINGS = {
    "Sexo": {
        0: "female",
        1: "male",
        "sentence": "sex: {value}",
        "group": "demographics",
    },
    "Área de residencia": {
        0: "rural",
        1: "urban",
        "sentence": "area of residence: {value}",
        "group": "demographics",
    },
    "Area de residencia": {
        0: "rural",
        1: "urban",
        "sentence": "area of residence: {value}",
        "group": "demographics",
    },
}

ONE_HOT_GROUPS = {
    "Tipo de seguridad social": {
        "prefixes": [
            "Tipo de seguridad social",
            "Tipo seguridad social",
            "Tipo de Seguridad Social",
        ],
        "mapping": {
            0: "subsidized insurance",
            1: "contributory insurance",
            2: "special insurance",
        },
        "sentence": "health insurance type: {value}",
        "group": "demographics",
    },
    "Escolaridad": {
        "prefixes": [
            "Escolaridad",
            "escolaridad",
        ],
        "mapping": {
            0: "none or primary education",
            1: "secondary education",
            2: "technical education",
            3: "university education",
            4: "unknown education level",
            5: "unknown education level",
        },
        "sentence": "education level: {value}",
        "group": "demographics",
    },
    "Estado civil": {
        "prefixes": [
            "Estado civil",
            "estado civil",
        ],
        "mapping": {
            0: "with partner",
            1: "without partner",
            2: "not reported",
        },
        "sentence": "marital status: {value}",
        "group": "demographics",
    },
}

CLINICAL_BINARY_KEYWORDS = [
    "antecedente",
    "uso de",
    "ingreso a uci",
    "uci",
    "arritmia",
    "hipertensión",
    "hipertension",
    "diabetes",
    "dislipidemia",
    "coronaria",
    "renal",
    "vascular",
    "pulmonar",
    "autoinmune",
    "hipotiroidismo",
    "insuficiencia",
    "demencia",
    "hepática",
    "hepatica",
    "antiarrítmicos",
    "antiarritmicos",
    "anticoagulantes",
    "betabloqueadores",
    "sglt2",
    "aldosterona",
    "renina",
]

# Variables sin cortes clínicos universales robustos.
# Se categorizan por cuantiles del train.
TRAIN_QUANTILE_COLS = [
    "MLR",
    "NLR",
    "PLR",
    "SII",
    "AISI",
    "SIRI",
    "MPR",
    "NPR",
    "MNR",
    "RPR",
]

# Variables que se reportan de forma neutral.
RAW_VALUE_COLS = [
    "Peso",
    "PESO (Kg)",
    "Talla",
    "TALLA (m)",
]

ENGLISH_COLNAMES = {
    "Edad al ingreso": "age at admission",
    "Frecuencia cardiaca": "heart rate",
    "Presión diastólica cardíaca": "diastolic blood pressure",
    "Presión sistólica cardíaca": "systolic blood pressure",
    "Peso": "weight",
    "PESO (Kg)": "weight",
    "Talla": "height",
    "TALLA (m)": "height",
    "IMC": "body mass index",
    "FEVI": "left ventricular ejection fraction",
    "Índice de comorbilidad de Charlson": "Charlson comorbidity index",
    "Potasio": "potassium",
    "Pro-BNP": "pro-BNP",
    "Creatinina": "creatinine",
    "Glóbulos rojos": "red blood cell count",
    "Leucocitos": "white blood cell count",
    "Hemoglobina": "hemoglobin",
    "Volumen corpuscular medio": "mean corpuscular volume",
    "Hemoglobina corpuscular media": "mean corpuscular hemoglobin",
    "Concentración de hemoglobina corpuscular media": "mean corpuscular hemoglobin concentration",
    "RDW": "red cell distribution width",
    "Volumen plaquetario": "mean platelet volume",
    "Neutrófilos": "neutrophils",
    "Linfocitos": "lymphocytes",
    "Monocitos": "monocytes",
    "Eosinófilos": "eosinophils",
    "Basófilos": "basophils",
    "BUN": "blood urea nitrogen",
    "Estancia hospitalaria": "hospital length of stay",
    "MLR": "monocyte-to-lymphocyte ratio",
    "NLR": "neutrophil-to-lymphocyte ratio",
    "PLR": "platelet-to-lymphocyte ratio",
    "SII": "systemic immune-inflammation index",
    "AISI": "aggregate index of systemic inflammation",
    "SIRI": "systemic inflammation response index",
    "MPR": "mean platelet volume-to-platelet ratio",
    "NPR": "neutrophil-to-platelet ratio",
    "MNR": "monocyte-to-neutrophil ratio",
    "RPR": "red cell distribution width-to-platelet ratio",
    "Antecedente de arritmia": "history of arrhythmia",
    "Antecedente de hipertensión arterial": "history of arterial hypertension",
    "Antecedente de diabetes mellitu": "history of diabetes mellitus",
    "Antecedente de dislipidemia": "history of dyslipidemia",
    "Antecedente de enfermedad coronaria": "history of coronary artery disease",
    "Antecedente de enfermedad renal crónica": "history of chronic kidney disease",
    "Antecedente de hipotiroidismo": "history of hypothyroidism",
    "Antecedente de hipertensión pulmonar": "history of pulmonary hypertension",
    "Antecedente de insuficiencia cardiaca congestiva": "history of congestive heart failure",
    "Antecedente de enfermedad vascular periférica": "history of peripheral vascular disease",
    "Antecedente de demencia": "history of dementia",
    "Antecedente de enfermedad pulmonar obstructiva crónica": "history of chronic obstructive pulmonary disease",
    "Antecedente de enfermedad autoinmune": "history of autoimmune disease",
    "Antecedente de patología hepática leve": "history of mild liver disease",
    "Antecedente de patología hepática moderada/grave": "history of moderate or severe liver disease",
    "Uso de inhibidores del sistema renina-angiotensina": "renin-angiotensin system inhibitor use",
    "Uso de betabloqueadores": "beta blocker use",
    "Uso de inhibidores de la aldosterona": "aldosterone antagonist use",
    "Uso de inhibidores SGLT2": "SGLT2 inhibitor use",
    "Uso de anticoagulantes": "anticoagulant use",
    "Uso de antiarrítmicos": "antiarrhythmic use",
    "Ingreso a UCI": "ICU admission",
}


# =====================================================
# FUNCIONES AUXILIARES
# =====================================================

def clean_colname(col):
    return str(col).strip().replace("_", " ")


def normalize_text(text):
    text = str(text).strip().lower()

    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)

    return text


def english_colname(col):
    col_clean = clean_colname(col)
    return ENGLISH_COLNAMES.get(col_clean, col_clean)


def parse_int_like(x):
    if pd.isna(x):
        return None

    try:
        return int(float(x))
    except Exception:
        return None


def get_sex_value(row):
    """
    Sexo: 1 = male, 0 = female.
    Retorna: 'male', 'female' o None.
    """
    if row is None:
        return None

    if "Sexo" not in row.index or pd.isna(row["Sexo"]):
        return None

    try:
        sexo = int(float(row["Sexo"]))

        if sexo == 1:
            return "male"
        if sexo == 0:
            return "female"

    except Exception:
        return None

    return None


def get_age_value(row):
    if row is None:
        return None

    if "Edad al ingreso" not in row.index or pd.isna(row["Edad al ingreso"]):
        return None

    try:
        return float(row["Edad al ingreso"])
    except Exception:
        return None


def is_column(col, names):
    col_norm = normalize_text(clean_colname(col))
    return any(col_norm == normalize_text(name) for name in names)


def contains_col(col, patterns):
    col_norm = normalize_text(clean_colname(col))
    return any(normalize_text(pattern) in col_norm for pattern in patterns)


# =====================================================
# RANGOS CLÍNICOS EN INGLÉS
# =====================================================

def classify_quantile_based(x, q):
    """
    q debe contener q20, q40, q60, q80.
    Usado para índices derivados sin rangos clínicos universales.
    """

    if pd.isna(x):
        return None

    if q is None:
        return "recorded"

    if x <= q["q20"]:
        return "very low within this cohort"
    elif x <= q["q40"]:
        return "low within this cohort"
    elif x <= q["q60"]:
        return "intermediate within this cohort"
    elif x <= q["q80"]:
        return "high within this cohort"
    else:
        return "very high within this cohort"


def clinical_numeric_to_text(col, x, row=None, quantiles=None):
    """
    English clinical categorization for Clinical-LongFormer.
    Keeps the original clinical logic but outputs English clinical pseudo-note text.
    """

    if pd.isna(x):
        return None

    sex = get_sex_value(row)
    age = get_age_value(row)

    # Edad
    if is_column(col, ["Edad al ingreso"]):
        if x < 40:
            return f"{x:.0f} years old, young adult"
        elif x < 60:
            return f"{x:.0f} years old, middle-aged adult"
        elif x < 75:
            return f"{x:.0f} years old, older adult"
        else:
            return f"{x:.0f} years old, very elderly adult"

    # Frecuencia cardiaca
    if contains_col(col, ["frecuencia cardiaca"]):
        if x < 50:
            category = "very low"
        elif x < 60:
            category = "low"
        elif x <= 100:
            category = "normal"
        elif x <= 120:
            category = "high"
        else:
            category = "very high"
        return f"{x:.0f} beats per minute, {category}"

    # Presión sistólica
    if contains_col(col, ["presion sistolica cardiaca", "presión sistólica cardíaca"]):
        if x < 90:
            category = "very low"
        elif x < 120:
            category = "normal"
        elif x < 130:
            category = "elevated"
        elif x < 140:
            category = "high"
        else:
            category = "very high"
        return f"{x:.0f} mmHg, {category}"

    # Presión diastólica
    if contains_col(col, ["presion diastolica cardiaca", "presión diastólica cardíaca"]):
        if x < 60:
            category = "very low"
        elif x < 80:
            category = "normal"
        elif x < 90:
            category = "high"
        else:
            category = "very high"
        return f"{x:.0f} mmHg, {category}"

    # Peso y talla
    if is_column(col, RAW_VALUE_COLS):
        return f"{x:.2f}, recorded"

    # IMC
    if is_column(col, ["IMC"]):
        if x < 18.5:
            category = "underweight"
        elif x < 25:
            category = "normal"
        elif x < 30:
            category = "overweight"
        elif x < 35:
            category = "class 1 obesity"
        elif x < 40:
            category = "class 2 obesity"
        else:
            category = "class 3 obesity"
        return f"{x:.1f} kg/m2, {category}"

    # FEVI
    if is_column(col, ["FEVI"]):
        if x < 30:
            category = "severely reduced"
        elif x <= 40:
            category = "reduced"
        elif x < 50:
            category = "mildly reduced"
        else:
            category = "preserved"
        return f"{x:.0f} percent, {category}"

    # Charlson
    if contains_col(col, ["charlson"]):
        if x == 0:
            category = "no comorbidity"
        elif x <= 2:
            category = "low"
        elif x <= 4:
            category = "moderate"
        elif x <= 6:
            category = "high"
        else:
            category = "very high"
        return f"{x:.0f}, {category}"

    # Potasio
    if is_column(col, ["Potasio"]):
        if x < 3.0:
            category = "very low"
        elif x < 3.5:
            category = "low"
        elif x <= 5.0:
            category = "normal"
        elif x < 6.0:
            category = "high"
        else:
            category = "very high"
        return f"{x:.2f} mEq/L, {category}"

    # Pro-BNP
    if contains_col(col, ["pro-bnp", "probnp", "nt-probnp", "nt probnp"]):
        if age is not None and age > 75:
            if x <= 450:
                category = "low for advanced age"
            elif x <= 1800:
                category = "elevated"
            else:
                category = "very elevated"
        else:
            if x <= 125:
                category = "low"
            elif x <= 450:
                category = "elevated"
            elif x <= 900:
                category = "high"
            else:
                category = "very high"
        return f"{x:.1f} pg/mL, {category}"

    # Creatinina
    if is_column(col, ["Creatinina"]):
        if sex == "female":
            if x < 0.5:
                category = "very low"
            elif x <= 0.70:
                category = "low"
            elif x <= 0.97:
                category = "normal"
            elif x < 1.05:
                category = "slightly high"
            elif x <= 1.5:
                category = "high"
            else:
                category = "very high"

        elif sex == "male":
            if x < 0.6:
                category = "very low"
            elif x <= 0.70:
                category = "low"
            elif x <= 1.10:
                category = "normal"
            elif x < 1.15:
                category = "slightly high"
            elif x <= 1.5:
                category = "high"
            else:
                category = "very high"

        else:
            if x < 0.6:
                category = "very low"
            elif x <= 0.70:
                category = "low"
            elif x <= 1.10:
                category = "normal"
            elif x <= 1.5:
                category = "high"
            else:
                category = "very high"

        return f"{x:.2f} mg/dL, {category}"

    # Glóbulos rojos
    if contains_col(col, ["globulos rojos", "glóbulos rojos"]):
        if sex == "female":
            if x < 3.8:
                category = "low"
            elif x <= 5.2:
                category = "normal"
            else:
                category = "high"

        elif sex == "male":
            if x < 4.2:
                category = "low"
            elif x <= 6.0:
                category = "normal"
            else:
                category = "high"

        else:
            if x < 4.0:
                category = "low"
            elif x <= 5.8:
                category = "normal"
            else:
                category = "high"

        return f"{x:.2f} million/uL, {category}"

    # Leucocitos
    if is_column(col, ["Leucocitos"]):
        val = x
        if x < 100:
            val = x * 1000.0

        if val < 3000:
            category = "very low"
        elif val < 4000:
            category = "low"
        elif val <= 11000:
            category = "normal"
        elif val <= 15000:
            category = "high"
        else:
            category = "very high"

        return f"{val:.0f} cells/uL, {category}"

    # Hemoglobina
    if is_column(col, ["Hemoglobina"]):
        if sex == "female":
            if x < 10:
                category = "very low"
            elif x < 12:
                category = "low"
            elif x <= 15.0:
                category = "normal"
            else:
                category = "high"

        elif sex == "male":
            if x < 11:
                category = "very low"
            elif x < 13.5:
                category = "low"
            elif x <= 18.0:
                category = "normal"
            else:
                category = "high"

        else:
            if x < 11:
                category = "very low"
            elif x < 13:
                category = "low"
            elif x <= 17:
                category = "normal"
            else:
                category = "high"

        return f"{x:.1f} g/dL, {category}"

    # MCV
    if contains_col(col, ["volumen corpuscular medio"]):
        if x < 75:
            category = "very low"
        elif x < 80:
            category = "low"
        elif x <= 100:
            category = "normal"
        elif x <= 110:
            category = "high"
        else:
            category = "very high"

        return f"{x:.1f} fL, {category}"

    # MCH
    if is_column(col, ["Hemoglobina corpuscular media"]):
        if x < 27:
            category = "low"
        elif x <= 33:
            category = "normal"
        else:
            category = "high"

        return f"{x:.1f} pg, {category}"

    # MCHC
    if contains_col(
        col,
        [
            "concentracion de hemoglobina corpuscular media",
            "concentración de hemoglobina corpuscular media",
        ],
    ):
        if x < 30.5:
            category = "low"
        elif x < 33:
            category = "low-normal"
        elif x <= 35.5:
            category = "normal"
        elif x >= 36.5:
            category = "high"
        else:
            category = "slightly high"

        return f"{x:.1f} g/dL, {category}"

    # RDW
    if is_column(col, ["RDW"]):
        if x <= 14.5:
            category = "normal"
        elif x <= 18:
            category = "high"
        else:
            category = "very high"

        return f"{x:.1f} percent, {category}"

    # MPV
    if contains_col(col, ["volumen plaquetario"]):
        if x < 7.5:
            category = "low"
        elif x <= 11.5:
            category = "normal"
        else:
            category = "high"

        return f"{x:.1f} fL, {category}"

    # Neutrófilos
    if is_column(col, ["Neutrófilos"]):
        if x < 40:
            category = "very low"
        elif x < 50:
            category = "low"
        elif x <= 70:
            category = "normal"
        elif x <= 80:
            category = "high"
        else:
            category = "very high"

        return f"{x:.1f} percent, {category}"

    # Linfocitos
    if is_column(col, ["Linfocitos"]):
        if x < 15:
            category = "very low"
        elif x < 20:
            category = "low"
        elif x <= 40:
            category = "normal"
        elif x <= 50:
            category = "high"
        else:
            category = "very high"

        return f"{x:.1f} percent, {category}"

    # Monocitos
    if is_column(col, ["Monocitos"]):
        if x < 2:
            category = "low"
        elif x <= 8:
            category = "normal"
        else:
            category = "high"

        return f"{x:.1f} percent, {category}"

    # Eosinófilos
    if is_column(col, ["Eosinófilos"]):
        category = "normal" if x <= 5 else "high"
        return f"{x:.1f} percent, {category}"

    # Basófilos
    if is_column(col, ["Basófilos"]):
        category = "normal" if x <= 1 else "high"
        return f"{x:.1f} percent, {category}"

    # BUN
    if is_column(col, ["BUN"]):
        if x < 6:
            category = "very low"
        elif x < 10:
            category = "low"
        elif x <= 20:
            category = "normal"
        elif x <= 25.6:
            category = "mildly high"
        elif x <= 40:
            category = "high"
        else:
            category = "very high"

        return f"{x:.1f} mg/dL, {category}"

    # Índices derivados por cuantiles
    if is_column(col, TRAIN_QUANTILE_COLS):
        q = None if quantiles is None else quantiles.get(col)
        return classify_quantile_based(x, q)

    # Estancia hospitalaria
    if contains_col(col, ["estancia hospitalaria"]):
        if x <= 3:
            category = "short"
        elif x <= 7:
            category = "intermediate"
        elif x <= 14:
            category = "prolonged"
        else:
            category = "very prolonged"

        return f"{x:.0f} days, {category}"

    return f"{x:.2f}, recorded"


# =====================================================
# DETECCIÓN DE TIPOS
# =====================================================

def is_binary_series(s):
    values = pd.Series(s.dropna().unique())

    if len(values) == 0:
        return False

    try:
        values_as_int = set(values.astype(float).astype(int).tolist())
        original_as_float = set(values.astype(float).tolist())

        return original_as_float.issubset({0.0, 1.0}) and values_as_int.issubset({0, 1})
    except Exception:
        return False


def get_existing_force_numeric_cols(df, feature_cols):
    return [
        col
        for col in FORCE_NUMERIC_COLS
        if col in feature_cols and col in df.columns
    ]


def find_one_hot_columns(feature_cols, prefixes):
    found = []

    for col in feature_cols:
        col_clean = clean_colname(col)

        for prefix in prefixes:
            prefix_clean = clean_colname(prefix)
            pattern = rf"^{re.escape(prefix_clean)}[\s_.-]+(\d+)$"
            match = re.match(pattern, col_clean)

            if match:
                category_code = int(match.group(1))
                found.append((col, category_code))
                break

    return found


def detect_one_hot_columns(feature_cols):
    one_hot_cols = set()
    one_hot_info = {}

    for group_name, cfg in ONE_HOT_GROUPS.items():
        cols = find_one_hot_columns(feature_cols, cfg["prefixes"])

        if len(cols) > 0:
            one_hot_info[group_name] = cols

            for col, _ in cols:
                one_hot_cols.add(col)

    return one_hot_cols, one_hot_info


def is_clinical_binary_col(col):
    col_lower = normalize_text(clean_colname(col))
    return any(normalize_text(keyword) in col_lower for keyword in CLINICAL_BINARY_KEYWORDS)


def infer_column_types(train_df, feature_cols):
    force_numeric_cols = set(get_existing_force_numeric_cols(train_df, feature_cols))
    one_hot_cols, one_hot_info = detect_one_hot_columns(feature_cols)

    special_binary_cols = set([
        col for col in SPECIAL_BINARY_MAPPINGS.keys()
        if col in feature_cols
    ])

    numeric_cols = []
    special_binary_cols_found = []
    clinical_binary_cols = []
    categorical_cols = []

    for col in feature_cols:
        if col in one_hot_cols:
            continue

        if col in force_numeric_cols:
            numeric_cols.append(col)
            continue

        if col in special_binary_cols:
            special_binary_cols_found.append(col)
            continue

        s = train_df[col]

        if pd.api.types.is_numeric_dtype(s):
            if is_binary_series(s) and is_clinical_binary_col(col):
                clinical_binary_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric_cols": numeric_cols,
        "special_binary_cols": special_binary_cols_found,
        "clinical_binary_cols": clinical_binary_cols,
        "categorical_cols": categorical_cols,
        "one_hot_info": one_hot_info,
    }


def build_quantiles(train_df, numeric_cols):
    quantiles = {}

    for col in numeric_cols:
        if not is_column(col, TRAIN_QUANTILE_COLS):
            continue

        values = pd.to_numeric(train_df[col], errors="coerce").dropna()

        if len(values) < 5:
            continue

        qs = values.quantile([0.20, 0.40, 0.60, 0.80])

        quantiles[col] = {
            "q20": float(qs.loc[0.20]),
            "q40": float(qs.loc[0.40]),
            "q60": float(qs.loc[0.60]),
            "q80": float(qs.loc[0.80]),
        }

    return quantiles


def add_sentence_to_group(groups, sentence, group):
    if group not in groups:
        group = "others"

    groups[group].append(sentence)


# =====================================================
# CONSTRUCCIÓN DE TEXTO
# =====================================================

def build_one_hot_sentences(row, one_hot_info):
    sentences = []

    for group_name, cols in one_hot_info.items():
        cfg = ONE_HOT_GROUPS[group_name]
        active_codes = []

        for col, code in cols:
            value = parse_int_like(row[col])

            if value == 1:
                active_codes.append(code)

        if len(active_codes) == 0:
            continue

        category_texts = [
            cfg["mapping"].get(code, f"category {code}")
            for code in sorted(active_codes)
        ]

        value_text = " and ".join(category_texts)
        sentence = cfg["sentence"].format(value=value_text)

        sentences.append((sentence, cfg["group"]))

    return sentences


def build_special_binary_sentence(col, value):
    cfg = SPECIAL_BINARY_MAPPINGS[col]
    v = parse_int_like(value)

    if v is None:
        return None, cfg["group"]

    value_text = cfg.get(v, f"code {v}")
    sentence = cfg["sentence"].format(value=value_text)

    return sentence, cfg["group"]


def build_clinical_binary_sentence(col, value):
    v = parse_int_like(value)

    if v is None:
        return None

    if v == 1:
        value_text = "present"
    elif v == 0:
        value_text = "absent"
    else:
        value_text = f"code {v}"

    return f"{english_colname(col)}: {value_text}"


def infer_group_for_column(col):
    col_clean = clean_colname(col)
    col_lower = normalize_text(col_clean)

    demographic_keywords = [
        "sexo",
        "edad",
        "escolaridad",
        "estado civil",
        "seguridad",
        "residencia",
        "area",
        "área",
    ]

    vital_keywords = [
        "frecuencia",
        "presion",
        "presión",
        "peso",
        "talla",
        "imc",
    ]

    clinical_keywords = [
        "aha",
        "nyha",
        "fevi",
        "arritmia",
        "hipertension",
        "hipertensión",
        "diabetes",
        "dislipidemia",
        "coronaria",
        "renal",
        "enfermedad",
        "antecedente",
        "uci",
        "cardiaca",
        "cardíaca",
        "insuficiencia",
        "pro-bnp",
        "creatinina",
        "bun",
        "potasio",
        "hemoglobina",
        "leucocitos",
        "neutrofilos",
        "neutrófilos",
        "linfocitos",
        "monocitos",
        "eosinofilos",
        "eosinófilos",
        "basofilos",
        "basófilos",
        "mlr",
        "nlr",
        "plr",
        "sii",
        "aisi",
        "siri",
        "mpr",
        "npr",
        "mnr",
        "rpr",
    ]

    if any(normalize_text(k) in col_lower for k in demographic_keywords):
        return "demographics"
    elif any(normalize_text(k) in col_lower for k in vital_keywords):
        return "vitals"
    elif any(normalize_text(k) in col_lower for k in clinical_keywords):
        return "clinical"
    else:
        return "others"


def row_to_text(row, column_types, quantiles):
    groups = {
        "demographics": [],
        "vitals": [],
        "clinical": [],
        "others": [],
    }

    numeric_cols = column_types["numeric_cols"]
    special_binary_cols = column_types["special_binary_cols"]
    clinical_binary_cols = column_types["clinical_binary_cols"]
    categorical_cols = column_types["categorical_cols"]
    one_hot_info = column_types["one_hot_info"]

    # One-hot categorical variables
    for sentence, group in build_one_hot_sentences(row, one_hot_info):
        add_sentence_to_group(groups, sentence, group)

    # Special binary variables: sex, residence area, etc.
    for col in special_binary_cols:
        if pd.isna(row[col]):
            continue

        sentence, group = build_special_binary_sentence(col, row[col])

        if sentence is not None:
            add_sentence_to_group(groups, sentence, group)

    # Numeric variables
    for col in numeric_cols:
        value = pd.to_numeric(row[col], errors="coerce")

        if pd.isna(value):
            continue

        value_text = clinical_numeric_to_text(
            col=col,
            x=value,
            row=row,
            quantiles=quantiles,
        )

        if value_text is None:
            continue

        sentence = f"{english_colname(col)}: {value_text}"
        group = infer_group_for_column(col)
        add_sentence_to_group(groups, sentence, group)

    # Clinical binary variables
    for col in clinical_binary_cols:
        if pd.isna(row[col]):
            continue

        sentence = build_clinical_binary_sentence(col, row[col])

        if sentence is not None:
            group = infer_group_for_column(col)
            add_sentence_to_group(groups, sentence, group)

    # Other categorical variables
    for col in categorical_cols:
        value = row[col]

        if pd.isna(value):
            continue

        sentence = f"{english_colname(col)}: {value}"
        group = infer_group_for_column(col)
        add_sentence_to_group(groups, sentence, group)

    paragraphs = []

    # No se menciona el desenlace objetivo para evitar que el embedding represente el prompt de predicción.
    paragraphs.append(
        "clinical note. patient with chronic chagas disease."
    )

    if groups["demographics"]:
        paragraphs.append(
            "demographics: "
            + "; ".join(groups["demographics"])
            + "."
        )

    if groups["vitals"]:
        paragraphs.append(
            "vital signs and anthropometrics: "
            + "; ".join(groups["vitals"])
            + "."
        )

    if groups["clinical"]:
        paragraphs.append(
            "clinical history, cardiovascular status, and laboratory findings: "
            + "; ".join(groups["clinical"])
            + "."
        )

    if groups["others"]:
        paragraphs.append(
            "other available clinical information: "
            + "; ".join(groups["others"])
            + "."
        )

    text = " ".join(paragraphs)

    # Clinical-LongFormer fue preentrenado con texto en minúsculas.
    text = text.lower()

    # Limpieza ligera
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_split(split_name, df, column_types, quantiles):
    texts = []
    labels = []

    for _, row in df.iterrows():
        text = row_to_text(
            row=row,
            column_types=column_types,
            quantiles=quantiles,
        )

        texts.append(text)
        labels.append(int(row[TARGET_COL]))

    out_df = pd.DataFrame(
        {
            "text": texts,
            "label": labels,
        }
    )

    out_path = TEXT_DIR / f"{split_name}_texts.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Guardado: {out_path} | shape={out_df.shape}")


def main():
    split_dfs = {
        split_name: pd.read_csv(OUTPUT_DIR / f"{split_name}.csv")
        for split_name in SPLITS
    }

    train_df = split_dfs["train"]

    feature_cols = pd.read_csv(
        OUTPUT_DIR / "feature_columns.csv",
        header=None,
    )[0].tolist()

    column_types = infer_column_types(train_df, feature_cols)
    quantiles = build_quantiles(train_df, column_types["numeric_cols"])

    metadata = {
        "language": "english",
        "text_style": "clinical pseudo-note",
        "model_target": "Clinical-LongFormer / clinical encoder embeddings",
        "target_prompt_removed": True,
        "lowercase": True,
        "numeric_cols": column_types["numeric_cols"],
        "special_binary_cols": column_types["special_binary_cols"],
        "clinical_binary_cols": column_types["clinical_binary_cols"],
        "categorical_cols": column_types["categorical_cols"],
        "one_hot_info": {
            group_name: [
                {"column": col, "category_code": code}
                for col, code in cols
            ]
            for group_name, cols in column_types["one_hot_info"].items()
        },
        "quantile_based_cols": TRAIN_QUANTILE_COLS,
        "raw_value_cols": RAW_VALUE_COLS,
        "english_colnames": ENGLISH_COLNAMES,
        "quantiles": quantiles,
        "range_strategy": {
            "clinical_ranges": [
                "Edad al ingreso",
                "Frecuencia cardiaca",
                "Presión sistólica cardíaca",
                "Presión diastólica cardíaca",
                "IMC",
                "FEVI",
                "Índice de comorbilidad de Charlson",
                "Potasio",
                "Pro-BNP",
                "Creatinina",
                "Glóbulos rojos",
                "Leucocitos",
                "Hemoglobina",
                "Volumen corpuscular medio",
                "Hemoglobina corpuscular media",
                "Concentración de hemoglobina corpuscular media",
                "RDW",
                "Volumen plaquetario",
                "Neutrófilos",
                "Linfocitos",
                "Monocitos",
                "Eosinófilos",
                "Basófilos",
                "BUN",
            ],
            "quantile_ranges_from_train": TRAIN_QUANTILE_COLS,
            "neutral_raw_values": RAW_VALUE_COLS,
        },
        "split_strategy": SPLIT_STRATEGY,
        "splits": SPLITS,
    }

    with open(TEXT_DIR / "text_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nResumen de tipos de variables")
    print("============================")
    print("Numéricas:", len(column_types["numeric_cols"]))
    print(column_types["numeric_cols"])

    print("\nBinarias especiales:", len(column_types["special_binary_cols"]))
    print(column_types["special_binary_cols"])

    print("\nBinarias clínicas:", len(column_types["clinical_binary_cols"]))
    print(column_types["clinical_binary_cols"])

    print("\nCategóricas:", len(column_types["categorical_cols"]))
    print(column_types["categorical_cols"])

    print("\nGrupos one-hot detectados:")
    for group_name, cols in column_types["one_hot_info"].items():
        print(group_name, ":", cols)

    print("\nVariables por cuantiles del train:")
    print(quantiles)

    for split_name in SPLITS:
        process_split(split_name, split_dfs[split_name], column_types, quantiles)

    print("\nEjemplo de texto:")
    example = pd.read_csv(TEXT_DIR / "train_texts.csv").iloc[0]["text"]
    print(example)


if __name__ == "__main__":
    main()
