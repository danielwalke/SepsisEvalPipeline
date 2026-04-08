import configparser

CBC = {
    "Hemoglobin (Blood)": 51222,
    "White Blood Cells (Blood)": 51301,
    "Red Blood Cells (Blood)": 51279,
    "Platelet Count (Blood)": 51265,
    "MCV (Blood)": 51250,
}


EXTCBC = {
    "Hemoglobin (Blood)": 51222,
    "White Blood Cells (Blood)": 51301,
    "Red Blood Cells (Blood)": 51279,
    "Platelet Count (Blood)": 51265,
    "MCV (Blood)": 51250,
    "MCH (Blood)": 51248,
    "MCHC (Blood)": 51249,
    "RDW (Blood)": 51277,
    "Hematocrit (Blood)": 51221,
}

BMP = {
    "Glucose (Blood)": 50931,
    "Calcium, Total (Blood)": 50893,
    "Sodium (Blood)": 50983,
    "Potassium (Blood)": 50971,
    "Chloride (Blood)": 50902,
    "Bicarbonate (Blood)": 50882,
    "Urea Nitrogen (Blood)": 51006,
    "Creatinine (Blood)": 50912
}


WBC_DIFFERENTIAL = {
    "Neutrophils (Blood)": 51256,
    "Lymphocytes (Blood)": 51244,
    "Monocytes (Blood)": 51254,
    "Eosinophils (Blood)": 51200,
    "Basophils (Blood)": 51146
}

EXTENDED_ELECTROLYTES = {
    "Magnesium (Blood)": 50960,
    "Phosphate (Blood)": 50970,
    "Anion Gap (Blood)": 50868
}

LIVER_PANEL = {
    "Alanine Aminotransferase (ALT) (Blood)": 50861,
    "Asparate Aminotransferase (AST) (Blood)": 50878,
    "Alkaline Phosphatase (Blood)": 50863,
    "Bilirubin, Total (Blood)": 50885,
    "Albumin (Blood)": 50862
}

COAGULATION = {
    "PT (Blood)": 51274,
    "INR(PT) (Blood)": 51237,
    "PTT (Blood)": 51275
}

KIDNEY_FUNCTION = {
    "Creatinine (Blood)": 50912,
    "Urea Nitrogen (Blood)": 51006,
    "Estimated GFR (MDRD equation) (Blood)": 52026
}

panel_name_to_feature_codes = {
    "CBC": [CBC[lab] for lab in CBC],
    "EXTCBC": [EXTCBC[lab] for lab in EXTCBC],
    "BMP": [BMP[lab] for lab in BMP],
    "WBCDIFF": [WBC_DIFFERENTIAL[lab] for lab in WBC_DIFFERENTIAL],
    "EXTELECT": [EXTENDED_ELECTROLYTES[lab] for lab in EXTENDED_ELECTROLYTES],
    "LIVER": [LIVER_PANEL[lab] for lab in LIVER_PANEL],
    "COAG": [COAGULATION[lab] for lab in COAGULATION],
    "KIDNEY": [KIDNEY_FUNCTION[lab] for lab in KIDNEY_FUNCTION]
}

config = configparser.ConfigParser()
config.read("config.ini")
panel_name = config["PANEL"]["panel_name"]
feature_codes = []
for panel in config["PANEL"]["panel_name"].split("_"):
    panel = panel.strip()
    if panel in panel_name_to_feature_codes:
        feature_codes.extend(panel_name_to_feature_codes[panel])
    else:
        raise ValueError(f"Panel name '{panel}' not found in panel_name_to_feature_codes mapping.")

with open("0_mimic_preprocess/features/feature_codes.csv", "w") as f:
    f.write("itemid\n")
    for code in feature_codes:
        f.write(f"{code}\n")