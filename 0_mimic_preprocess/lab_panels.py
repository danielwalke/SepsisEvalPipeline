CBC = [
    "Hematocrit (Blood)",
    "Hemoglobin (Blood)",
    "White Blood Cells (Blood)",
    "Red Blood Cells (Blood)",
    "Platelet Count (Blood)",
    "MCV (Blood)",
    "MCH (Blood)",
    "MCHC (Blood)",
    "RDW (Blood)",
    "RDW-SD (Blood)"
]

WBC_DIFFERENTIAL = [
    "Neutrophils (Blood)",
    "Lymphocytes (Blood)",
    "Monocytes (Blood)",
    "Eosinophils (Blood)",
    "Basophils (Blood)"
]

BMP = [
    "Glucose (Blood)",
    "Calcium, Total (Blood)",
    "Sodium (Blood)",
    "Potassium (Blood)",
    "Chloride (Blood)",
    "Bicarbonate (Blood)",
    "Urea Nitrogen (Blood)",   # BUN
    "Creatinine (Blood)"
]

EXTENDED_ELECTROLYTES = [
    "Magnesium (Blood)",
    "Phosphate (Blood)",
    "Anion Gap (Blood)"
]

LIVER_PANEL = [
    "Alanine Aminotransferase (ALT) (Blood)",
    "Asparate Aminotransferase (AST) (Blood)",
    "Alkaline Phosphatase (Blood)",
    "Bilirubin, Total (Blood)",
    "Albumin (Blood)"
]

COAGULATION = [
    "PT (Blood)",
    "INR(PT) (Blood)",
    "PTT (Blood)"
]

KIDNEY_FUNCTION = [
    "Creatinine (Blood)",
    "Urea Nitrogen (Blood)",
    "Estimated GFR (MDRD equation) (Blood)"
]

HIL_INDICES = [
    "H (Blood)",  # Hemolysis
    "L (Blood)",  # Lipemia
    "I (Blood)"   # Icterus
]

LAB_PANELS = {
    "CBC": CBC,
    "WBC_DIFFERENTIAL": WBC_DIFFERENTIAL,
    "BMP": BMP,
    "EXTENDED_ELECTROLYTES": EXTENDED_ELECTROLYTES,
    "LIVER_PANEL": LIVER_PANEL,
    "COAGULATION": COAGULATION,
    "KIDNEY_FUNCTION": KIDNEY_FUNCTION,
    "HIL_INDICES": HIL_INDICES
}


mimic_iv_lab_itemids = {
    "Hematocrit (Blood)": 51221,
    "Hemoglobin (Blood)": 51222,
    "White Blood Cells (Blood)": 51301,
    "Red Blood Cells (Blood)": 51279,
    "Platelet Count (Blood)": 51265,
    "MCV (Blood)": 51250,
    "MCH (Blood)": 51248,
    "MCHC (Blood)": 51249,
    "RDW (Blood)": 51277,
    "RDW-SD (Blood)": 52159,
    "Neutrophils (Blood)": 51256,
    "Lymphocytes (Blood)": 51244,
    "Monocytes (Blood)": 51254,
    "Eosinophils (Blood)": 51200,
    "Basophils (Blood)": 51146,
    "Glucose (Blood)": 50931,
    "Calcium, Total (Blood)": 50893,
    "Sodium (Blood)": 50983,
    "Potassium (Blood)": 50971,
    "Chloride (Blood)": 50902,
    "Bicarbonate (Blood)": 50882,
    "Urea Nitrogen (Blood)": 51006,
    "Creatinine (Blood)": 50912,
    "Magnesium (Blood)": 50960,
    "Phosphate (Blood)": 50970,
    "Anion Gap (Blood)": 50868,
    "Alanine Aminotransferase (ALT) (Blood)": 50861,
    "Asparate Aminotransferase (AST) (Blood)": 50878,
    "Alkaline Phosphatase (Blood)": 50863,
    "Bilirubin, Total (Blood)": 50885,
    "Albumin (Blood)": 50862,
    "PT (Blood)": 51274,
    "INR(PT) (Blood)": 51237,
    "PTT (Blood)": 51275,
    "Estimated GFR (MDRD equation) (Blood)": 52026,
    "H (Blood)": 50934,
    "L (Blood)": 51678,
    "I (Blood)": 50947
}