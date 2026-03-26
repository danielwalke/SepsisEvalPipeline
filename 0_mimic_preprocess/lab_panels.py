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