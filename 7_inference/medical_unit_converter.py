# medical_unit_converter.py
import re

MIMIC_TARGET_UNITS = {
    "Hemoglobin (Blood)": "mmol/l",
    "White Blood Cells (Blood)": "K/uL",
    "Red Blood Cells (Blood)": "m/uL",
    "Platelet Count (Blood)": "K/uL",
    "MCV (Blood)": "fL",
    "Glucose (Blood)": "mg/dL",
    "Calcium, Total (Blood)": "mg/dL",
    "Sodium (Blood)": "mEq/L",
    "Potassium (Blood)": "mEq/L",
    "Chloride (Blood)": "mEq/L",
    "Bicarbonate (Blood)": "mEq/L",
    "Urea Nitrogen (Blood)": "mg/dL",
    "Creatinine (Blood)": "mg/dL",
    "Alanine Aminotransferase (ALT) (Blood)": "IU/L",
    "Asparate Aminotransferase (AST) (Blood)": "IU/L",
    "Alkaline Phosphatase (Blood)": "IU/L",
    "Bilirubin, Total (Blood)": "mg/dL",
    "Albumin (Blood)": "g/dL"
}

def normalize_unit_string(unit: str) -> str:
    """Cleans LLM unit variations, unicode superscripts, and typos into a predictable format."""
    if not unit: return ""
    u = str(unit).lower().strip()
    
    # Map unicode superscripts
    u = u.translate(str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789'))
    
    # Standardize syntax
    u = re.sub(r'[\*x]10\^?', '10^', u)  # *10^9, x10^9 -> 10^9
    u = u.replace("µ", "u").replace("mc", "u") # micro
    u = u.replace("cells/", "").replace("cell/", "") # remove 'cells/'
    
    # Common replacements
    replacements = {
        "liter": "l", "deciliter": "dl", "milliliter": "ml", "microliter": "ul",
        "cc": "ml", "mm3": "ul", "mcl": "ul", "iu/l": "u/l"
    }
    for old, new in replacements.items():
        u = u.replace(old, new)
        
    return u.strip()

# Matrix of normalized input units to multipliers to reach MIMIC target units
CONVERSION_MATRIX = {
    "Hemoglobin (Blood)": {
        "mmol/l": 1.0, "g/dl": 0.6206, "g/l": 0.06206
    },
    "White Blood Cells (Blood)": {
        "k/ul": 1.0, "10^3/ul": 1.0, "10^9/l": 1.0,
        "ul": 0.001, "/ul": 0.001  # e.g., 12,000 /uL -> 12 K/uL
    },
    "Red Blood Cells (Blood)": {
        "m/ul": 1.0, "10^6/ul": 1.0, "10^12/l": 1.0,
        "ul": 0.000001, "/ul": 0.000001
    },
    "Platelet Count (Blood)": {
        "k/ul": 1.0, "10^3/ul": 1.0, "10^9/l": 1.0,
        "ul": 0.001, "/ul": 0.001
    },
    "MCV (Blood)": {
        "fl": 1.0, "um3": 1.0
    },
    "Glucose (Blood)": {
        "mg/dl": 1.0, "mmol/l": 18.0182
    },
    "Calcium, Total (Blood)": {
        "mg/dl": 1.0, "mmol/l": 4.0078
    },
    # Electrolytes are generally 1:1 between mEq/L and mmol/L
    "Sodium (Blood)": { "meq/l": 1.0, "mmol/l": 1.0 },
    "Potassium (Blood)": { "meq/l": 1.0, "mmol/l": 1.0 },
    "Chloride (Blood)": { "meq/l": 1.0, "mmol/l": 1.0 },
    "Bicarbonate (Blood)": { "meq/l": 1.0, "mmol/l": 1.0 },
    
    "Urea Nitrogen (Blood)": {
        "mg/dl": 1.0, "mmol/l": 2.801 # Blood Urea (mmol/L) to BUN (mg/dL)
    },
    "Creatinine (Blood)": {
        "mg/dl": 1.0, "umol/l": 0.011312
    },
    "Alanine Aminotransferase (ALT) (Blood)": { "u/l": 1.0, "ukat/l": 60.0 },
    "Asparate Aminotransferase (AST) (Blood)": { "u/l": 1.0, "ukat/l": 60.0 },
    "Alkaline Phosphatase (Blood)": { "u/l": 1.0, "ukat/l": 60.0 },
    
    "Bilirubin, Total (Blood)": {
        "mg/dl": 1.0, "umol/l": 0.05847
    },
    "Albumin (Blood)": {
        "g/dl": 1.0, "g/l": 0.1
    }
}

def convert_lab_value(lab_name: str, value: float, unit: str):
    """Returns (converted_value, target_unit, success_bool, error_string)"""
    if value is None or not unit:
        return value, unit, False, None
        
    target_unit = MIMIC_TARGET_UNITS.get(lab_name)
    if not target_unit:
        return value, unit, True, None # Not in our target list, leave as is

    norm_input_unit = normalize_unit_string(unit)
    norm_target_unit = normalize_unit_string(target_unit)

    # Already the correct unit
    if norm_input_unit == norm_target_unit:
        return value, target_unit, True, None

    # Retrieve conversion rules for this specific lab
    rules = CONVERSION_MATRIX.get(lab_name, {})
    
    if norm_input_unit in rules:
        multiplier = rules[norm_input_unit]
        print(f"Converting {lab_name}: {value} {unit} -> {value * multiplier} {target_unit} using multiplier {multiplier}")
        converted_val = round(value * multiplier, 4)
        return converted_val, target_unit, True, None
    else:
        error_msg = f"Unknown unit '{unit}' (normalized to '{norm_input_unit}') for {lab_name}. Expected {target_unit}."
        return value, unit, False, error_msg
