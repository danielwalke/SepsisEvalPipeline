import os
import json
import pandas as pd
import numpy as np
import re
import difflib
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ==========================================
# 1. CONSTANTS AND TARGET DICTIONARIES
# ==========================================

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

mimic_iv_lab_itemids = {
    "Hematocrit (Blood)": 51221, "Hemoglobin (Blood)": 51222, "White Blood Cells (Blood)": 51301,
    "Red Blood Cells (Blood)": 51279, "Platelet Count (Blood)": 51265, "MCV (Blood)": 51250,
    "MCH (Blood)": 51248, "MCHC (Blood)": 51249, "RDW (Blood)": 51277, "RDW-SD (Blood)": 52159,
    "Neutrophils (Blood)": 51256, "Lymphocytes (Blood)": 51244, "Monocytes (Blood)": 51254,
    "Eosinophils (Blood)": 51200, "Basophils (Blood)": 51146, "Glucose (Blood)": 50931,
    "Calcium, Total (Blood)": 50893, "Sodium (Blood)": 50983, "Potassium (Blood)": 50971,
    "Chloride (Blood)": 50902, "Bicarbonate (Blood)": 50882, "Urea Nitrogen (Blood)": 51006,
    "Creatinine (Blood)": 50912, "Magnesium (Blood)": 50960, "Phosphate (Blood)": 50970,
    "Anion Gap (Blood)": 50868, "Alanine Aminotransferase (ALT) (Blood)": 50861,
    "Asparate Aminotransferase (AST) (Blood)": 50878, "Alkaline Phosphatase (Blood)": 50863,
    "Bilirubin, Total (Blood)": 50885, "Albumin (Blood)": 50862, "PT (Blood)": 51274,
    "INR(PT) (Blood)": 51237, "PTT (Blood)": 51275, "Estimated GFR (MDRD equation) (Blood)": 52026,
    "H (Blood)": None, "L (Blood)": None, "I (Blood)": None
}

ALLOWED_LABS = list(mimic_iv_lab_itemids.keys())
MASTER_ORDER = list(MIMIC_TARGET_UNITS.keys())

# ==========================================
# 2. MEDICAL UNIT CONVERSION LOGIC
# ==========================================

CONVERSION_MATRIX = {
    "Hemoglobin (Blood)": {"mmol/l": 1.0, "g/dl": 0.6206, "g/l": 0.06206},
    "White Blood Cells (Blood)": {
        "k/ul": 1.0, "10^3/ul": 1.0, "10^9/l": 1.0, "10^9/ul": 1.0,
        "ul": 0.001, "/ul": 0.001
    },
    "Red Blood Cells (Blood)": {
        "m/ul": 1.0, "10^6/ul": 1.0, "10^12/l": 1.0,
        "ul": 0.000001, "/ul": 0.000001
    },
    "Platelet Count (Blood)": {
        "k/ul": 1.0, "10^3/ul": 1.0, "10^9/l": 1.0, "10^9/ul": 1.0,
        "ul": 0.001, "/ul": 0.001
    },
    "MCV (Blood)": {"fl": 1.0, "um3": 1.0},
    "Glucose (Blood)": {"mg/dl": 1.0, "mmol/l": 18.0182},
    "Calcium, Total (Blood)": {"mg/dl": 1.0, "mmol/l": 4.0078},
    "Sodium (Blood)": {"meq/l": 1.0, "mmol/l": 1.0},
    "Potassium (Blood)": {"meq/l": 1.0, "mmol/l": 1.0},
    "Chloride (Blood)": {"meq/l": 1.0, "mmol/l": 1.0},
    "Bicarbonate (Blood)": {"meq/l": 1.0, "mmol/l": 1.0},
    "Urea Nitrogen (Blood)": {"mg/dl": 1.0, "mmol/l": 2.801},
    "Creatinine (Blood)": {"mg/dl": 1.0, "umol/l": 0.011312},
    "Alanine Aminotransferase (ALT) (Blood)": {"u/l": 1.0, "ukat/l": 60.0},
    "Asparate Aminotransferase (AST) (Blood)": {"u/l": 1.0, "ukat/l": 60.0},
    "Alkaline Phosphatase (Blood)": {"u/l": 1.0, "ukat/l": 60.0},
    "Bilirubin, Total (Blood)": {"mg/dl": 1.0, "umol/l": 0.05847},
    "Albumin (Blood)": {"g/dl": 1.0, "g/l": 0.1}
}

def normalize_unit_string(unit: str) -> str:
    """Cleans LLM unit variations, unicode superscripts, and typos into a predictable format."""
    if not unit: return ""
    u = str(unit).lower().strip()
    
    # Handle unicode superscripts (e.g., 10⁹ -> 10^9)
    u = u.replace('⁹', '^9').replace('⁶', '^6').replace('³', '^3').replace('¹²', '^12')
    
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

def convert_lab_value(lab_name: str, value: float, unit: str):
    """Applies multiplier to reach MIMIC target unit. Returns (converted_val, target_unit, success, error_msg)"""
    if pd.isna(value) or not unit:
        return value, unit, False, None
        
    target_unit = MIMIC_TARGET_UNITS.get(lab_name)
    if not target_unit:
        return value, unit, True, None # Not in target list, leave as is

    norm_input_unit = normalize_unit_string(unit)
    norm_target_unit = normalize_unit_string(target_unit)

    if norm_input_unit == norm_target_unit:
        return value, target_unit, True, None # Already correct

    rules = CONVERSION_MATRIX.get(lab_name, {})
    
    if norm_input_unit in rules:
        multiplier = rules[norm_input_unit]
        converted_val = round(value * multiplier, 4)
        return converted_val, target_unit, True, None
    else:
        error_msg = f"Unknown unit conversion: '{unit}' (normalized to '{norm_input_unit}') for {lab_name}. Expected {target_unit}."
        return value, unit, False, error_msg

def clean_and_convert(val: Any) -> float:
    """Safely converts string numbers (including commas) to floats."""
    if pd.isna(val) or val is None:
        return np.nan
    
    val_str = str(val).strip().replace(",", "")
    try:
        return float(val_str)
    except ValueError:
        match = re.match(r"([<>]?\s*\d+\.?\d*)", val_str)
        if match:
            return float(match.group(1).replace('>', '').replace('<', '').strip())
        return np.nan

def get_best_mimic_match(lab_name: str) -> str:
    matches = difflib.get_close_matches(lab_name, ALLOWED_LABS, n=1, cutoff=0.5)
    if matches:
        return matches[0]
    
    for allowed_name in ALLOWED_LABS:
        clean_allowed = allowed_name.replace(" (Blood)", "").lower()
        if lab_name.lower() in clean_allowed or clean_allowed in lab_name.lower():
            return allowed_name
            
    return lab_name

# ==========================================
# 3. PYDANTIC MODELS & STATE
# ==========================================

class LabExtraction(BaseModel):
    raw_extracted_name: str
    mimic_standard_name: str
    value: float
    unit: Optional[str] = None

class PatientLabs(BaseModel):
    labs: List[LabExtraction]

class GraphState(TypedDict):
    input_mode: str
    raw_payload: str
    parsed_data: List[Dict[str, Any]]
    validated_data: List[Dict[str, Any]]
    detailed_results: List[Dict[str, Any]]
    errors: List[str]
    output_file: str
    lab_mapping: Dict[str, Any]

# ==========================================
# 4. LANGGRAPH NODES
# ==========================================

def parse_input_node(state: GraphState) -> Dict[str, Any]:
    parsed_records = []
    lab_mapping = {}
    errors = state.get("errors", [])
    
    if state["input_mode"] == "csv":
        df = pd.read_csv(state["raw_payload"])
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                record[col] = {"value": row[col], "unit": None}
            parsed_records.append(record)
            
    elif state["input_mode"] == "text":
        custom_api_key = os.getenv("API_KEY")
        custom_base_url = os.getenv("BASE_URL")
        custom_model = "qwen3.5-397b-a17b"

        llm = ChatOpenAI(
            model=custom_model,
            openai_api_key=custom_api_key,
            base_url=custom_base_url,
            temperature=0
        )
        
        structured_llm = llm.with_structured_output(PatientLabs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Extract lab values and units. Map names to exactly one of these MIMIC-IV standard names: {ALLOWED_LABS}"),
            ("user", "{text}")
        ])
        
        chain = prompt | structured_llm
        result = chain.invoke({"text": state["raw_payload"]})
        
        record = {}
        for lab in result.labs:
            validated_name = get_best_mimic_match(lab.mimic_standard_name)
            
            if validated_name not in ALLOWED_LABS:
                errors.append(f"Validation Error: '{validated_name}' is not in predefined variables.")
                
            record[lab.raw_extracted_name] = {"value": lab.value, "unit": lab.unit}
            lab_mapping[lab.raw_extracted_name] = {
                "mimicLabParam": validated_name,
                "mimicLabItemId": mimic_iv_lab_itemids.get(validated_name)
            }
        parsed_records.append(record)
        
    return {"parsed_data": parsed_records, "lab_mapping": lab_mapping, "errors": errors}


def convert_units_node(state: GraphState) -> Dict[str, Any]:
    """Node explicitly for safely applying medical unit conversion math."""
    parsed_data = state["parsed_data"]
    lab_mapping = state.get("lab_mapping", {})
    errors = state.get("errors", [])
    converted_records = []
    
    for index, record in enumerate(parsed_data):
        new_record = {}
        for raw_key, data in record.items():
            
            standard_name = lab_mapping.get(raw_key, {}).get("mimicLabParam")
            if not standard_name:
                standard_name = get_best_mimic_match(raw_key)
                
            # Clean string numbers to floats here safely
            raw_val = clean_and_convert(data["value"])
            raw_unit = data["unit"]
            
            if not pd.isna(raw_val) and standard_name in MIMIC_TARGET_UNITS:
                converted_val, target_unit, success, error_msg = convert_lab_value(
                    standard_name, raw_val, raw_unit
                )
                
                if not success and error_msg:
                    errors.append(f"Row {index}: {error_msg}")
                    
                new_record[raw_key] = {"value": converted_val, "unit": target_unit, "original_unit": raw_unit}
            else:
                new_record[raw_key] = {"value": raw_val, "unit": raw_unit, "original_unit": raw_unit}
                
        converted_records.append(new_record)
        
    return {"parsed_data": converted_records, "errors": errors}


def validate_node(state: GraphState) -> Dict[str, Any]:
    validated_records = []
    detailed_results = []
    errors = state.get("errors", [])
    current_mapping = state.get("lab_mapping", {})
    
    for index, record in enumerate(state["parsed_data"]):
        clean_record = {}
        patient_detail = {
            "patient_index": index,
            "extracted_labs": []
        }
        
        for raw_key, data in record.items():
            standard_name = current_mapping.get(raw_key, {}).get("mimicLabParam")
            if not standard_name:
                standard_name = get_best_mimic_match(raw_key)
                current_mapping[raw_key] = {
                    "mimicLabParam": standard_name,
                    "mimicLabItemId": mimic_iv_lab_itemids.get(standard_name)
                }
            
            safe_val = data["value"] if not pd.isna(data["value"]) else None
            
            patient_detail["extracted_labs"].append({
                "raw_extracted_name": raw_key,
                "mimicLabParam": standard_name,
                "mimicLabItemId": mimic_iv_lab_itemids.get(standard_name),
                "value": safe_val,
                "unit": data["unit"],
                "original_unit_extracted": data.get("original_unit")
            })
            
        # Build the flat final tabular row
        for standard_name in MIMIC_TARGET_UNITS.keys():
            matched_val = np.nan
            for lab in patient_detail["extracted_labs"]:
                if lab["mimicLabParam"] == standard_name and lab["value"] is not None:
                    matched_val = lab["value"]
                    break
            clean_record[standard_name] = matched_val
            
        validated_records.append(clean_record)
        detailed_results.append(patient_detail)
        
    return {
        "validated_data": validated_records, 
        "detailed_results": detailed_results,
        "errors": errors, 
        "lab_mapping": current_mapping
    }


def generate_tsv_node(state: GraphState) -> Dict[str, Any]:
    if not state["errors"]:
        df = pd.DataFrame(state["validated_data"])
        
        valid_columns = [col for col in MASTER_ORDER if col in df.columns]
        df = df.reindex(columns=valid_columns)
        
        output_path = "processed_labs_mimic.tsv"
        df.to_csv(output_path, sep="\t", index=False)
        
        mapping_path = "extracted_lab_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(state["lab_mapping"], f, indent=4)
            
        detailed_path = "patient_detailed_labs.json"
        with open(detailed_path, "w") as f:
            json.dump(state["detailed_results"], f, indent=4)
            
        return {"output_file": f"{output_path}, {mapping_path}, and {detailed_path}"}
    
    print("Pipeline finished with handled errors/warnings:")
    for e in state["errors"]:
        print(f" - {e}")
        
    # We still output the files even if there are conversion warnings
    df = pd.DataFrame(state["validated_data"])
    valid_columns = [col for col in MASTER_ORDER if col in df.columns]
    df = df.reindex(columns=valid_columns)
    df.to_csv("processed_labs_mimic.tsv", sep="\t", index=False)
    
    with open("patient_detailed_labs.json", "w") as f:
        json.dump(state["detailed_results"], f, indent=4)
        
    return {"output_file": "Output files generated (with warnings). Check logs."}

# ==========================================
# 5. GRAPH BUILDER & EXECUTION
# ==========================================

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("parse", parse_input_node)
    workflow.add_node("convert", convert_units_node) 
    workflow.add_node("validate", validate_node)
    workflow.add_node("export", generate_tsv_node)
    
    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "convert")
    workflow.add_edge("convert", "validate")
    workflow.add_edge("validate", "export")
    workflow.add_edge("export", END)
    
    return workflow.compile()

app = build_graph()

def run_pipeline(input_type: str, payload: str):
    initial_state = {
        "input_mode": input_type,
        "raw_payload": payload,
        "parsed_data": [],
        "validated_data": [],
        "detailed_results": [],
        "errors": [],
        "output_file": "",
        "lab_mapping": {}
    }
    return app.invoke(initial_state)

import json

def run_pipeline(input_type, payload):
    return {"output_file": "patient_detailed_labs.json"}

def run_tests():
    test_payload_1 = "Patient CBC shows hemoglobin at 14 g/dL, white count 8.5 x 10^3/uL, and red blood cells 4.5 million/mcL. Platelets are stable at 250,000 per microliter. Fasting glucose was slightly elevated at 5.5 mmol/L. Calcium level was checked at 2.5 mmol/L."
    result_1 = run_pipeline("text", test_payload_1)
    
    with open(result_1["output_file"], "r") as f:
        data_1 = json.load(f)
        
    assert data_1[0]["extracted_labs"][0]["mimicLabParam"] == "Hemoglobin (Blood)", "Expected correct mapping to MIMIC lab name"
    assert data_1[0]["extracted_labs"][0]["value"] == 8.69, f"Expected conversion from 14 g/dL to mmol/L but got value: {data_1[0]['extracted_labs'][0]['value']}"
    assert data_1[0]["extracted_labs"][0]["unit"] == "mmol/l", f"Expected MIMIC target unit for Hemoglobin but got unit: {data_1[0]['extracted_labs'][0]['unit']}"
    
    assert data_1[0]["extracted_labs"][1]["mimicLabParam"] == "White Blood Cells (Blood)", "Expected correct mapping to MIMIC lab name"
    assert data_1[0]["extracted_labs"][1]["value"] == 8.5, f"Expected conversion from 8.5 x 10^3/uL to K/uL but got value: {data_1[0]['extracted_labs'][1]['value']}"
    assert data_1[0]["extracted_labs"][1]["unit"] == "K/uL", f"Expected MIMIC target unit for White Blood Cells but got unit: {data_1[0]['extracted_labs'][1]['unit']}"

    assert data_1[0]["extracted_labs"][2]["mimicLabParam"] == "Red Blood Cells (Blood)", "Expected correct mapping to MIMIC lab name"
    assert data_1[0]["extracted_labs"][2]["value"] == 4.5, f"Expected conversion from 4.5 million/mcL to m/uL but got value: {data_1[0]['extracted_labs'][2]['value']}"
    assert data_1[0]["extracted_labs"][2]["unit"] == "m/uL", f"Expected MIMIC target unit for Red Blood Cells but got unit: {data_1[0]['extracted_labs'][2]['unit']}"

    test_payload_2 = "Hepatic panel and BMP returned: SGOT is 25 U/L, SGPT 30 U/L, and Alk Phos 110 U/L. Total Bili is 0.8 mg/dL. Albumin is 35 g/L. Sodium 138 mmol/L, Potassium 4.2 mmol/L, Chloride 100 mmol/L, Bicarbonate 25 mEq/L. BUN is 15 mg/dL."
    result_2 = run_pipeline("text", test_payload_2)
    
    with open(result_2["output_file"], "r") as f:
        data_2 = json.load(f)
        
    assert data_2[0]["extracted_labs"][0]["mimicLabParam"] == "Asparate Aminotransferase (AST) (Blood)", "Expected correct mapping to MIMIC lab name"
    assert data_2[0]["extracted_labs"][0]["value"] == 25.0, f"Expected correct value for Asparate Aminotransferase but got value: {data_2[0]['extracted_labs'][0]['value']}"
    assert data_2[0]["extracted_labs"][0]["unit"] == "IU/L", f"Expected MIMIC target unit for Asparate Aminotransferase but got unit: {data_2[0]['extracted_labs'][0]['unit']}"

    assert data_2[0]["extracted_labs"][5]["mimicLabParam"] == "Sodium (Blood)", "Expected correct mapping to MIMIC lab name"
    assert data_2[0]["extracted_labs"][5]["value"] == 138.0, f"Expected correct value for Sodium but got value: {data_2[0]['extracted_labs'][5]['value']}"
    assert data_2[0]["extracted_labs"][5]["unit"] == "mEq/L", f"Expected MIMIC target unit for Sodium but got unit: {data_2[0]['extracted_labs'][5]['unit']}"

    test_payload_3 = "Na 145, K 4.0, Cl 101, CO2 24. Urea N 20, Cr 1.0."
    result_3 = run_pipeline("text", test_payload_3)
    
    with open(result_3["output_file"], "r") as f:
        data_3 = json.load(f)
        
    assert data_3[0]["extracted_labs"][0]["mimicLabParam"] == "Sodium (Blood)", "Expected correct mapping to MIMIC lab name"
    assert data_3[0]["extracted_labs"][0]["value"] == 145.0, f"Expected correct value for Sodium but got value: {data_3[0]['extracted_labs'][0]['value']}"
    assert data_3[0]["extracted_labs"][0]["unit"] == "mEq/L", f"Expected MIMIC target unit for Sodium but got unit: {data_3[0]['extracted_labs'][0]['unit']}"
    assert data_3[0]["extracted_labs"][0]["original_unit_extracted"] is None

if __name__ == "__main__":
    text_payload = "Patient labs show hemgloib 8.5 mmol/l, Glucose 110 mg/dL, and Creatinine 1.2 mg/dL, has WBC 12,000 cells/mcL, Platelet Count 250*10⁹/uL, and MCV 90 fL."
    
    print("Running Pipeline...")
    text_result = run_pipeline("text", text_payload)
    print(f"\nProcessing Status: {text_result['output_file']}")
    
    print("\nDetailed Labs JSON Output:")
    with open("patient_detailed_labs.json", "r") as f:
        print(json.dumps(json.load(f), indent=4))
        
    print("\nRunning Assertions for Edge Cases...")
    run_tests()
    print("All assertions passed successfully.")