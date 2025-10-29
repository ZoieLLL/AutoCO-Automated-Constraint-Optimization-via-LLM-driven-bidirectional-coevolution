import re
import os
import inspect
from typing import Dict, List, Optional, Any, Tuple


def extract_code_information(code_content: str) -> Dict:
    result = {
        "function_signatures": [],
        "class_attributes": [],
        "parameters": [],
        "imports": []
    }

    function_matches = re.finditer(
        r'def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\):', code_content)
    for match in function_matches:
        func_name = match.group(1)
        params = match.group(2).strip()
        result["function_signatures"].append({
            "name": func_name,
            "params": params,
            "full_signature": f"def {func_name}({params}):"
        })

    class_attr_matches = re.finditer(
        r'self\.([a-zA-Z0-9_]+)\s*=', code_content)
    for match in class_attr_matches:
        attr_name = match.group(1)
        if attr_name not in result["class_attributes"]:
            result["class_attributes"].append(attr_name)

    param_matches = re.finditer(
        r'([A-Z_]+)\s*=\s*["\']?([^,\s"\']+)["\']?', code_content)
    for match in param_matches:
        param_name = match.group(1)
        param_value = match.group(2)
        result["parameters"].append({
            "name": param_name,
            "value": param_value
        })

    import_matches = re.finditer(
        r'(from\s+[a-zA-Z0-9_.]+\s+import\s+[a-zA-Z0-9_.,\s]+|import\s+[a-zA-Z0-9_.,\s]+)', code_content)
    for match in import_matches:
        import_stmt = match.group(1).strip()
        result["imports"].append(import_stmt)

    return result


def get_function_signature_by_name(code_content: str, function_name: str) -> Optional[str]:

    pattern = rf'def\s+{function_name}\s*\(([^)]*)\):'
    match = re.search(pattern, code_content)
    if match:
        params = match.group(1).strip()
        return f"def {function_name}({params}):"
    return None


def extract_llm_fill_area(code_content: str) -> Optional[str]:

    start_marker = "# ============ LLM Fill Area - Start ============"
    end_marker = "# ============ LLM Fill Area - End ============"

    if start_marker in code_content and end_marker in code_content:
        start_idx = code_content.find(start_marker)
        end_idx = code_content.find(end_marker) + len(end_marker)
        if start_idx != -1 and end_idx != -1:
            return code_content[start_idx:end_idx]

    return None


def format_code_information_for_prompt(code_info: Dict) -> str:

    result = "Key Code Information:\n\n"
    if code_info["function_signatures"]:
        result += "Function Signatures:\n"
        for func in code_info["function_signatures"]:
            result += f"```python\n{func['full_signature']}\n```\n"
    if code_info["class_attributes"]:
        result += "\nClass Attributes:\n"
        for attr in code_info["class_attributes"]:
            result += f"- self.{attr}\n"
    if code_info["parameters"]:
        result += "\nConstant Parameters:\n"
        for param in code_info["parameters"]:
            result += f"- {param['name']}: {param['value']}\n"

    return result
