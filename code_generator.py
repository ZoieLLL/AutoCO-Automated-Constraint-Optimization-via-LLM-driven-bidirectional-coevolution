from typing import Dict, List, Tuple, Any, Optional, Set, Union
import os
import re
import time
import uuid
from datetime import datetime
import logging
import tempfile
from output_manager import OutputManager
from llm_client import CodeLLMClient
from models import TextStrategy
from copy import deepcopy
import json


class CodeGenerator:
    def __init__(self, llm_client: CodeLLMClient,
                 problem_info: Dict,
                 output_manager: OutputManager):
        self.llm_client = llm_client
        self.problem_info = problem_info
        self.output_manager = output_manager
        self.code_template_path = os.path.join(os.path.dirname(
            __file__), "templates", "solution_template.py")
        self.code_template = self._load_template()
        self.generated_codes = {}
        self.problem_data = {}

    def _load_template(self) -> str:
        try:
            if os.path.exists(self.code_template_path):
                with open(self.code_template_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                return self._get_default_template()
        except Exception as e:
            self.output_manager.log_error(
                "code_generator", "template_loading_error",
                f"template_loading_error: {str(e)}"
            )
            return self._get_default_template()

    def _get_default_template(self) -> str:
        try:
            if hasattr(self, 'algorithm_loader') and self.algorithm_loader:
                available_algorithms = self.algorithm_loader.get_available_algorithms()
                if available_algorithms:
                    template_name = available_algorithms[0]
                    template_content = self.algorithm_loader.get_template(
                        template_name)
                    return template_content

        except Exception as e:
            self.output_manager.log(
                f"Cannot read template from algorithm_loader: {str(e)}", level="warning")
        return """
    import math
    from typing import Dict, List, Tuple, Any

    class Solution:
        def __init__(self, problem_data: Dict, problem_params: Dict = None, relaxation_strategy = None):
            self.problem_data = problem_data
            self.parameters = problem_data.get("parameters", {}) if problem_params is None else problem_params
            self.relaxation_strategy = relaxation_strategy or {}
        
        def solve(self) -> Dict:
            solution = {}
            
            return solution
    """

    def generate_code4new(self, strategy: Dict, variant_index: int = 0) -> Dict:
        strategy_id = strategy.get("id", str(uuid.uuid4()))
        strategy_text = strategy.get("text", "")
        constraint_order = strategy.get("constraint_order", [])
        relaxation_factors = strategy.get("relaxation_factors", {})
        code_prompt = self._build_code_generation_prompt_new(
            strategy_text,
            constraint_order,
            relaxation_factors,
            variant_index
        )
        start_time = time.time()

        try:
            code = self.llm_client.generate_code(
                code_prompt, self.problem_info)
            processed_code = self._process_generated_code(code)
            if "def solve" not in processed_code:
                self.output_manager.log_warning(
                    "code_generator", "missing_solve_function",
                    f"Missing solve function"
                )
                processed_code = self._add_default_solve_function(
                    processed_code)
            code_id = f"{strategy_id}_variant_{variant_index}_{int(time.time())}"
            file_path = self.output_manager.save_generated_code(
                processed_code, code_id)
            code_info = {
                "id": code_id,
                "strategy_id": strategy_id,
                "variant_index": variant_index,
                "code": processed_code,
                "file_path": file_path,
                "generation_time": time.time() - start_time,
                "constraint_order": constraint_order,
                "relaxation_factors": relaxation_factors
            }
            self.generated_codes[code_id] = code_info
            return code_info

        except Exception as e:
            self.output_manager.log_error(
                "code_generator", "generation_error",
                f"Generation failed: {str(e)}"
            )
            code = self._get_default_solution_code(
                strategy_text,
                constraint_order,
                relaxation_factors
            )
            code_id = f"{strategy_id}_variant_{variant_index}_default_{int(time.time())}"
            file_path = self.output_manager.save_generated_code(code, code_id)
            code_info = {
                "id": code_id,
                "strategy_id": strategy_id,
                "variant_index": variant_index,
                "code": code,
                "file_path": file_path,
                "generation_time": time.time() - start_time,
                "constraint_order": constraint_order,
                "relaxation_factors": relaxation_factors,
                "is_default": True
            }
            self.generated_codes[code_id] = code_info

            return code_info

    def generate_variants(self, strategy: Dict, count: int = 3) -> List[Dict]:
        variants = []

        for i in range(count):
            code_info = self.generate_code4new(strategy, i)
            variants.append(code_info)

        return variants

    def _build_code_generation_prompt_new(self, strategy_text: str,
                                          constraint_order: List[str],
                                          relaxation_factors: Dict[str, float],
                                          variant_index: int) -> str:
        limited_problem_data = self.load_limited_problem_data(max_points=5)
        constraints_details = ""
        for i, constraint in enumerate(constraint_order):
            relaxation = relaxation_factors.get(constraint, 1.0)
            constraint_found = False
            for c in limited_problem_data.get("detailed_description", {}).get("constraints", []):
                if c.get("name") == constraint:
                    constraints_details += f"{i+1}. {constraint} (Relaxation factor: {relaxation}):\n"
                    constraints_details += f"   Description: {c.get('description')}\n"
                    constraints_details += f"   Calculation method: {c.get('calculation')}\n"
                    constraints_details += f"   Formula: {c.get('formula')}\n\n"
                    constraint_found = True
                    break
            if not constraint_found and hasattr(self, 'problem_info') and self.problem_info:
                constraint_info = self.problem_info.get(
                    "constraint_info", {}).get(constraint, {})
                if constraint_info:
                    constraints_details += f"{i+1}. {constraint} (Relaxation factor: {relaxation}):\n"
                    constraints_details += f"   Description: {constraint_info.get('description', '')}\n"
                    constraints_details += f"   Calculation method: {constraint_info.get('calculation', '')}\n"
                    constraints_details += f"   Formula: {constraint_info.get('formula', '')}\n\n"
                    constraint_found = True
            if not constraint_found:
                constraints_details += f"{i+1}. {constraint} (Relaxation factor: {relaxation}):\n"
                constraints_details += f"   Description: Constraint condition\n\n"
        parameters_info = ""
        for param_name, param_info in limited_problem_data.get("parameters", {}).items():
            if isinstance(param_info, dict) and "value" in param_info and "description" in param_info:
                parameters_info += f"{param_name} = {param_info['value']} - {param_info['description']}\n"
            else:
                parameters_info += f"{param_name} = {param_info}\n"
        data_structure_info = json.dumps(
            limited_problem_data.get("data", []), indent=2)
        objectives_info = ""
        for obj in limited_problem_data.get("detailed_description", {}).get("objectives", []):
            objectives_info += f"Objective: {obj.get('name')} - {obj.get('description')}\n"
            objectives_info += f"Direction: {obj.get('direction')}\n"
            objectives_info += f"Calculation: {obj.get('calculation')}\n"
            objectives_info += f"Formula: {obj.get('formula')}\n\n"
        auxiliary_info = ""
        for func in limited_problem_data.get("detailed_description", {}).get("auxiliary_functions", []):
            auxiliary_info += f"Function: {func.get('name')} - {func.get('description')}\n"
            auxiliary_info += f"Formula: {func.get('formula')}\n\n"
        full_code_template = self._get_default_template()

        diversity_prompt = ""
        if variant_index == 0:
            diversity_prompt = "Please prioritize algorithm efficiency and minimize computational complexity."
        elif variant_index == 1:
            diversity_prompt = "Please prioritize constraint satisfaction and ensure solution feasibility."
        else:
            diversity_prompt = "Please find a balance between algorithm efficiency and solution quality, and try innovative heuristic methods."

        problem_description = limited_problem_data.get(
            "description", "Combinatorial optimization problem")
        prompt = f"""
Please generate Python implementation code for the following {problem_description}. Please carefully read the problem description and parameter definitions.

【Problem Overview】
{limited_problem_data.get("detailed_description", {}).get("overview", "")}

【Strategy Description】
{strategy_text}

【Constraint Processing Order and Relaxation Factors】
{constraints_details}

【Available Parameters】
{parameters_info}

【Data Format】
{data_structure_info}

【Objective Functions】
{objectives_info}

【Auxiliary Functions】
{auxiliary_info}

【Solution Format Requirements】
{json.dumps(limited_problem_data.get("solution_format", {}) , indent=2)}

【Special Requirements】
{diversity_prompt}

【Complete Code Framework】
```python
{full_code_template}
```

⚠️ **Important Code Generation Requirements** ⚠️
1. **Generate only one select_next_point function**: Please concentrate all algorithm logic and functionality within this single function
2. **Prohibit generating multiple def functions**: Do not generate auxiliary functions like _calculate_density, _calculate_balance, etc.
3. **Implement all logic within the function**: If auxiliary calculations are needed, please implement them directly inside the select_next_point function

This function should:
- Follow the constraint processing order and relaxation factors in the strategy
- Place all required computational logic directly inside the function
- Not depend on any external auxiliary functions

Please ensure the code can handle problem inputs of different scales, efficiently and accurately generate drone paths. The code should be complete, executable, and contain necessary comments to explain the algorithm logic.

Please keep the code framework unchanged and only fill in the select_next_point function. No need to rewrite the entire class or add new classes.

⚠️ **Emphasize again**: Generate only one select_next_point function, put all logic inside this function!

Please provide the complete implementation code directly, no explanations or descriptions needed.
"""

        return prompt

    def _process_generated_code(self, code: str) -> str:
        extracted = self.llm_client.extract_content(code)
        if extracted["code"]:
            processed_code = self._fix_multiple_def_indentation(
                extracted["code"])
            return processed_code
        code_pattern = r"```python(.*?)```|```(.*?)```"
        matches = re.findall(code_pattern, code, re.DOTALL)

        if matches:
            for match in matches:
                if match[0]:
                    extracted_code = match[0].strip()
                    return self._fix_multiple_def_indentation(extracted_code)
                elif match[1]:
                    extracted_code = match[1].strip()
                    return self._fix_multiple_def_indentation(extracted_code)
        lines = code.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith("class Solution") or line.strip().startswith("import "):
                in_code = True

            if in_code:
                code_lines.append(line)

        if code_lines:
            extracted_code = "\n".join(code_lines)
            return self._fix_multiple_def_indentation(extracted_code)
        cleaned_code = re.sub(r'^```\s*python\s*', '', code)
        cleaned_code = re.sub(r'^```', '', code)

        lines = code.splitlines()
        if lines and lines[-1].strip() == '}':
            left_braces = code.count('{')
            right_braces = code.count('}')
            if right_braces > left_braces:
                lines = lines[:-1]
        fixed_code = '\n'.join(lines)

        if not fixed_code.endswith('\n'):
            fixed_code += '\n'
        return self._fix_multiple_def_indentation(fixed_code)

    def _add_default_solve_function(self, code: str) -> str:
        if "class Solution" not in code:
            return self._get_default_template()
        lines = code.split("\n")
        solution_class_started = False
        solution_class_end_line = len(lines)
        indentation = "    "

        for i, line in enumerate(lines):
            if "class Solution" in line:
                solution_class_started = True
                indentation_match = re.match(
                    r"^(\s+)def", "\n".join(lines[i:i+10]), re.MULTILINE)
                if indentation_match:
                    indentation = indentation_match.group(1)
            elif solution_class_started and i < len(lines) - 1:
                next_line = lines[i + 1]
                if next_line and not next_line.startswith(" ") and not next_line.startswith("\t"):
                    solution_class_end_line = i + 1
                    break
        default_solve = f"""
{indentation}def solve(self) -> Dict:
{indentation}    solution = {{
{indentation}        "drone_routes": {{}},
{indentation}        "unassigned_points": []
{indentation}    }}
{indentation}    
{indentation}    delivery_points = self.delivery_points
{indentation}    num_drones = min(self.MAX_DRONES, len(delivery_points))
{indentation}    for i, point in enumerate(delivery_points):
{indentation}        drone_id = i % num_drones
{indentation}        drone_key = f"drone_{{drone_id}}"
{indentation}        
{indentation}        if drone_key not in solution["drone_routes"]:
{indentation}            solution["drone_routes"][drone_key] = []
{indentation}            
{indentation}        solution["drone_routes"][drone_key].append(point["id"])
{indentation}    
{indentation}    return solution
"""
        result = lines[:solution_class_end_line]
        result.append(default_solve)
        result.extend(lines[solution_class_end_line:])

        return "\n".join(result)

    def _get_default_solution_code(self, strategy_text: str,
                                   constraint_order: List[str],
                                   relaxation_factors: Dict[str, float]) -> str:

        constraints_comment = ""
        for i, constraint in enumerate(constraint_order):
            relaxation = relaxation_factors.get(constraint, 1.0)
            constraints_comment += f"# {i+1}. {constraint}"
            if relaxation != 1.0:
                constraints_comment += f" (Relaxation factor: {relaxation})"
            constraints_comment += "\n"

        code = f"""
# Based on strategy: 
# {strategy_text}
#
# Constraint processing order:
{constraints_comment}

import math
from typing import Dict, List, Tuple, Any

class Solution:
    def __init__(self, problem_data: Dict):
        # Initialize solution
        self.problem_data = problem_data
        self.depot = problem_data.get("depot", {{"x": 0, "y": 0}})
        self.delivery_points = problem_data.get("delivery_points", [])
        self.parameters = problem_data.get("parameters", {{}})
        
        # Extract parameters
        self.MAX_DRONES = self.parameters.get("MAX_DRONES", 3)
        self.MAX_PAYLOAD = self.parameters.get("MAX_PAYLOAD", 50.0)
        self.MAX_BATTERY = self.parameters.get("MAX_BATTERY", 1000.0)
        self.DRONE_SPEED = self.parameters.get("DRONE_SPEED", 1.0)
        self.BATTERY_PER_KM = self.parameters.get("BATTERY_PER_KM", 10.0)
        self.relaxation_factors = {{
            {", ".join([f'"{constraint}": {relaxation}' for constraint, relaxation in relaxation_factors.items()])}
        }}
        
    def calculate_distance(self, point1: Dict, point2: Dict) -> float:
        return math.sqrt((point1["x"] - point2["x"]) ** 2 + (point1["y"] - point2["y"]) ** 2)
    
    def calculate_total_distance(self, route: List[int]) -> float:
        if not route:
            return 0
        
        total = 0
        prev = self.depot
        
        for point_id in route:
            point = next((p for p in self.delivery_points if p["id"] == point_id), None)
            if point:
                total += self.calculate_distance(prev, point)
                prev = point
        total += self.calculate_distance(prev, self.depot)
        
        return total
    
    def calculate_total_demand(self, route: List[int]) -> float:
        return sum(next((p["demand"] for p in self.delivery_points if p["id"] == point_id), 0) for point_id in route)
    
    def check_payload_constraint(self, route: List[int]) -> bool:
        relaxation = self.relaxation_factors.get("payload_constraint", 1.0)
        return self.calculate_total_demand(route) <= self.MAX_PAYLOAD * relaxation
    
    def check_battery_constraint(self, route: List[int]) -> bool:
        relaxation = self.relaxation_factors.get("battery_constraint", 1.0)
        total_distance = self.calculate_total_distance(route)
        return total_distance * self.BATTERY_PER_KM <= self.MAX_BATTERY * relaxation
    
    def solve(self) -> Dict:
        solution = {{
            "drone_routes": {{}},
            "unassigned_points": []
        }}
        delivery_points = sorted(self.delivery_points, key=lambda p: p["demand"], reverse=True)
        num_drones = min(self.MAX_DRONES, len(delivery_points))
        for i in range(num_drones):
            solution["drone_routes"][f"drone_{{i}}"] = []
        constraint_order = {constraint_order}
        for point in delivery_points:
            point_id = point["id"]
            assigned = False
            drone_loads = [(drone_id, self.calculate_total_demand(route)) 
                           for drone_id, route in solution["drone_routes"].items()]
            drone_loads.sort(key=lambda x: x[1])  
            for drone_id, load in drone_loads:
                route = solution["drone_routes"][drone_id]
                new_route = route + [point_id]
                valid = True
                for constraint in constraint_order:
                    if constraint == "payload_constraint" and not self.check_payload_constraint(new_route):
                        valid = False
                        break
                    elif constraint == "battery_constraint" and not self.check_battery_constraint(new_route):
                        valid = False
                        break
                
                if valid:
                    solution["drone_routes"][drone_id] = new_route
                    assigned = True
                    break
            if not assigned:
                solution["unassigned_points"].append(point_id)
        
        return solution
"""
        return code

    def generate_code_from_template(self, strategy: Dict, algorithm_name: str, variant_index: int = 0) -> Dict:
        strategy_id = strategy.get("id", str(uuid.uuid4()))
        strategy_text = strategy.get("text", "")
        constraint_order = strategy.get("constraint_order", [])
        relaxation_factors = strategy.get("relaxation_factors", {})
        template = self.algorithm_loader.get_template(algorithm_name)
        prompt = self._build_template_completion_prompt(
            template,
            strategy_text,
            constraint_order,
            relaxation_factors,
            algorithm_name,
            variant_index
        )
        start_time = time.time()

        try:
            generated_parts = self.llm_client.generate_code(
                prompt, self.problem_info)
            final_code = self._integrate_generated_parts(
                template, generated_parts)

            code_id = f"{strategy_id}_{algorithm_name}_{variant_index}_{int(time.time())}"
            code_path = self.output_manager.save_generated_code(
                final_code, code_id)

            generation_time = time.time() - start_time
            return {
                "code_id": code_id,
                "strategy_id": strategy_id,
                "algorithm": algorithm_name,
                "variant_index": variant_index,
                "code_path": code_path,
                "code": final_code,
                "generation_time": generation_time
            }

        except Exception as e:
            self.output_manager.log_error(
                "code_generator", "template_generation_error",
                f"Error generating code based on template: {str(e)}"
            )
            raise

    def _build_template_completion_prompt(self, template, strategy_text, constraint_order, relaxation_factors, algorithm_name, variant_index):

        limited_problem_data = self.load_limited_problem_data(max_points=5)
        data_access_guide = self._generate_data_access_guide(
            limited_problem_data)
        function_templates = self._generate_common_function_templates()
        key_paths = self._extract_key_data_paths(limited_problem_data)

        data_structure_examples = "【Data Access Examples】\n"
        for path_name, path_info in key_paths.items():
            path = path_info["path"]
            data_type = path_info["type"]

            if data_type == "dict":
                data_structure_examples += f"# Access {path_name} dictionary\n{path_name} = self.problem_data{path}\n\n"
            elif data_type == "list":
                data_structure_examples += f"# Access {path_name} list\n{path_name}_list = self.problem_data{path}\n\n"
                data_structure_examples += f"# Get {path_name} list length\n{path_name}_count = len(self.problem_data{path})\n\n"
                data_structure_examples += f"# Access single element in {path_name} list\n{path_name}_item = self.problem_data{path}[i]\n\n"
                try:
                    current_data = limited_problem_data
                    path_parts = path.strip('[]').split('][')
                    for part in path_parts:
                        if part.startswith('[') and part.endswith(']'):
                            part = part[1:-1]
                        if part.isdigit():
                            current_data = current_data[int(part)]
                        else:
                            current_data = current_data.get(part, [])

                    if current_data and isinstance(current_data, list) and len(current_data) > 0:
                        first_item = current_data[0]
                        data_structure_examples += f"# {path_name} item example\n# {json.dumps(first_item, indent=2)}\n\n"
                except Exception:
                    pass
            else:
                data_structure_examples += f"# Access {path_name}\n{path_name}_value = self.problem_data{path}\n\n"
        fill_parts = self._extract_fill_parts(template)
        problem_type = self.problem_info.get("problem_type", "")
        problem_description = self.problem_info.get("description", "")
        detailed_description = limited_problem_data.get(
            "detailed_description", {}).get("overview", "")
        constraints_text = ""
        for i, constraint in enumerate(constraint_order):
            relaxation = relaxation_factors.get(constraint, 1.0)
            constraint_info = None
            for c in limited_problem_data.get("detailed_description", {}).get("constraints", []):
                if c.get("name") == constraint:
                    constraint_info = c
                    break

            if constraint_info:
                constraints_text += f"{i+1}. {constraint} (Relaxation factor: {relaxation}):\n"
                constraints_text += f"   Description: {constraint_info.get('description')}\n"
                constraints_text += f"   Calculation method: {constraint_info.get('calculation')}\n"
                constraints_text += f"   Formula: {constraint_info.get('formula')}\n\n"
            else:
                constraints_text += f"{i+1}. {constraint} Relaxation factor: {relaxation}\n"
        parameters_info = "【Available Parameters】\n"
        for param_name, param_info in limited_problem_data.get("parameters", {}).items():
            if isinstance(param_info, dict) and "value" in param_info and "description" in param_info:
                parameters_info += f"{param_name} = {param_info['value']} - {param_info['description']}\n"
            else:
                parameters_info += f"{param_name} = {param_info}\n"
        objectives_info = "【Objective Functions】\n"
        for obj in limited_problem_data.get("detailed_description", {}).get("objectives", []):
            objectives_info += f"Objective: {obj.get('name')} - {obj.get('description')}\n"
            objectives_info += f"Direction: {obj.get('direction')}\n"
            objectives_info += f"Calculation: {obj.get('calculation')}\n"
            objectives_info += f"Formula: {obj.get('formula')}\n\n"
        auxiliary_info = "【Auxiliary Functions】\n"
        for func in limited_problem_data.get("detailed_description", {}).get("auxiliary_functions", []):
            auxiliary_info += f"Function: {func.get('name')} - {func.get('description')}\n"
            auxiliary_info += f"Formula: {func.get('formula')}\n\n"
        data_structure_text = "【Data Structure Access Guide】\n"
        if hasattr(self.algorithm_loader, 'single_algorithm_file') and self.algorithm_loader.single_algorithm_file:
            data_structure_text += "**Important: Please use the helper methods defined in the algorithm template to access data, do not access self.problem_data structure directly**\n\n"
            template_methods = self._extract_template_methods()
            if template_methods:
                data_structure_text += "【Available Methods in Algorithm Template】\n"
                for method_name, method_info in template_methods.items():
                    data_structure_text += f"- {method_name}: {method_info}\n"
                data_structure_text += "\n"
        else:
            data_structure_text += "The following shows how to access various parts of problem_data (limited to first 5 points for example data):\n"
            data_structure_text += self._generate_data_access_guide(
                limited_problem_data)
        needed_imports_text = """
【Required Import Instructions】
Please include all required module import statements at the beginning of your returned code, for example:
import random
import math
import copy
import numpy as np 
import time 
"""
        solution_format_text = "【Solution Format】\n"
        solution_format = limited_problem_data.get("solution_format", {})
        for key, info in solution_format.items():
            solution_format_text += f"{key}: {info.get('description', '')}\n"
            if "example" in info:
                solution_format_text += f"  Example: {json.dumps(info['example'])}\n"

        prompt = f"""
Please implement key functions in the {algorithm_name} algorithm for {problem_type}.

【Important Note: The returned methods will be integrated into Python classes, so no class indentation is needed, but please ensure proper indentation within methods】

【Problem Description】
{detailed_description if detailed_description else problem_description}

【Strategy Description】
{strategy_text}

【Constraint Processing Order and Relaxation Factors】
{constraints_text}


{parameters_info}

{data_structure_text}

【Important Instructions】
The structure of relaxation_strategy is a list, where each element is a dictionary containing 'constraint' and 'relaxation' keys:
[{{'constraint': 'constraint_name1', 'relaxation': 0.9}}, {{'constraint': 'constraint_name2', 'relaxation': 1.1}}, ...]
When processing constraint relaxation strategies, you must use the get_relaxation_factor method provided in the class to obtain relaxation factors, rather than directly accessing the self.relaxation_dict dictionary. For example:
```python
# Wrong way - may cause KeyError
max_unassigned = len(self.delivery_points) * (1 - self.relaxation_dict['allocation_constraint'])

# Correct way - safe access
allocation_factor = self.get_relaxation_factor('allocation_constraint')
max_unassigned = len(self.delivery_points) * (1 - allocation_factor)
Please always use the get_relaxation_factor method in your code to access relaxation factors, this method handles all possible key name variants and provides default values.
{objectives_info}

{auxiliary_info}

{solution_format_text}

{needed_imports_text}



【Complete Code Framework】
{template}
The parts you need to implement are between "# ============ LLM Fill Area - Start ============" and "# ============ LLM Fill Area - End ============".
These functions to implement include: {fill_parts}

When implementing these functions, please ensure:
1. Include all necessary import statements at the beginning of the code
2. Strictly follow constraint processing order: {constraint_order}
3. Correctly apply relaxation factors: {relaxation_factors} 
4. Do not use undefined attributes or methods
5. Write efficient, readable, and well-commented code

This is algorithm variant #{variant_index}, you can try different implementation methods to explore more possible solutions.


"""

        return prompt

    def _extract_fill_parts(self, template):
        start_marker = "# ============ LLM Fill Area - Start ============"
        end_marker = "# ============ LLM Fill Area - End ============"

        start_index = template.find(start_marker)
        end_index = template.find(end_marker)

        if start_index == -1 or end_index == -1:
            return "Cannot find LLM fill area, please check template format"
        fill_part = template[start_index + len(start_marker):end_index].strip()
        return fill_part

    def _integrate_generated_parts(self, template, generated_code):
        import re

        start_marker = "# ============ LLM Fill Area - Start ============"
        end_marker = "# ============ LLM Fill Area - End ============"
        start_index = template.find(start_marker) + len(start_marker)
        end_index = template.find(end_marker)

        if start_index == -1 or end_index == -1:
            return template
        cleaned_code = generated_code
        cleaned_code = re.sub(r'^```\s*(?:python)?\s*\n',
                              '', cleaned_code, flags=re.MULTILINE)
        cleaned_code = re.sub(r'\n\s*```\s*$', '',
                              cleaned_code, flags=re.MULTILINE)
        import_lines = []
        other_code_lines = []

        for line in cleaned_code.split('\n'):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_lines.append(line)
            else:
                other_code_lines.append(line)
        lines_before_marker = template[:start_index].split('\n')
        if lines_before_marker:
            last_line = lines_before_marker[-1]
            indentation = ' ' * 4
            for line in reversed(lines_before_marker[:-1]):
                stripped = line.lstrip()
                if stripped and stripped.startswith('def '):
                    indentation = line[:len(line)-len(stripped)]
                    break
        else:
            indentation = '    '
        indented_code = ""
        if import_lines:
            for line in import_lines:
                indented_code += indentation + line + '\n'
            indented_code += '\n'
        for line in other_code_lines:
            if line.lstrip().startswith('def ') or not line.strip() or line.lstrip().startswith('#'):
                indented_code += indentation + line + '\n'
            else:
                indented_code += indentation + '    ' + line + '\n'
        final_code = template[:start_index] + '\n\n' + \
            indented_code + '\n' + template[end_index:]

        return final_code

    def _generate_data_access_guide(self, data, prefix="self.problem_data", max_depth=3):
        guide = ["【Data Structure Guide】"]
        guide.append(f"{prefix} contains the following structure:")

        def analyze_structure(data, path, depth=0):
            if depth >= max_depth:
                return []

            lines = []
            indent = "  " * depth

            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}['{key}']"
                    type_info = type(value).__name__

                    if isinstance(value, dict):
                        lines.append(f"{indent}- {current_path}: dictionary")
                        lines.extend(analyze_structure(
                            value, current_path, depth + 1))
                    elif isinstance(value, list) and value:
                        lines.append(
                            f"{indent}- {current_path}: list (length: {len(value)})")
                        if value and isinstance(value[0], dict):
                            elem_path = f"{current_path}[0]"
                            lines.append(
                                f"{indent}  Example element access: {elem_path}")
                            lines.extend(analyze_structure(
                                value[0], elem_path, depth + 2))
                    else:
                        value_str = repr(value) if len(
                            repr(value)) < 50 else f"{repr(value)[:47]}..."
                        lines.append(
                            f"{indent}- {current_path}: {type_info} = {value_str}")

            elif isinstance(data, list) and data and max_depth > 1:
                if isinstance(data[0], dict):
                    lines.append(f"{indent}- {path}[0]: List element example")
                    lines.extend(analyze_structure(
                        data[0], f"{path}[0]", depth + 1))

            return lines

        guide.extend(analyze_structure(data, prefix))

        guide.append("\n【Common Access Patterns】")
        guide.append("- Access dictionary: `value = self.problem_data['key']`")
        guide.append(
            "- Access nested dictionary: `value = self.problem_data['outer']['inner']`")
        guide.append(
            "- Access list: `item = self.problem_data['list_key'][index]`")
        guide.append(
            "- Get list length: `length = len(self.problem_data['list_key'])`")
        guide.append(
            "- Iterate through list: `for item in self.problem_data['list_key']:`")

        return "\n".join(guide)

    def _find_path_to_key(self, data, target_key, current_path=""):
        if isinstance(data, dict):
            if target_key in data:
                return f"{current_path}['{target_key}']"

            for key, value in data.items():
                path = self._find_path_to_key(
                    value, target_key, f"{current_path}['{key}']")
                if path:
                    return path

        elif isinstance(data, list):
            for i, item in enumerate(data):
                path = self._find_path_to_key(
                    item, target_key, f"{current_path}[{i}]")
                if path:
                    return path

        return ""

    def _generate_common_function_templates(self):
        templates = {}
        key_paths = self._extract_key_data_paths(self.problem_data)
        point_paths = []
        for name, info in key_paths.items():
            try:
                path = info["path"]
                value = eval(f"self.problem_data{path}")
                if isinstance(value, dict) and "x" in value and "y" in value:
                    point_paths.append((name, path))
            except Exception:
                continue

        if point_paths:
            templates["calculate_distance"] = f"""
    def calculate_distance(self, point1, point2):
        \"\"\"Calculate Euclidean distance between two points\"""
        return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

    # Example:
    # Assuming the problem data contains point coordinates in the following format
    # Calculate distance between two points
    """
            if len(point_paths) >= 1:
                path1_name, path1 = point_paths[0]
                templates["calculate_distance"] += f"# point1 = self.problem_data{path1}\n"

                if len(point_paths) >= 2:
                    path2_name, path2 = point_paths[1]
                    templates["calculate_distance"] += f"# point2 = self.problem_data{path2}\n"
                else:
                    templates["calculate_distance"] += f"# point2 = other_point_coordinates\n"

                templates[
                    "calculate_distance"] += f"# distance = self.calculate_distance(point1, point2)\n"

        return templates

    def _extract_key_data_paths(self, data, prefix="", max_depth=4):
        if max_depth <= 0:
            return {}

        result = {}

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{prefix}['{key}']"
                if isinstance(value, dict):
                    result[key] = {"path": current_path, "type": "dict"}
                    sub_paths = self._extract_key_data_paths(
                        value, current_path, max_depth - 1)
                    result.update(sub_paths)

                elif isinstance(value, list) and value:
                    result[key] = {"path": current_path, "type": "list"}
                    if value and isinstance(value[0], dict):
                        elem_path = f"{current_path}[0]"
                        sub_paths = self._extract_key_data_paths(
                            value[0], elem_path, max_depth - 1)
                        result.update(sub_paths)

                else:
                    result[key] = {"path": current_path, "type": "value"}
        elif isinstance(data, list) and data and max_depth > 1:
            if isinstance(data[0], dict):
                current_path = f"{prefix}[0]"
                sub_paths = self._extract_key_data_paths(
                    data[0], current_path, max_depth - 1)
                result.update(sub_paths)

        return result

    def generate_variants_from_snippet(self, strategy: Dict, count: int = 1) -> List[Dict]:
        strategy_id = strategy.get("id", str(uuid.uuid4()))
        code_snippet = strategy.get("code_snippet", "")
        if not code_snippet:
            self.output_manager.log_warning(
                "code_generator", "empty_code_snippet",
                f"Strategy {strategy_id} has empty code snippet, cannot generate variants"
            )
            return []

        variants = []
        actual_count = max(1, count)

        for i in range(actual_count):
            variant_id = f"{strategy_id}_v{i}"
            if hasattr(self, 'algorithm_loader') and self.algorithm_loader:
                try:
                    available_algorithms = self.algorithm_loader.get_available_algorithms()
                    if available_algorithms:
                        algorithm_name = available_algorithms[0]
                        template_content = self.algorithm_loader.get_template(
                            algorithm_name)
                    else:
                        template_content = self._get_default_template()
                except Exception as e:
                    self.output_manager.log_error(
                        "code_generator", "template_load_error",
                        f"Unable to load template from algorithm loader: {e}"
                    )
                    template_content = self._get_default_template()
            else:
                template_content = self._get_default_template()
            code_snippet = self._preprocess_code_file(code_snippet)
            full_code = self._integrate_code_to_llm_area(
                template_content, code_snippet)
            file_path = os.path.join(
                self.output_manager.base_dir, "codes", f"{variant_id}.py")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(full_code)
            variant_info = {
                "id": variant_id,
                "strategy_id": strategy_id,
                "file_path": file_path,
                "variant_index": i,
                "method": "direct_snippet"
            }

            variants.append(variant_info)
        return variants

    def _integrate_code_to_llm_area(self, template_content: str, code_snippet: str) -> str:
        llm_start = template_content.find(
            "# ============ LLM Fill Area - Start ============")
        llm_end = template_content.find(
            "# ============ LLM Fill Area - End ============")

        if llm_start == -1 or llm_end == -1:
            return template_content + "\n\n" + code_snippet
        before_llm = template_content[:llm_start +
                                      len("# ============ LLM Fill Area - Start ============")]
        after_llm = template_content[llm_end:]
        indented_code_lines = []
        for line in code_snippet.split('\n'):
            if not line.strip():
                indented_code_lines.append("")
            else:
                indented_code_lines.append("    " + line)

        indented_code = "\n".join(indented_code_lines)

        return before_llm + "\n" + indented_code + "\n" + after_llm

    def _preprocess_code_file(self, code_snippet: str) -> str:
        try:

            cleaned_code = re.sub(r'^```\s*python\s*', '', code_snippet)
            cleaned_code = re.sub(r'^```', '', cleaned_code)
            lines = cleaned_code.splitlines()
            if lines and lines[-1].strip() == '}':
                left_braces = cleaned_code.count('{')
                right_braces = cleaned_code.count('}')
                if right_braces > left_braces:
                    lines = lines[:-1]
            fixed_code = '\n'.join(lines)
            if not fixed_code.endswith('\n'):
                fixed_code += '\n'

            return fixed_code

        except Exception as e:
            self.output_manager.log_error(
                "code_generator", "code_preprocess_error",
                f"Error preprocessing code snippet: {str(e)}"
            )
            return code_snippet

    def _limit_problem_data_points(self, problem_data: Dict, max_points: int = 5) -> Dict:
        if not problem_data:
            return {}
        limited_data = deepcopy(problem_data)
        if "data" in limited_data and isinstance(limited_data["data"], list):
            for i, data_item in enumerate(limited_data["data"]):
                if isinstance(data_item, dict):
                    original_item = problem_data.get("data", [])[i] if i < len(
                        problem_data.get("data", [])) else {}
                    self._limit_points_in_dict(
                        data_item, max_points, original_item)

        for field_name in ["parameters", "constraints", "objectives"]:
            if field_name in limited_data and isinstance(limited_data[field_name], dict):
                original_field = problem_data.get(field_name, {})
                self._limit_points_in_dict(
                    limited_data[field_name], max_points, original_field)

        return limited_data

    def _limit_points_in_dict(self, data_dict: Dict, max_points: int, original_dict: Dict = None):
        if original_dict is None:
            original_dict = data_dict

        for field_name, field_value in list(data_dict.items()):
            if field_name in ["description", "type", "name", "id"]:
                continue

            if field_name.endswith("_limitation_note"):
                continue

            if isinstance(field_value, list) and len(field_value) > 0:
                original_value = original_dict.get(field_name, [])
                original_count = len(original_value)

                if self._is_object_list(field_value):
                    if len(field_value) > max_points:
                        data_dict[field_name] = field_value[:max_points]
                        note_key = f"_{field_name}_limitation_note"
                        data_dict[note_key] = f"Only showing first {max_points} {field_name}, actual data contains {original_count} items"

                elif self._is_matrix(field_value):

                    original_rows = len(original_value)
                    original_cols = len(original_value[0]) if original_value and len(
                        original_value) > 0 else 0

                    if len(field_value) > max_points:

                        data_dict[field_name] = field_value[:max_points]

                    for i, row in enumerate(data_dict[field_name]):
                        if isinstance(row, list) and len(row) > max_points:
                            data_dict[field_name][i] = row[:max_points]

                    if original_rows > max_points or original_cols > max_points:
                        note_key = f"_{field_name}_limitation_note"
                        data_dict[field_name +
                                  "_limitation_note"] = f"Only showing first {max_points}x{max_points} of {field_name} matrix, actual matrix size is {original_rows}x{original_cols}"

                elif self._is_large_array(field_value, threshold=max_points * 2):
                    # Large one-dimensional array
                    if len(field_value) > max_points:
                        data_dict[field_name] = field_value[:max_points]
                        note_key = f"_{field_name}_limitation_note"
                        data_dict[note_key] = f"Only showing first {max_points} {field_name} elements, actual array contains {original_count} elements"

    def _is_object_list(self, data_list: list) -> bool:

        if not data_list:
            return False

        sample_size = min(3, len(data_list))
        dict_count = sum(
            1 for item in data_list[:sample_size] if isinstance(item, dict))
        return dict_count > 0

    def _is_matrix(self, data_list: list) -> bool:

        if not data_list:
            return False

        if not all(isinstance(item, list) for item in data_list):
            return False

        first_row = data_list[0]
        return len(first_row) > 0 and isinstance(first_row[0], (int, float))

    def _is_large_array(self, data_list: list, threshold: int = 10) -> bool:

        if not data_list or len(data_list) <= threshold:
            return False

        sample_size = min(3, len(data_list))
        simple_types = (int, float, str, bool)
        simple_count = sum(
            1 for item in data_list[:sample_size] if isinstance(item, simple_types))
        return simple_count == sample_size

    def generate_limited_problem_data_file(self, output_path: str = None, max_points: int = 5) -> str:

        if output_path is None:

            problem_type = self.problem_info.get("problem_type", "default")
            output_path = os.path.join(os.path.dirname(
                __file__), f"problem_data_limited_{problem_type}_{max_points}.json")

        limited_data = self._limit_problem_data_points(
            self.problem_data, max_points)

        limited_data["_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "max_points_limit": max_points,
            "purpose": "Limited data for LLM prompt generation",
            "original_data_source": "problem_data.json"
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(limited_data, f, indent=2, ensure_ascii=False)

            return output_path

        except Exception as e:

            raise

    def load_limited_problem_data(self, file_path: str = None, max_points: int = 5) -> Dict:

        if not self.problem_data or self.problem_data == {}:

            return {}

        if file_path is None:

            problem_type = self.problem_info.get("problem_type", "default")
            file_path = os.path.join(os.path.dirname(
                __file__), f"problem_data_limited_{problem_type}_{max_points}.json")

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    limited_data = json.load(f)

                metadata = limited_data.get("_metadata", {})
                if metadata.get("max_points_limit") == max_points:
                    return limited_data

            except Exception as e:
                self.output_manager.log_warning(
                    "code_generator", "limited_data_load_error",
                    f"Failed to load limited data file: {str(e)}, will regenerate"
                )

        self.generate_limited_problem_data_file(file_path, max_points)

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_template_methods(self):

        methods = {}

        if not hasattr(self.algorithm_loader, 'single_algorithm_file') or not self.algorithm_loader.single_algorithm_file:
            return methods

        try:

            available_algorithms = self.algorithm_loader.get_available_algorithms()
            if not available_algorithms:
                return methods

            algorithm_name = available_algorithms[0]
            template_content = self.algorithm_loader.get_template(
                algorithm_name)
            if not template_content:
                return methods

            import re

            method_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\):'
            matches = re.findall(method_pattern, template_content)
            for method_name in matches:
                if method_name.startswith('_extract'):
                    methods[method_name] = "Data extraction method for retrieving problem data"
                elif method_name == 'generate_solution':
                    methods[method_name] = "Main method for generating solutions"
                elif method_name == 'select_next_point':
                    methods[method_name] = "Main method for generating solutions"
                elif method_name.startswith('calculate'):
                    methods[method_name] = "Calculation-related auxiliary method"
                elif method_name.startswith('_'):
                    methods[method_name] = "Internal auxiliary method"
                else:
                    methods[method_name] = "Algorithm implementation method"

        except Exception as e:
            self.output_manager.log_error(
                "code_generator", "template_method_extraction_failed",
                f"Failed to extract template methods: {e}"
            )

        return methods
