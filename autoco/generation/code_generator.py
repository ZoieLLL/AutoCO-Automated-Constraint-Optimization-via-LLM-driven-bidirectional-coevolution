from typing import Dict, List
import os
import re
import time
import uuid
from datetime import datetime
from autoco import PROJECT_ROOT
from autoco.output_manager import OutputManager
from autoco.llm.llm_client import CodeLLMClient
from copy import deepcopy
import json


class CodeGenerator:
    def __init__(self, llm_client: CodeLLMClient,
                 problem_info: Dict,
                 output_manager: OutputManager):
        self.llm_client = llm_client
        self.problem_info = problem_info
        self.output_manager = output_manager
        self.generated_codes = {}
        self.problem_data = {} 

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
                f"Template generation error: {str(e)}"
            )
            raise

    def _build_template_completion_prompt(self, template, strategy_text, constraint_order, relaxation_factors, algorithm_name, variant_index):

        limited_problem_data = self.load_limited_problem_data(max_points=5)

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
                constraints_text += f"{i+1}. {constraint} (relaxation factor: {relaxation}):\n"
                constraints_text += f"   Description: {constraint_info.get('description')}\n"
                constraints_text += f"   Calculation: {constraint_info.get('calculation')}\n"
                constraints_text += f"   Formula: {constraint_info.get('formula')}\n\n"
            else:
                constraints_text += f"{i+1}. {constraint} relaxation factor: {relaxation}\n"

        parameters_info = "[Available Parameters]\n"
        for param_name, param_info in limited_problem_data.get("parameters", {}).items():
            if isinstance(param_info, dict) and "value" in param_info and "description" in param_info:
                parameters_info += f"{param_name} = {param_info['value']} - {param_info['description']}\n"
            else:
                parameters_info += f"{param_name} = {param_info}\n"

        objectives_info = "[Objective Functions]\n"
        for obj in limited_problem_data.get("detailed_description", {}).get("objectives", []):
            objectives_info += f"Objective: {obj.get('name')} - {obj.get('description')}\n"
            objectives_info += f"Direction: {obj.get('direction')}\n"
            objectives_info += f"Calculation: {obj.get('calculation')}\n"
            objectives_info += f"Formula: {obj.get('formula')}\n\n"

        auxiliary_info = "[Auxiliary Functions]\n"
        for func in limited_problem_data.get("detailed_description", {}).get("auxiliary_functions", []):
            auxiliary_info += f"Function: {func.get('name')} - {func.get('description')}\n"
            auxiliary_info += f"Formula: {func.get('formula')}\n\n"
        data_structure_text = "[Data Structure Access Guide]\n"

        if hasattr(self.algorithm_loader, 'single_algorithm_file') and self.algorithm_loader.single_algorithm_file:
            data_structure_text += "**Important: Use the helper methods defined in the algorithm template to access data; do NOT access self.problem_data directly!**\n\n"
            data_structure_text += "Data access methods provided for you:\n"
            data_structure_text += "- self._extract_direct_info(): Extract basic problem information (e.g., node coordinates, scale, etc.)\n"
            data_structure_text += "- self._extract_instance_info(): Extract detailed information for a specific instance\n"
            data_structure_text += "- Other methods defined in the algorithm template\n\n"
            data_structure_text += "**Please prefer these methods to obtain data instead of directly accessing paths like self.problem_data['data'][0]!**\n\n"

            template_methods = self._extract_template_methods()
            if template_methods:
                data_structure_text += "[Methods Available in the Algorithm Template]\n"
                for method_name, method_info in template_methods.items():
                    data_structure_text += f"- {method_name}: {method_info}\n"
                data_structure_text += "\n"
        else:
            data_structure_text += "The following shows how to access each part of problem_data (limited to the first 5 points of sample data):\n"
            data_structure_text += self._generate_data_access_guide(
                limited_problem_data)

        algorithm_specific = ""
        if algorithm_name == "AntColonyOptimization":
            algorithm_specific = "For the ant colony algorithm, pay special attention to pheromone initialization and update strategies."
        elif algorithm_name == "GeneticAlgorithm":
            algorithm_specific = "For the genetic algorithm, pay special attention to individual representation, crossover, and mutation operations."
        elif algorithm_name == "NeighborhoodSearch":
            algorithm_specific = "For variable neighborhood search, pay special attention to neighborhood structure definition and move strategies."

        needed_imports_text = """
[Required Imports]
Include all required module import statements at the beginning of the code you return, for example:
import random
import math
import copy
import numpy as np  # if needed
import time  # for timing
Add any other packages you need according to your implementation.
"""
        solution_format_text = "[Solution Format]\n"
        solution_format = limited_problem_data.get("solution_format", {})
        for key, info in solution_format.items():
            solution_format_text += f"{key}: {info.get('description', '')}\n"
            if "example" in info:
                solution_format_text += f"  Example: {json.dumps(info['example'])}\n"

        prompt = f"""
Implement the key functions of the {algorithm_name} algorithm for {problem_type}.

[Important: The returned methods will be integrated into a Python class, so class-level indentation is not required, but ensure the code inside the methods is correctly indented]

[Problem Description]
{detailed_description if detailed_description else problem_description}

[Strategy Description]
{strategy_text}

[Constraint Handling Order and Relaxation Factors]
{constraints_text}


{parameters_info}

{data_structure_text}

[Important]
The structure of relaxation_strategy is a list, where each element is a dict containing the keys 'constraint' and 'relaxation':
[{{'constraint': 'constraint_name1', 'relaxation': 0.9}}, {{'constraint': 'constraint_name2', 'relaxation': 1.1}}, ...]
When handling the constraint relaxation strategy, you must use the get_relaxation_factor method provided by the class to obtain the relaxation factor instead of directly accessing the self.relaxation_dict dictionary. For example:
```python
# Wrong way - may cause KeyError
max_unassigned = len(self.delivery_points) * (1 - self.relaxation_dict['allocation_constraint'])

# Correct way - safe access
allocation_factor = self.get_relaxation_factor('allocation_constraint')
max_unassigned = len(self.delivery_points) * (1 - allocation_factor)
Always use the get_relaxation_factor method in your code to access relaxation factors; it handles all possible key-name variants and provides default values.
{objectives_info}

{auxiliary_info}

{solution_format_text}

{needed_imports_text}

[Algorithm Characteristics]
{algorithm_specific}


[Complete Code Skeleton]
{template}
The parts you need to implement are between "# ============ LLM Fill Area - START ============" and "# ============ LLM Fill Area - END ============".
The functions you need to implement include: {fill_parts}

When implementing these functions, please ensure:
1. Include all necessary import statements at the very beginning of the code
2. Strictly follow the constraint handling order: {constraint_order}
3. Correctly apply the relaxation factors: {relaxation_factors} 
4. Do not use undefined attributes or methods
5. The code should be efficient, readable, and commented

This is algorithm variant #{variant_index}; you can try different implementation approaches to explore more possible solutions.


"""

        return prompt

    def _extract_fill_parts(self, template):
        start_marker = "# ============ LLM Fill Area - START ============"
        end_marker = "# ============ LLM Fill Area - END ============"

        start_index = template.find(start_marker)
        end_index = template.find(end_marker)

        if start_index == -1 or end_index == -1:
            return "Cannot find the LLM fill area; please check the template format"

        fill_part = template[start_index + len(start_marker):end_index].strip()
        return fill_part

    def _integrate_generated_parts(self, template, generated_code):
        import re

        start_marker = "# ============ LLM Fill Area - START ============"
        end_marker = "# ============ LLM Fill Area - END ============"

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
        guide = ["[Data Structure Guide]"]
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
                        lines.append(f"{indent}- {current_path}: dict")
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
                    lines.append(f"{indent}- {path}[0]: list element example")
                    lines.extend(analyze_structure(
                        data[0], f"{path}[0]", depth + 1))

            return lines

        guide.extend(analyze_structure(data, prefix))

        guide.append("\n[Common Access Patterns]")
        guide.append("- Access a dict: `value = self.problem_data['key']`")
        guide.append("- Access a nested dict: `value = self.problem_data['outer']['inner']`")
        guide.append("- Access a list: `item = self.problem_data['list_key'][index]`")
        guide.append("- Get list length: `length = len(self.problem_data['list_key'])`")
        guide.append("- Iterate over a list: `for item in self.problem_data['list_key']:`")

        return "\n".join(guide)

    def generate_variants_from_snippet(self, strategy: Dict, count: int = 1) -> List[Dict]:
        strategy_id = strategy.get("id", str(uuid.uuid4()))
        code_snippet = strategy.get("code_snippet", "")

        if not code_snippet:
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
                except Exception:
                    template_content = self._get_default_template()
            else:
                template_content = self._get_default_template()
            code_snippet = self._preprocess_code_file(code_snippet)
            full_code = self._integrate_code_to_llm_area(
                template_content, code_snippet)
            file_path = os.path.join(
                self.output_manager.code_dir, f"{variant_id}.py")
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
            "# ============ LLM Fill Area - START ============")
        llm_end = template_content.find(
            "# ============ LLM Fill Area - END ============")

        if llm_start == -1 or llm_end == -1:
            return template_content + "\n\n" + code_snippet

        before_llm = template_content[:llm_start +
                                      len("# ============ LLM Fill Area - START ============")]
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

        except Exception:
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
                        data_dict[note_key] = f"Only the first {max_points} {field_name} are shown here; the actual data contains {original_count} items"

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
                                  "_limitation_note"] = f"Only the first {max_points}x{max_points} of the {field_name} matrix are shown here; the actual matrix size is {original_rows}x{original_cols}"

                elif self._is_large_array(field_value, threshold=max_points * 2):
                    if len(field_value) > max_points:
                        data_dict[field_name] = field_value[:max_points]
                        note_key = f"_{field_name}_limitation_note"
                        data_dict[note_key] = f"Only the first {max_points} elements of {field_name} are shown here; the actual array contains {original_count} elements"

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
            output_path = os.path.join(PROJECT_ROOT, f"problem_data_limited_{problem_type}_{max_points}.json")

        
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
            file_path = os.path.join(PROJECT_ROOT, f"problem_data_limited_{problem_type}_{max_points}.json")

       
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    limited_data = json.load(f)

               
                metadata = limited_data.get("_metadata", {})
                if metadata.get("max_points_limit") == max_points:
                    self.output_manager.log_info(
                        "code_generator", "limited_data_loaded",
                        f"Loaded existing limited data file: {file_path}"
                    )
                    return limited_data

            except Exception as e:
                self.output_manager.log_warning(
                    "code_generator", "limited_data_load_error",
                    f"Failed to load the limited data file: {str(e)}; will regenerate"
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
                    methods[method_name] = "Data extraction method for obtaining problem data"
                elif method_name == 'generate_solution':
                    methods[method_name] = "Main method for generating the solution"
                elif method_name == 'select_next_point':
                    methods[method_name] = "Main method for generating the solution"
                elif method_name.startswith('calculate'):
                    methods[method_name] = "Helper method for calculations"
                elif method_name.startswith('_'):
                    methods[method_name] = "Internal helper method"
                else:
                    methods[method_name] = "Algorithm implementation method"

        except Exception as e:
            self.output_manager.log_error(
                "code_generator", "template_method_extraction_failed",
                f"Failed to extract template methods: {e}"
            )

        return methods
