from typing import Dict, Optional
import json
import time
import re
import os
from datetime import datetime
from autoco import PROJECT_ROOT
from autoco.llm.llm_client import LLMClient
from autoco.output_manager import OutputManager


class RepairAgent:
    def __init__(self, llm_client: LLMClient,
                 output_manager: OutputManager,
                 config: Dict,
                 algorithm_loader=None):
        self.llm_client = llm_client
        self.output_manager = output_manager
        self.config = config
        self.algorithm_loader = algorithm_loader
        self.repair_timeout = config.get('repair_timeout', 60)

        self.include_full_template = config.get(
            'include_full_template', False)  
        self.max_template_length = config.get(
            'max_template_length', 20000) 

    def repair_strategy_and_code(self, strategy_data: Dict,
                                 violation_analysis: Dict) -> Optional[Dict]:
        try:
            repair_prompt = self.generate_repair_prompt(
                violation_analysis, strategy_data
            )

            llm_response = self.llm_client.generate_text(repair_prompt)

            repaired_data = self.parse_repair_result(llm_response)

            if repaired_data:
                return repaired_data
            else:
                default_repair = self.apply_default_repair(violation_analysis)
                return default_repair

        except Exception as e:
            self.output_manager.log_error(
                "repair_agent", "repair_failed",
                f"repair failed: {str(e)}"
            )
            return self.apply_default_repair(violation_analysis)

    def generate_repair_prompt(self, violation_analysis: Dict,
                               strategy_data: Dict) -> str:
        current_violations = self.format_violations(violation_analysis)
        strategy_text = strategy_data.get('text', '')
        current_code = strategy_data.get(
            'code_snippet', strategy_data.get('generated_code', ''))
        relaxation_strategy = strategy_data.get('relaxation_strategy', [])
        constraint_order = strategy_data.get('constraint_order', [])
        relaxation_factors = strategy_data.get('relaxation_factors', {})

        framework_context = self._get_framework_context()

        prompt = f"""

=== Current Complete Framework Details ===
{framework_context}
=== Current Strategy Execution Results ===
The following constraint violations occurred after strategy execution:
{current_violations}

=== Current Strategy Information ===
Strategy Text: {strategy_text}
Constraint Order: {constraint_order}
Relaxation Factors: {relaxation_factors}

=== [Code to Repair] select_next_point Function ===
The following is the current problematic select_next_point function implementation (needs repair):
```python
{current_code}
```

=== Repair Requirements ===
Please analyze the causes of the above violations and provide a repair plan:

⚠️ **Critical Repair Requirements** ⚠️
1. **Generate only one select_next_point function**: Implement all repair logic and algorithm improvements within this single function
2. **Do not generate multiple def functions**: Absolutely do not create auxiliary functions such as _calculate_density or _calculate_balance
3. **Implement all logic inside the function**: All required computation, judgment, and processing logic must be implemented directly inside the select_next_point function
4. **Keep correct indentation**: Function indentation must be 4 spaces (class method level)

Specific repair content:
- Adjust the relaxation factors of the relaxation strategy (increase relaxation factors for severely violated constraints)
- Modify the algorithm logic of the select_next_point function to better handle constraints
- Rebalance constraint priorities
- The code must be strictly wrapped between ```python and ```
- The code must contain a complete and self-contained select_next_point function implementation

Please strictly return the repair result in the following JSON format:
{{
    "relaxation_strategy": [
        {{"constraint": "constraint name", "relaxation": relaxation factor value}},
        {{"constraint": "constraint name", "relaxation": relaxation factor value}}
    ],
    "text": "repaired strategy text description",
    "generated_code": "```python\\n\\n    def select_next_point(self,\\n                          current_drone_id=None,\\n                          available_points=None,\\n                          current_position=None,\\n                          current_load=None,\\n                          solution=None,\\n                          violations=None,\\n                          iteration=None,\\n                          max_iterations=None):\\n        \\\"\\\"\\\"Repaired select_next_point function implementation\\\"\\\"\\\"\\n        # Implement all repair logic inside this function\\n        # Use the repaired relaxation factors to adjust constraint limits\\n        # Handle four scenarios: select next point, apply relaxation strategy, repair solution, adjust relaxation factors\\n        # If auxiliary logic such as density computation is needed, implement it directly in this function\\n        # Do not create additional auxiliary functions\\n        pass\\n```"
}}

⚠️ **Repeated Important Reminders** ⚠️
- **Repair focus**: Only repair the select_next_point function; keep other framework code unchanged
- **Single function principle**: The generated code will be added to the LLM fill area of the base template and must contain only one select_next_point function
- **Completeness requirement**: Although there is only one function, it must contain all repaired logic to ensure completeness and independence of the code
- **Indentation consistency**: Function indentation must be correct (4 spaces), consistent with class methods
- **Framework compatibility**: You may use all properties and methods provided by the framework (as shown above)

Repair guidelines:
- For severely violated constraints, increase the relaxation factor by 1.5-2.0x
- For mildly violated constraints, increase the relaxation factor by 1.2-1.5x
- Ensure the repaired code is syntactically correct and logically complete
- All logic must be implemented inside the select_next_point function without relying on external auxiliary functions
"""
        return prompt

    def format_violations(self, violation_analysis: Dict) -> str:
        violations = []

        if 'violation_details' in violation_analysis:
            violation_details = violation_analysis['violation_details']
            if isinstance(violation_details, dict):
                for constraint_name, details in violation_details.items():
                    violations.append(f"• {constraint_name}: {details}")

        for constraint_name, details in violation_analysis.items():
            if constraint_name in ['has_violations', 'violation_count', 'violation_summary',
                                   'severity_level', 'suggested_relaxations', 'violation_details']:
                continue

            if isinstance(details, dict) and details.get('violated', False):
                violations.append(
                    f"• {constraint_name}: {details.get('description', '')}"
                    f" (violation degree: {details.get('total_violation_degree', 0)})"
                )

        if not violations and violation_analysis.get('has_violations'):
            violations.append(
                f"• Overall violation: {violation_analysis.get('violation_summary', 'unknown violation')}")

        return "\n".join(violations) if violations else "No specific violation information found"

    def parse_repair_result(self, llm_response: str) -> Optional[Dict]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                if 'relaxation_strategy' in result or 'text' in result or 'generated_code' in result:
                    standardized_result = {}

                    if 'relaxation_strategy' in result:
                        standardized_result['relaxation_strategy'] = result['relaxation_strategy']

                        constraint_order = []
                        for item in result['relaxation_strategy']:
                            if isinstance(item, dict) and 'constraint' in item:
                                constraint_order.append(item['constraint'])
                        standardized_result['constraint_order'] = constraint_order

                    if 'text' in result:
                        standardized_result['text'] = result['text']
                    if 'generated_code' in result:
                        generated_code = result['generated_code']
                        if isinstance(generated_code, str):
                            if hasattr(self, 'llm_client') and hasattr(self.llm_client, 'extract_content'):
                                extracted = self.llm_client.extract_content(
                                    generated_code)
                                if extracted.get('code'):
                                    standardized_result['generated_code'] = extracted['code']
                                else:
                                    extracted_code = self._extract_code_from_text(
                                        generated_code)
                                    standardized_result['generated_code'] = extracted_code if extracted_code else generated_code
                            else:
                                extracted_code = self._extract_code_from_text(
                                    generated_code)
                                standardized_result['generated_code'] = extracted_code if extracted_code else generated_code
                        else:
                            standardized_result['generated_code'] = generated_code

                    if not standardized_result.get('generated_code'):
                        extracted_code = self._extract_code_from_text(
                            llm_response)
                        if extracted_code:
                            standardized_result['generated_code'] = extracted_code

                    for key in ['code_snippet', 'algorithm_design']:
                        if key in result:
                            standardized_result[key] = result[key]


                    return standardized_result

            text_result = self.parse_text_repair_result(llm_response)
            if text_result:
                return text_result

            return None

        except Exception as e:
            self.output_manager.log_error(
                "repair_agent", "parse_error",
                f"parse error: {str(e)}"
            )
            return None

    def _extract_code_from_text(self, text: str) -> Optional[str]:
        code_patterns = [
            r'```python\s*\n(.*?)\n```',  
            r'```python(.*?)```',         
            r'```Python\s*\n(.*?)\n```', 
            r'```Python(.*?)```',         
            r"'''python\s*\n(.*?)\n'''",  
            r"'''python(.*?)'''",         
            r"'''Python\s*\n(.*?)\n'''", 
            r"'''Python(.*?)'''",         
            r'```\s*python\s*\n(.*?)\n```',  
            r'```\s*python\s*(.*?)```',   
            r'```\s*\n(.*?)\n```',        
            r'```(.*?)```',             
            r"'''\s*\n(.*?)\n'''",        
            r"'''(.*?)'''"                
        ]

        for pattern in code_patterns:
            code_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if code_match:
                extracted_code = code_match.group(1).strip()
                if extracted_code and len(extracted_code) > 10:  
                    return extracted_code

        return None

    def parse_text_repair_result(self, llm_response: str) -> Optional[Dict]:
        try:
            result = {}
            relaxation_pattern = r'(\w+_constraint).*?(\d+\.\d+)'
            relaxation_matches = re.findall(relaxation_pattern, llm_response)

            if relaxation_matches:
                relaxation_strategy = []
                for constraint, factor in relaxation_matches:
                    relaxation_strategy.append({
                        "constraint": constraint,
                        "relaxation": float(factor)
                    })
                result['relaxation_strategy'] = relaxation_strategy
                result['constraint_order'] = [item['constraint']
                                              for item in relaxation_strategy]
            if 'strategy' in llm_response.lower() or 'repair' in llm_response.lower():
                text_lines = llm_response.split('\n')
                strategy_text = []
                for line in text_lines:
                    if line.strip() and not line.startswith('{') and not line.startswith('}'):
                        strategy_text.append(line.strip())
                        if len(strategy_text) >= 5:  
                            break

                if strategy_text:
                    result['text'] = '\n'.join(strategy_text)
            code_patterns = [
                r'```python\s*\n(.*?)\n```',  
                r'```python(.*?)```',         
                r'```Python\s*\n(.*?)\n```', 
                r'```Python(.*?)```',        
                r"'''python\s*\n(.*?)\n'''",  
                r"'''python(.*?)'''",         
                r"'''Python\s*\n(.*?)\n'''",  
                r"'''Python(.*?)'''",         
                r'```\s*python\s*\n(.*?)\n```',  
                r'```\s*python\s*(.*?)```',   
                r'```\s*\n(.*?)\n```',       
                r'```(.*?)```',               
                r"'''\s*\n(.*?)\n'''",       
                r"'''(.*?)'''"               
            ]

            extracted_code = None
            for pattern in code_patterns:
                code_match = re.search(
                    pattern, llm_response, re.DOTALL | re.IGNORECASE)
                if code_match:
                    extracted_code = code_match.group(1).strip()
                    if extracted_code:  # ensure the extracted code is not empty
                        break

            if extracted_code:
                result['generated_code'] = extracted_code

            return result if result else None

        except Exception as e:
            self.output_manager.log_error(
                "repair_agent", "text_parse_error",
                f"text parse error: {str(e)}"
            )
            return None

    def apply_default_repair(self, violation_analysis: Dict) -> Dict:
        default_relaxation = []
        constraint_order = []
        if 'violation_details' in violation_analysis:
            violation_details = violation_analysis['violation_details']
            if isinstance(violation_details, dict):
                for constraint_name, details in violation_details.items():
                    default_relaxation.append({
                        "constraint": constraint_name,
                        "relaxation": 2.0
                    })
                    constraint_order.append(constraint_name)
        for constraint_name, details in violation_analysis.items():
            if constraint_name in ['has_violations', 'violation_count', 'violation_summary',
                                   'severity_level', 'suggested_relaxations', 'violation_details']:
                continue

            if isinstance(details, dict) and details.get('violated', False):
                violation_degree = details.get('total_violation_degree', 0)
                if violation_degree > 10000:
                    relaxation_factor = 3.0
                elif violation_degree > 1000:
                    relaxation_factor = 2.0
                else:
                    relaxation_factor = 1.5

                default_relaxation.append({
                    "constraint": constraint_name,
                    "relaxation": relaxation_factor
                })
                constraint_order.append(constraint_name)
        if not default_relaxation and violation_analysis.get('has_violations'):
            default_relaxation = [
                {"constraint": "battery_constraint", "relaxation": 1.5},
                {"constraint": "payload_constraint", "relaxation": 1.5},
                {"constraint": "time_constraint", "relaxation": 1.5}
            ]
            constraint_order = ["battery_constraint",
                                "payload_constraint", "time_constraint"]

        repair_text = "Applied default repair strategy: adjust constraint relaxation factors based on violation severity"
        if default_relaxation:
            relaxation_desc = []
            for item in default_relaxation:
                relaxation_desc.append(
                    f"{item['constraint']} relaxed by {item['relaxation']}")
            repair_text += "\n" + "\n".join(relaxation_desc)

        return {
            "relaxation_strategy": default_relaxation,
            "constraint_order": constraint_order,
            "text": repair_text,
            "generated_code": None  
        }

    def _get_framework_context(self) -> str:
        try:
            template_content = None
            if self.algorithm_loader:
                available_algorithms = self.algorithm_loader.get_available_algorithms()
                if available_algorithms:
                    algorithm_name = "PointSelectionAlgorithm"
                    if algorithm_name not in available_algorithms:
                        algorithm_name = available_algorithms[0]

                    template_content = self.algorithm_loader.get_template(
                        algorithm_name)
                else:
                    self.output_manager.log_warning(
                        "repair_agent", "no_algorithms_available",
                        "no algorithms available"
                    )

            if not template_content:
                base_template_path = os.path.join(
                    PROJECT_ROOT, 'base_algorithm', 'PointSelectionAlgorithm.py'
                )

                if not os.path.exists(base_template_path):
                    return "Unable to read the base template file"

                with open(base_template_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()

            if self.include_full_template:
                if len(template_content) > self.max_template_length:
                    template_content = template_content[:self.max_template_length] + \
                        "\n# ... Template content truncated ..."

                context = f"""
=== Framework Context Information ===

You are repairing the select_next_point function for the Solution class of the current optimization problem.

**Full content of the current base template**:
```python
{template_content}
```

**Important notes**:
- The template above shows the complete framework structure and available functionality
- You need to repair the select_next_point function implementation within it
- You may use all properties and methods defined in the template
- Please keep the interface consistent with the one defined in the template
- The generated code will replace the content of the LLM fill area in the template

**Repair focus**:
- Only repair the select_next_point function; keep other framework code unchanged
- Ensure the repaired code is compatible with the template's interfaces and data structures
- Leverage all available functionality provided in the template to improve the algorithm logic
"""
            else:
                key_sections = self._extract_key_sections(template_content)

                context = f"""
=== Framework Context Information ===

You are repairing the select_next_point function for the Solution class of the current optimization problem.

**Key information of the current base template**:
{key_sections}

**Important notes**:
- You need to repair the select_next_point function implementation
- You may use all properties and methods defined in the template
- Please keep the interface consistent with the one defined in the template
- The generated code will replace the content of the LLM fill area in the template

**Repair focus**:
- Only repair the select_next_point function; keep other framework code unchanged
- Ensure the repaired code is compatible with the template's interfaces and data structures
- Leverage all available functionality provided in the template to improve the algorithm logic
"""
            return context

        except Exception as e:
            return f"Failed to get framework context: {str(e)}"

    def _extract_key_sections(self, template_content: str) -> str:
        try:
            lines = template_content.split('\n')
            key_sections = []
            init_start = None
            for i, line in enumerate(lines):
                if 'def __init__(self' in line:
                    init_start = i
                    break

            if init_start:
                init_section = []
                for i in range(init_start, min(init_start + 100, len(lines))):
                    line = lines[i]
                    init_section.append(line)
                    if line.strip().startswith('def ') and i > init_start:
                        break

                key_sections.append("**1. Class initialization and available attributes**:")
                key_sections.append("```python")
                key_sections.extend(init_section)
                key_sections.append("```")
            utility_start = None
            for i, line in enumerate(lines):
                if 'Utility Function Area' in line and 'START' in line:
                    utility_start = i
                    break

            if utility_start:
                utility_section = []
                for i in range(utility_start, min(utility_start + 200, len(lines))):
                    line = lines[i]
                    utility_section.append(line)
                    if 'Utility Function Area' in line and 'END' in line:
                        break

                key_sections.append("\n**2. Utility function area (available methods)**:")
                key_sections.append("```python")
                key_sections.extend(utility_section)
                key_sections.append("```")

            select_start = None
            for i, line in enumerate(lines):
                if 'def select_next_point(' in line:
                    select_start = i
                    break

            if select_start:
                select_section = []

                for i in range(select_start, min(select_start + 300, len(lines))):
                    line = lines[i]
                    select_section.append(line)

                    if line.strip().startswith('def ') and i > select_start:
                        break
                    if '# ============ LLM Fill Area - END ============' in line:
                        select_section.append(line)
                        break

                key_sections.append("\n**3. select_next_point function interface definition**:")
                key_sections.append("```python")
                key_sections.extend(select_section)
                key_sections.append("```")
            solve_start = None
            for i, line in enumerate(lines):
                if 'def solve(' in line:
                    solve_start = i
                    break

            if solve_start:
                solve_section = []
                for i in range(solve_start, min(solve_start + 50, len(lines))):
                    line = lines[i]
                    solve_section.append(line)
                    if line.strip().startswith('def ') and i > solve_start:
                        break

                key_sections.append("\n**4. solve method signature (overall flow)**:")
                key_sections.append("```python")
                key_sections.extend(solve_section)
                key_sections.append("```")

            return '\n'.join(key_sections)

        except Exception as e:
            return f"```python\n{template_content[:5000]}...\n```\n\nNote: The template content has been truncated. Please refer to the base template file for the full content."

    def generate_repaired_complete_code_file(self, strategy_id: str, strategy_data: Dict) -> Optional[str]:
        try:
            code_snippet = strategy_data.get('code_snippet', '')
            if not code_snippet:
                code_snippet = strategy_data.get('generated_code', '')

            if not code_snippet:
                return None

            template_content = None

            if self.algorithm_loader:
                available_algorithms = self.algorithm_loader.get_available_algorithms()
                if available_algorithms:
                    algorithm_name = "PointSelectionAlgorithm"
                    if algorithm_name not in available_algorithms:
                        algorithm_name = available_algorithms[0]

                    template_content = self.algorithm_loader.get_template(
                        algorithm_name)
                else:
                    self.output_manager.log_warning(
                        "repair_agent", "no_algorithms_for_repair",
                        "no algorithms for repair"
                    )

            if not template_content:
                base_template_path = os.path.join(
                    PROJECT_ROOT, 'base_algorithm', 'PointSelectionAlgorithm.py'
                )

                if not os.path.exists(base_template_path):
                    self.output_manager.log_error(
                        "repair_agent", "template_not_found",
                        f"template not found: {base_template_path}"
                    )
                    return None

                with open(base_template_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()

            complete_code = self._integrate_repaired_code_to_template(
                template_content, code_snippet)

            import time
            timestamp = int(time.time())
            file_name = f"{strategy_id}_repaired_{timestamp}.py"
            output_dir = self.output_manager.code_dir
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(complete_code)

            if not os.path.exists(file_path):
                self.output_manager.log_error(
                    "repair_agent", "file_creation_failed",
                    f"file creation failed: {file_path}"
                )
                return None
            file_size = os.path.getsize(file_path)
            self._validate_generated_code_file(file_path)

            return file_path

        except Exception as e:
            self.output_manager.log_error(
                "repair_agent", "repaired_code_generation_failed",
                f"repaired code generation failed: {str(e)}"
            )
            return None

    def _integrate_repaired_code_to_template(self, template_content: str, code_snippet: str) -> str:
        try:
            llm_start_marker = "# ============ LLM Fill Area - START ============"
            llm_end_marker = "# ============ LLM Fill Area - END ============"

            llm_start = template_content.find(llm_start_marker)
            llm_end = template_content.find(llm_end_marker)

            if llm_start == -1 or llm_end == -1:
                return template_content + "\n\n" + code_snippet

            before_llm = template_content[:llm_start + len(llm_start_marker)]
            after_llm = template_content[llm_end:]
            processed_code = self._preprocess_repaired_code_snippet(
                code_snippet)

            indented_code_lines = []
            for line in processed_code.split('\n'):
                if line.strip():  
                    indented_code_lines.append("    " + line)
                else: 
                    indented_code_lines.append("")

            indented_code = "\n".join(indented_code_lines)
            complete_code = before_llm + "\n" + indented_code + "\n" + after_llm
            return complete_code

        except Exception as e:
            return template_content + "\n\n# Repaired code (fallback handling)\n" + code_snippet

    def _preprocess_repaired_code_snippet(self, code_snippet: str) -> str:
        try:
            cleaned_code = re.sub(r'^```\s*python\s*\n?',
                                  '', code_snippet, flags=re.MULTILINE)
            cleaned_code = re.sub(
                r'^```\s*\n?', '', cleaned_code, flags=re.MULTILINE)
            cleaned_code = re.sub(r'\n?\s*```\s*$', '',
                                  cleaned_code, flags=re.MULTILINE)

            lines = cleaned_code.splitlines()
            if lines and lines[-1].strip() == '}':
                left_braces = cleaned_code.count('{')
                right_braces = cleaned_code.count('}')

                if right_braces > left_braces:
                    lines = lines[:-1]

            if 'class ' in cleaned_code and 'def select_next_point' in cleaned_code:
                function_code = self._extract_functions_from_class(
                    cleaned_code)
                if function_code:
                    cleaned_code = function_code

            processed_code = '\n'.join(lines) if isinstance(
                lines, list) else cleaned_code
            if not processed_code.endswith('\n'):
                processed_code += '\n'
            processed_code = re.sub(r'\n\s*\n\s*\n', '\n\n', processed_code)

            return processed_code

        except Exception as e:
            return code_snippet

    def _extract_functions_from_class(self, code_with_class: str) -> Optional[str]:
        try:
            lines = code_with_class.splitlines()
            function_lines = []
            in_function = False
            function_indent = 0

            for line in lines:
                if re.match(r'\s*def\s+', line):
                    in_function = True
                    function_indent = len(line) - len(line.lstrip())
                    adjusted_line = '    ' + line.lstrip()
                    function_lines.append(adjusted_line)
                elif in_function:
                    if line.strip() == '':
                        function_lines.append('')
                    elif len(line) - len(line.lstrip()) <= function_indent and line.strip():
                        if re.match(r'\s*def\s+', line):
                            function_indent = len(line) - len(line.lstrip())
                            adjusted_line = '    ' + line.lstrip()
                            function_lines.append(adjusted_line)
                        else:
                            in_function = False
                    else:
                        adjusted_line = '    ' + line.lstrip()
                        function_lines.append(adjusted_line)

            if function_lines:
                return '\n'.join(function_lines)
            else:
                return None

        except Exception as e:
            return None

    def _validate_generated_code_file(self, file_path: str) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            checks = [
                ('class Solution:', 'Solution class definition'),
                ('def __init__(', '__init__ method'),
                ('def select_next_point(', 'select_next_point method'),
                ('def build_solution(', 'build_solution method'),
                ('def solve(', 'solve method')
            ]

            missing_elements = []
            for check_str, description in checks:
                if check_str not in content:
                    missing_elements.append(description)

            if missing_elements:
                return False

            file_size = len(content)
            if file_size < 1000: 
                self.output_manager.log_warning(
                    "repair_agent", "code_file_too_small",
                    f"Generated code file may be too small ({file_size} chars), please check code integrity"
                )
            return True

        except Exception as e:
            self.output_manager.log_error(
                "repair_agent", "code_validation_error",
                f"Error validating code file: {str(e)}"
            )
            return False

    def debug_repair_process(self, strategy_id: str, strategy_data: Dict) -> Dict:
        debug_info = {
            'strategy_id': strategy_id,
            'timestamp': datetime.now().isoformat(),
            'input_data': {},
            'code_extraction': {},
            'file_generation': {},
            'issues': []
        }

        try:
            debug_info['input_data'] = {
                'has_code_snippet': 'code_snippet' in strategy_data,
                'has_generated_code': 'generated_code' in strategy_data,
                'code_snippet_length': len(strategy_data.get('code_snippet', '')),
                'generated_code_length': len(strategy_data.get('generated_code', '')),
                'has_relaxation_strategy': 'relaxation_strategy' in strategy_data,
                'has_text': 'text' in strategy_data
            }

            code_snippet = strategy_data.get('code_snippet', '')
            if not code_snippet:
                code_snippet = strategy_data.get('generated_code', '')

            debug_info['code_extraction'] = {
                'extracted_code_length': len(code_snippet),
                'has_select_next_point': 'def select_next_point' in code_snippet,
                'has_code_blocks': '```' in code_snippet,
                'has_class_definition': 'class ' in code_snippet,
                'code_preview': code_snippet[:200] + '...' if len(code_snippet) > 200 else code_snippet
            }

            return debug_info

        except Exception as e:
            debug_info['issues'].append(f"Error during debugging: {str(e)}")
            return debug_info
