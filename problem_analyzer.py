from typing import Dict, List, Any, Optional, Tuple
import json
import re
import inspect
from output_manager import OutputManager
from llm_client import ProblemAnalysisLLMClient


class ProblemAnalyzer:
    def __init__(self, problem_data: Dict, problem_params: Dict,
                 llm_client: ProblemAnalysisLLMClient,
                 output_manager: OutputManager):
        self.problem_data = problem_data
        self.problem_params = problem_params
        self.llm_client = llm_client
        self.output_manager = output_manager

    def analyze(self) -> Dict:

        problem_info = {
            "problem_type": self.problem_data.get("problem_type", "general_optimization"),
            "hard_constraints": [],
            "soft_constraints": [],
            "constraint_info": {},
            "objective_functions": [],
            "data_structures": {}
        }

        if ("detailed_description" in self.problem_data and
                "constraints" in self.problem_data.get("detailed_description", {})) or "description" in self.problem_data:
            description_data = self.problem_data.get(
                "detailed_description", self.problem_data.get("description", ""))
            llm_analysis = self._analyze_with_llm(description_data)
            problem_info = self._merge_analysis_results(
                problem_info, llm_analysis)
        if not problem_info["hard_constraints"] and "detailed_description" in self.problem_data:
            detailed_desc = self.problem_data["detailed_description"]
            if "constraints" in detailed_desc:
                constraints = detailed_desc["constraints"]
                for constraint in constraints:
                    if isinstance(constraint, dict) and "name" in constraint:
                        constraint_name = constraint["name"]
                        problem_info["hard_constraints"].append(
                            constraint_name)
                        problem_info["constraint_info"][constraint_name] = {
                            "description": constraint.get("description", ""),
                            "is_hard": True,
                            "weight": 1.0,
                            "formula": constraint.get("formula", ""),
                            "calculation": constraint.get("calculation", "")
                        }
            if "objectives" in detailed_desc:
                objectives = detailed_desc["objectives"]
                for objective in objectives:
                    if isinstance(objective, dict):
                        problem_info["objective_functions"].append(objective)

        relaxation_suggestions = self._suggest_relaxation_factors(problem_info)
        problem_info["relaxation_info"] = relaxation_suggestions

        self.output_manager.save_problem_analysis(problem_info)

        hard_constraints_count = len(problem_info["hard_constraints"])

        params_type = type(self.problem_params).__name__
        self.output_manager.log_info(
            "problem_analyzer",
            "params_type",
            f"params_type: {params_type}"
        )

        if hasattr(self.problem_params, 'update_hard_constraint_count'):
            try:
                import inspect
                method_sig = str(inspect.signature(
                    self.problem_params.update_hard_constraint_count))

                self.problem_params.update_hard_constraint_count(
                    hard_constraints_count)
                self.output_manager.log_info(
                    "problem_analyzer",
                    "update_hard_constraint_count",
                    f"update_hard_constraint_count: {hard_constraints_count}"
                )
            except Exception as e:
                self.output_manager.log_error(
                    "problem_analyzer",
                    "update_hard_constraint_count_error",
                    f"update_hard_constraint_count_error: {str(e)}, type: {type(e).__name__}"
                )
        else:
            attrs = dir(self.problem_params)
            public_methods = [attr for attr in attrs if callable(
                getattr(self.problem_params, attr)) and not attr.startswith('_')]

            self.output_manager.log_warning(
                "problem_analyzer",
                "update_hard_constraint_count_failed",
                f"update_hard_constraint_count_failed。public_methods: {public_methods}"
            )
            try:
                if hasattr(self.problem_params, '__dict__'):
                    self.problem_params.__dict__[
                        '_HARD_CONSTRAINT_COUNT'] = hard_constraints_count
            except Exception as e:
                self.output_manager.log_error(
                    "problem_analyzer",
                    "direct_update_error",
                    f"Set_HARD_CONSTRAINT_COUN failed: {str(e)}"
                )

        problem_info["problem_data"] = self.problem_data

        return problem_info

    def _analyze_with_llm(self, problem_description) -> Dict:
        try:

            system_prompt = """
            You are an optimization problem analysis expert. Your task is to analyze the given optimization problem description, extract constraints and objective functions.
            
            Please strictly follow this JSON format for your results:
            
            ```python
            {
                "hard_constraints": {
                    "hard_constraint1_name": "hard_constraint1_description", 
                    "hard_constraint2_name": "hard_constraint2_description"
                },
                "objectives": ["objective_function_description"],
                "constraint_importance": {
                    "constraint1": priority(1-10),
                    "constraint2": priority(1-10)
                },
                "hard_constraints_name": ["hard_constraint1_name","hard_constraint2_name"]
            }
            ```
            
            Hard constraints are constraints that must be satisfied.
            Please extract constraints and objectives based on the complete problem description, do not add constraints that don't exist.
            Do not provide any additional explanations, only return the analysis results in JSON format.
            """

            if isinstance(problem_description, dict):
                detailed_desc = problem_description
                basic_description = self.problem_data.get("description", "")
            else:
                basic_description = problem_description
                detailed_desc = self.problem_data.get(
                    "detailed_description", {})

            problem_type = self.problem_data.get("problem_type", "")

            constraints_info = detailed_desc.get("constraints", [])
            constraints_str = json.dumps(
                constraints_info, indent=2, ensure_ascii=False)

            objectives_info = detailed_desc.get("objectives", [])
            objectives_str = json.dumps(
                objectives_info, indent=2, ensure_ascii=False)

            parameters = self.problem_data.get("parameters", {})
            parameters_str = json.dumps(
                parameters, indent=2, ensure_ascii=False)

            data_samples = self.problem_data.get("data", [])

            if data_samples and isinstance(data_samples, list) and len(data_samples) > 0:
                temp_problem_data = {"data": data_samples}
                limited_problem_data = self.llm_client._limit_problem_data_points(
                    temp_problem_data, max_points=5)
                limited_data_samples = limited_problem_data.get("data", [])
                data_str = json.dumps(
                    limited_data_samples, indent=2, ensure_ascii=False)
            else:
                data_str = json.dumps(
                    data_samples, indent=2, ensure_ascii=False)
            user_prompt = f"""
            Please analyze the following combinatorial optimization problem and extract key information:
            
            Basic description: {basic_description}
            
            Problem type: {problem_type}
            
            Detailed description: {detailed_desc.get('overview', '')}
            
            Constraints:
            {constraints_str}
            
            Objective functions:
            {objectives_str}
            
            Problem parameters:
            {parameters_str}
            
            Data samples:
            {data_str}
            
            Solution format:
            {json.dumps(self.problem_data.get('solution_format', {}), indent=2, ensure_ascii=False)}
            
            Please identify all hard constraints, soft constraints, and optimization objectives, and return the results in JSON format.
            Hard constraints are conditions that must be strictly satisfied, soft constraints are conditions that are desired to be satisfied but can be relaxed.
            Assign importance scores of 1-10 to constraints, with 10 being the most important.
            """
            return self.llm_client.call_json(system_prompt, user_prompt)
        except Exception as e:
            self.output_manager.log_error(
                "problem_analyzer",
                "llm_analysis_error",
                f"LLM分析问题时出错: {str(e)}"
            )
            return {
                "hard_constraints": [],
                "soft_constraints": [],
                "objectives": [],
                "constraint_importance": {}
            }

    def _merge_analysis_results(self, code_analysis: Dict, llm_analysis: Dict) -> Dict:
        result = {
            "problem_type": code_analysis.get("problem_type", ""),
            "hard_constraints": [],
            "soft_constraints": [],
            "constraint_info": {},
            "objective_functions": code_analysis.get("objective_functions", []),
            "data_structures": code_analysis.get("data_structures", {})
        }
        if "hard_constraints_name" in llm_analysis:
            for constraint_name in llm_analysis.get("hard_constraints_name", []):
                result["hard_constraints"].append(constraint_name)
                description = ""
                hard_constraints = llm_analysis.get("hard_constraints", [])
                if isinstance(hard_constraints, list):
                    for constraint in hard_constraints:
                        if isinstance(constraint, dict) and constraint_name in constraint:
                            description = constraint[constraint_name]
                            break
                        elif isinstance(constraint, str) and constraint_name.lower() in constraint.lower():
                            description = constraint
                            break
                result["constraint_info"][constraint_name] = {
                    "description": description,
                    "is_hard": True,
                    "weight": 1.0
                }

        elif "hard_constraints" in llm_analysis:
            hard_constraints = llm_analysis.get("hard_constraints", [])
            if isinstance(hard_constraints, list):
                for constraint in hard_constraints:
                    if isinstance(constraint, dict):
                        for constraint_name, description in constraint.items():
                            if constraint_name not in result["hard_constraints"]:
                                result["hard_constraints"].append(
                                    constraint_name)
                                result["constraint_info"][constraint_name] = {
                                    "description": description,
                                    "is_hard": True,
                                    "weight": 1.0
                                }
                    elif isinstance(constraint, str):
                        constraint_name = self._normalize_constraint_name(
                            constraint)
                        if constraint_name not in result["hard_constraints"]:
                            result["hard_constraints"].append(constraint_name)
                            result["constraint_info"][constraint_name] = {
                                "description": constraint,
                                "is_hard": True,
                                "weight": 1.0
                            }

        for constraint in llm_analysis.get("soft_constraints", []):
            if isinstance(constraint, dict):
                for constraint_name, description in constraint.items():
                    if constraint_name not in result["soft_constraints"] and constraint_name not in result["hard_constraints"]:
                        result["soft_constraints"].append(constraint_name)
                        result["constraint_info"][constraint_name] = {
                            "description": description,
                            "is_hard": False,
                            "weight": 0.5
                        }
            else:
                constraint_name = self._normalize_constraint_name(constraint)
                if constraint_name not in result["soft_constraints"] and constraint_name not in result["hard_constraints"]:
                    result["soft_constraints"].append(constraint_name)
                    result["constraint_info"][constraint_name] = {
                        "description": constraint,
                        "is_hard": False,
                        "weight": 0.5
                    }

        for constraint, importance in llm_analysis.get("constraint_importance", {}).items():
            if constraint in result["constraint_info"]:
                result["constraint_info"][constraint]["weight"] = importance / 10.0

        return result

    def _normalize_constraint_name(self, constraint: str) -> str:

        name = re.sub(r'[^a-z0-9]', '_', constraint.lower())
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        if not name:
            name = "generic"

        if not name.endswith('_constraint'):
            name += '_constraint'

        return name

    def _suggest_relaxation_factors(self, problem_info: Dict) -> Dict:
        relaxation_info = {}
        for constraint_name in problem_info["hard_constraints"]:
            constraint_info = problem_info["constraint_info"].get(
                constraint_name, {})

            relaxation_info[constraint_name] = {
                "description": constraint_info.get("description", ""),
                "possible_relaxations": [0.9, 0.95, 1.0, 1.05, 1.1]
            }

        try:

            system_prompt = """
            You are a constraint optimization expert. You need to provide relaxation factor suggestions for the given optimization problem constraints.
            
            Please strictly follow the JSON format below for your results:
            
            ```json
            {
                "constraint_name1": {
                    "possible_relaxations": [relaxation_factor1, relaxation_factor2, ...]
                },
                "constraint_name2": {
                    "possible_relaxations": [relaxation_factor1, relaxation_factor2, ...]
                },
                ...
            }
            ```
            
            Do not provide any other explanations, only return the suggestions in JSON format.
            """

            problem_description = self.problem_data.get(
                "description", "Problem Description")
            problem_type = self.problem_data.get("problem_type", "")
            detailed_desc = self.problem_data.get("detailed_description", {})
            constraints_info = detailed_desc.get("constraints", [])
            constraints_str = json.dumps(
                constraints_info, indent=2, ensure_ascii=False)

            objectives_info = detailed_desc.get("objectives", [])
            objectives_str = json.dumps(
                objectives_info, indent=2, ensure_ascii=False)
            parameters = self.problem_data.get("parameters", {})
            parameters_str = json.dumps(
                parameters, indent=2, ensure_ascii=False)
            user_prompt = f"""
            Please provide appropriate relaxation factor suggestions for each constraint based on the complete background information of the following optimization problem:
            
            Basic description: {problem_description}
            
            Problem type: {problem_type}
            
            Detailed description: {detailed_desc.get('overview', '')}
            
            Constraints:
            {constraints_str}
            
            Objective functions:
            {objectives_str}
            
            Problem parameters:
            {parameters_str}
            
            Constraint information to consider for relaxation factors:
            {json.dumps(problem_info["constraint_info"], indent=2, ensure_ascii=False)}
            
            For each hard constraint, please suggest 5 different relaxation factors based on the nature and importance of the constraint, with values ranging from 0.5 to 20.
            Relaxation factors less than 1 indicate tightening the constraint, equal to 1 means keeping the original constraint, and greater than 1 means relaxing the constraint.
            During early exploration, you can try bold constraint relaxations.
            Please ensure that the suggested relaxation factors are reasonable, taking into account the characteristics of the constraints and the practical context of the problem.
            """

            llm_suggestions = self.llm_client.call_json(
                system_prompt, user_prompt)

            for constraint_name, suggestion in llm_suggestions.items():
                if constraint_name in relaxation_info:
                    if "possible_relaxations" in suggestion:
                        relaxation_info[constraint_name]["possible_relaxations"] = suggestion["possible_relaxations"]
        except Exception as e:
            self.output_manager.log_error(
                "problem_analyzer",
                "relaxation_suggestion_error",
                f"Error getting constraint relaxation suggestions: {str(e)}"
            )

        return relaxation_info
