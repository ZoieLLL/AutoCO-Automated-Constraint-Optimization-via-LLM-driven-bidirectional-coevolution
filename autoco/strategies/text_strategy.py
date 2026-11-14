from typing import List, Dict, Optional, Tuple
import re
import uuid
import random
import copy
from autoco import PROJECT_ROOT
from autoco.output_manager import OutputManager
from autoco.llm.llm_client import StrategyLLMClient
from autoco.models import TextStrategy
import time


class TextStrategyManager:
    def __init__(self, problem_info: Dict,
                 llm_client: StrategyLLMClient,
                 output_manager: OutputManager,
                 algorithm_loader=None):
        self.problem_info = problem_info
        self.llm_client = llm_client
        self.output_manager = output_manager
        self.algorithm_loader = algorithm_loader
        self.problem_data = problem_info.get("problem_data", {})
        self.strategies: Dict[str, TextStrategy] = {}
        self.current_population: List[str] = []

        self.generation = 0

        self.crossover_prompts = {
            "Crossover1": """
            Constraint Relaxation Expert: Create a totally different constraint relaxation approach.
            
            Focus: constraint_order + relaxation_factors 
            Combine parent strategies but use completely different constraint handling sequence.
            """,

            "Crossover2": """
            Code Implementation Expert: Create a totally different code implementation approach.
            
            Focus: select_next_point function logic
            Combine parent algorithms but use completely different scoring/selection mechanism.
            """,

            "Crossover3": """
            Algorithm Design Expert: Create a totally different algorithm design approach.
            
            Focus: algorithm_design philosophy 
            Combine parent ideas but design totally different optimization strategy.
            """,

            "Crossover4": """
            Strength-Weakness Fusion Expert: High-fitness dominates + low-fitness supplements.
            
            Take best parent as backbone, enhance with good parts from weaker parent.
            Create totally different combination that preserves strengths.
            """,

            "Crossover5": """
            Symmetric Crossover Expert: Balanced fusion of both parents equally.
            
            Alternate features between parents across all 3 layers.
            Create totally different balanced combination.
            """,

            "Crossover6": """
            Complementary Optimization Expert: Identify differences and create synergy.
            
            Find what parent1 lacks that parent2 has, and vice versa.
            Create totally different complementary combination.
            """
        }
        self.mutation_prompts = {
            "Mutation1": """
            Constraint Relaxation Mutation Expert: Create a totally different constraint relaxation strategy.
            
            Modify constraint_order and relaxation_factors.
            Keep algorithm implementation, focus constraint innovation.
            """,

            "Mutation2": """
            Code Implementation Mutation Expert: Create totally different implementation logic.
            
            Modify select_next_point function completely.
            Keep constraint strategy, focus code innovation.
            """,

            "Mutation3": """
            System Architecture Mutation Expert: Create a totally different system architecture.
            
            Redesign both constraint handling and algorithm implementation.
            Create completely new coordination mechanism.
            """
        }

    def _extract_llm_area_from_template(self) -> str:

        try:
            if self.algorithm_loader:
                try:
                    available_algorithms = self.algorithm_loader.get_available_algorithms()
                    if available_algorithms:
                        algorithm_name = available_algorithms[0]
                        template_content = self.algorithm_loader.get_template(
                            algorithm_name)

                        self.output_manager.log_info(
                            "strategy_manager", "using_algorithm_loader_template",
                            f"Getting the template using the algorithm loader: {algorithm_name}."
                        )
                    else:
                        template_content = ""
                except Exception as e:
                    self.output_manager.log_error(
                        "strategy_manager", "algorithm_loader_error",
                        f"Failed to get the template via the algorithm loader: {e}."
                    )
                    template_content = ""
            else:
                self.output_manager.log_warning(
                    "strategy_manager", "no_algorithm_loader",
                    "No algorithm_loader was provided, algorithm template cannot be retrieved."
                )
                return ""

            llm_area_pattern = r'# ============ LLM Fill Area - START ============(.*?)# ============ LLM Fill Area - END ============'
            match = re.search(llm_area_pattern, template_content, re.DOTALL)

            if match:
                llm_area = match.group(1)

                llm_area = re.sub(
                    r'def\s+(\w+\([^)]*\)):\s*\n\s*pass', r'def \1:\n        # You need to implement this function', llm_area)
                return llm_area
            self.output_manager.log_error(
                "strategy_manager", "llm_area_not_found",
                "LLM fill area not found in template file."
            )
            return ""

        except Exception as e:
            self.output_manager.log_error(
                "strategy_manager", "extract_template_error",
                f"Error extracting template function: {str(e)}."
            )
            return ""

    def generate_initial_strategies(self, count: int = 4, outer_iteration=0, best_code_snippet=None) -> List[str]:
        self.output_manager.log_info("strategy_manager", "generating_initial_strategies",
                                     f"Generating {count} initial strategies (two-step method), outer iteration: {outer_iteration}.")

        mcts_hints = self.problem_info.get("mcts_exploration_hints", [])
        strategy_texts = self.llm_client.generate_initial_strategy(
            self.problem_info,
            strategy_count=count,
            best_code_snippet=best_code_snippet if outer_iteration > 0 else None,
            problem_data=self.problem_data
        )
        strategy_ids = []
        for idx, text in enumerate(strategy_texts):
            constraint_order, relaxation_factors, _, _ = self._parse_strategy_text(
                text)
            algorithm_design, code_snippet = self._generate_algorithm_and_code(
                text, constraint_order, relaxation_factors,
                reference_code=best_code_snippet if outer_iteration > 0 else None,
                problem_data=self.problem_data
            )
            strategy_id = self._create_strategy(
                text=text,
                method="initial",
                generation=0,
                parent_ids=[],
                outer_iteration=outer_iteration,
                constraint_order=constraint_order,
                relaxation_factors=relaxation_factors,
                algorithm_design=algorithm_design,
                code_snippet=code_snippet
            )
            self.output_manager.log_info(
                "strategy_manager", "initial_strategy_created",
                f"Creating initial strategy {idx+1}/{count}, ID: {strategy_id}, "
                f"Number of constraints: {len(constraint_order)}, relaxation factors: {relaxation_factors}."
            )

            strategy_ids.append(strategy_id)
        if "mcts_exploration_hints" in self.problem_info:
            self.problem_info["mcts_exploration_hints"] = []
        self.current_population = strategy_ids

        return strategy_ids

    def _generate_algorithm_and_code(self, strategy_text: str,
                                     constraint_order: List[str],
                                     relaxation_factors: Dict[str, float],
                                     reference_code: str = None,
                                     problem_data: Dict = None) -> Tuple[str, str]:
        import os
        import json

        self.output_manager.log_info(
            "strategy_manager", "generating_algorithm_and_code",
            f"Generating algorithm design and code snippet for the strategy; reference code length: {len(reference_code) if reference_code else 0}."
        )
        if problem_data:
            actual_problem_data = problem_data
        else:
            if hasattr(self, 'problem_data') and self.problem_data:
                actual_problem_data = self.problem_data
            else:
                actual_problem_data = {}
                problem_data_path = os.path.join(PROJECT_ROOT, "problem_data.json")
                if os.path.exists(problem_data_path):
                    try:
                        with open(problem_data_path, 'r', encoding='utf-8') as f:
                            actual_problem_data = json.load(f)
                    except Exception as e:
                        self.output_manager.log_error(
                            "text_strategy_manager", "problem_data_error",
                            f"Cannot load problem data from file: {str(e)}."
                        )
        limited_problem_data = self._limit_problem_data_points(
            actual_problem_data, max_points=5)
        problem_data_json = json.dumps(
            limited_problem_data, indent=2) if limited_problem_data else "{}"
        system_prompt = """You are an optimization algorithm expert. Based on the given constraint-handling strategy and template functions, you need to provide:
        1. A brief description of the algorithm design approach
        2. The corresponding Python code implementation snippet
        
        Please ensure that your algorithm design and code implementation can effectively handle the constraints and reflect the constraint-handling order and relaxation factors in the strategy.
        Your code must not directly use or copy the template functions; the implementation logic, structure, and content of the code you design must differ significantly from the template functions in at least one of these three aspects.
        Also keep the function signatures unchanged, but adjust the function content according to the strategy.
        You may refer to the following innovation ideas:
            1. Scoring function innovation:
            - Go beyond the simple "profit-distance" calculation
            - Introduce multi-factor weighted combinations (e.g., density, clustering, load balancing)
            - Consider dynamic weight adjustment mechanisms

            2. Selection strategy innovation:
            - Avoid purely greedy selection
            - Introduce probabilistic selection, multi-candidate comparison, or local search
            - Consider a global optimization perspective

            3. Constraint-handling innovation:
            - Design intelligent repair mechanisms
            - Implement constraint-violation prevention strategies
            - Optimize resource allocation methods"""

        code_template = ""
        if reference_code:
            code_template = reference_code
            self.output_manager.log_info(
                "strategy_manager", "using_reference_code",
                f"Using the best individual's code as the template function reference, length: {len(reference_code)}."
            )
        else:
            code_template = self._extract_llm_area_from_template()
            self.output_manager.log_info(
                "strategy_manager", "using_template_functions",
                "Using template function as reference."
            )
        base_algorithm_info = ""
        if reference_code:  
            try:
                base_algorithm_code = ""
                try:
                    if hasattr(self, 'algorithm_loader'):
                        loader = self.algorithm_loader
                    elif hasattr(self.llm_client, 'algorithm_loader'):
                        loader = self.llm_client.algorithm_loader

                    if loader:
                        available_algorithms = loader.get_available_algorithms()
                        if "PointSelectionAlgorithm" in available_algorithms:
                            base_algorithm_code = loader.get_template(
                                "PointSelectionAlgorithm")
                        elif available_algorithms:
                            base_algorithm_code = loader.get_template(
                                available_algorithms[0])
                except Exception as e:
                    self.output_manager.log_warning(
                        "strategy_manager", "algorithm_loader_error",
                        f"algorithm loader error: {str(e)}"
                    )
                if not base_algorithm_code:
                    self.output_manager.log_warning(
                        "strategy_manager", "no_algorithm_template",
                        "No algorithm template"
                    )

                if base_algorithm_code:
                    from autoco.generation.code_analyzer import extract_code_information, get_function_signature_by_name, format_code_information_for_prompt

                    code_info = extract_code_information(base_algorithm_code)
                    select_next_point_signature = get_function_signature_by_name(
                        base_algorithm_code, "select_next_point")

                    base_algorithm_info = format_code_information_for_prompt(
                        code_info)

            except Exception as e:
                self.output_manager.log_warning(
                    "strategy_manager", "code_analysis_error",
                    f"Code analysis error: {str(e)}"
                )
        algorithm_info_text = ""
        if reference_code:
            algorithm_info_text = "The complete code framework information is as follows: please ensure that the parameters and functions used in your code have names consistent with the corresponding parameter and function names in the code framework.\n" + \
                base_algorithm_info

        innovation_focus = "balanced"
        strategy_text_lower = strategy_text.lower()
        if "constraint" in strategy_text_lower and ("relax" in strategy_text_lower or "order" in strategy_text_lower):
            if "algorithm" in strategy_text_lower and ("score" in strategy_text_lower or "select" in strategy_text_lower or "optimize" in strategy_text_lower):
                innovation_focus = "balanced"
            else:
                innovation_focus = "constraint_relaxation"
        elif "algorithm" in strategy_text_lower and ("score" in strategy_text_lower or "select" in strategy_text_lower or "optimize" in strategy_text_lower):
            innovation_focus = "algorithm_logic"

        algorithm_requirements = ""
        if innovation_focus == "constraint_relaxation":
            algorithm_requirements = """
        🎯 Constraint Relaxation Innovation Focus - Algorithm Design Requirements:
        1. Constraint-handling innovation:
           - Implement dynamic constraint priority adjustment or grouping mechanisms
           - Design intelligent constraint-violation prevention and recovery strategies
           - Introduce coordination and dependency handling between constraints
        
        2. Relaxation strategy innovation:
           - Implement progressive, adaptive, or conditional relaxation mechanisms
           - Design constraint-satisfaction monitoring and feedback adjustment
           - Explore the timing and interactions of constraint relaxation
        
        3. Algorithm adaptation: keep a stable algorithm framework while focusing on optimizing the constraint-handling logic
        """
        elif innovation_focus == "algorithm_logic":
            algorithm_requirements = """
        🔧 Algorithm Logic Innovation Focus - Algorithm Design Requirements:
        1. Scoring mechanism innovation (must go beyond the simple profit-distance model):
           - Design multi-factor composite scoring systems (profit + distance + density + load balancing, etc.)
           - Introduce dynamic weight adjustment or adaptive scoring mechanisms
           - Implement hierarchical evaluation or multi-objective optimization scoring
        
        2. Selection strategy innovation:
           - Go beyond the greedy mode: probabilistic selection, multi-candidate comparison, local search, etc.
           - Introduce intelligent decision-making: predictive selection, backtracking optimization, strategy switching, etc.
           - Implement multi-stage decision or hierarchical selection mechanisms
        
        3. Algorithm architecture innovation: restructure the overall logic of select_next_point to make it modular and intelligent
        """
        else:  # balanced
            algorithm_requirements = """
        ⚖️ Synergistic Innovation Mode - Algorithm Design Requirements:
        1. Constraint-algorithm synergy: design a linkage mechanism between constraint state and algorithm behavior
        2. Adaptive coordination: implement relaxation-algorithm collaborative optimization based on solving progress
        3. Overall performance: redesign the coordination between constraint handling and algorithm implementation from a global perspective
        """

        user_prompt = f"""Here is a strategy for an optimization problem, including the constraint-handling order and relaxation factors:
        
        Strategy text:
            Constraint-handling order: {constraint_order}
            Relaxation factors: {relaxation_factors}
        
        {algorithm_requirements}
        
        Problem background:
        {self.problem_info.get('description', 'optimization problem')}
        Problem data:
        {problem_data_json}
        {algorithm_info_text}
        {'Reference code' if reference_code else 'Template function reference'} (these are the key functions you need to implement or modify):
        ```python
{code_template}
        ```
        
        ⚠️ Strict requirements:
        1. The algorithm design must reflect the innovation focus above and must not be a simple variant of the standard greedy algorithm.
        2. The code implementation must show significant innovation in at least two of the following aspects: logical structure, scoring mechanism, or selection strategy.
        3. Do not generate an implementation highly similar to the reference code (similarity > 80%).
        
        Please provide:
        
        1. Algorithm design: describe the innovations, implementation approach, and expected effects in detail.
        
        2. Code snippet: Python code implementing the innovative algorithm design above, ensuring the function names and inputs/outputs remain unchanged.
        
        {'Note: the reference code is only for understanding the function interface; you must redesign the algorithm logic according to the innovation requirements.' if reference_code else ''}
        
        All parameters and functions used must have the same names as in the base algorithm framework. If you introduce new functions or parameters, you must implement their complete functionality and definitions in the code snippet you provide.
        Return in the following format:
        
        Algorithm Design:
        ```json
        {{
          "innovation_type": "{innovation_focus}",
          "core_innovations": ["innovation 1", "innovation 2", "innovation 3"],
          "algorithm_description": "detailed algorithm design approach",
          "expected_advantages": "expected advantages and improvement effects"
        }}
        ```
        
        Code Snippet:
        ```python
        [innovative select_next_point implementation that reflects the design approach above]
        ```
        
        """

        try:
            response = self.llm_client.call(system_prompt, user_prompt)

            # Extract the algorithm design and code snippet
            algo_match = re.search(
                r'Algorithm Design:\s*```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
            code_match = re.search(
                r'Code Snippet:\s*```python\s*(.*?)\s*```', response, re.DOTALL)

            algorithm_design = algo_match.group(
                1).strip() if algo_match else ""
            code_snippet = code_match.group(1).strip() if code_match else ""

            return algorithm_design, code_snippet
        except Exception as e:
            self.output_manager.log_error(
                "strategy_manager", "generate_algorithm_code_error",
                f"Error generating the algorithm design and code snippet: {str(e)}"
            )
            return "", ""

    def _create_strategy(self, text: str, method: str = "initial",
                         generation: int = 0, parent_ids: List[str] = None, outer_iteration=0,
                         constraint_order: List[str] = None, relaxation_factors: Dict[str, float] = None,
                         algorithm_design: str = None, code_snippet: str = None) -> str:

        import uuid
        import traceback
        if not isinstance(self.problem_info, dict):
            self.output_manager.log_error(
                "strategy_manager", "invalid_problem_info",
                f"invalid problem info: {type(self.problem_info)}"
            )
            self.problem_info = {}  

        strategy_id = str(uuid.uuid4())
        cleaned_text = text
        extracted_code = None
        import re
        code_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
        if method == "initial":
            if code_match:
                cleaned_text = extracted_code 
            else:
                cleaned_text = re.sub(
                    r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        else:
            cleaned_text = re.sub(r'<think>.*?</think>',
                                  '', text, flags=re.DOTALL).strip()
        if constraint_order is None or relaxation_factors is None or algorithm_design is None or code_snippet is None:
            parsed_constraint_order, parsed_relaxation_factors, parsed_algorithm_design, parsed_code_snippet = self._parse_strategy_text(
                cleaned_text)
            constraint_order = constraint_order or parsed_constraint_order
            relaxation_factors = relaxation_factors or parsed_relaxation_factors
            algorithm_design = algorithm_design or parsed_algorithm_design
            code_snippet = code_snippet or extracted_code or parsed_code_snippet

        self.output_manager.log_info(
            "strategy_manager", "text_processing",
            f"Method: {method}, generation: {generation}, text length: {len(text)}. "
            f"Length after cleanup: {len(cleaned_text)}, code length: {len(code_snippet or '')}."
        )
        strategy = TextStrategy(
            id=strategy_id,
            text=cleaned_text, 
            constraint_order=constraint_order,
            relaxation_factors=relaxation_factors,
            algorithm_design=algorithm_design, 
            code_snippet=code_snippet,  
            parent_ids=parent_ids or [],
            generation=generation,
            method=method,
            outer_iteration=outer_iteration
        )
        self.strategies[strategy_id] = strategy
        try:
            strategy_dict = strategy.to_dict()
            self.output_manager.log_info(
                "strategy_manager", "strategy_content",
                f"Strategy ID: {strategy_id}, method: {method}, constraint order: {constraint_order}., "
                f"Text length: {len(cleaned_text)}, algorithm design length: {len(algorithm_design or '')}., "
                f"Code snippet length: {len(code_snippet or '')}."
            )
            self.output_manager.save_strategy(strategy_dict, method)
            if "strategies" not in self.problem_info:
                self.problem_info["strategies"] = {}
            if not isinstance(self.problem_info["strategies"], dict):
                self.problem_info["strategies"] = {}

            self.problem_info["strategies"][strategy_id] = strategy_dict
        except Exception as e:
            self.output_manager.log_error( 
                "strategy_manager", "save_strategy_error",
                f"Error saving strategy: {str(e)}\n{traceback.format_exc()}."
            )

        return strategy_id

    def _parse_strategy_text(self, text: str) -> Tuple[List[str], Dict[str, float], str, str]:
        all_constraints = self.problem_info.get(
            "hard_constraints", []) + self.problem_info.get("soft_constraints", [])

        constraint_order = []
        relaxation_factors = {}
        algorithm_design = ""
        code_snippet = ""
        initial_format_pattern = r'(\d+)[.．、) ]+([a-zA-Z_]+)\s+relax\s*(\d+\.?\d*)'
        initial_matches = re.findall(initial_format_pattern, text)

        if initial_matches:
            for _, constraint_name, factor_str in initial_matches:
                for constraint in all_constraints:
                    if constraint_name in constraint or constraint in constraint_name:
                        constraint_order.append(constraint)
                        try:
                            relaxation_factors[constraint] = float(factor_str)
                        except ValueError:
                            relaxation_factors[constraint] = 1.0
                        break
        if not (constraint_order and relaxation_factors):
            original_extraction_success = False
            try:
                constraint_section_patterns = [
                    r'Constraint Handling Order[:：](.*?)(?=Algorithm Design|$)',
                    r'Constraint Handling Strategy[:：](.*?)(?=Algorithm Design|$)',
                ]

                constraints_text = ""
                for pattern in constraint_section_patterns:
                    match = re.search(pattern, text, re.DOTALL)
                    if match:
                        constraints_text = match.group(1).strip()
                        break

                if constraints_text:
                    constraints_lines = constraints_text.split('\n')
                    for line in constraints_lines:
                        line = line.strip()
                        if line and any(c in line for c in all_constraints):
                            for constraint in all_constraints:
                                if constraint in line:
                                    constraint_order.append(constraint)
                                    factor_match = re.search(
                                        r'(\d+\.?\d*)', line)
                                    if factor_match:
                                        relaxation_factors[constraint] = float(
                                            factor_match.group(1))
                                    else:
                                        relaxation_factors[constraint] = 1.0
                                    break
                    if constraint_order:
                        original_extraction_success = True

                algo_match = re.search(
                    r'Algorithm Design[:：](.*?)(?=Code Snippet|$)', text, re.DOTALL)
                if algo_match:
                    algorithm_design = algo_match.group(1).strip()
                code_match = re.search(
                    r'```python\s*(.*?)\s*```', text, re.DOTALL)
                if code_match:
                    code_snippet = code_match.group(1).strip()

            except Exception as e:
                self.output_manager.log_warning(
                    "strategy_manager", "original_extraction_error",
                    f"Extraction error: {str(e)}"
                )

            if not original_extraction_success:
                if not algorithm_design:
                    algo_match = re.search(
                        r'Algorithm Design[:：](.*?)(?=Code Snippet|```python|$)', text, re.DOTALL)
                    if algo_match:
                        algorithm_design = algo_match.group(1).strip()
                    elif 'Algorithm Design' in text:
                        parts = text.split('Algorithm Design', 1)
                        if len(parts) > 1:
                            next_part_match = re.search(
                                r'(Code Snippet|Constraint Handling|```python)', parts[1])
                            if next_part_match:
                                algorithm_design = parts[1][:next_part_match.start()].strip(
                                )
                            else:
                                algorithm_design = parts[1].strip()

                if not code_snippet:
                    code_match = re.search(
                        r'```python\s*(.*?)\s*```', text, re.DOTALL)
                    if code_match:
                        code_snippet = code_match.group(1).strip()
                    elif 'Code Snippet' in text and '```' in text:
                        parts = text.split('Code Snippet', 1)
                        if len(parts) > 1:
                            code_parts = parts[1].split('```', 2)
                            if len(code_parts) > 2:
                                code_snippet = code_parts[1].strip()
                                if code_snippet.startswith('python'):
                                    code_snippet = code_snippet[6:].strip()

                if not constraint_order:
                    constraint_section_patterns = [
                        r'Constraint Handling Order(?: and )?Relaxation Factors[:：](.*?)(?=Algorithm Design|Code Snippet|```python|$)',
                        r'Constraint Handling Strategy[:：](.*?)(?=Algorithm Design|Code Snippet|```python|$)',
                        r'Constraint Handling Order[:：](.*?)(?=Algorithm Design|Code Snippet|```python|$)',
                        r'Constraint[:：](.*?)(?=Algorithm Design|Code Snippet|```python|$)',
                        r'Constraint Relaxation Factors[:：](.*?)(?=Algorithm Design|Code Snippet|```python|$)',
                        r'Constraint(?: and )?Relaxation[:：](.*?)(?=Algorithm Design|Code Snippet|```python|$)'
                    ]

                    constraints_text = ""
                    for pattern in constraint_section_patterns:
                        match = re.search(pattern, text, re.DOTALL)
                        if match:
                            constraints_text = match.group(1).strip()
                            break

                    if not constraints_text and 'constraint' in text.lower():
                        parts = text.split('constraint', 1)
                        if len(parts) > 1:
                            next_part_match = re.search(
                                r'(Algorithm Design|Code Snippet|```python)', parts[1])
                            if next_part_match:
                                constraints_text = parts[1][:next_part_match.start()].strip(
                                )
                            else:
                                constraints_text = parts[1].strip()

                    if constraints_text:
                        pattern1 = r'(\d+)[.．、) ]+([a-zA-Z_]+)\s*[:：]?\s*relax\s*(\d+\.?\d*)'
                        matches1 = re.findall(pattern1, constraints_text)
                        pattern2 = r'([a-zA-Z_]+)\s*[:：]?\s*relax\s*(\d+\.?\d*)'
                        matches2 = re.findall(pattern2, constraints_text)
                        pattern3 = r'([a-zA-Z_]+)[_]?constraint\s*[:：]?\s*(\d+\.?\d*)'
                        matches3 = re.findall(pattern3, constraints_text)
                        pattern4 = r'(\d+)[.．、)]\s*([^,，:：]*?)[,，:：]\s*(\d+\.?\d*)'
                        matches4 = re.findall(pattern4, constraints_text)
                        all_matches = []
                        for _, constraint, factor in matches1:
                            all_matches.append((constraint, factor))
                        all_matches.extend(matches2)
                        all_matches.extend(matches3)
                        for _, constraint_text, factor in matches4:
                            for constraint in all_constraints:
                                if constraint.lower() in constraint_text.lower() or constraint_text.lower() in constraint.lower():
                                    all_matches.append((constraint, factor))
                                    break
                        for constraint_name, factor_str in all_matches:
                            best_match = self._find_best_match_constraint(
                                constraint_name, all_constraints)
                            if best_match and best_match not in constraint_order:
                                constraint_order.append(best_match)
                                try:
                                    relaxation_factors[best_match] = float(
                                        factor_str)
                                except ValueError:
                                    relaxation_factors[best_match] = 1.0
        else:
            algo_match = re.search(r'Algorithm Design[:：](.*?)(?=Code Snippet|$)', text, re.DOTALL)
            if algo_match:
                algorithm_design = algo_match.group(1).strip()
            code_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
            if code_match:
                code_snippet = code_match.group(1).strip()
        if not constraint_order and all_constraints:
            constraint_order = all_constraints.copy()
            for constraint in constraint_order:
                relaxation_factors[constraint] = 1.0

        return constraint_order, relaxation_factors, algorithm_design, code_snippet

    def _find_best_match_constraint(self, name: str, constraints: List[str]) -> Optional[str]:
        for constraint in constraints:
            if constraint.lower() in name.lower() or name.lower() in constraint.lower():
                return constraint
        name_lower = name.lower()
        for constraint in constraints:
            constraint_keywords = re.findall(r'[a-zA-Z]+', constraint.lower())
            matches = sum(
                1 for keyword in constraint_keywords if keyword in name_lower)

            if matches > 0:
                return constraint
        constraint_info = self.problem_info.get("constraint_info", {})
        for constraint, info in constraint_info.items():
            if constraint in constraints:
                description = info.get("description", "").lower()
                if any(word in description for word in name_lower.split()):
                    return constraint

        for constraint in constraints:
            if self._calculate_similarity(name.lower(), constraint.lower()) > 0.5:
                return constraint

        return None

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        words1 = set(re.findall(r'[a-zA-Z]+', str1.lower()))
        words2 = set(re.findall(r'[a-zA-Z]+', str2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        return len(intersection) / (len(words1) + len(words2) - len(intersection))

    def crossover_strategies(self, parent_ids: List[str], outer_iteration=0) -> str:
        if len(parent_ids) < 2:
            return parent_ids[0] if parent_ids else str(uuid.uuid4())

        parent1 = self.strategies.get(parent_ids[0])
        parent2 = self.strategies.get(parent_ids[1])

        if not parent1 or not parent2:
            return parent_ids[0] if parent1 else (parent_ids[1] if parent2 else str(uuid.uuid4()))

        self.output_manager.log_info(
            "strategy_manager", "crossover_start",
            f"Starting crossover strategies {parent1.id} and {parent2.id}."
        )

        crossover_type = random.choice(list(self.crossover_prompts.keys()))
        system_prompt = self.crossover_prompts[crossover_type]
        self.output_manager.log_info(
            "strategy_manager", "crossover_strategy_selected",
            f"Selected crossover strategy: {crossover_type}."
        )
        fitness_info = f"Parent strategy 1 (fitness: {parent1.fitness}):\n{parent1.text}\n\n" \
            f"Parent strategy 2 (fitness: {parent2.fitness}):\n{parent2.text}"
        code_info = f"\n\nParent strategy 1 code snippet:\n```python\n{parent1.code_snippet}\n```\n\n" + \
                    f"Parent strategy 2 code snippet:\n```python\n{parent2.code_snippet}\n```"
        prompt = fitness_info + code_info
        crossover_text = self.llm_client.cross_strategies(
            [parent1.text, parent2.text],
            self.problem_info,
            [parent1.fitness, parent2.fitness],
            [{"code_snippet": parent1.code_snippet}, {
                "code_snippet": parent2.code_snippet}],
            system_prompt=system_prompt, 
            problem_data=self.problem_data
        )

        constraints_text, algorithm_design, code_snippet, relaxation_factors = self._process_crossover_mutation_response(
            crossover_text)
        constraint_order = list(relaxation_factors.keys())
        new_strategy_id = self._create_strategy(
            text=constraints_text,
            method=f"crossover_{crossover_type}", 
            generation=self.generation + 1,
            parent_ids=parent_ids,
            outer_iteration=outer_iteration,
            constraint_order=constraint_order,
            relaxation_factors=relaxation_factors,
            algorithm_design=algorithm_design,
            code_snippet=code_snippet
        )

        return new_strategy_id

    def mutate_strategy(self, strategy_id: str, intensity: float = 0.3, outer_iteration=0) -> str:
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return str(uuid.uuid4())

        mutation_type = random.choice(list(self.mutation_prompts.keys()))
        system_prompt = self.mutation_prompts[mutation_type]
        self.output_manager.log_info(
            "strategy_manager", "mutation_strategy_selected",
            f"Selected mutation strategy: {mutation_type}."
        )

        parent_info = f"strategy (fitness: {strategy.fitness}):\n{strategy.text}\n\n" + \
            f"strategy code snippet:\n```python\n{strategy.code_snippet}\n```"
        
        mutated_text = self.llm_client.mutate_strategy(
            strategy.text,
            intensity,
            self.problem_info,
            strategy.fitness,
            strategy.code_snippet,
            system_prompt=system_prompt,  
            problem_data=self.problem_data
        )

        constraints_text, algorithm_design, code_snippet, relaxation_factors = self._process_crossover_mutation_response(
            mutated_text)

        constraint_order = list(relaxation_factors.keys())
        new_strategy_id = self._create_strategy(
            text=constraints_text,
            method=f"mutated_{mutation_type}", 
            generation=self.generation + 1,
            parent_ids=[strategy_id],
            outer_iteration=outer_iteration,
            constraint_order=constraint_order,
            relaxation_factors=relaxation_factors,
            algorithm_design=algorithm_design,
            code_snippet=code_snippet
        )

        return new_strategy_id

    def _process_crossover_mutation_response(self, response: str) -> Tuple[str, str, str, Dict[str, float]]:

        original_response = response
        constraints_text = ""
        algorithm_design = ""
        code_snippet = ""
        relaxation_factors = {}
        outer_code_match = re.search(
            r'^```(?:python)?\s*(.*?)\s*```$', response, re.DOTALL)
        if outer_code_match:
            response = outer_code_match.group(1).strip()

        full_text = response

        constraints_pattern = r'\{Constraint Handling Order and Relaxation Factors:(.*?)\}'
        constraints_match = re.search(constraints_pattern, response, re.DOTALL)
        if constraints_match:
            constraints_text = constraints_match.group(1).strip()
        else:
            alt_constraints_pattern = r'Constraint Handling Order and Relaxation Factors[：:](.*?)(?=Algorithm Design[：:]|$)'
            alt_match = re.search(alt_constraints_pattern, response, re.DOTALL)
            if alt_match:
                constraints_text = alt_match.group(1).strip()

        algorithm_pattern = r'Algorithm Design[：:]\s*```json\s*(.*?)\s*```'
        algorithm_match = re.search(algorithm_pattern, response, re.DOTALL)
        if algorithm_match:
            algorithm_design = algorithm_match.group(1).strip()
        else:
            alt_algorithm_pattern = r'Algorithm Design[：:](.*?)(?=Code Snippet[：:]|$)'
            alt_match = re.search(alt_algorithm_pattern, response, re.DOTALL)
            if alt_match:
                algorithm_text = alt_match.group(1).strip()
                nested_json = re.search(
                    r'```(?:json)?\s*(.*?)\s*```', algorithm_text, re.DOTALL)
                if nested_json:
                    algorithm_design = nested_json.group(1).strip()
                else:
                    algorithm_design = algorithm_text

        code_pattern = r'Code Snippet[：:]\s*```python\s*(.*?)\s*```'
        code_match = re.search(code_pattern, response, re.DOTALL)
        if code_match:
            code_snippet = code_match.group(1).strip()
        else:
            alt_code_pattern = r'Code Snippet[：:](.*?)(?=$)'
            alt_match = re.search(alt_code_pattern, response, re.DOTALL)
            if alt_match:
                code_text = alt_match.group(1).strip()
                nested_code = re.search(
                    r'```(?:python)?\s*(.*?)\s*```', code_text, re.DOTALL)
                if nested_code:
                    code_snippet = nested_code.group(1).strip()
                else:
                    code_snippet = code_text
        if not code_snippet:
            function_pattern = r'(def\s+\w+\s*\(.*?(?:\n\s*return.*?)?)'
            function_matches = re.findall(
                function_pattern, response, re.DOTALL)
            if function_matches:
                code_snippet = '\n'.join(function_matches)
        combined_pattern = r'(\d+)[.．、)\s]+([a-zA-Z_]+)(?:_constraint)?(?:\s+relax\s*|\s*[：:]\s*)(\d+\.?\d*)'
        relaxation_matches = re.findall(combined_pattern, constraints_text)

        for match in relaxation_matches:
            _, constraint, factor = match
            constraint_name = constraint
            if not constraint_name.endswith('_constraint'):
                constraint_name += '_constraint'
            try:
                relaxation_factors[constraint_name] = float(factor)
            except ValueError:
                continue

        self.output_manager.log_info(
            "strategy_manager", "crossover_mutation_extraction",
            f"Original response length:: {len(original_response)}, processed response length: {len(response)}, "
            f"constraint text length: {len(constraints_text)}, algorithm design length: {len(algorithm_design)}, "
            f"code length: {len(code_snippet)}, number of relaxation factors: {len(relaxation_factors)}"
        )

        return constraints_text, algorithm_design, code_snippet, relaxation_factors

    def generate_strategy_from_mcts(self, mcts_info: Dict, outer_iteration=0) -> str:
        new_strategy_text = self.llm_client.generate_strategy_from_mcts(
            mcts_info,
            self.problem_info,
            outer_iteration=outer_iteration,
            use_llm=False
        )

        new_strategy_id = self._create_strategy(
            new_strategy_text,
            method="mcts_guided",
            generation=self.generation + 1,
            outer_iteration=outer_iteration
        )

        return new_strategy_id

    def update_strategy_fitness(self, strategy_id: str, fitness: float, solution_data: Dict = None) -> None:
        if strategy_id in self.strategies:
            self.output_manager.log_info(
                "text_strategy", "fitness_update_start",
                f"Starting to update strategy {strategy_id} fitness to {fitness}" +
                (", including solution data" if solution_data else "")
            )

            self.strategies[strategy_id].fitness = fitness
            self.strategies[strategy_id].evaluated = True
            if solution_data and isinstance(solution_data, dict):
                self.strategies[strategy_id].solution = solution_data
                self.output_manager.log_info(
                    "text_strategy", "solution_added",
                    f"Added solution data to strategy {strategy_id}, data size: {len(str(solution_data))}."
                )

            strategy_dict = {
                "id": self.strategies[strategy_id].id,
                "text": self.strategies[strategy_id].text,
                "constraint_order": self.strategies[strategy_id].constraint_order,
                "relaxation_factors": self.strategies[strategy_id].relaxation_factors,
                "algorithm_design": self.strategies[strategy_id].algorithm_design,
                "code_snippet": self.strategies[strategy_id].code_snippet,
                "fitness": self.strategies[strategy_id].fitness,
                "created_at": self.strategies[strategy_id].created_at.isoformat(),
                "parent_ids": self.strategies[strategy_id].parent_ids,
                "generation": self.strategies[strategy_id].generation,
                "method": self.strategies[strategy_id].method,
                "evaluated": self.strategies[strategy_id].evaluated,
                "outer_iteration": self.strategies[strategy_id].outer_iteration,
                "solution": self.strategies[strategy_id].solution
            }

            if hasattr(self.strategies[strategy_id], 'violation_analysis'):
                strategy_dict["violation_analysis"] = self.strategies[strategy_id].violation_analysis

            if hasattr(self.strategies[strategy_id], 'repair_info'):
                strategy_dict["repair_info"] = self.strategies[strategy_id].repair_info

            if solution_data and isinstance(solution_data, dict):
                strategy_dict["solution"] = solution_data
            elif hasattr(self.strategies[strategy_id], 'solution') and self.strategies[strategy_id].solution:
                strategy_dict["solution"] = self.strategies[strategy_id].solution
            else:
                strategy_dict["solution"] = {}

            if 'solution' in strategy_dict:
                solution_type = type(strategy_dict['solution'])
                self.output_manager.log_info(
                    "text_strategy", "solution_in_dict",
                    f"The solution field is already included in the strategy dictionary, type: {solution_type}."
                )
            else:
                self.output_manager.log_warning(
                    "text_strategy", "solution_missing",
                    f"The solution field is missing from the strategy dictionary."
                )

            try:
                self.output_manager.update_strategy(strategy_dict)
                self.output_manager.log_info(
                    "text_strategy", "fitness_update_complete",
                    f"The fitness of strategy {strategy_id} has been updated to {fitness}." +
                    (", and saved the solution data." if solution_data else "")
                )
            except Exception as e:
                self.output_manager.log_error(
                    "text_strategy", "fitness_update_error",
                    f"Error updating strategy fitness.: {str(e)}"
                )
        else:
            self.output_manager.log_warning(
                "text_strategy", "update_nonexistent",
                f"Attempted to update fitness of a nonexistent strategy: {strategy_id}"
            )

    def update_strategy_violation_info(self, strategy_id: str, violation_info: Dict) -> None:
        if strategy_id in self.strategies:
            self.output_manager.log_info(
                "text_strategy", "violation_info_update_start",
                f"Starting to update violation information for strategy {strategy_id}."
            )

            self.strategies[strategy_id].violation_analysis = violation_info.get(
                'violation_analysis', {})

            if hasattr(self.strategies[strategy_id], 'has_violations'):
                self.strategies[strategy_id].has_violations = violation_info.get(
                    'has_violations', False)
            else:

                setattr(self.strategies[strategy_id], 'has_violations', violation_info.get(
                    'has_violations', False))

            try:
                strategy_obj = self.strategies[strategy_id]
                strategy_dict = {
                    "id": strategy_obj.id,
                    "text": strategy_obj.text,
                    "constraint_order": strategy_obj.constraint_order,
                    "relaxation_factors": strategy_obj.relaxation_factors,
                    "algorithm_design": strategy_obj.algorithm_design,
                    "code_snippet": strategy_obj.code_snippet,
                    "fitness": strategy_obj.fitness,
                    "created_at": strategy_obj.created_at.isoformat(),
                    "parent_ids": strategy_obj.parent_ids,
                    "generation": strategy_obj.generation,
                    "method": strategy_obj.method,
                    "evaluated": strategy_obj.evaluated,
                    "outer_iteration": strategy_obj.outer_iteration,
                    "solution": strategy_obj.solution,
                    "violation_analysis": violation_info.get('violation_analysis', {}),
                    "has_violations": violation_info.get('has_violations', False),
                    "constraint_violations": violation_info.get('constraint_violations', 0),
                    "objective": violation_info.get('objective'),
                    "violation_update_timestamp": violation_info.get('timestamp', time.time())
                }

                self.output_manager.append_strategy(strategy_dict)

            except Exception as e:
                self.output_manager.log_error(
                    "text_strategy", "violation_info_save_error",
                    f"Error saving strategy violation information: {str(e)}."
                )
        else:
            self.output_manager.log_warning(
                "text_strategy", "update_violation_nonexistent",
                f"Attempted to update violation information for a nonexistent strategy: {strategy_id}."
            )

    def select_parents(self, population: List[str], tournament_size: int = 3) -> List[str]:
        if len(population) < 2:
            return population * 2  
        evaluated_strategies = [
            s_id for s_id in population
            if (s_id in self.strategies and
                self.strategies[s_id].evaluated)
        ]

        if not evaluated_strategies:
            evaluated_strategies = population

        selected_parents = []
        if tournament_size > len(evaluated_strategies):
            tournament_size = len(evaluated_strategies)

        tournament = random.sample(evaluated_strategies, tournament_size)

        best_parent = None
        best_fitness = float('-inf')

        for parent_id in tournament:
            if parent_id in self.strategies:
                strategy = self.strategies[parent_id]
                fitness = strategy.fitness if hasattr(
                    strategy, 'fitness') else float('-inf')

                if not isinstance(fitness, (int, float)) or fitness is None:
                    fitness = float('-inf')

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_parent = parent_id

        if best_parent is None:
            best_parent = random.choice(tournament)

        selected_parents.append(best_parent)
        attempts = 0
        max_attempts = 5  
        second_parent = best_parent  
        while second_parent == best_parent and attempts < max_attempts:
            attempts += 1
            tournament = random.sample(evaluated_strategies, tournament_size)
            second_best = None
            second_best_fitness = float('-inf')

            for parent_id in tournament:
                if parent_id in self.strategies and parent_id != best_parent:
                    strategy = self.strategies[parent_id]
                    fitness = strategy.fitness if hasattr(
                        strategy, 'fitness') else float('-inf')
                    if not isinstance(fitness, (int, float)) or fitness is None:
                        fitness = float('-inf')

                    if fitness > second_best_fitness:
                        second_best_fitness = fitness
                        second_best = parent_id

            if second_best is not None:
                second_parent = second_best

        if second_parent == best_parent:
            candidates = [p for p in evaluated_strategies if p != best_parent]
            if candidates:
                second_parent = random.choice(candidates)

        selected_parents.append(second_parent)
        return selected_parents

    def select_next_generation(self, candidates: List[str], elite_count: int,
                               population_size: int) -> List[str]:
        valid_candidates = [
            cid for cid in candidates if self.strategies[cid].evaluated]

        if len(valid_candidates) == 0:
            return self.current_population

        def get_sort_key(cid):
            fitness = self.strategies[cid].fitness
            if not isinstance(fitness, (int, float)) or fitness is None:
                return float('-inf')
            return fitness

        sorted_candidates = sorted(
            valid_candidates,
            key=get_sort_key,
            reverse=True 
        )

        elites = sorted_candidates[:min(elite_count, len(sorted_candidates))]

        remaining = population_size - len(elites)
        next_gen = elites.copy()

        if remaining > 0 and len(sorted_candidates) > len(elites):
            fitness_values = []
            for cid in sorted_candidates:
                fitness = self.strategies[cid].fitness
                if not isinstance(fitness, (int, float)) or fitness is None:
                    fitness = float('-inf')
                if fitness == float('-inf'):
                    fitness_values.append(0.001)  
                else:
                    fitness_values.append(max(0.01, fitness))

            min_fitness = min(fitness_values)

            if min_fitness <= 0:
                offset = abs(min_fitness) + 0.01
                fitness_values = [f + offset for f in fitness_values]

            total_fitness = sum(fitness_values)
            if total_fitness > 0:
                probabilities = [f / total_fitness for f in fitness_values]
            else:
                probabilities = [
                    1.0 / len(fitness_values)] * len(fitness_values)

            remaining_candidates = []
            for _ in range(remaining):
                r = random.random()
                cumulative = 0
                for i, prob in enumerate(probabilities):
                    cumulative += prob
                    if r <= cumulative:
                        remaining_candidates.append(sorted_candidates[i])
                        break
                else:
                    remaining_candidates.append(
                        random.choice(sorted_candidates))

            next_gen.extend(remaining_candidates)

        self.generation += 1
        self.current_population = next_gen

        return next_gen

    def prune_strategies(self, max_keep: int = 100) -> None:
        if len(self.strategies) <= max_keep:
            return

        all_strategies = list(self.strategies.values())
        sorted_strategies = sorted(
            all_strategies,
            key=lambda s: (s.id not in self.current_population, -
                           s.generation, -s.fitness)
        )
        keep_ids = set(
            strategy.id for strategy in sorted_strategies[:max_keep])

        for strategy_id in list(self.strategies.keys()):
            if strategy_id not in keep_ids:
                del self.strategies[strategy_id]

    def has_strategy(self, strategy_id: str) -> bool:
        return strategy_id in self.strategies

    def _limit_problem_data_points(self, problem_data: Dict, max_points: int = 5) -> Dict:
        if not problem_data:
            return {}

        limited_data = copy.deepcopy(problem_data)
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
                        data_dict[note_key] = f"Only the first {max_points} {field_name} shown to save tokens; actual data has {original_count} items."

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
                        data_dict[note_key] = f"Shown: {max_points}x{max_points} of {field_name} (actual {original_rows}x{original_cols}) to save tokens."

                elif self._is_large_array(field_value, threshold=max_points * 2):
                    if len(field_value) > max_points:
                        data_dict[field_name] = field_value[:max_points]
                        note_key = f"_{field_name}_limitation_note"
                        data_dict[note_key] = f"Shown: first {max_points} {field_name} (of {original_count}) to save tokens."

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
