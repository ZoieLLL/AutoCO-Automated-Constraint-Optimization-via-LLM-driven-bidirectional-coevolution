from typing import Dict, List, Tuple, Optional, Any, Callable
from copy import deepcopy
import inspect
import json


class BaseStrategy:
    def __init__(self, problem_data: Dict, problem_params: Dict):
        self.problem_data = problem_data
        self.problem_params = problem_params
        self.constraint_functions = {}
        self.objective_functions = {}
        self.constraint_info = {}

    def get_hard_constraints(self) -> List[str]:
        return [name for name, info in self.constraint_info.items() if info["is_hard"]]

    def get_soft_constraints(self) -> List[str]:
        return [name for name, info in self.constraint_info.items() if not info["is_hard"]]

    def evaluate_constraint(self, constraint_name: str, solution: Dict, relaxation_factor: float = 1.0) -> Tuple[bool, float]:
        if constraint_name not in self.constraint_functions:
            raise ValueError(
                f"Constraint function not found: {constraint_name}")

        constraint_func = self.constraint_functions[constraint_name]

        try:
            result = constraint_func(
                solution, self.problem_data, self.problem_params, relaxation_factor)
            if isinstance(result, tuple) and len(result) == 2:
                return result
            elif isinstance(result, bool):
                return (result, 0.0 if result else 1.0)
            else:
                return (bool(result), 0.0 if bool(result) else 1.0)
        except Exception as e:
            print(f"Error evaluating constraint {constraint_name}: {e}")
            return (False, float('inf'))

    def evaluate_objective(self, objective_name: str, solution: Dict) -> float:
        if objective_name not in self.objective_functions:
            raise ValueError(f"Objective function not found: {objective_name}")

        objective_func = self.objective_functions[objective_name]
        try:
            return objective_func(solution, self.problem_data, self.problem_params)
        except Exception as e:
            print(f"Error evaluating objective {objective_name}: {e}")
            return float('-inf')

    def evaluate_solution(self, solution: Dict, relaxation_factors: Dict[str, float] = None) -> Dict:
        relaxation_factors = relaxation_factors or {}

        result = {
            "constraints_satisfied": True,
            "constraint_details": {},
            "objective_values": {},
            "total_objective": 0.0,
            "feasible": True
        }

        for name in self.constraint_functions:
            relaxation = relaxation_factors.get(name, 1.0)
            satisfied, violation = self.evaluate_constraint(
                name, solution, relaxation)

            result["constraint_details"][name] = {
                "satisfied": satisfied,
                "violation": violation,
                "relaxation_factor": relaxation,
                "is_hard": self.constraint_info.get(name, {}).get("is_hard", False)
            }

            if self.constraint_info.get(name, {}).get("is_hard", False) and not satisfied:
                result["feasible"] = False
                result["constraints_satisfied"] = False

        for name in self.objective_functions:
            value = self.evaluate_objective(name, solution)
            result["objective_values"][name] = value
            result["total_objective"] += value
        return result
