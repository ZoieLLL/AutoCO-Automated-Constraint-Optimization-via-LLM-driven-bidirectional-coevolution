from typing import Tuple, Dict, Set, List, Any, Optional
from output_manager import OutputManager
from base_strategy import BaseStrategy


class SolutionValidator:

    def __init__(self, problem_data: Dict, problem_params: Dict,
                 output_manager: OutputManager, strategy: Optional[BaseStrategy] = None):
        self.problem_data = problem_data
        self.problem_params = problem_params
        self.output_manager = output_manager

        self.strategy = strategy if strategy else BaseStrategy(
            problem_data, problem_params)

    def validate_solution(self, solution: Dict) -> Tuple[bool, Dict]:

        try:
            evaluation_result = self.strategy.evaluate_solution(solution)

            self._log_validation_result(solution, evaluation_result)

            return evaluation_result["feasible"], evaluation_result

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.output_manager.log_error(
                "solution_validator", "validation_error", error_msg)
            return False, {"error": error_msg}

    def _log_validation_result(self, solution: Dict, evaluation: Dict) -> None:
        feasible = evaluation["feasible"]
        constraints_satisfied = evaluation["constraints_satisfied"]
        total_objective = evaluation["total_objective"]

        log_message = {
            "feasible": feasible,
            "constraints_satisfied": constraints_satisfied,
            "total_objective": total_objective,
            "constraint_details": {
                name: {
                    "satisfied": details["satisfied"],
                    "violation": details["violation"]
                } for name, details in evaluation["constraint_details"].items()
            },
            "objective_values": evaluation["objective_values"]
        }

        self.output_manager.log_validation_result(log_message)

    def _log_validation_result_with_relaxations(self, solution: Dict,
                                                evaluation: Dict,
                                                relaxation_factors: Dict[str, float]) -> None:

        log_message = {
            "feasible": evaluation["feasible"],
            "constraints_satisfied": evaluation["constraints_satisfied"],
            "total_objective": evaluation["total_objective"],
            "relaxation_factors": relaxation_factors,
            "constraint_details": evaluation["constraint_details"],
            "objective_values": evaluation["objective_values"]
        }

        self.output_manager.log_relaxed_validation_result(log_message)