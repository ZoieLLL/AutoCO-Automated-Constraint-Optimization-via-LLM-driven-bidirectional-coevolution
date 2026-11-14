from typing import Dict, List, Optional
import time
import json
import os
import numpy as np


class MultiDatasetEvaluator:

    def __init__(self, dataset_manager, solution_runner, output_manager):

        self.dataset_manager = dataset_manager
        self.solution_runner = solution_runner
        self.output_manager = output_manager

    def evaluate_on_multiple_datasets(self, code_file: str, dataset_ids: List[str] = None,
                                      timeout: int = None) -> Dict:
        if dataset_ids is None:
            self.dataset_manager.load_all_datasets()
            dataset_ids = self.dataset_manager.get_all_dataset_ids()

        results = {}

        for dataset_id in dataset_ids:
            dataset = self.dataset_manager.get_dataset(dataset_id)
            if not dataset:
                results[dataset_id] = {
                    "success": False,
                    "error": f"Dataset {dataset_id} does not exist",
                    "score": float('-inf')
                }
                continue

            try:
                if self.dataset_manager.is_multi_instance_dataset(dataset_id):
                    result = self._evaluate_multi_instance_dataset(
                        code_file, dataset_id, timeout
                    )
                else:
                    result = self._evaluate_single_instance_dataset(
                        code_file, dataset_id, timeout
                    )

                results[dataset_id] = result

            except Exception as e:
                self.output_manager.log_error(
                    "multi_dataset_evaluator", "evaluation_error",
                    f"Error evaluating dataset {dataset_id}: {str(e)}"
                )
                results[dataset_id] = {
                    "success": False,
                    "error": str(e),
                    "score": float('-inf')
                }

        aggregated_results = self._aggregate_results(results)

        return {
            "by_dataset": results,
            "aggregated": aggregated_results
        }

    def _evaluate_multi_instance_dataset(self, code_file: str, dataset_id: str, timeout: int = None) -> Dict:
        instance_count = self.dataset_manager.get_instance_count(dataset_id)
        instance_results = []
        instance_scores = []

        for i in range(instance_count):
            instance_data = self.dataset_manager.get_instance_data(
                dataset_id, i)
            if not instance_data:
                continue

            result = self.solution_runner.run_solution_code(
                code_file, instance_data, timeout
            )

            instance_results.append({
                "instance_index": i,
                "result": result
            })

            score = self._extract_objective(result)
            if score is not None:
                instance_scores.append(score)

        if instance_scores:
            average_score = np.mean(instance_scores)
            success_rate = len(instance_scores) / instance_count
        else:
            average_score = float('-inf')
            success_rate = 0.0

        self.output_manager.log_info(
            "multi_dataset_evaluator", "multi_instance_complete",
            f"Multi-instance evaluation complete for {dataset_id}: average score={average_score:.4f}, "
            f"success rate={success_rate:.2%} ({len(instance_scores)}/{instance_count})"
        )

        return {
            "success": success_rate > 0,
            "score": average_score,
            "instance_count": instance_count,
            "successful_instances": len(instance_scores),
            "success_rate": success_rate,
            "instance_results": instance_results,
            "is_multi_instance": True
        }

    def _evaluate_single_instance_dataset(self, code_file: str, dataset_id: str, timeout: int = None) -> Dict:
        dataset = self.dataset_manager.get_dataset(dataset_id)
        result = self.solution_runner.run_solution_code(
            code_file, dataset, timeout)
        if isinstance(result, dict):
            result["is_multi_instance"] = False

        return result

    def _aggregate_results(self, results: Dict[str, Dict]) -> Dict:

        if not results:
            return {
                "success": False,
                "message": "No results to aggregate"
            }

        objectives = []
        execution_times = []
        success_count = 0

        for dataset_id, result in results.items():
            if result.get("execution_success", False):
                success_count += 1
                objective = self._extract_objective(result)
                if objective is not None:
                    objectives.append(objective)

                if "execution_time" in result:
                    execution_times.append(result["execution_time"])

        aggregated = {
            "success_rate": success_count / len(results) if results else 0,
            "dataset_count": len(results),
            "success_count": success_count
        }

        if objectives:
            aggregated["average_objective"] = np.mean(objectives)
            aggregated["median_objective"] = np.median(objectives)
            aggregated["min_objective"] = min(objectives)
            aggregated["max_objective"] = max(objectives)
            aggregated["std_objective"] = np.std(objectives)

        if execution_times:
            aggregated["average_execution_time"] = np.mean(execution_times)
            aggregated["total_execution_time"] = sum(execution_times)

        return aggregated

    def _extract_objective(self, result: Dict) -> Optional[float]:
        for field in ['profit_objective', 'objective', 'total_cost', 'fitness']:
            if field in result and isinstance(result[field], (int, float)):
                return float(result[field])
        for container in ['evaluation', 'statistics', 'costs']:
            if container in result and isinstance(result[container], dict):
                for field in ['objective', 'total', 'fitness', 'best_objective']:
                    if field in result[container] and isinstance(result[container][field], (int, float)):
                        return float(result[container][field])
        if 'stdout' in result and isinstance(result['stdout'], str):
            import re
            patterns = [
                r'total cost:\s*(-?\d+\.?\d*)',
                r'objective:\s*(-?\d+\.?\d*)',
                r'fitness:\s*(-?\d+\.?\d*)',
                r'profit:\s*(-?\d+\.?\d*)'
            ]

            for pattern in patterns:
                match = re.search(pattern, result['stdout'])
                if match:
                    try:
                        return float(match.group(1))
                    except (ValueError, IndexError):
                        pass

        return None

    def save_evaluation_results(self, results: Dict, code_file: str) -> str:
        code_name = os.path.basename(code_file)
        results_dir = os.path.join(
            self.output_manager.base_dir, "multi_dataset_results")
        os.makedirs(results_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            results_dir, f"{code_name}_{timestamp}_results.json")

        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            return result_file
        except Exception as e:
            self.output_manager.log_error(
                "multi_dataset_evaluator", "save_results_error",
                f"Failed to save evaluation results: {e}"
            )
            return None
