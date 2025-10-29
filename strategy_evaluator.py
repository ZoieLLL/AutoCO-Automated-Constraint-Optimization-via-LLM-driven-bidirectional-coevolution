from typing import Dict, List, Tuple, Any, Optional, Set, Union, Callable
import os
import re
import importlib.util
import sys
import time
import copy
import traceback
import subprocess
import tempfile
import uuid
import math
import numpy as np
from output_manager import OutputManager
from models import TextStrategy
from code_generator import CodeGenerator
import json
from datetime import datetime


class StrategyEvaluator:
    def __init__(self, problem_data: Dict,
                 code_generator: CodeGenerator,
                 output_manager: OutputManager,
                 optimization_problem: Dict = None):

        self.problem_data = problem_data
        self.code_generator = code_generator
        self.output_manager = output_manager
        self.optimization_problem = optimization_problem or {}
        self.output_manager.log_info(
            "strategy_evaluator", "init_debug",
            f"StrategyEvaluator初始化 - problem_data keys: {list(problem_data.keys()) if isinstance(problem_data, dict) else 'Not a dict'}"
        )
        self.output_manager.log_info(
            "strategy_evaluator", "init_debug",
            f"is_multi_instance: {problem_data.get('is_multi_instance', 'Not found') if isinstance(problem_data, dict) else 'N/A'}"
        )
        if isinstance(problem_data, dict) and problem_data.get('is_multi_instance'):
            instances = problem_data.get('instances', [])
            self.output_manager.log_info(
                "strategy_evaluator", "init_debug",
                f"instances数量: {len(instances) if isinstance(instances, list) else 'Not a list'}"
            )
        self.repair_config = {}
        self.repair_controller = None
        self.repair_agent = None
        self.evaluation_cache = {}
        self._repair_components_initialized = False
        self.best_fitness = float('-inf')
        self.best_strategy_id = None
        self.best_solution = None
        self.execution_timeout = optimization_problem.get(
            "execution_timeout", 30)
        self.objective_function = self.optimization_problem.get(
            "objective_function", "default")
        self.objective_direction = self.optimization_problem.get(
            "objective_direction", "maximize")
        self.logger = self.output_manager
        self.is_multi_instance = problem_data.get("is_multi_instance", False)
        self.multi_instance_evaluation = self.is_multi_instance
        self.problem_type = problem_data.get("problem_type", "unknown")
        if self.is_multi_instance:
            self.output_manager.log_info(
                "strategy_evaluator", "multi_instance_detected",
                f"Detected multi-instance dataset, instance count: {problem_data.get('instance_count', 0)}"
            )

        self.multi_instance_evaluation = problem_data.get('evaluation', {}).get(
            'evaluation_method') == 'multi_instance_average'
        self.output_manager.log_info(
            "strategy_evaluator", "optimization_objective",
            f"Optimization objective: {self.objective_direction} {self.objective_function}"
        )

        from solution_runner import SolutionRunner
        self.solution_runner = SolutionRunner(output_manager)
        self.mcts_evolution = None

    def evaluate_strategy(self, strategy_id: str,
                          force_regenerate: bool = False,
                          variant_count: int = 1) -> Dict:
        if not isinstance(strategy_id, str) or not strategy_id:
            self.output_manager.log_error(
                "strategy_evaluator", "invalid_strategy_id",
                f"Invalid strategy ID: {strategy_id}"
            )
            return {
                'fitness': float('-inf'),
                'violation_analysis': {'has_violations': False},
                'strategy_text': '',
                'solution': {},
                'metrics': {}
            }
        if strategy_id in self.evaluation_cache and not force_regenerate:
            cached_result = self.evaluation_cache[strategy_id]
            if isinstance(cached_result, dict) and 'fitness' in cached_result:
                return cached_result
            else:
                fitness = cached_result.get("fitness", float(
                    '-inf')) if isinstance(cached_result, dict) else cached_result
                return {
                    'fitness': fitness,
                    'violation_analysis': {'has_violations': False},
                    'strategy_text': '',
                    'solution': cached_result.get('solution', {}) if isinstance(cached_result, dict) else {},
                    'metrics': cached_result.get('metrics', {}) if isinstance(cached_result, dict) else {}
                }
        strategy_dict = None
        if strategy_id in self.code_generator.problem_info.get("strategies", {}):
            strategy_dict = self.code_generator.problem_info["strategies"][strategy_id]
        else:
            if hasattr(self, 'mcts_evolution') and hasattr(self.mcts_evolution, 'problem_info'):
                if strategy_id in self.mcts_evolution.problem_info.get("strategies", {}):
                    strategy_dict = self.mcts_evolution.problem_info["strategies"][strategy_id]
                    if "strategies" not in self.code_generator.problem_info:
                        self.code_generator.problem_info["strategies"] = {}
                    self.code_generator.problem_info["strategies"][strategy_id] = strategy_dict

                    self.output_manager.log_info(
                        "strategy_evaluator", "strategy_found_in_mcts",
                        f"Found strategy {strategy_id} in MCTS problem_info"
                    )
            if not strategy_dict:
                try:
                    strategy_manager = None
                    if hasattr(self.code_generator.llm_client, "text_strategy_manager"):
                        strategy_manager = self.code_generator.llm_client.text_strategy_manager
                    elif hasattr(self.code_generator, "strategy_manager"):
                        strategy_manager = self.code_generator.strategy_manager
                    if strategy_manager:
                        if hasattr(strategy_manager, "has_strategy"):
                            has_strategy = strategy_manager.has_strategy(
                                strategy_id)
                        else:
                            has_strategy = strategy_id in getattr(
                                strategy_manager, "strategies", {})

                        if has_strategy and strategy_id in strategy_manager.strategies:
                            strategy_obj = strategy_manager.strategies[strategy_id]
                            strategy_dict = strategy_obj.to_dict()
                            if "strategies" not in self.code_generator.problem_info:
                                self.code_generator.problem_info["strategies"] = {
                                }
                            self.code_generator.problem_info["strategies"][strategy_id] = strategy_dict
                except Exception as e:
                    self.output_manager.log_error(
                        "strategy_evaluator", "strategy_lookup_error",
                        f"Error occurred while looking up strategy: {str(e)}"
                    )
        if not strategy_dict:
            if hasattr(self, 'create_mcts_strategy'):
                try:
                    temp_strategy_id = self.create_mcts_strategy(strategy_id)
                    if temp_strategy_id and temp_strategy_id in self.code_generator.problem_info.get("strategies", {}):
                        strategy_dict = self.code_generator.problem_info["strategies"][temp_strategy_id]
                        strategy_id = temp_strategy_id
                except Exception as e:
                    self.output_manager.log_warning(
                        "strategy_evaluator", "temp_strategy_creation_failed",
                        f"Failed to create temporary strategy: {str(e)}"
                    )
        if not strategy_dict:
            self.output_manager.log_warning(
                "strategy_evaluator", "strategy_not_found",
                f"Strategy {strategy_id} not found, returning default fitness -inf"
            )
            return {
                'fitness': float('-inf'),
                'violation_analysis': {'has_violations': False},
                'strategy_text': '',
                'solution': {},
                'metrics': {}
            }
        has_code_snippet = (
            'code_snippet' in strategy_dict and
            strategy_dict['code_snippet'] and
            len(strategy_dict['code_snippet']) > 30
        )
        code_variants = []
        if force_regenerate or strategy_id not in self.evaluation_cache:
            if has_code_snippet:
                code_variants = self.code_generator.generate_variants_from_snippet(
                    strategy_dict,
                    count=variant_count
                )
            else:
                code_variants = self.code_generator.generate_variants(
                    strategy_dict,
                    count=variant_count
                )
        else:
            cached_code_info = self.evaluation_cache[strategy_id]["code_info"]
            if cached_code_info:
                code_variants = [cached_code_info]

        if not code_variants:
            self.output_manager.log_error(
                "strategy_evaluator", "code_generation_failed",
                f"Code generation failed for strategy {strategy_id}"
            )
            return {
                'fitness': float('-inf'),
                'violation_analysis': {'has_violations': False},
                'strategy_text': strategy_dict.get('text', ''),
                'solution': {},
                'metrics': {}
            }
        best_solution = None
        best_fitness = float('-inf')
        best_code_info = None
        best_metrics = None

        for code_info in code_variants:
            try:
                result = self._execute_code(code_info["file_path"])
                solution = result.get("solution", {})
                metrics = result.get("metrics", {})
                self.output_manager.log_info(
                    "strategy_evaluator", "debug_result_objective_values",
                    f"[DEBUG_RESULT] result objective: objective={result.get('objective')}, "
                    f"profit_objective={result.get('profit_objective')}, "
                    f"result_keys={list(result.keys())}"
                )

                if not metrics and "evaluation" in result:
                    metrics = result["evaluation"]
                metrics["has_violations"] = result.get("has_violations", False)
                metrics["violation_analysis"] = result.get(
                    "violation_analysis", {})
                metrics["constraint_violations"] = result.get(
                    "constraint_violations", 0)
                metrics["violations_detail"] = result.get(
                    "violations_detail", {})

                if "objective" in result and result["objective"] is not None:
                    metrics["objective"] = result["objective"]
                    self.output_manager.log_info(
                        "strategy_evaluator", "debug_objective_fix_path1",
                        f"[DEBUG_FIX] Path 1: Set metrics.objective={result['objective']} from result.objective"
                    )
                elif "profit_objective" in result and result["profit_objective"] is not None:
                    metrics["profit_objective"] = result["profit_objective"]
                    metrics["objective"] = result["profit_objective"]
                    self.output_manager.log_info(
                        "strategy_evaluator", "debug_objective_fix_path2",
                        f"[DEBUG_FIX] Path 2: Set metrics.objective={result['profit_objective']} from result.profit_objective"
                    )
                else:
                    self.output_manager.log_warning(
                        "strategy_evaluator", "debug_objective_fix_no_path",
                        f"[DEBUG_FIX] No path: No valid objective or profit_objective in result"
                    )

                self.output_manager.log_info(
                    "strategy_evaluator", "debug_objective_fix_after",
                    f"[DEBUG_FIX] After fix: metrics.objective={metrics.get('objective')}, "
                    f"metrics.profit_objective={metrics.get('profit_objective')}"
                )
                self.output_manager.log_info(
                    "strategy_evaluator", "debug_metrics_after_fix",
                    f"[DEBUG] After fix metrics: has_violations={metrics.get('has_violations')}, "
                    f"constraint_violations={metrics.get('constraint_violations')}, "
                    f"violation_analysis_keys={list(metrics.get('violation_analysis', {}).keys())}"
                )
                self.output_manager.log_info(
                    "strategy_evaluator", "solution_extracted",
                    f"Extracted solution from execution result, type: {type(solution)}, " +
                    f"is empty: {not bool(solution)}"
                )
                fitness = self._calculate_fitness(solution, metrics)
                if fitness >= best_fitness:
                    best_fitness = fitness
                    best_solution = solution
                    best_code_info = code_info
                    best_metrics = metrics
            except Exception as e:
                error_msg = str(e)
                tb = traceback.format_exc()
                self.output_manager.log_error(
                    "strategy_evaluator", "execution_error",
                    f"Error occurred while executing code {code_info['id']}: {error_msg}\n{tb}"
                )
        self.output_manager.log_info(
            "strategy_evaluator", "debug_best_metrics_status",
            f"[DEBUG] best_metrics status: is_none={best_metrics is None}, "
            f"type={type(best_metrics) if best_metrics is not None else 'None'}"
        )

        if best_metrics is not None:
            self.output_manager.log_info(
                "strategy_evaluator", "debug_best_metrics_content",
                f"[DEBUG] best_metrics content: has_violations={best_metrics.get('has_violations')}, "
                f"violation_analysis={best_metrics.get('violation_analysis')}"
            )

        evaluation_result = {
            "strategy_id": strategy_id,
            "fitness": best_fitness,
            "metrics": best_metrics if best_metrics is not None else {},
            "code_info": best_code_info,
            "solution": best_solution,
            "strategy_text": strategy_dict.get('text', ''),
            "violation_analysis": self._analyze_constraint_violations(best_solution, best_metrics if best_metrics is not None else {})
        }
        if best_metrics is not None:
            self.output_manager.log_info(
                "strategy_evaluator", "debug_best_metrics",
                f"[DEBUG] best_metrics内容: has_violations={best_metrics.get('has_violations')}, "
                f"constraint_violations={best_metrics.get('constraint_violations')}, "
                f"violation_analysis={best_metrics.get('violation_analysis')}"
            )
        has_violations = evaluation_result['violation_analysis'].get(
            'has_violations', False)
        repair_enabled = self.repair_config.get('enabled', False)
        repair_completed = False
        if hasattr(self, 'repair_controller') and self.repair_controller:
            repair_completed = self.repair_controller.is_repair_completed(
                strategy_id)

        if (has_violations and repair_enabled and not repair_completed):
            if not self._repair_components_initialized:
                self._initialize_repair_components()
            if hasattr(self, 'repair_controller') and self.repair_controller:
                repair_result = self.execute_repair_process(
                    strategy_id, evaluation_result)
                if repair_result.get('success'):
                    repaired_code_file_path = self._get_repaired_code_file_path(
                        strategy_id)

                    if repaired_code_file_path and os.path.exists(repaired_code_file_path):
                        repaired_evaluation_result = self._evaluate_with_repaired_code_file(
                            strategy_id, repaired_code_file_path
                        )
                        repaired_evaluation_result['repair_info'] = {
                            'repaired': True,
                            'repair_attempts': self.repair_controller.repair_history.get(strategy_id, 0),
                            'repair_successful': True,
                            'used_repaired_code_file': True,
                            'original_violations': evaluation_result['violation_analysis']
                        }
                        self.evaluation_cache[strategy_id] = repaired_evaluation_result
                        return repaired_evaluation_result
                    else:
                        self.output_manager.log_warning(
                            "strategy_evaluator", "repaired_code_file_missing",
                            f"Repaired code file does not exist, using original evaluation result: {strategy_id}"
                        )
                        evaluation_result['repair_info'] = {
                            'repaired': True,
                            'repair_attempts': self.repair_controller.repair_history.get(strategy_id, 0),
                            'repair_successful': False,
                            'used_repaired_code_file': False,
                            'error': 'repaired_code_file_missing'
                        }
                else:
                    evaluation_result['repair_info'] = {
                        'repaired': False,
                        'repair_failed': True,
                        'repair_attempts': self.repair_controller.repair_history.get(strategy_id, 0)
                    }
            else:
                self.output_manager.log_error(
                    "strategy_evaluator", "repair_controller_unavailable",
                    f"Repair controller unavailable, skipping repair strategy {strategy_id}"
                )
        elif repair_completed:
            self.output_manager.log_info(
                "strategy_evaluator", "repair_completed_unsuccessful",
                f"Strategy {strategy_id} has completed repair process but failed, fitness remains {best_fitness}"
            )
            evaluation_result['repair_info'] = {
                'repaired': True,
                'repair_completed': True,
                'repair_successful': False,
                'repair_attempts': self.repair_controller.repair_history.get(strategy_id, 0) if hasattr(self, 'repair_controller') else 0,
                'max_attempts_reached': True
            }
        else:
            self.output_manager.log_info(
                "strategy_evaluator", "repair_not_needed",
                f"Strategy {strategy_id} does not need repair (has_violations: {has_violations}, repair_enabled: {repair_enabled}, repair_completed: {repair_completed})"
            )

        self.evaluation_cache[strategy_id] = evaluation_result
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_strategy_id = strategy_id
            self.best_solution = best_solution
            self.output_manager.log_info(
                "strategy_evaluator", "new_best_strategy",
                f"Found new best strategy {strategy_id}, fitness: {best_fitness}"
            )
        if hasattr(self, 'code_generator') and hasattr(self.code_generator, 'strategy_manager'):
            self.code_generator.strategy_manager.update_strategy_fitness(
                strategy_id, best_fitness, best_solution
            )
            if strategy_id in self.code_generator.strategy_manager.strategies:
                strategy_obj = self.code_generator.strategy_manager.strategies[strategy_id]
                strategy_obj.violation_analysis = evaluation_result["violation_analysis"]
        if hasattr(self, 'code_generator') and hasattr(self.code_generator, 'strategy_manager'):
            strategy_manager = self.code_generator.strategy_manager
            if strategy_id in strategy_manager.strategies:
                strategy_obj = strategy_manager.strategies[strategy_id]
                strategy_obj.violation_analysis = evaluation_result.get(
                    'violation_analysis', {})
                if 'repair_info' in evaluation_result:
                    strategy_obj.repair_info = evaluation_result['repair_info']
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
                    "violation_analysis": getattr(strategy_obj, 'violation_analysis', {}),
                }
                if hasattr(strategy_obj, 'repair_info'):
                    strategy_dict["repair_info"] = strategy_obj.repair_info
                self.output_manager.update_strategy(strategy_dict)
        return evaluation_result

    def evaluate_strategy_with_algorithms(self, strategy_id: str) -> Dict:
        strategy_dict = None
        if hasattr(self, 'code_generator') and hasattr(self.code_generator, 'problem_info'):
            strategies = self.code_generator.problem_info.get("strategies", {})
            if strategy_id in strategies:
                strategy_dict = strategies[strategy_id]
            else:
                self.output_manager.log_error(
                    "strategy_evaluator", "strategy_not_found",
                    f"Cannot find strategy {strategy_id}"
                )
                return float('-inf')
        if strategy_dict is None:
            self.output_manager.log_error(
                "strategy_evaluator", "strategy_not_found",
                f"Cannot find strategy {strategy_id}"
            )
            return {
                'fitness': float('-inf'),
                'violation_analysis': {'has_violations': False},
                'strategy_text': '',
                'solution': {},
                'metrics': {}
            }
        has_code_snippet = (
            'code_snippet' in strategy_dict and
            strategy_dict['code_snippet'] and
            len(strategy_dict['code_snippet']) > 50
        )

        available_algorithms = self.code_generator.algorithm_loader.get_available_algorithms()
        all_results = []
        algorithm_results = {}
        problem_data_for_code = getattr(
            self.code_generator, 'problem_data', self.problem_data)
        if has_code_snippet:
            try:
                code_variants = self.code_generator.generate_variants_from_snippet(
                    strategy_dict, count=1
                )

                if code_variants and len(code_variants) > 0:
                    code_info = code_variants[0]
                    metrics = self._execute_code(code_info["file_path"])
                    fitness = metrics.get("objective",  float('-inf'))
                    if fitness is not None and fitness > 1e9:
                        fitness = -float('inf')
                        metrics["objective"] = -float('inf')
                    self.output_manager.log_info(
                        "strategy_evaluator", "existing_code_result",
                        f"Existing code snippet score: {fitness}"
                    )

                    if fitness is not None and fitness > -1e9:
                        all_results.append(fitness)
                        algorithm_results["existing_snippet"] = {
                            "fitness": fitness,
                            "metrics": metrics,
                            "solution": metrics.get("solution", {})
                        }

                    else:
                        self.output_manager.log_warning(
                            "strategy_evaluator", "existing_code_invalid_fitness",
                            f"Existing code snippet fitness value is invalid or extremely low: {fitness}"
                        )
                else:
                    self.output_manager.log_warning(
                        "strategy_evaluator", "existing_code_generation_failed",
                        f"Failed to generate variants using existing code snippet, falling back to template generation"
                    )
            except Exception as e:
                self.output_manager.log_error(
                    "strategy_evaluator", "existing_code_evaluation_error",
                    f"Error occurred while evaluating existing code snippet: {str(e)}\n{traceback.format_exc()}"
                )
                self.output_manager.log_warning(
                    "strategy_evaluator", "fallback_to_templates",
                    f"fallback_to_templates: Falling back to template-based code generation"
                )
        if not has_code_snippet or not all_results:
            for algorithm_name in available_algorithms:
                try:
                    code_info = self.code_generator.generate_code_from_template(
                        strategy_dict, algorithm_name, variant_index=0
                    )
                    metrics = self._execute_code(code_info["code_path"])

                    fitness = metrics.get("objective",  -float('inf'))
                    if fitness is not None and fitness > 1e9:
                        self.output_manager.log_warning(
                            "strategy_evaluator", "abnormal_fitness",
                            f"Algorithm {algorithm_name} fitness value {fitness} exceeds 1e9, setting it to 0"
                        )
                        fitness = -float('inf')
                        metrics["objective"] = -float('inf')

                    if fitness is not None and fitness > -1e9:
                        all_results.append(fitness)
                        algorithm_results[algorithm_name] = {
                            "fitness": fitness,
                            "metrics": metrics,
                            "solution": metrics.get("solution", {})
                        }
                    else:
                        self.output_manager.log_warning(
                            "strategy_evaluator", "algorithm_invalid_fitness",
                            f"Algorithm {algorithm_name} fitness value is invalid or extremely low: {fitness}"
                        )

                except Exception as e:
                    self.output_manager.log_error(
                        "strategy_evaluator", "algorithm_evaluation_error",
                        f"Error occurred while evaluating algorithm {algorithm_name}: {str(e)}\n{traceback.format_exc()}"
                    )
        if all_results:
            avg_fitness = sum(all_results) / len(all_results)
            best_algorithm_result = None
            best_algorithm_fitness = float('-inf')
            for algorithm_name, result in algorithm_results.items():
                if result.get("fitness", float('-inf')) > best_algorithm_fitness:
                    best_algorithm_fitness = result.get("fitness")
                    best_algorithm_result = result
            best_solution = None
            if best_algorithm_result and "solution" in best_algorithm_result:
                best_solution = best_algorithm_result["solution"]

            evaluation_result = {
                "strategy_id": strategy_id,
                "fitness": avg_fitness,
                "algorithm_results": algorithm_results,
                "algorithms_count": len(all_results),
                "evaluation_time": time.time(),
                "best_solution": best_solution,
                "strategy_text": strategy_dict.get('text', ''),
                "violation_analysis": self._analyze_constraint_violations(best_solution, best_algorithm_result.get('metrics', {}) if best_algorithm_result else {})
            }
        else:
            self.output_manager.log_warning(
                "strategy_evaluator", "debug_all_results_empty",
                f"[DEBUG] all_results为空，algorithm_results={list(algorithm_results.keys())}"
            )
            best_algorithm_result = None
            best_solution = None

            if algorithm_results:
                first_algorithm = list(algorithm_results.keys())[0]
                best_algorithm_result = algorithm_results[first_algorithm]
                best_solution = best_algorithm_result.get("solution", {})

            evaluation_result = {
                "strategy_id": strategy_id,
                "fitness": float('-inf'),
                "algorithm_results": algorithm_results,
                "algorithms_count": 0,
                "evaluation_time": time.time(),
                "best_solution": best_solution,
                "strategy_text": strategy_dict.get('text', ''),
                "violation_analysis": self._analyze_constraint_violations(best_solution, best_algorithm_result.get('metrics', {}) if best_algorithm_result else {})
            }

        has_violations = evaluation_result['violation_analysis'].get(
            'has_violations', False)
        repair_enabled = self.repair_config.get('enabled', False)

        self.output_manager.log_info(
            "strategy_evaluator", "debug_violation_analysis",
            f"[DEBUG] violation_analysis: {evaluation_result['violation_analysis']}"
        )
        if (has_violations and repair_enabled):
            self.output_manager.log_info(
                "strategy_evaluator", "debug_repair_conditions_met",
                f"[DEBUG] Repair conditions met: has_violations={has_violations}, repair_enabled={repair_enabled}"
            )
            self.output_manager.log_info(
                "strategy_evaluator", "repair_conditions_met_multi_algo",
                f"Multi-algorithm evaluation strategy {strategy_id} meets repair conditions, starting repair process"
            )
            if not self._repair_components_initialized:
                self._initialize_repair_components()
            if hasattr(self, 'repair_controller') and self.repair_controller:
                repair_result = self.execute_repair_process(
                    strategy_id, evaluation_result)
                if repair_result.get('success'):
                    repaired_code_file_path = self._get_repaired_code_file_path(
                        strategy_id)

                    if repaired_code_file_path and os.path.exists(repaired_code_file_path):
                        repaired_evaluation_result = self._evaluate_with_repaired_code_file(
                            strategy_id, repaired_code_file_path
                        )
                        repaired_evaluation_result['repair_info'] = {
                            'repaired': True,
                            'repair_attempts': self.repair_controller.repair_history.get(strategy_id, 0),
                            'repair_successful': True,
                            'used_repaired_code_file': True,
                            'original_violations': evaluation_result['violation_analysis']
                        }
                        self.evaluation_cache[strategy_id] = repaired_evaluation_result
                        return repaired_evaluation_result
                    else:
                        evaluation_result['repair_info'] = {
                            'repaired': True,
                            'repair_attempts': self.repair_controller.repair_history.get(strategy_id, 0),
                            'repair_successful': False,
                            'used_repaired_code_file': False,
                            'error': 'repaired_code_file_missing'
                        }
                else:
                    self.output_manager.log_warning(
                        "strategy_evaluator", "repair_failed_multi_algo",
                        f"Multi-algorithm evaluation strategy {strategy_id} repair failed"
                    )
                    evaluation_result['repair_info'] = {
                        'repaired': False,
                        'repair_failed': True,
                        'repair_attempts': self.repair_controller.repair_history.get(strategy_id, 0)
                    }
            else:
                self.output_manager.log_error(
                    "strategy_evaluator", "debug_repair_controller_unavailable",
                    f"[DEBUG] Repair controller unavailable: has_repair_controller={hasattr(self, 'repair_controller')}, "
                    f"repair_controller_not_none={hasattr(self, 'repair_controller') and self.repair_controller is not None}"
                )
        if all_results:
            self.evaluation_cache[strategy_id] = evaluation_result
            if hasattr(self, 'strategy_manager') and self.strategy_manager:
                self.strategy_manager.update_strategy_fitness(
                    strategy_id, avg_fitness, best_solution)
            elif hasattr(self.code_generator, 'strategy_manager') and self.code_generator.strategy_manager:
                self.code_generator.strategy_manager.update_strategy_fitness(
                    strategy_id, avg_fitness, best_solution)

            self.output_manager.log_info(
                "strategy_evaluator", "multi_algorithm_evaluation",
                f"Multi-algorithm evaluation strategy {strategy_id} average fitness: {avg_fitness} (based on {len(all_results)} algorithms)"
            )

            return evaluation_result

        return {
            'fitness': float('-inf'),
            'violation_analysis': {'has_violations': False},
            'strategy_text': strategy_dict.get('text', ''),
            'solution': {},
            'metrics': {}
        }

    def _execute_code(self, file_path: str) -> Dict:
        from solution_runner import SolutionRunner
        runner = SolutionRunner(self.output_manager)
        try:
            if self.is_multi_instance:
                return self._execute_multi_instance_code(file_path, runner)
            result = runner.run_solution_code(
                file_path,
                self.problem_data,
                timeout=60
            )
            metrics = {
                "success": result.get("success", False),
                "size": len(str(result)),
                "violations_detail_size": 0,
                "constraint_violations": result.get("constraint_violations", 0),
                "constraints_checked": 0,
                "violations_detail": result.get("violations_detail", {}),
                "has_violations": result.get("has_violations", False),
                "violation_analysis": result.get("violation_analysis", {}),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "")
            }

            objective = None
            if "profit_objective" in result and result["profit_objective"] is not None:
                objective = result["profit_objective"]
                self.output_manager.log_info(
                    "strategy_evaluator", "objective_extracted",
                    f"Extracted objective value from profit_objective: {objective}"
                )
            elif "objective" in result and result["objective"] is not None:
                objective = result["objective"]
                self.output_manager.log_info(
                    "strategy_evaluator", "objective_extracted",
                    f"Extracted objective value from objective: {objective}"
                )
            elif "statistics" in result and "best_objective" in result["statistics"]:
                objective = result["statistics"]["best_objective"]
                self.output_manager.log_info(
                    "strategy_evaluator", "objective_extracted",
                    f"Extracted objective value from statistics.best_objective: {objective}"
                )
            elif "evaluation" in result and "profit" in result["evaluation"]:
                objective = result["evaluation"]["profit"]
                self.output_manager.log_info(
                    "strategy_evaluator", "objective_extracted",
                    f"Extracted objective value from evaluation.profit: {objective}"
                )
            stdout = result.get("stdout", "")
            if objective is None and stdout:
                patterns = [
                    r'BestObjective:\s*(-?\d+\.?\d*)',
                    r'Fitness:\s*(-?\d+\.?\d*)',
                    r'ObjectiveFunctionValue:\s*(-?\d+\.?\d*)',
                    r'profit:\s*(-?\d+\.?\d*)',
                    r'Done.*?(-?\d+\.?\d+)'
                ]

                for pattern in patterns:
                    match = re.search(pattern, stdout)
                    if match:
                        objective = float(match.group(1))
                        self.output_manager.log_info(
                            "strategy_evaluator", "objective_extracted",
                            f"Extracted objective value from stdout (pattern: {pattern}): {objective}"
                        )
                        break
            if objective is not None:
                metrics["objective"] = float(objective)
                if objective == float('-inf') or objective <= -1e8:
                    self.output_manager.log_warning(
                        "strategy_evaluator", "invalid_objective",
                        f"Detected invalid objective value: {objective}, marked as constraint violation"
                    )
                    metrics["has_violations"] = True
                    metrics["constraint_violations"] = max(
                        1, metrics["constraint_violations"])
                    if not metrics["violation_analysis"]:
                        metrics["violation_analysis"] = {}
                    metrics["violation_analysis"]["invalid_objective"] = f"Objective value is invalid: {objective}"
                else:
                    metrics["success"] = True
            else:
                self.output_manager.log_warning(
                    "strategy_evaluator", "missing_objective",
                    f"Unable to extract the objective function value, be marked as constraint violation"
                )
                metrics["has_violations"] = True
                metrics["constraint_violations"] = max(
                    1, metrics["constraint_violations"])
                if not metrics["violation_analysis"]:
                    metrics["violation_analysis"] = {}
                metrics["violation_analysis"]["missing_objective"] = "Unable to extract the objective function value"
            self.output_manager.log_info(
                "strategy_evaluator", "debug_final_metrics",
                f"[DEBUG] Final constructed metrics: has_violations={metrics.get('has_violations')}, "
                f"objective={metrics.get('objective')}, constraint_violations={metrics.get('constraint_violations')}"
            )
            try:
                if hasattr(self, 'code_generator') and hasattr(self.code_generator, 'strategy_manager'):
                    import re
                    strategy_id_match = re.search(
                        r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', file_path)
                    if strategy_id_match:
                        strategy_id = strategy_id_match.group(1)

                        violation_info = {
                            'has_violations': metrics.get('has_violations', False),
                            'violation_analysis': metrics.get('violation_analysis', {}),
                            'constraint_violations': metrics.get('constraint_violations', 0),
                            'objective': metrics.get('objective'),
                            'timestamp': time.time()
                        }

                        self.code_generator.strategy_manager.update_strategy_violation_info(
                            strategy_id, violation_info)

                        self.output_manager.log_info(
                            "strategy_evaluator", "violation_info_saved_immediately",
                            f"Strategy {strategy_id} violation info saved immediately: {violation_info}"
                        )
            except Exception as e:
                self.output_manager.log_warning(
                    "strategy_evaluator", "immediate_violation_save_failed",
                    f"Failed to save violation info immediately: {str(e)}"
                )
            return {
                "solution": result.get("solution", {}),
                "metrics": metrics,
                "success": result.get("success", False),
                "has_violations": metrics.get("has_violations", False),
                "violation_analysis": metrics.get("violation_analysis", {}),
                "constraint_violations": metrics.get("constraint_violations", 0),
                "violations_detail": metrics.get("violations_detail", {}),
                "profit_objective": result.get("profit_objective"),
                "objective": metrics.get("objective"),
                "evaluation": result.get("evaluation", {}),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "")
            }
        except Exception as e:
            self.output_manager.log_error(
                "strategy_evaluator", "algorithm_evaluation_error",
                f"Error occurred while evaluating algorithm {os.path.basename(file_path).split('_')[1]}: {str(e)}"
            )
            return {
                "solution": {},
                "metrics": {
                    "success": False,
                    "size": 0,
                    "violations_detail_size": 0,
                    "constraint_violations": 0,
                    "constraints_checked": 0,
                    "violations_detail": {}
                },
                "success": False,
                "has_violations": False,
                "violation_analysis": {},
                "constraint_violations": 0,
                "violations_detail": {},
                "profit_objective": None,
                "objective": None,
                "evaluation": {}
            }

    def _execute_multi_instance_code(self, file_path: str, runner) -> Dict:

        try:
            all_instances = []
            if "instances" in self.problem_data and self.problem_data["instances"]:
                all_instances = self.problem_data["instances"]
                self.output_manager.log_info(
                    "strategy_evaluator", "pkl_data_loaded",
                    f"Using instance data from problem_data: {len(all_instances)} complete instances (each instance is {len(all_instances[0]) if all_instances else 0} element tuple)"
                )
            for idx, inst in enumerate(all_instances[:5]):
                self.output_manager.log_info(
                    "debug", f"instance_{idx}_type",
                    f"Instance {idx} type: {type(inst)}"
                )
                if isinstance(inst, (tuple, list)):
                    self.output_manager.log_info(
                        "debug", f"instance_{idx}_len",
                        f"Instance {idx} length: {len(inst)}"
                    )
                else:
                    self.output_manager.log_info(
                        "debug", f"instance_{idx}_content",
                        f"Instance {idx} content: {inst}"
                    )
            if not all_instances:
                all_instances = self.problem_data.get("instances", [])
                self.output_manager.log_warning(
                    "strategy_evaluator", "using_json_instances",
                    f"Using JSON instance data, a total of {len(all_instances)} instances (type: {type(all_instances[0]) if all_instances else 'None'})"
                )
                for idx, inst in enumerate(all_instances[:5]):
                    self.output_manager.log_info(
                        "debug", f"json_instance_{idx}_type",
                        f"JSON instance {idx} type: {type(inst)}"
                    )
                    if isinstance(inst, (tuple, list)):
                        self.output_manager.log_info(
                            "debug", f"json_instance_{idx}_len",
                            f"JSON instance {idx} length: {len(inst)}"
                        )
                    else:
                        self.output_manager.log_info(
                            "debug", f"json_instance_{idx}_content",
                            f"JSON instance {idx} content: {inst}"
                        )

            if not all_instances:
                return self._execute_single_instance_fallback(file_path, runner)

            self.output_manager.log_info(
                "strategy_evaluator", "multi_instance_start",
                f"Starting multi-instance evaluation, a total of {len(all_instances)} instances"
            )

            instance_results = []
            instance_scores = []
            instance_violations = []

            for i, instance_data in enumerate(all_instances):
                if isinstance(instance_data, (tuple, list)):
                    self.output_manager.log_info(
                        "debug", f"eval_instance_{i}_len",
                        f"Instance {i} length: {len(instance_data)}"
                    )
                else:
                    self.output_manager.log_info(
                        "debug", f"eval_instance_{i}_content",
                        f"Instance {i} content: {instance_data}"
                    )
                try:
                    complete_instance_data = self._prepare_instance_data(
                        instance_data, i)
                    result = runner.run_solution_code(
                        file_path,
                        complete_instance_data,
                        timeout=self.execution_timeout
                    )

                    instance_results.append({
                        "instance_index": i,
                        "success": result.get("success", False),
                        "result": result
                    })

                    score = self._extract_objective_from_result(result)
                    has_violations = result.get("has_violations", False)
                    violations = result.get("violations_detail", {})
                    instance_violations.append({
                        "instance_index": i,
                        "has_violations": has_violations,
                        "violations": violations,
                        "success": result.get("success", False)
                    })

                    if score is not None and result.get("success", False) and not has_violations:
                        instance_scores.append(score)

                except Exception as e:
                    self.output_manager.log_error(
                        "strategy_evaluator", "instance_evaluation_error",
                        f"Error occurred while evaluating instance {i}: {str(e)}"
                    )
                    instance_results.append({
                        "instance_index": i,
                        "success": False,
                        "error": str(e)
                    })
                    instance_violations.append({
                        "instance_index": i,
                        "has_violations": True,
                        "violations": {"execution_error": str(e)},
                        "success": False
                    })
            successful_instances = len(instance_scores)
            total_instances = len(all_instances)
            violated_instances = len(
                [v for v in instance_violations if v["has_violations"]])
            if successful_instances > 0:
                average_score = float(np.mean(instance_scores))
                success_rate = successful_instances / total_instances
                overall_has_violations = False
                overall_constraint_violations = 0
                overall_violations_detail = {}

                self.output_manager.log_info(
                    "strategy_evaluator", "multi_instance_partial_success",
                    f"Multi-instance evaluation partial success: {successful_instances}/{total_instances} instances without violations, "
                    f"average score={average_score:.4f}"
                )
            else:
                average_score = float('-inf')
                success_rate = 0.0
                overall_has_violations = True
                all_violations = {}
                constraint_count = 0
                for violation_info in instance_violations:
                    if violation_info["has_violations"]:
                        for violation_type, details in violation_info["violations"].items():
                            if violation_type not in all_violations:
                                all_violations[violation_type] = []
                            all_violations[violation_type].append({
                                "instance": violation_info["instance_index"],
                                "details": details
                            })
                            constraint_count += 1

                overall_constraint_violations = constraint_count
                overall_violations_detail = all_violations

                self.output_manager.log_warning(
                    "strategy_evaluator", "multi_instance_all_failed",
                    f"Multi-instance evaluation all failed: {total_instances}/{total_instances} instances violated constraints,"
                )
            if instance_scores:
                average_score = float(np.mean(instance_scores))
                success_rate = len(instance_scores) / len(all_instances)
            else:
                average_score = float('-inf')
                success_rate = 0.0

            self.output_manager.log_info(
                "strategy_evaluator", "multi_instance_complete",
                f"Multi-instance evaluation complete: average score={average_score:.4f}, "
                f"success rate={success_rate:.2%} ({len(instance_scores)}/{len(all_instances)}), "
                f"overall violations: {overall_has_violations}"
            )
            return {
                "solution": {"average_score": average_score},
                "metrics": {
                    "success": success_rate > 0,
                    "average_score": average_score,
                    "success_rate": success_rate,
                    "total_instances": len(all_instances),
                    "successful_instances": len(instance_scores),
                    "instance_results": instance_results,
                    "size": sum(len(str(r.get("result", {}))) for r in instance_results),
                    "violations_detail_size": len(str(overall_violations_detail)),
                    "constraint_violations": overall_constraint_violations,
                    "constraints_checked": len(all_instances),
                    "violations_detail": overall_violations_detail,
                    "has_violations": overall_has_violations,
                    "objective": average_score,
                    "profit_objective": average_score,
                    "violation_analysis": {
                        "multi_instance_summary": {
                            "total_instances": total_instances,
                            "successful_instances": successful_instances,
                            "violated_instances": violated_instances,
                            "overall_violations": overall_has_violations
                        }
                    }
                },
                "success": success_rate > 0,
                "has_violations": overall_has_violations,
                "violation_analysis": {
                    "multi_instance_summary": {
                        "total_instances": total_instances,
                        "successful_instances": successful_instances,
                        "violated_instances": violated_instances,
                        "overall_violations": overall_has_violations
                    }
                },
                "constraint_violations": overall_constraint_violations,
                "violations_detail": overall_violations_detail,
                "objective": average_score,
                "profit_objective": average_score,
                "evaluation": {
                    "multi_instance_evaluation": True,
                    "average_score": average_score,
                    "success_rate": success_rate
                }
            }

        except Exception as e:
            self.output_manager.log_error(
                "strategy_evaluator", "multi_instance_error",
                f"Multi-instance evaluation failed: {str(e)}"
            )
            return self._execute_single_instance_fallback(file_path, runner)

    def _prepare_instance_data(self, instance_data: Any, instance_index: int) -> Dict:
        if isinstance(instance_data, dict):
            complete_data = self.problem_data.get("problem_config", {}).copy()
            complete_data.update(instance_data)
            complete_data['instances'] = [instance_data]
            complete_data['is_multi_instance'] = False
            return complete_data
        else:
            if isinstance(instance_data, (tuple, list)) and len(instance_data) == 7:
                coordinates, distance_matrix, demands, capacity, service_time, time_windows, max_vehicles = instance_data

                complete_data = self.problem_data.get(
                    "problem_config", {}).copy()
                complete_data.update({
                    "coordinates": coordinates,
                    "distance_matrix": distance_matrix,
                    "demands": demands,
                    "capacity": capacity,
                    "service_time": service_time,
                    "time_windows": time_windows,
                    "max_vehicles": max_vehicles,
                    "instance_data": instance_data,
                    "instance_index": instance_index,
                    "instances": [instance_data],
                    "is_multi_instance": False
                })
                return complete_data
            else:
                complete_data = self.problem_data.get(
                    "problem_config", {}).copy()
                complete_data["instance_data"] = instance_data
                complete_data["instance_index"] = instance_index
                complete_data['instances'] = [instance_data]
                complete_data['is_multi_instance'] = False
                return complete_data

    def _load_pkl_instances(self, pkl_path: str) -> List:
        import pickle
        import os

        try:
            if not os.path.exists(pkl_path):
                self.output_manager.log_error(
                    "strategy_evaluator", "pkl_file_not_found",
                    f"pkl_file_not_found: {pkl_path}"
                )
                return []

            with open(pkl_path, 'rb') as f:
                instances = pickle.load(f)

            self.output_manager.log_info(
                "strategy_evaluator", "pkl_instances_loaded",
                f"Loaded {len(instances)} instances from {pkl_path}"
            )
            if instances and len(instances) > 0:
                first_instance = instances[0]
                if isinstance(first_instance, (tuple, list)) and len(first_instance) == 7:
                    self.output_manager.log_info(
                        "strategy_evaluator", "pkl_format_validated",
                        f"PKL instance format validation passed: 7-element tuple/list"
                    )
                else:
                    self.output_manager.log_warning(
                        "strategy_evaluator", "pkl_format_unexpected",
                        f"PKL instance format anomaly: type={type(first_instance)}, length={len(first_instance) if hasattr(first_instance, '__len__') else 'N/A'}"
                    )

            return instances

        except Exception as e:
            self.output_manager.log_error(
                "strategy_evaluator", "pkl_load_error",
                f"PKL loading failed: {e}"
            )
            return []

    def _extract_objective_from_result(self, result: Dict) -> Optional[float]:
        for field in ["objective", "profit_objective", "total_distance", "cost", "score"]:
            if field in result and result[field] is not None:
                try:
                    return float(result[field])
                except (ValueError, TypeError):
                    continue
        solution = result.get("solution", {})
        if isinstance(solution, dict):
            for field in ["objective", "total_distance", "cost", "score"]:
                if field in solution and solution[field] is not None:
                    try:
                        return float(solution[field])
                    except (ValueError, TypeError):
                        continue

        return None

    def _execute_single_instance_fallback(self, file_path: str, runner) -> Dict:
        self.output_manager.log_info(
            "strategy_evaluator", "single_instance_fallback",
            "Falling back to single instance evaluation"
        )

        evaluation_data = self.problem_data.get(
            "problem_config", self.problem_data)

        result = runner.run_solution_code(
            file_path,
            evaluation_data,
            timeout=self.execution_timeout
        )

        return result

    def _collect_metrics_structure(self, metrics: Dict) -> Dict:
        result = {
            "success": metrics.get("execution_success", False)
        }

        def collect_structure(data, prefix=""):
            if isinstance(data, dict):
                result[f"{prefix}size"] = len(data)
                for key, value in data.items():
                    collect_structure(value, f"{prefix}{key}_")
            elif isinstance(data, list):
                result[f"{prefix}size"] = len(data)
                if data and isinstance(data[0], (dict, list)):
                    collect_structure(data[0], f"{prefix}0_")

        collect_structure(metrics)

        if "constraint_violations" in metrics:
            result["constraint_violations"] = metrics["constraint_violations"]
        if "constraints_checked" in metrics:
            result["constraints_checked"] = metrics["constraints_checked"]
        if "violations_detail" in metrics:
            result["violations_detail"] = metrics["violations_detail"]

        return result

    def _calculate_fitness(self, solution: Dict, metrics: Dict) -> float:
        self.output_manager.log_info(
            "strategy_evaluator", "debug_fitness_input",
            f"[DEBUG_FITNESS] Input metrics keys: {list(metrics.keys())}"
        )
        self.output_manager.log_info(
            "strategy_evaluator", "debug_fitness_objective_values",
            f"[DEBUG_FITNESS] objective={metrics.get('objective')}, "
            f"profit_objective={metrics.get('profit_objective')}, "
            f"average_score={metrics.get('average_score')}"
        )
        if "evaluation" in metrics:
            self.output_manager.log_info(
                "strategy_evaluator", "debug_fitness_evaluation",
                f"[DEBUG_FITNESS] evaluation keys: {list(metrics['evaluation'].keys()) if isinstance(metrics['evaluation'], dict) else type(metrics['evaluation'])}"
            )

        if "stdout" in metrics and isinstance(metrics["stdout"], str):
            import re
            stdout = metrics["stdout"]
            best_match = re.search(r'BestObjective:\s*(-?\d+\.\d+)', stdout)
            if best_match:
                value = float(best_match.group(1))
                self.output_manager.log_info(
                    "strategy_evaluator", "stdout_direct_extract",
                    f"Directly extracted objective value from stdout: {value}"
                )
                return value

            # Search for more possible patterns
            alt_match = re.search(r'Done.*?(\d+\.\d+)', stdout)
            if alt_match:
                value = float(alt_match.group(1))
                self.output_manager.log_info(
                    "strategy_evaluator", "stdout_alt_extract",
                    f"Directly extracted objective value from stdout (alternative pattern): {value}"
                )
                return value

        if "error" in metrics or metrics.get("timeout", False):
            self.output_manager.log_warning(
                "strategy_evaluator", "fitness_error",
                f"Execution error or timeout: {metrics.get('error', 'Timeout')}"
            )
            return float('-inf')
        objective_value = None

        if "objective" in metrics and metrics["objective"] is not None:
            objective_value = metrics["objective"]
            self.output_manager.log_info(
                "strategy_evaluator", "fitness_from_metrics_objective",
                f"get fitness from metrics.objective: {objective_value}"
            )

        elif "profit_objective" in metrics and metrics["profit_objective"] is not None:
            objective_value = metrics["profit_objective"]
            self.output_manager.log_info(
                "strategy_evaluator", "fitness_from_metrics_profit_objective",
                f"get fitness from metrics.profit_objective: {objective_value}"
            )

        elif "average_score" in metrics and metrics["average_score"] is not None and metrics["average_score"] != float('-inf'):
            objective_value = metrics["average_score"]
            self.output_manager.log_info(
                "strategy_evaluator", "fitness_from_metrics_average_score",
                f"get fitness from metrics.average_score: {objective_value}"
            )
        elif "evaluation" in metrics and isinstance(metrics["evaluation"], dict):
            evaluation = metrics["evaluation"]
            if "objective" in evaluation:
                objective_value = evaluation["objective"]
            elif "raw_objective" in evaluation:
                objective_value = evaluation["raw_objective"]

        if objective_value is None:
            self.output_manager.log_warning(
                "strategy_evaluator", "missing_objective",
                f"Unable to extract objective value, returning -inf. Metrics structure: {json.dumps(self._collect_metrics_structure(metrics), indent=2)}"
            )
            return float('-inf')

        # Ensure the return value is numeric
        try:
            return float(objective_value)
        except (ValueError, TypeError):
            return float('-inf')

    def get_best_solution(self) -> Tuple[Optional[Dict], Optional[str], float]:

        if self.best_fitness == float('-inf') or self.best_strategy_id is None:
            return None, None, float('-inf')

        solution = None
        if self.best_strategy_id in self.evaluation_cache:
            cache_entry = self.evaluation_cache[self.best_strategy_id]
            solution = cache_entry.get("solution")

        return solution, self.best_strategy_id, self.best_fitness

    def create_mcts_strategy(self, strategy_id: str) -> Optional[str]:

        try:
            if hasattr(self, 'mcts_evolution'):
                if "strategies" in self.mcts_evolution.problem_info and strategy_id in self.mcts_evolution.problem_info["strategies"]:
                    strategy_dict = self.mcts_evolution.problem_info["strategies"][strategy_id]
                    if "strategies" not in self.code_generator.problem_info:
                        self.code_generator.problem_info["strategies"] = {}
                    self.code_generator.problem_info["strategies"][strategy_id] = strategy_dict

                    return strategy_id

        except Exception as e:
            self.output_manager.log_warning(
                "strategy_evaluator", "mcts_strategy_error",
                f"mcts_strategy_error: {str(e)}"
            )

        return None

    def evaluate_strategy_on_multiple_datasets(self, strategy_id: str,
                                               dataset_ids: List[str] = None,
                                               force_regenerate: bool = False,
                                               variant_count: int = 1) -> Dict:
        if not hasattr(self, 'multi_dataset_evaluator'):
            from dataset_manager import DatasetManager
            from multi_dataset_evaluator import MultiDatasetEvaluator

            dataset_manager = DatasetManager()
            self.multi_dataset_evaluator = MultiDatasetEvaluator(
                dataset_manager,
                self.solution_runner,
                self.output_manager
            )

        if dataset_ids is None:
            self.multi_dataset_evaluator.dataset_manager.load_all_datasets()
            dataset_ids = self.multi_dataset_evaluator.dataset_manager.get_all_dataset_ids()

        if len(dataset_ids) > 10:  
            self.output_manager.log_warning(
                "strategy_evaluator", "too_many_datasets",
                f"Too many datasets ({len(dataset_ids)}), which may affect performance"
            )

        baseline_result = self.evaluate_strategy(
            strategy_id, force_regenerate, variant_count)

        if baseline_result.get('fitness') == float('-inf'):
            return baseline_result

        if strategy_id in self.evaluation_cache:
            code_info = self.evaluation_cache[strategy_id].get("code_info")
            if code_info and "file_path" in code_info:
                code_file = code_info["file_path"]
                multi_results = self.multi_dataset_evaluator.evaluate_on_multiple_datasets(
                    code_file, dataset_ids
                )

                self.multi_dataset_evaluator.save_evaluation_results(
                    multi_results, code_file)
                avg_objective = multi_results["aggregated"].get(
                    "average_objective")
                if avg_objective is not None:
                    multi_dataset_result = {
                        "strategy_id": strategy_id,
                        "fitness": avg_objective,
                        "multi_dataset_results": multi_results,
                        "strategy_text": baseline_result.get('strategy_text', ''),
                        "violation_analysis": baseline_result.get('violation_analysis', {'has_violations': False}),
                        "solution": baseline_result.get('solution', {}),
                        "metrics": baseline_result.get('metrics', {})
                    }


                    self.evaluation_cache[strategy_id] = multi_dataset_result

                    if avg_objective > self.best_fitness:
                        self.best_fitness = avg_objective
                        self.best_strategy_id = strategy_id

                        self.output_manager.log_info(
                            "strategy_evaluator", "new_best_strategy_multi_dataset",
                            f"Found new best strategy {strategy_id} with average fitness: {avg_objective}"
                        )

                    return multi_dataset_result
        return baseline_result

    def set_repair_config(self, repair_config: Dict):
        self.repair_config = repair_config
        if repair_config.get('enabled', False):
            self.initialize_repair_components()

    def _initialize_repair_components(self):
        if self._repair_components_initialized:
            return

        try:
            from repair_agent import RepairAgent
            from repair_controller import RepairController
            self.repair_agent = RepairAgent(
                self.code_generator.llm_client,
                self.output_manager,
                self.repair_config,
                self.code_generator.algorithm_loader
            )

            self.repair_controller = RepairController(
                self.repair_agent,
                self.output_manager,
                self.repair_config
            )

            self.repair_controller.set_code_generator(self.code_generator)
            self._repair_components_initialized = True

        except Exception as e:
            self.output_manager.log_error(
                "strategy_evaluator", "repair_init_failed",
                f"Repair mechanism initialization failed: {str(e)}"
            )
            self._repair_components_initialized = False

    def initialize_repair_components(self):
        return self._initialize_repair_components()

    def _analyze_constraint_violations(self, solution: Dict, metrics: Dict) -> Dict:
        violation_analysis = {
            'has_violations': False,
            'violation_count': 0,
            'violation_details': {},
            'violation_summary': '',
            'severity_level': 'none',
            'suggested_relaxations': []
        }

        if isinstance(metrics, dict):
            has_violations_direct = metrics.get('has_violations', False)
            nested_analysis = metrics.get('violation_analysis', {})
            has_nested_violations = False
            if isinstance(nested_analysis, dict) and nested_analysis:
                has_nested_violations = any(
                    isinstance(v, dict) and v.get('violated', False)
                    for v in nested_analysis.values()
                )
            objective_value = metrics.get('objective')
            has_invalid_objective = (objective_value is not None and
                                     (objective_value == float('-inf') or objective_value <= -1e8))
            self.output_manager.log_info(
                "strategy_evaluator", "debug_violation_check_detailed",
                f"[DEBUG] debug_violation_check_detailed:\n"
                f"  - has_violations_direct: {has_violations_direct}\n"
                f"  - nested_analysis: {nested_analysis}\n"
                f"  - has_nested_violations: {has_nested_violations}\n"
                f"  - objective_value: {objective_value}\n"
                f"  - has_invalid_objective: {has_invalid_objective}"
            )

            if has_violations_direct or has_nested_violations or has_invalid_objective:
                violation_analysis['has_violations'] = True

                if isinstance(nested_analysis, dict) and nested_analysis:
                    violation_analysis['violation_details'] = nested_analysis
                    violation_analysis['violation_summary'] = 'Detected constraint violations'
                    violation_analysis['severity_level'] = 'medium'
                    violation_count = sum(1 for k, v in nested_analysis.items()
                                          if isinstance(v, dict) and v.get('violated', False))
                    violation_analysis['violation_count'] = max(
                        1, violation_count)
                else:
                    violation_analysis['violation_details'] = metrics.get(
                        'violations_detail', {})
                    violation_analysis['violation_count'] = max(
                        1, metrics.get('constraint_violations', 1))
                    violation_analysis['violation_summary'] = 'Detected constraint violations'
                    violation_analysis['severity_level'] = 'medium'
                if has_invalid_objective:
                    if 'invalid_objective' not in violation_analysis['violation_details']:
                        violation_analysis['violation_details'][
                            'invalid_objective'] = f'Invalid objective value: {objective_value}'
                    violation_analysis['severity_level'] = 'critical'
                violation_analysis['suggested_relaxations'] = []
                return violation_analysis
            else:
                self.output_manager.log_info(
                    "strategy_evaluator", "debug_no_violation_confirmed",
                    f"[DEBUG] debug_no_violation_confirmed: has_violations_direct={has_violations_direct}, "
                    f"has_nested_violations={has_nested_violations}, has_invalid_objective={has_invalid_objective}"
                )
            if not solution and not metrics.get('objective'):
                violation_analysis.update({
                    'has_violations': True,
                    'violation_count': 1,
                    'violation_details': {'solution_generation': 'No valid solution generated'},
                    'violation_summary': 'No valid solution generated',
                    'severity_level': 'critical'
                })

        return violation_analysis

    def execute_repair_process(self, strategy_id: str, evaluation_result: Dict) -> Dict:
        repair_result = {'success': False, 'message': ''}

        try:
            if not hasattr(self, 'repair_controller') or not self.repair_controller:
                repair_result['message'] = 'Repair controller not initialized'
                return repair_result
            if not self.repair_controller.should_trigger_repair(evaluation_result, strategy_id):
                repair_result['message'] = 'Repair trigger conditions not met'
                return repair_result

            strategy_data = {
                'id': strategy_id,
                'text': evaluation_result.get('strategy_text', ''),
                'relaxation_strategy': evaluation_result.get('relaxation_strategy', []),
                'generated_code': evaluation_result.get('generated_code', '')
            }
            original_strategy = None
            try:
                if strategy_id in self.code_generator.problem_info.get("strategies", {}):
                    original_strategy = self.code_generator.problem_info["strategies"][strategy_id]
                elif hasattr(self, 'mcts_evolution') and hasattr(self.mcts_evolution, 'problem_info'):
                    if strategy_id in self.mcts_evolution.problem_info.get("strategies", {}):
                        original_strategy = self.mcts_evolution.problem_info["strategies"][strategy_id]
                elif hasattr(self.code_generator, 'llm_client') and hasattr(self.code_generator.llm_client, 'text_strategy_manager'):
                    strategy_manager = self.code_generator.llm_client.text_strategy_manager
                    if hasattr(strategy_manager, 'strategies') and strategy_id in strategy_manager.strategies:
                        strategy_obj = strategy_manager.strategies[strategy_id]
                        original_strategy = strategy_obj.to_dict() if hasattr(
                            strategy_obj, 'to_dict') else strategy_obj
                if original_strategy and isinstance(original_strategy, dict):
                    if 'code_snippet' in original_strategy and original_strategy['code_snippet']:
                        strategy_data['code_snippet'] = original_strategy['code_snippet']
                        if not strategy_data['generated_code']:
                            strategy_data['generated_code'] = original_strategy['code_snippet']
                    elif 'generated_code' in original_strategy and original_strategy['generated_code']:
                        strategy_data['code_snippet'] = original_strategy['generated_code']
                        if not strategy_data['generated_code']:
                            strategy_data['generated_code'] = original_strategy['generated_code']
                    for field in ['constraint_order', 'relaxation_factors']:
                        if field in original_strategy:
                            strategy_data[field] = original_strategy[field]
            except Exception as e:
                self.output_manager.log_warning(
                    "strategy_evaluator", "strategy_data_enrichment_failed",
                    f"strategy_data_enrichment_failed: {str(e)}"
                )
            max_repair_attempts = self.repair_config.get(
                'max_repair_attempts', 2)
            current_attempts = self.repair_controller.repair_history.get(
                strategy_id, 0)
            while current_attempts < max_repair_attempts:

                updated_strategy = self.repair_controller.execute_repair_cycle(
                    strategy_id, strategy_data, evaluation_result['violation_analysis']
                )

                if not updated_strategy:
                    repair_result['message'] = f'Repair attempt {current_attempts + 1} failed'
                    break

                current_attempts = self.repair_controller.repair_history.get(
                    strategy_id, 0)

                repaired_code_file_path = self._get_repaired_code_file_path(
                    strategy_id)
                if not repaired_code_file_path or not os.path.exists(repaired_code_file_path):
                    self.output_manager.log_warning(
                        "strategy_evaluator", "repaired_code_file_missing",
                        f"Repair attempt {current_attempts} failed: Repaired code file missing: {repaired_code_file_path}"
                    )
                    continue
                repaired_result = self._execute_code(repaired_code_file_path)
                repaired_metrics = repaired_result.get("metrics", {})
                repaired_solution = repaired_result.get("solution", {})
                self.output_manager.log_info(
                    "strategy_evaluator", "debug_repaired_result_objective_values",
                    f"[DEBUG_REPAIRED] repaired_result_objective: objective={repaired_result.get('objective')}, "
                    f"profit_objective={repaired_result.get('profit_objective')}, "
                    f"repaired_result_keys={list(repaired_result.keys())}"
                )

                if "objective" in repaired_result and repaired_result["objective"] is not None:
                    repaired_metrics["objective"] = repaired_result["objective"]
                    self.output_manager.log_info(
                        "strategy_evaluator", "debug_repaired_objective_fix_path1",
                        f"[DEBUG_REPAIRED] path1: Set repaired_metrics.objective = {repaired_result['objective']} from repaired_result.objective"
                    )
                elif "profit_objective" in repaired_result and repaired_result["profit_objective"] is not None:
                    repaired_metrics["profit_objective"] = repaired_result["profit_objective"]
                    repaired_metrics["objective"] = repaired_result["profit_objective"]
                    self.output_manager.log_info(
                        "strategy_evaluator", "debug_repaired_objective_fix_path2",
                        f"[DEBUG_REPAIRED] path2: Set repaired_metrics.objective = {repaired_result['profit_objective']} from repaired_result.profit_objective"
                    )
                else:
                    self.output_manager.log_warning(
                        "strategy_evaluator", "debug_repaired_objective_fix_no_path",
                        f"[DEBUG_REPAIRED] no path: No valid objective or profit_objective found in repaired_result"
                    )

                self.output_manager.log_info(
                    "strategy_evaluator", "debug_repaired_objective_fix_after",
                    f"[DEBUG_REPAIRED] After repair: repaired_metrics.objective={repaired_metrics.get('objective')}, "
                    f"repaired_metrics.profit_objective={repaired_metrics.get('profit_objective')}"
                )

                repaired_fitness = self._calculate_fitness(
                    repaired_solution, repaired_metrics)
                repaired_violation_analysis = self._analyze_constraint_violations(
                    repaired_solution, repaired_metrics)
                try:
                    if hasattr(self, 'code_generator') and hasattr(self.code_generator, 'strategy_manager'):
                        violation_info = {
                            'has_violations': repaired_violation_analysis.get('has_violations', False),
                            'violation_analysis': repaired_violation_analysis,
                            'constraint_violations': repaired_metrics.get('constraint_violations', 0),
                            'objective': repaired_metrics.get('objective'),
                            'fitness': repaired_fitness,
                            'repair_attempt': current_attempts,
                            'timestamp': time.time()
                        }
                        self.code_generator.strategy_manager.update_strategy_violation_info(
                            strategy_id, violation_info)
                except Exception as e:
                    self.output_manager.log_warning(
                        "strategy_evaluator", "repair_violation_save_failed",
                        f"Repair attempt {current_attempts} failed: {str(e)}"
                    )
                has_violations = repaired_violation_analysis.get(
                    'has_violations', False)
                if not has_violations and repaired_fitness > float('-inf'):
                    repair_result['success'] = True
                    repair_result['message'] = f'Repair attempt {current_attempts} succeeded: Feasible solution found'
                    repair_result['updated_strategy'] = updated_strategy
                    repair_result['final_fitness'] = repaired_fitness
                    repair_result['repair_attempts'] = current_attempts
                    break
                else:
                    evaluation_result = {
                        'strategy_id': strategy_id,
                        'fitness': repaired_fitness,
                        'metrics': repaired_metrics,
                        'solution': repaired_solution,
                        'strategy_text': evaluation_result.get('strategy_text', ''),
                        'violation_analysis': repaired_violation_analysis
                    }
                    strategy_data = updated_strategy
            if current_attempts >= max_repair_attempts:
                repair_result['message'] = f'Maximum repair attempts {max_repair_attempts} reached; repair process ends.'
                repair_result['repair_attempts'] = current_attempts

        except Exception as e:
            repair_result['message'] = f'Repair process error: {str(e)}'
            self.output_manager.log_error(
                "strategy_evaluator", "repair_process_error",
                f"Repair process error: {str(e)}"
            )

        return repair_result

    def _get_repaired_code_file_path(self, strategy_id: str) -> Optional[str]:
        try:
            strategy_dict = None
            if strategy_id in self.code_generator.problem_info.get("strategies", {}):
                strategy_dict = self.code_generator.problem_info["strategies"][strategy_id]
            elif hasattr(self, 'mcts_evolution') and hasattr(self.mcts_evolution, 'problem_info'):
                if strategy_id in self.mcts_evolution.problem_info.get("strategies", {}):
                    strategy_dict = self.mcts_evolution.problem_info["strategies"][strategy_id]

            if strategy_dict and 'repaired_code_file_path' in strategy_dict:
                file_path = strategy_dict['repaired_code_file_path']
                if os.path.exists(file_path):
                    return file_path
                else:
                    self.output_manager.log_warning(
                        "strategy_evaluator", "repaired_code_file_not_found",
                        f"The post-fix code file for strategy {strategy_id} does not exist: {file_path}"
                    )

            strategy_file_path = self._find_strategy_file_path(strategy_id)
            if strategy_file_path:
                try:
                    with open(strategy_file_path, 'r', encoding='utf-8') as f:
                        strategy_file = json.load(f)

                    if 'repaired_code_file_path' in strategy_file:
                        file_path = strategy_file['repaired_code_file_path']
                        if os.path.exists(file_path):
                            return file_path
                        else:
                            self.output_manager.log_warning(
                                "strategy_evaluator", "repaired_code_file_not_found_in_file",
                                f"The post-fix code file recorded in the strategy file does not exist: {file_path}"
                            )
                except Exception as e:
                    self.output_manager.log_warning(
                        "strategy_evaluator", "strategy_file_read_error",
                        f"Error reading strategy file: {str(e)}"
                    )

            codes_dir = os.path.join(self.output_manager.base_dir, "codes")
            if os.path.exists(codes_dir):
                import glob
                pattern = os.path.join(
                    codes_dir, f"{strategy_id}_repaired_*.py")
                repaired_files = glob.glob(pattern)
                if repaired_files:
                    latest_file = max(repaired_files, key=os.path.getmtime)
                    self.output_manager.log_info(
                        "strategy_evaluator", "found_repaired_code_file",
                        f"Repaired code file found by pattern matching: {latest_file}"
                    )
                    return latest_file

            return None

        except Exception as e:
            self.output_manager.log_warning(
                "strategy_evaluator", "get_repaired_code_file_error",
                f"Error getting post-fix code file path: {str(e)}"
            )
            return None

    def _find_strategy_file_path(self, strategy_id: str) -> Optional[str]:
        try:
            for root, dirs, files in os.walk(self.output_manager.base_dir):
                for file in files:
                    if file.endswith('.json') and strategy_id in file:
                        return os.path.join(root, file)
            return None
        except Exception as e:
            self.output_manager.log_warning(
                "strategy_evaluator", "find_strategy_file_error",
                f"Error finding strategy file: {str(e)}"
            )
            return None

    def _evaluate_with_repaired_code_file(self, strategy_id: str, code_file_path: str) -> Dict:
        try:
            strategy_dict = None
            if strategy_id in self.code_generator.problem_info.get("strategies", {}):
                strategy_dict = self.code_generator.problem_info["strategies"][strategy_id]
            elif hasattr(self, 'mcts_evolution') and hasattr(self.mcts_evolution, 'problem_info'):
                if strategy_id in self.mcts_evolution.problem_info.get("strategies", {}):
                    strategy_dict = self.mcts_evolution.problem_info["strategies"][strategy_id]

            strategy_text = strategy_dict.get(
                'text', '') if strategy_dict else ''
            result = self._execute_code(code_file_path)
            solution = result.get("solution", {})
            metrics = result.get("metrics", {})
            if not metrics and "evaluation" in result:
                metrics = result["evaluation"]
            fitness = self._calculate_fitness(solution, metrics)
            evaluation_result = {
                "strategy_id": strategy_id,
                "fitness": fitness,
                "metrics": metrics if metrics is not None else {},
                "solution": solution,
                "strategy_text": strategy_text,
                "violation_analysis": self._analyze_constraint_violations(solution, metrics) if solution else {'has_violations': False},
                "code_file_path": code_file_path,
                "evaluation_method": "repaired_code_file"
            }

            try:
                if hasattr(self, 'code_generator') and hasattr(self.code_generator, 'strategy_manager'):
                    violation_info = {
                        'has_violations': evaluation_result['violation_analysis'].get('has_violations', False),
                        'violation_analysis': evaluation_result['violation_analysis'],
                        'constraint_violations': metrics.get('constraint_violations', 0),
                        'objective': metrics.get('objective'),
                        'fitness': fitness,
                        'is_repaired': True,  
                        'timestamp': time.time()
                    }
                    self.code_generator.strategy_manager.update_strategy_violation_info(
                        strategy_id, violation_info)

                    self.output_manager.log_info(
                        "strategy_evaluator", "repaired_violation_info_saved",
                        f"Repaired strategy {strategy_id} violation info saved: has_violations={violation_info['has_violations']}, "
                        f"fitness={fitness}, objective={violation_info['objective']}"
                    )
            except Exception as e:
                self.output_manager.log_warning(
                    "strategy_evaluator", "repaired_violation_save_failed",
                    f"Error saving repaired strategy {strategy_id} violation info: {str(e)}"
                )

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_strategy_id = strategy_id
                self.best_solution = solution

                self.output_manager.log_info(
                    "strategy_evaluator", "new_best_strategy_repaired",
                    f"Repaired strategy {strategy_id} is the new best strategy, fitness: {fitness}"
                )

            return evaluation_result

        except Exception as e:
            self.output_manager.log_error(
                "strategy_evaluator", "repaired_code_evaluation_error",
                f"Error evaluating strategy {strategy_id} with post-fix code file: {str(e)}"
            )
            return self.evaluate_strategy(strategy_id, force_regenerate=True, variant_count=1)
