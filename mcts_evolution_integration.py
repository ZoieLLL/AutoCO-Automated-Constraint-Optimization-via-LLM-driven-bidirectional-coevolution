
from typing import Dict, List, Tuple, Any, Optional, Set, Callable, Union
import time
import os
import random
import uuid
from datetime import datetime
from models import OptimizationParams, TextStrategy, MCTSNode
from output_manager import OutputManager
from mcts import MCTSTree
from evolutionary_algorithm import EvolutionaryAlgorithm


class MCTSEvolutionIntegration:

    def __init__(self, mcts: MCTSTree,
                 evolution: EvolutionaryAlgorithm,
                 params: OptimizationParams,
                 output_manager: OutputManager):

        self.mcts = mcts
        self.evolution = evolution
        self.params = params
        self.output_manager = output_manager

        self.iteration = 0
        self.last_mcts_iteration = -1
        self.mcts_count = 0
        self.evolution_count = 0
        self.stagnation_count = 0

        self.best_fitness = float('-inf')
        self.best_strategy_id = None

        self.evaluated_paths = {}

        self.feedback_strategy_ids = set()

        self.problem_info = self.evolution.problem_info if hasattr(
            self.evolution, 'problem_info') else {}

        if hasattr(evolution, 'strategy_manager'):
            self.mcts.text_strategy_manager = evolution.strategy_manager
            if hasattr(evolution.strategy_manager.llm_client, 'algorithm_loader'):
                self.mcts.algorithm_loader = evolution.strategy_manager.llm_client.algorithm_loader
            self.mcts.strategy_evaluator = evolution.evaluator if hasattr(
                evolution, 'evaluator') else None
            self.mcts.outer_iteration = 0

    def run_integrated_optimization(self, evaluator,
                                    time_limit: float = 3600.0) -> Dict:

        start_time = time.time()
        for i in range(self.params.MAX_OUTER_ITERATIONS):
            if time.time() - start_time > time_limit:
                self.output_manager.log_info(
                    "mcts_evolution_integration", "time_limit_reached",
                    f"finished {i}/{self.params.MAX_OUTER_ITERATIONS} oUTER_ITERATIONS"
                )
                break

            self.iteration = i + 1
            iteration_start_time = time.time()

            self._feedback_all_evaluated_to_mcts()

            should_run_mcts = self._should_run_mcts()

            if should_run_mcts:

                mcts_result = self._run_mcts_search(
                    evaluator, time_limit - (time.time() - start_time))

                self._inject_mcts_strategy(mcts_result, outer_iteration=i+1)

                self.last_mcts_iteration = self.iteration
                self.mcts_count += 1

            best_code_snippet = None
            if i > 0 and self.best_strategy_id:
                strategy_manager = self.evolution.strategy_manager
                if self.best_strategy_id in strategy_manager.strategies:
                    best_strategy = strategy_manager.strategies[self.best_strategy_id]
                    best_code_snippet = best_strategy.code_snippet

            if i == 0:
                population = self.evolution.initialize_population(
                    outer_iteration=i+1)
            else:
                if should_run_mcts:
                    self._inject_mcts_strategy(mcts_result)
                population = self.evolution.initialize_population(
                    outer_iteration=i+1,
                    best_code_snippet=best_code_snippet
                )

            self._evaluate_population(population, evaluator)

            self.output_manager.log_info(
                "mcts_evolution_integration", "run_evolution",
                f"run_evolution, best_fitness: {self.best_fitness}"
            )

            evolution_result = self.evolution.run_evolution(
                max_iterations=self.params.MAX_INNER_ITERATIONS,
                evaluator=evaluator,
                time_limit=time_limit,
                convergence_threshold=self.params.MAX_STAGNATION_COUNT,
                outer_iteration=i+1
            )
            self._feedback_all_evaluated_to_mcts()

            self.evolution_count += 1

            previous_best = self.best_fitness
            self._update_best_fitness()

            if self.best_fitness > previous_best:
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            iteration_time = time.time() - iteration_start_time

            iteration_data = {
                "outer_iteration": self.iteration,
                "global_best_fitness": self.best_fitness,
                "best_strategy_id": self.best_strategy_id,
                "mcts_triggered": should_run_mcts,
                "stagnation_count": self.stagnation_count,
                "stagnation_threshold": self.params.MAX_STAGNATION_COUNT,
                "mcts_count": self.mcts_count,
                "evolution_count": self.evolution_count,
                "iteration_time": iteration_time,
                "timestamp": datetime.now().isoformat(),
                "feedback_strategies_count": len(self.feedback_strategy_ids),
                "best_code_provided": best_code_snippet is not None
            }

            self.output_manager.save_iteration_stats(iteration_data)

            self.output_manager.log_info(
                "mcts_evolution_integration", "iteration_end",
                f"finished {self.iteration} iteration, time: {iteration_time:.2f}"
            )

            if self.stagnation_count >= self.params.MAX_STAGNATION_COUNT:
                self.output_manager.log_info(
                    "mcts_evolution_integration", "convergence",
                    f"stagnation_count: {self.stagnation_count}/{self.params.MAX_STAGNATION_COUNT}"
                )
                break

        total_time = time.time() - start_time

        best_strategy = None
        if self.best_strategy_id is not None:
            strategy_manager = self.evolution.strategy_manager
            if self.best_strategy_id in strategy_manager.strategies:
                best_strategy = strategy_manager.strategies[self.best_strategy_id].to_dict(
                )

        result = {
            "best_strategy": best_strategy,
            "best_fitness": self.best_fitness,
            "iterations": self.iteration,
            "mcts_count": self.mcts_count,
            "evolution_count": self.evolution_count,
            "total_time": total_time,
            "converged": self.stagnation_count >= self.params.MAX_STAGNATION_COUNT,
            "feedback_strategies_count": len(self.feedback_strategy_ids)
        }

        self.output_manager.log_info(
            "mcts_evolution_integration", "optimization_end",
            f"best_fitness: {self.best_fitness}，iteration: {self.iteration}, Time: {total_time:.2f}"
        )

        self.output_manager.save_optimization_result(result)

        return result

    def _feedback_all_evaluated_to_mcts(self) -> None:

        strategy_manager = self.evolution.strategy_manager

        feedback_count = 0
        for strategy_id, strategy in strategy_manager.strategies.items():
            if strategy.evaluated and strategy_id not in self.feedback_strategy_ids:
                constraint_order = strategy.constraint_order
                relaxation_factors = strategy.relaxation_factors
                fitness = strategy.fitness

                self.mcts.update_statistics(
                    constraint_order, relaxation_factors, fitness)

                self.feedback_strategy_ids.add(strategy_id)
                feedback_count += 1

        if feedback_count > 0:
            self.output_manager.log_info(
                "mcts_evolution_integration", "feedback_to_mcts",
                f"feedback to MCTS {feedback_count} strategies"
            )

    def _inject_mcts_strategy(self, mcts_result: Dict, outer_iteration=0) -> None:

        best_paths = mcts_result.get("best_paths", [])
        if not best_paths:
            self.output_manager.log_warning(
                "mcts_evolution_integration", "mcts_no_strategy",
                "MCTS don't find any strategy to inject into evolution algorithm"
            )
            return

        exploration_hints = []

        for path_info in best_paths:
            constraint_order = path_info.get("constraint_order", [])
            relaxation_factors = path_info.get("relaxation_factors", {})

            if not constraint_order and "path" in path_info:
                path = path_info["path"]
                if isinstance(path, list):
                    if len(path) >= 2 and isinstance(path[0], list) and isinstance(path[1], dict):
                        constraint_order = path[0]
                        relaxation_factors = path[1]
                    else:
                        idx = 1 if path and path[0] == "root" else 0
                        new_constraint_order = []
                        new_relaxation_factors = {}
                        while idx < len(path) - 1:
                            if isinstance(path[idx], str) and path[idx] != "root":
                                constraint = path[idx]
                                relaxation = path[idx + 1] if idx + \
                                    1 < len(path) else 0.5
                                new_constraint_order.append(constraint)
                                new_relaxation_factors[constraint] = relaxation
                            idx += 2
                        constraint_order = new_constraint_order
                        relaxation_factors = new_relaxation_factors

            if not constraint_order and "nodes" in path_info:
                nodes = path_info.get("nodes", [])
                new_constraint_order = []
                new_relaxation_factors = {}

                for node in nodes:
                    if "constraint" in node:
                        constraint = node.get("constraint")
                        new_constraint_order.append(constraint)
                        if "relaxation" in node:
                            new_relaxation_factors[constraint] = node.get(
                                "relaxation")

                if new_constraint_order:
                    constraint_order = new_constraint_order
                    relaxation_factors = new_relaxation_factors

            if constraint_order:
                hint = {
                    "path_type": path_info.get("type", "exploitation"),
                    "value": path_info.get("average_reward", path_info.get("value", 0)),
                    "constraint_order": constraint_order,
                    "relaxation_factors": relaxation_factors
                }
                exploration_hints.append(hint)

        mcts_statistics = mcts_result.get('statistics', {})

        if exploration_hints:
            self.problem_info["mcts_exploration_hints"] = exploration_hints
            self.problem_info["mcts_statistics"] = {
                "constraint_frequency": mcts_statistics.get("constraint_frequency", {}),
                "relaxation_stats": mcts_statistics.get("relaxation_stats", {}),
                "total_nodes": mcts_statistics.get("total_nodes", 0),
                "max_depth": mcts_statistics.get("max_depth", 0),
                "best_reward": mcts_result.get("best_reward", 0)
            }

            if hasattr(self.evolution, 'problem_info'):
                self.evolution.problem_info["mcts_exploration_hints"] = exploration_hints
                self.evolution.problem_info["mcts_statistics"] = self.problem_info["mcts_statistics"]

                self.output_manager.log_info(
                    "mcts_evolution_integration", "mcts_info_updated",
                    f"updated MCTS exploration hints to evolution algorithm, hints count: {len(exploration_hints)}"
                )

    def _evaluate_population(self, population: List[str], evaluator, outer_iteration: int = 0) -> None:

        self.output_manager.log_info(
            "mcts_evolution_integration", "evaluate_population",
            f"population: {len(population)}, outer_iteration: {outer_iteration}"
        )

        for strategy_id in population:
            evaluation_result = evaluator(strategy_id)

            if isinstance(evaluation_result, dict):
                fitness = evaluation_result.get('fitness', float('-inf'))
            else:
                fitness = evaluation_result

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_strategy_id = strategy_id

                self.output_manager.log_info(
                    "mcts_evolution_integration", "new_best_strategy",
                    f"New best strategy, ID: {strategy_id}, fitness: {fitness}, outer_iteration: {outer_iteration}"
                )

    def _update_best_fitness(self) -> None:

        strategy_manager = self.evolution.strategy_manager

        for strategy_id, strategy in strategy_manager.strategies.items():
            if strategy.evaluated and strategy.fitness > self.best_fitness:
                self.best_fitness = strategy.fitness
                self.best_strategy_id = strategy_id

                self.output_manager.log_info(
                    "mcts_evolution_integration", "update_best",
                    f"updated global best strategy, ID: {strategy_id}, fitness: {strategy.fitness}"
                )

    def _should_run_mcts(self) -> bool:

        if self.iteration == 1:

            return False

        if self.stagnation_count >= self.params.MCTS_TRIGGER_STAGNATION:
            self.output_manager.log_info(
                "mcts_evolution_integration", "mcts_stagnation_trigger",
                f"stagnation_count: {self.stagnation_count}/{self.params.MCTS_TRIGGER_STAGNATION}, run MCTS exploration"
            )
            return True

        if self.iteration - self.last_mcts_iteration >= self.params.MCTS_INTERVAL:
            self.output_manager.log_info(
                "mcts_evolution_integration", "mcts_interval_trigger",
                f"interval: {self.iteration - self.last_mcts_iteration}/{self.params.MCTS_INTERVAL}, run MCTS exploration"
            )
            return True

        return False

    def _get_evaluation_result(self, strategy_id: str):

        try:
            if hasattr(self, 'framework_controller') and hasattr(self.framework_controller, 'strategy_evaluator'):
                return self.framework_controller.strategy_evaluator.evaluate_strategy(strategy_id)

            elif hasattr(self.evolution, 'evaluator'):
                return self.evolution.evaluator(strategy_id)

            return float('-inf')

        except Exception as e:
            self.output_manager.log_error(
                "mcts_evolution_integration", "evaluation_error",
                f"Error occurred while evaluating strategy {strategy_id}: {str(e)}"
            )
            return float('-inf')

    def _evaluate_mcts_path(self, constraint_order: List[str], relaxation_factors: Dict[str, float]) -> float:

        try:
            path_key = str((tuple(constraint_order), tuple(
                sorted(relaxation_factors.items()))))

            if path_key in self.evaluated_paths:
                return self.evaluated_paths[path_key]

            path_info = {
                "type": "simulation",
                "constraint_order": constraint_order,
                "relaxation_factors": relaxation_factors
            }

            mcts_info = {
                "best_paths": [path_info],
                "statistics": {
                    "constraint_frequency": {constraint: 1.0 for constraint in constraint_order},
                    "relaxation_stats": {c: {"mean": relaxation_factors.get(c, 0.5), "std": 0.0} for c in constraint_order},
                    "total_nodes": 1,
                    "max_depth": len(constraint_order)
                }
            }

            strategy_text = self._evaluate_mcts_path_with_custom_prompt(
                constraint_order, relaxation_factors)

            problem_info_summary = {}
            if hasattr(self, 'problem_info'):
                problem_info_summary = {
                    "description": self.problem_info.get("description", "Optimization Problem"),
                    "hard_constraints": self.problem_info.get("hard_constraints", []),
                    "soft_constraints": self.problem_info.get("soft_constraints", [])
                }

            strategy_id = str(uuid.uuid4())

            from models import TextStrategy
            strategy = TextStrategy(
                id=strategy_id,
                text=strategy_text,
                constraint_order=constraint_order,
                relaxation_factors=relaxation_factors,
                algorithm_design="mcts_based_strategy",
                method="mcts_evaluation",
                generation=0,
                outer_iteration=self.iteration
            )

            self.evolution.strategy_manager.strategies[strategy_id] = strategy

            llm_client = self.evolution.strategy_manager.llm_client

            if hasattr(llm_client, 'generate_strategy_from_mcts'):
                code_text = llm_client.generate_strategy_from_mcts(
                    mcts_info=mcts_info,
                    problem_info=problem_info_summary,
                    outer_iteration=self.iteration,
                    use_llm=True
                )

                strategy.code_snippet = code_text

                if hasattr(self.evolution, 'code_generator'):
                    code_generator = self.evolution.code_generator

                    if hasattr(code_generator, 'algorithm_loader'):
                        algorithm_template = code_generator.algorithm_loader.get_template(
                            'PointSelectionAlgorithm')

                    if algorithm_template and hasattr(code_generator, '_integrate_code_to_llm_area'):
                        integrated_code = code_generator._integrate_code_to_llm_area(
                            algorithm_template, code_text)

                        temp_dir = os.path.join(
                            self.output_manager.base_dir, "temp")
                        os.makedirs(temp_dir, exist_ok=True)
                        code_file = os.path.join(
                            temp_dir, f"mcts_code_{strategy_id}.py")

                        with open(code_file, 'w', encoding='utf-8') as f:
                            f.write(integrated_code)

                        strategy.metadata["code_file"] = code_file

            if not hasattr(self, 'problem_info'):
                self.problem_info = {}
            if "strategies" not in self.problem_info:
                self.problem_info["strategies"] = {}

            self.problem_info["strategies"][strategy_id] = strategy.to_dict()

            if hasattr(self.evolution, 'code_generator'):
                if not hasattr(self.evolution.code_generator, 'problem_info'):
                    self.evolution.code_generator.problem_info = {}
                if "strategies" not in self.evolution.code_generator.problem_info:
                    self.evolution.code_generator.problem_info["strategies"] = {
                    }
                self.evolution.code_generator.problem_info["strategies"][strategy_id] = strategy.to_dict(
                )

            fitness = float('-inf')

            if hasattr(self, 'framework_controller') and hasattr(self.framework_controller, 'strategy_evaluator'):
                evaluation_result = self.framework_controller.strategy_evaluator.evaluate_strategy(
                    strategy_id)
                if isinstance(evaluation_result, dict):
                    fitness = evaluation_result.get('fitness', float('-inf'))
                else:
                    fitness = evaluation_result
            elif hasattr(self.evolution, 'evaluator'):
                evaluation_result = self.evolution.evaluator(strategy_id)
                if isinstance(evaluation_result, dict):
                    fitness = evaluation_result.get('fitness', float('-inf'))
                else:
                    fitness = evaluation_result
            else:
                self.output_manager.log_error(
                    "mcts_evolution_integration", "evaluator_not_found",
                    "evaluator_not_found"
                )

            self.evaluated_paths[path_key] = fitness

            return fitness
        except Exception as e:
            self.output_manager.log_error(
                "mcts_evolution_integration", "mcts_path_evaluation_error",
                f"Wrong: {str(e)}"
            )
            return float('-inf')

    def _evaluate_mcts_path_with_custom_prompt(self, constraint_order: List[str], relaxation_factors: Dict[str, float]) -> str:

        import json

        path_info = {
            "type": "simulation",
            "constraint_order": constraint_order,
            "relaxation_factors": relaxation_factors
        }

        mcts_info = {
            "best_paths": [path_info],
            "statistics": {
                "constraint_frequency": {constraint: 1.0 for constraint in constraint_order},
                "relaxation_stats": {c: {"mean": relaxation_factors.get(c, 0.5), "std": 0.0} for c in constraint_order},
                "total_nodes": 1,
                "max_depth": len(constraint_order)
            }
        }

        problem_info_summary = {}
        if hasattr(self, 'problem_info'):
            problem_info_summary = {
                "description": self.problem_info.get("description", "Optimization Problem"),
                "hard_constraints": self.problem_info.get("hard_constraints", []),
                "soft_constraints": self.problem_info.get("soft_constraints", [])
            }

        strategy_text = f"MCTS path\n\n"
        strategy_text += "Constraint Processing Order and Relaxation Factors:\n"

        for i, constraint in enumerate(constraint_order):
            factor = relaxation_factors.get(constraint, 1.0)
            strategy_text += f"{i+1}. {constraint}: {factor}\n"

        strategy_text += "\nAlgorithm Design:\n"
        strategy_text += "problem feature\n"

        strategy_text += "- constraint_frequency: " + json.dumps(mcts_info["statistics"]["constraint_frequency"],
                                                                 indent=2, ensure_ascii=False) + "\n"

        strategy_text += "- relaxation_stats: " + json.dumps(mcts_info["statistics"]["relaxation_stats"],
                                                             indent=2, ensure_ascii=False) + "\n"

        strategy_text += "\nProblem:\n"
        strategy_text += problem_info_summary.get(
            "description", "Optimization Problem")

        return strategy_text

    def _run_mcts_search(self, evaluator, time_limit: float) -> Dict:

        self.output_manager.log_info(
            "mcts_evolution_integration", "mcts_search_start",
            f"StarMCTS, time_limit: {time_limit},SIMULATIONS: {self.params.MCTS_SIMULATIONS}"
        )
        simulation_count = max(1, self.params.MCTS_SIMULATIONS)

        if not isinstance(self.problem_info, dict):
            self.problem_info = {}

        if "strategies" not in self.problem_info:
            self.problem_info["strategies"] = {}

        def mcts_evaluator(constraint_order: List[str], relaxation_factors: Dict[str, float]) -> float:

            try:
                path_key = str((tuple(constraint_order), tuple(
                    sorted(relaxation_factors.items()))))

                if hasattr(self, 'evaluated_paths') and path_key in self.evaluated_paths:
                    return self.evaluated_paths[path_key]

                fitness = self._evaluate_mcts_path(
                    constraint_order, relaxation_factors)

                if not hasattr(self, 'evaluated_paths'):
                    self.evaluated_paths = {}
                self.evaluated_paths[path_key] = fitness

                return fitness
            except Exception as e:
                self.output_manager.log_error(
                    "mcts_evolution_integration", "mcts_evaluation_error",
                    f"mcts_evaluation_errorg: {str(e)}"
                )
                return float('-inf')
        mcts_result = self.mcts.run_search(
            simulation_count=self.params.MCTS_SIMULATIONS,
            evaluator=mcts_evaluator,
            time_limit=min(time_limit, self.params.MCTS_TIME_LIMIT)
        )
        best_paths = self.mcts.get_best_paths(top_n=5, exploration_weight=1.0)
        mcts_result["best_paths"] = best_paths

        return mcts_result

    def _log_population_iteration_distribution(self):

        if not self.evolution.population:
            return

        iteration_counts = {}
        strategy_manager = self.evolution.strategy_manager

        for strategy_id in self.evolution.population:
            if strategy_id in strategy_manager.strategies:
                strategy = strategy_manager.strategies[strategy_id]
                outer_iteration = getattr(strategy, 'outer_iteration', 0)
                if outer_iteration in iteration_counts:
                    iteration_counts[outer_iteration] += 1
                else:
                    iteration_counts[outer_iteration] = 1

        distribution_text = ", ".join([f"Iteration {iter_num}: {count} individuals"
                                       for iter_num, count in sorted(iteration_counts.items())])
