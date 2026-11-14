
from typing import Dict, List
import time
import uuid
from autoco.models import OptimizationParams
from autoco.output_manager import OutputManager
from autoco.core.mcts import MCTSTree
from autoco.core.evolutionary_algorithm import EvolutionaryAlgorithm


class MCTSEvolutionIntegration:
    """
    MCTS and Evolutionary Algorithm Integration: Combines the advantages of both algorithms, enabling bidirectional information exchange
    """

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

        self.output_manager.log_info(
            "mcts_evolution_integration", "optimization_start",
            f"Starting integrated optimization, time limit: {time_limit} seconds"
        )

        start_time = time.time()
        last_checkpoint_time = start_time
        checkpoint_interval = 1800  
        for i in range(self.params.MAX_OUTER_ITERATIONS):
            if time.time() - start_time > time_limit:
                self.output_manager.log_info(
                    "mcts_evolution_integration", "time_limit_reached",
                    f"Time limit reached, completed {i}/{self.params.MAX_OUTER_ITERATIONS} outer iterations"
                )
                break

            current_time = time.time()
            if current_time - last_checkpoint_time > checkpoint_interval:
                if hasattr(self, 'framework_controller') and hasattr(self.framework_controller, 'save_checkpoint'):
                    try:
                        self.framework_controller.save_checkpoint()
                        self.output_manager.log_info(
                            "mcts_evolution_integration", "checkpoint_saved",
                            f"{self.iteration} checkpoint_saved"
                        )
                        last_checkpoint_time = current_time
                    except Exception as e:
                        self.output_manager.log_error(
                            "mcts_evolution_integration", "checkpoint_save_error",
                            f"Error saving checkpoint: {str(e)}"
                        )
            self.iteration = i + 1
            total_iters = self.params.MAX_OUTER_ITERATIONS
            if total_iters > 0:
                pct = (i + 1) / total_iters
                width = 20
                filled = int(width * pct)
                bar = '#' * filled + '-' * (width - filled)
                best = self.best_fitness
                if isinstance(best, float) and best not in (float('-inf'), float('inf')):
                    best_str = f"{best:.4f}"
                else:
                    best_str = "N/A"
                print(f"\rProgress [{bar}] {i+1}/{total_iters}  Best fitness: {best_str}", end='', flush=True)

            self.output_manager.log_info(
                "mcts_evolution_integration", "iteration_start",
                f"Starting outer iteration {self.iteration}/{self.params.MAX_OUTER_ITERATIONS}"
            )

            iteration_start_time = time.time()

            self._feedback_all_evaluated_to_mcts()

            should_run_mcts = self._should_run_mcts()

            if should_run_mcts:
                self.output_manager.log_info(
                    "mcts_evolution_integration", "run_mcts",
                    f"Running MCTS search, stagnation count: {self.stagnation_count}"
                )
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
                    self.output_manager.log_info(
                        "mcts_evolution_integration", "best_code_found",
                        f"Providing best code snippet for outer iteration {i+1}, length: {len(best_code_snippet) if best_code_snippet else 0}"
                    )

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
                f"Running evolutionary algorithm, current best fitness: {self.best_fitness}"
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
                self.output_manager.log_info(
                    "mcts_evolution_integration", "improvement",
                    f"Found improved solution, fitness: {self.best_fitness}"
                )
            else:
                self.stagnation_count += 1
                self.output_manager.log_info(
                    "mcts_evolution_integration", "stagnation",
                    f"No improvement, stagnation count: {self.stagnation_count}/{self.params.MAX_STAGNATION_COUNT}"
                )

            iteration_time = time.time() - iteration_start_time

            self.output_manager.log_info(
                "mcts_evolution_integration", "iteration_end",
                f"Completed outer iteration {self.iteration}, time taken: {iteration_time:.2f} seconds"
            )

            if self.stagnation_count >= self.params.MAX_STAGNATION_COUNT:
                self.output_manager.log_info(
                    "mcts_evolution_integration", "convergence",
                    f"Reached convergence threshold, stagnation count: {self.stagnation_count}/{self.params.MAX_STAGNATION_COUNT}"
                )
                break

        print()

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
            f"Integrated optimization completed, best fitness: {self.best_fitness}, iterations: {self.iteration}, time taken: {total_time:.2f} seconds"
        )

        self.output_manager.save_optimization_result(result)

        return result

    def _feedback_all_evaluated_to_mcts(self) -> None:
        """Feed back all evaluated strategies from the evolutionary algorithm to the MCTS tree"""
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
                f"Fed back results of {feedback_count} evaluated strategies to MCTS"
            )

    def _inject_mcts_strategy(self, mcts_result: Dict, outer_iteration=0) -> None:
        best_paths = mcts_result.get("best_paths", [])
        if not best_paths:
            self.output_manager.log_warning(
                "mcts_evolution_integration", "mcts_no_strategy",
                "MCTS didn't find valid exploration directions, skipping prompts"
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
            self.output_manager.log_info(
                "mcts_evolution_integration", "mcts_hints_provided",
                f"Provided {len(exploration_hints)} MCTS exploration hints"
            )

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
                    f"Updated MCTS exploration hints to evolutionary algorithm, hint count: {len(exploration_hints)}"
                )

    def _evaluate_population(self, population: List[str], evaluator, outer_iteration: int = 0) -> None:
        self.output_manager.log_info(
            "mcts_evolution_integration", "evaluate_population",
            f"Evaluating population, size: {len(population)}, outer iteration: {outer_iteration}"
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
                    f"Found new best strategy, ID: {strategy_id}, fitness: {fitness}, outer iteration: {outer_iteration}"
                )

    def _update_best_fitness(self) -> None:
        strategy_manager = self.evolution.strategy_manager

        for strategy_id, strategy in strategy_manager.strategies.items():
            if strategy.evaluated and strategy.fitness > self.best_fitness:
                self.best_fitness = strategy.fitness
                self.best_strategy_id = strategy_id

                self.output_manager.log_info(
                    "mcts_evolution_integration", "update_best",
                    f"Updated global best strategy, ID: {strategy_id}, fitness: {strategy.fitness}"
                )

    def _should_run_mcts(self) -> bool:
        if self.iteration == 1:
            self.output_manager.log_info(
                "mcts_evolution_integration", "mcts_skip_first",
                "First iteration, skipping MCTS for direct cold start"
            )
            return False

        if self.stagnation_count >= self.params.MCTS_TRIGGER_STAGNATION:
            self.output_manager.log_info(
                "mcts_evolution_integration", "mcts_stagnation_trigger",
                f"Stagnation count ({self.stagnation_count}) reached threshold ({self.params.MCTS_TRIGGER_STAGNATION}), running MCTS exploration"
            )
            return True

        if self.iteration - self.last_mcts_iteration >= self.params.MCTS_INTERVAL:
            self.output_manager.log_info(
                "mcts_evolution_integration", "mcts_interval_trigger",
                f"Interval reached ({self.iteration - self.last_mcts_iteration}/{self.params.MCTS_INTERVAL}), running MCTS exploration"
            )
            return True

        return False

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
                    "description": self.problem_info.get("description", "problem"),
                    "hard_constraints": self.problem_info.get("hard_constraints", []),
                    "soft_constraints": self.problem_info.get("soft_constraints", [])
                }

            strategy_id = str(uuid.uuid4())

            from autoco.models import TextStrategy
            strategy = TextStrategy(
                id=strategy_id,
                text=strategy_text,
                constraint_order=constraint_order,
                relaxation_factors=relaxation_factors,
                algorithm_design="from MCTS path evaluation",
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
                f"Mcts path evaluation error: {str(e)}"
            )
            return float('-inf')

    def _evaluate_mcts_path_with_custom_prompt(self, constraint_order: List[str], relaxation_factors: Dict[str, float]) -> str:
        """Build strategy text using MCTS information"""
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
                "description": self.problem_info.get("description", "problem"),
                "hard_constraints": self.problem_info.get("hard_constraints", []),
                "soft_constraints": self.problem_info.get("soft_constraints", [])
            }

        strategy_text = f"Strategy: MCTS Path\n\n"
        strategy_text += "Constraint Processing Order and Relaxation Factors:\n"

        for i, constraint in enumerate(constraint_order):
            factor = relaxation_factors.get(constraint, 1.0)
            strategy_text += f"{i+1}. {constraint}: {factor}\n"

        strategy_text += "\nAlgorithm Design:\n"
        strategy_text += "Strategy implemented based on MCTS search results, with the following features:\n"
        strategy_text += "- Constraint Frequency: " + json.dumps(mcts_info["statistics"]["constraint_frequency"],
                                                                 indent=2, ensure_ascii=False) + "\n"
        strategy_text += "- Relaxation Factor Statistics: " + json.dumps(mcts_info["statistics"]["relaxation_stats"],
                                                                         indent=2, ensure_ascii=False) + "\n"

        strategy_text += "\nProblem Background:\n"
        strategy_text += problem_info_summary.get(
            "description", "Optimization Problem")
        self.output_manager.log_info(
            "mcts_evolution_integration", "strategy_text_generated",
            f"Generated strategy text based on MCTS path information, constraint count: {len(constraint_order)}"
        )

        return strategy_text

    def _run_mcts_search(self, evaluator, time_limit: float) -> Dict:
        self.output_manager.log_info(
            "mcts_evolution_integration", "mcts_search_start",
            f"Starting MCTS search, time limit: {time_limit} seconds, simulation count: {self.params.MCTS_SIMULATIONS}"
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
                    f"mcts_evaluation_error: {str(e)}"
                )
                return float('-inf')

        mcts_result = self.mcts.run_search(
            simulation_count=self.params.MCTS_SIMULATIONS,
            evaluator=mcts_evaluator,
            time_limit=min(time_limit, self.params.MCTS_TIME_LIMIT)
        )
        best_paths = self.mcts.get_best_paths(top_n=5, exploration_weight=1.0)
        mcts_result["best_paths"] = best_paths

        self.output_manager.log_info(
            "mcts_evolution_integration", "mcts_search_complete",
            f"MCTS search completed, best reward: {mcts_result.get('best_reward', 'N/A')}, "
            f"found {len(best_paths)} valuable paths"
        )
        return mcts_result
