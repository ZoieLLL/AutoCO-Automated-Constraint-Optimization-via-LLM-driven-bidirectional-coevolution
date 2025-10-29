from typing import Dict, List, Any, Optional, Tuple
import time
import json
import os
import re
from datetime import datetime
from models import OptimizationParams, OptimizationRun
from output_manager import OutputManager
from mcts_evolution_integration import MCTSEvolutionIntegration
from code_generator import CodeGenerator
from solution_runner import SolutionRunner
from strategy_evaluator import StrategyEvaluator


class FrameworkController:

    def __init__(self, mcts_evolution: MCTSEvolutionIntegration,
                 code_generator: CodeGenerator,
                 solution_runner: SolutionRunner,
                 strategy_evaluator: StrategyEvaluator,
                 optimization_params: OptimizationParams,
                 optimization_run: OptimizationRun,
                 output_manager: OutputManager,
                 repair_config: Dict = None):

        self.mcts_evolution = mcts_evolution
        self.code_generator = code_generator
        self.solution_runner = solution_runner
        self.strategy_evaluator = strategy_evaluator
        self.params = optimization_params
        self.optimization_run = optimization_run
        self.output_manager = output_manager
        self.repair_config = repair_config or {}

        self.start_time = None
        self.total_time_limit = None
        self.best_strategy_id = None
        self.best_fitness = float('-inf')
        self.best_solution = None

        self.mcts_evolution.strategy_evaluator = strategy_evaluator
        self.mcts_evolution.framework_controller = self

        if hasattr(self.strategy_evaluator, 'set_repair_config'):
            self.strategy_evaluator.set_repair_config(self.repair_config)

    def run_optimization(self, time_limit: float = None) -> Dict:

        self.total_time_limit = time_limit or self.params.time_limit

        checkpoint_loaded = self.load_checkpoint(os.path.join(
            self.output_manager.base_dir, "checkpoint.json"))
        if not checkpoint_loaded:
            self.start_time = time.time()
        evaluator = self._create_strategy_evaluator()

        try:

            checkpoint_interval = 100
            last_checkpoint_time = time.time()

            integration_result = self.mcts_evolution.run_integrated_optimization(
                evaluator=evaluator,
                time_limit=self.total_time_limit
            )

            self.best_strategy_id = self.mcts_evolution.best_strategy_id
            self.best_fitness = self.mcts_evolution.best_fitness

            self.best_solution, _, _ = self.strategy_evaluator.get_best_solution()

            self.optimization_run.update_best(
                self.best_strategy_id,
                "best_solution",
                self.best_fitness
            )

            self.optimization_run.mark_completed()

            total_execution_time = time.time() - self.start_time

            result = {
                "best_strategy_id": self.best_strategy_id,
                "best_fitness": self.best_fitness,
                "best_solution": self.best_solution,
                "total_execution_time": total_execution_time,
                "iterations_completed": integration_result.get("iterations", 0),
                "mcts_count": integration_result.get("mcts_count", 0),
                "evolution_count": integration_result.get("evolution_count", 0),
                "converged": integration_result.get("converged", False)
            }

            self.output_manager.log_info(
                "framework_controller", "optimization_complete",
                f"Optimization complete, best fitness: {self.best_fitness}, total time: {total_execution_time:.2f} seconds"
            )

            self._save_final_result(result)

            return result

        except KeyboardInterrupt:
            self.save_checkpoint()

            return {
                "interrupted": True,
                "best_strategy_id": self.best_strategy_id,
                "best_fitness": self.best_fitness,
                "best_solution": self.best_solution,
                "total_execution_time": time.time() - self.start_time
            }
        except Exception as e:
            self.output_manager.log_error(
                "framework_controller", "optimization_error",
                f"Error during optimization: {str(e)}"
            )

            try:
                self.save_checkpoint()
            except:
                pass

            return {
                "error": str(e),
                "best_strategy_id": self.best_strategy_id,
                "best_fitness": self.best_fitness,
                "best_solution": self.best_solution,
                "total_execution_time": time.time() - self.start_time
            }

    def _create_strategy_evaluator(self):

        use_multi_dataset = getattr(self.params, 'use_multi_dataset', False)
        datasets = getattr(self.params, 'datasets', None)
        dataset_ids = datasets.split(',') if datasets else None

        def evaluate_strategy(strategy_id: str,
                              force_regenerate: bool = False,
                              variant_count: int = 1) -> float:

            evaluation_result = execute_original_evaluation(
                strategy_id, force_regenerate, variant_count)

            return evaluation_result.get('fitness', float('-inf'))

        def execute_original_evaluation(strategy_id: str,
                                        force_regenerate: bool = False,
                                        variant_count: int = 1) -> Dict:

            remaining_time = self.total_time_limit - \
                (time.time() - self.start_time)
            if remaining_time <= 0:
                self.output_manager.log_warning(
                    "framework_controller", "time_limit_reached",
                    "Time limit reached during strategy evaluation, returning default fitness"
                )
                return {'fitness': float('-inf')}

            try:
                if use_multi_dataset:
                    evaluation_result = self.strategy_evaluator.evaluate_strategy_on_multiple_datasets(
                        strategy_id=strategy_id,
                        dataset_ids=dataset_ids,
                        force_regenerate=force_regenerate,
                        variant_count=variant_count or self.params.CODE_VARIANTS
                    )
                else:
                    evaluation_result = self.strategy_evaluator.evaluate_strategy(
                        strategy_id=strategy_id,
                        force_regenerate=force_regenerate,
                        variant_count=variant_count or self.params.CODE_VARIANTS
                    )

                if isinstance(evaluation_result, (int, float)):

                    fitness = evaluation_result
                    evaluation_dict = {
                        'fitness': fitness,
                        'strategy_id': strategy_id,
                        'profit_objective': fitness
                    }

                    if (hasattr(self.strategy_evaluator, 'evaluation_cache') and
                            strategy_id in self.strategy_evaluator.evaluation_cache):
                        cached_result = self.strategy_evaluator.evaluation_cache[strategy_id]
                        if isinstance(cached_result, dict):
                            evaluation_dict.update(cached_result)

                    evaluation_result = evaluation_dict
                elif isinstance(evaluation_result, dict):

                    if 'fitness' not in evaluation_result:
                        evaluation_result['fitness'] = evaluation_result.get(
                            'profit_objective', float('-inf'))
                else:

                    evaluation_result = {'fitness': float('-inf')}

                fitness = evaluation_result.get('fitness', float('-inf'))
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_strategy_id = strategy_id

                    if 'solution' in evaluation_result:
                        self.best_solution = evaluation_result['solution']
                    elif (hasattr(self.strategy_evaluator, 'evaluation_cache') and
                          strategy_id in self.strategy_evaluator.evaluation_cache):
                        cached_result = self.strategy_evaluator.evaluation_cache[strategy_id]
                        if isinstance(cached_result, dict) and 'solution' in cached_result:
                            self.best_solution = cached_result['solution']

                return evaluation_result

            except Exception as e:
                self.output_manager.log_error(
                    "framework_controller", "evaluation_error",
                    f"Error evaluating strategy {strategy_id}: {str(e)}"
                )
                return {'fitness': float('-inf'), 'error': str(e)}

        return evaluate_strategy

    def _save_final_result(self, result: Dict) -> None:

        final_result = {
            "optimization_run_id": self.optimization_run.id,
            "timestamp": datetime.now().isoformat(),
            "best_strategy": result.get("best_strategy_id"),
            "best_fitness": result.get("best_fitness"),
            "execution_stats": {
                "total_time": result.get("total_execution_time", 0),
                "iterations": result.get("iterations_completed", 0),
                "mcts_count": result.get("mcts_count", 0),
                "evolution_count": result.get("evolution_count", 0),
                "converged": result.get("converged", False)
            }
        }

        if result.get("best_solution"):

            solution_copy = {k: v for k, v in result.get("best_solution", {}).items()
                             if isinstance(v, (dict, list, str, int, float, bool, type(None)))}
            final_result["best_solution"] = solution_copy

        self.output_manager.save_final_result(final_result)

        print("\n" + "="*80)
        print(f"Optimization complete!")
        print(f"Best fitness: {result.get('best_fitness')}")
        print("="*80)

    def save_checkpoint(self):

        checkpoint_data = {
            "start_time": self.start_time,
            "best_strategy_id": self.best_strategy_id,
            "best_fitness": self.best_fitness,
            "mcts_evolution_state": {
                "iteration": self.mcts_evolution.iteration,
                "last_mcts_iteration": self.mcts_evolution.last_mcts_iteration,
                "mcts_count": self.mcts_evolution.mcts_count,
                "evolution_count": self.mcts_evolution.evolution_count,
                "stagnation_count": self.mcts_evolution.stagnation_count,
                "evaluated_paths": list(self.mcts_evolution.evaluated_paths.keys()),
                "feedback_strategy_ids": list(self.mcts_evolution.feedback_strategy_ids)
            },
            "optimization_run": self.optimization_run.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

        checkpoint_file = os.path.join(
            self.output_manager.base_dir, "checkpoint.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_checkpoint(self, checkpoint_file):

        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            if 'start_time' in checkpoint_data:
                start_time = checkpoint_data['start_time']
                if isinstance(start_time, str):

                    try:
                        dt = datetime.fromisoformat(start_time)
                        self.start_time = dt.timestamp()
                    except Exception as e:
                        self.output_manager.log_warning(
                            "framework_controller", "start_time_parse_error",
                            f"Failed to convert ISO format start_time to timestamp: {str(e)}, using current timestamp instead"
                        )
                        self.start_time = time.time()
                elif isinstance(start_time, (int, float)):

                    self.start_time = start_time
                else:
                    self.output_manager.log_warning(
                        "framework_controller", "start_time_unknown_type",
                        f"Unknown start_time type: {type(start_time)}, using current timestamp instead"
                    )
                    self.start_time = time.time()
            else:
                self.start_time = time.time()

            self.best_strategy_id = checkpoint_data.get("best_strategy_id")
            self.best_fitness = checkpoint_data.get(
                "best_fitness", float('-inf'))

            mcts_evolution_state = checkpoint_data.get(
                "mcts_evolution_state", {})
            self.mcts_evolution.iteration = mcts_evolution_state.get(
                "iteration", 0)
            self.mcts_evolution.last_mcts_iteration = mcts_evolution_state.get(
                "last_mcts_iteration", 0)
            self.mcts_evolution.mcts_count = mcts_evolution_state.get(
                "mcts_count", 0)
            self.mcts_evolution.evolution_count = mcts_evolution_state.get(
                "evolution_count", 0)
            self.mcts_evolution.stagnation_count = mcts_evolution_state.get(
                "stagnation_count", 0)

            optimization_run_data = checkpoint_data.get("optimization_run", {})
            for key, value in optimization_run_data.items():
                if key != "id" and hasattr(self.optimization_run, key):
                    setattr(self.optimization_run, key, value)

            self.output_manager.log_info(
                "framework_controller", "checkpoint_loaded",
                f"Resumed from checkpoint, continuing from iteration {self.mcts_evolution.iteration}"
            )
            return True
        except Exception as e:
            self.output_manager.log_error(
                "framework_controller", "checkpoint_load_error",
                f"Error loading checkpoint: {str(e)}"
            )
            return False
