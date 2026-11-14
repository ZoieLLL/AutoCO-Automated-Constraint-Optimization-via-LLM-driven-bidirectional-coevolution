from typing import Dict, List, Tuple
import random
import time
from autoco.models import OptimizationParams
from autoco.output_manager import OutputManager
from autoco.strategies.text_strategy import TextStrategyManager


class EvolutionaryAlgorithm:
    def __init__(self, strategy_manager: TextStrategyManager,
                 problem_info: Dict,
                 params: OptimizationParams,
                 output_manager: OutputManager):
        self.strategy_manager = strategy_manager
        self.problem_info = problem_info
        self.params = params
        self.output_manager = output_manager
        self.population: List[str] = []
        self.generation = 0
        self.stagnation_count = 0
        self.best_fitness = float('-inf')
        self.best_strategy_id = None

    def initialize_population(self, population_size: int = None, outer_iteration=0, best_code_snippet=None) -> List[str]:
        size = population_size or self.params.POPULATION_SIZE

        self.population = self.strategy_manager.generate_initial_strategies(
            size,
            outer_iteration=outer_iteration,
            best_code_snippet=best_code_snippet
        )

    
        self.generation = 0
        self.stagnation_count = 0
        self.best_fitness = float('-inf')
        self.best_strategy_id = None

        return self.population

    def evolve_one_generation(self, evaluator=None, outer_iteration=0) -> Tuple[List[str], bool]:
        start_time = time.time()

        previous_best = self.best_fitness

    
        if evaluator:
            for strategy_id in self.population:
                if strategy_id in self.strategy_manager.strategies and not self.strategy_manager.strategies[strategy_id].evaluated:
                    evaluation_result = evaluator(strategy_id)

                    
                    if isinstance(evaluation_result, dict):
                        fitness = evaluation_result.get(
                            'fitness', float('-inf'))
                        solution_data = evaluation_result.get(
                            'best_solution', {})
                    else:
                        fitness = evaluation_result
                        solution_data = {}

                    self.strategy_manager.update_strategy_fitness(
                        strategy_id, fitness, solution_data)

                    
                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_strategy_id = strategy_id

    
        offspring = self._generate_offspring(outer_iteration=outer_iteration)

    
        if evaluator:
            for strategy_id in offspring:
                if strategy_id in self.strategy_manager.strategies and not self.strategy_manager.strategies[strategy_id].evaluated:
                    evaluation_result = evaluator(strategy_id)

                    
                    if isinstance(evaluation_result, dict):
                        fitness = evaluation_result.get(
                            'fitness', float('-inf'))
                        solution_data = evaluation_result.get(
                            'best_solution', {})
                    else:
                        fitness = evaluation_result
                        solution_data = {}

                    self.strategy_manager.update_strategy_fitness(
                        strategy_id, fitness, solution_data)

                    
                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_strategy_id = strategy_id

    
        all_candidates = self.population + offspring
        next_population = self.strategy_manager.select_next_generation(
            all_candidates,
            self.params.ELITE_COUNT,
            self.params.POPULATION_SIZE
        )

        self.population = next_population

    
        improved = False
        if self.best_fitness > previous_best:
            improved = True
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        self.generation += 1

        return next_population, improved

    def _generate_offspring(self, outer_iteration=0) -> List[str]:
    
        offspring = []

    
        offspring_count = self.params.OFFSPRING_COUNT


        for i in range(offspring_count):
    
            strategy_type = None

    
            if random.random() < self.params.CROSSOVER_RATE: 
    
                parents = self.strategy_manager.select_parents(
                    self.population,
                    self.params.TOURNAMENT_SIZE
                )

                try:
    
                    new_strategy_id = self.strategy_manager.crossover_strategies(
                        parents, outer_iteration=outer_iteration)
                    offspring.append(new_strategy_id)

    
                    if new_strategy_id in self.strategy_manager.strategies:
                        strategy = self.strategy_manager.strategies[new_strategy_id]
                        if strategy.method.startswith("crossover_"):
                            strategy_type = strategy.method
                except Exception as e:
                    self.output_manager.log_error(
                        "evolutionary_algorithm", "crossover_error",
                        f"Crossover operation failed: {str(e)}"
                    )
    
                    offspring.append(parents[0])
            else: 

                evaluated_strategies = [sid for sid in self.population if
                                        sid in self.strategy_manager.strategies and
                                        self.strategy_manager.strategies[sid].evaluated]

    
                if not evaluated_strategies:
                    self.output_manager.log_warning(
                        "evolutionary_algorithm", "mutation_skip",
                        "No evaluated strategy available for mutation, skipping this mutation"
                    )
                    continue


                strategy_to_mutate = self._select_strategy_for_mutation(
                    evaluated_strategies)

                try:

                    new_strategy_id = self.strategy_manager.mutate_strategy(
                        strategy_to_mutate,
                        self.params.MUTATION_STRENGTH,
                        outer_iteration=outer_iteration
                    )
                    offspring.append(new_strategy_id)


                    if new_strategy_id in self.strategy_manager.strategies:
                        strategy = self.strategy_manager.strategies[new_strategy_id]
                        if strategy.method.startswith("mutated_"):
                            strategy_type = strategy.method

                except Exception as e:
                    self.output_manager.log_error(
                        "evolutionary_algorithm", "mutation_error",
                        f"Mutation operation failed: {str(e)}"
                    )


        self.strategy_manager.prune_strategies(100)

        return offspring

    def run_evolution(self, max_iterations=None, evaluator=None, time_limit=600.0, convergence_threshold=None, outer_iteration=0):


        if max_iterations is None:
            max_iter = self.params.MAX_INNER_ITERATIONS
        else:

            if callable(max_iterations):
                max_iter = self.params.MAX_INNER_ITERATIONS
            else:
                try:
                    max_iter = int(max_iterations)
                except (ValueError, TypeError):
                    max_iter = self.params.MAX_INNER_ITERATIONS


        if convergence_threshold is None:
            conv_threshold = self.params.MAX_STAGNATION_COUNT
        else:

            if callable(convergence_threshold):
                conv_threshold = self.params.MAX_STAGNATION_COUNT
            else:
                try:
                    conv_threshold = int(convergence_threshold)
                except (ValueError, TypeError):
                    conv_threshold = self.params.MAX_STAGNATION_COUNT


        if not self.population:
            self.initialize_population()


        if evaluator:
            initial_evaluation_count = 0
            for strategy_id in self.population:
                if strategy_id in self.strategy_manager.strategies and not self.strategy_manager.strategies[strategy_id].evaluated:
                    try:
                        evaluation_result = evaluator(strategy_id)


                        if isinstance(evaluation_result, dict):
                            fitness = evaluation_result.get(
                                'fitness', float('-inf'))
                            solution_data = evaluation_result.get(
                                'best_solution', {})
                        else:
                            fitness = evaluation_result
                            solution_data = {}

                        self.strategy_manager.update_strategy_fitness(
                            strategy_id, fitness, solution_data)
                        initial_evaluation_count += 1


                        if fitness > self.best_fitness:
                            self.best_fitness = fitness
                            self.best_strategy_id = strategy_id
                    except Exception as e:
                        self.output_manager.log_error(
                            "evolutionary_algorithm", "initial_evaluation_error",
                            f"Initial population evaluation failed, strategy ID: {strategy_id}, error: {str(e)}"
                        )

                        continue




        start_time = time.time()

        for i in range(max_iter):

            if time.time() - start_time > time_limit:

                break


            _, improved = self.evolve_one_generation(
                evaluator, outer_iteration=outer_iteration)


            if self.stagnation_count >= conv_threshold:

                break

        total_time = time.time() - start_time
        completed_iterations = i + 1


        best_strategy = None
        if self.best_strategy_id:
            best_strategy = self.strategy_manager.strategies[self.best_strategy_id].to_dict(
            )

        result = {
            "best_strategy": best_strategy,
            "best_fitness": self.best_fitness,
            "iterations": completed_iterations,
            "total_time": total_time,
            "converged": self.stagnation_count >= conv_threshold,
            "generation": self.generation,
            "population_size": len(self.population)
        }



        return result

    def _select_strategy_for_mutation(self, evaluated_strategies: List[str]) -> str:

        if not evaluated_strategies:
            raise ValueError("No options for mutation")

        fitness_values = []
        valid_strategies = []

        for sid in evaluated_strategies:
            fitness = self.strategy_manager.strategies[sid].fitness

            if not isinstance(fitness, (int, float)) or fitness is None:
                fitness = float('-inf')

            fitness_values.append(fitness)
            valid_strategies.append(sid)

        valid_fitness_strategies = []
        inf_fitness_strategies = []

        for i, (sid, fitness) in enumerate(zip(valid_strategies, fitness_values)):
            if fitness > float('-inf'):
                valid_fitness_strategies.append((sid, fitness))
            else:
                inf_fitness_strategies.append(sid)

        if valid_fitness_strategies:

            min_fitness = min(
                fitness for _, fitness in valid_fitness_strategies)
            if min_fitness < 0:

                offset = abs(min_fitness) + 1
                adjusted_fitness = [(sid, fitness + offset)
                                    for sid, fitness in valid_fitness_strategies]
            else:
                adjusted_fitness = valid_fitness_strategies


            total_fitness = sum(fitness for _, fitness in adjusted_fitness)

            if total_fitness > 0:

                pick = random.uniform(0, total_fitness)
                current = 0
                for sid, fitness in adjusted_fitness:
                    current += fitness
                    if current >= pick:
                        return sid

                return adjusted_fitness[0][0]
            else:
           
                return random.choice([sid for sid, _ in valid_fitness_strategies])

        if inf_fitness_strategies:
            return random.choice(inf_fitness_strategies)


        return random.choice(valid_strategies)
