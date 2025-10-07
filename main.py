import os
import time
import argparse
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
from models import OptimizationParams, OptimizationRun
from output_manager import OutputManager
from llm_client import LLMClient, StrategyLLMClient, CodeLLMClient, ProblemAnalysisLLMClient
from problem_analyzer import ProblemAnalyzer
from mcts import MCTSTree
from evolutionary_algorithm import EvolutionaryAlgorithm
from text_strategy import TextStrategyManager
from code_generator import CodeGenerator
from solution_runner import SolutionRunner
from strategy_evaluator import StrategyEvaluator
from framework_controller import FrameworkController
from mcts_evolution_integration import MCTSEvolutionIntegration
from algorithm_loader import AlgorithmLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_problem_data(data_path: str) -> Dict:
    """
    Load problem data

    Args:
        data_path: Path to data file

    Returns:
        Dict: Problem data
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        logger.error(f"Failed to load problem data: {e}")
        raise


def save_problem_data(problem_data: Dict, file_path: str) -> None:
    """
    Save problem data to file

    Args:
        problem_data: Problem data
        file_path: File path
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(problem_data, file, indent=2, ensure_ascii=False)
        logger.info(f"Problem data saved to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save problem data: {e}")
        raise


def load_config(config_path="config.json"):
    """Load configuration file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        logger.warning(
            f"Configuration file {config_path} does not exist, will use command line args or environment variables")
        return {}
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return {}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Optimization Framework')

    parser.add_argument('--data_path', type=str, default='problem_data.json',
                        help='Problem data file path')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--config_path', type=str, default='config.json',
                        help='Configuration file path')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Default API key (can be overridden by component-specific config)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='Default API base URL')
    parser.add_argument('--model', type=str, default=None,
                        help='Default LLM model name')
    parser.add_argument('--max_outer_iterations', '--outer-iterations', type=int, default=8,
                        help='Maximum outer iterations')
    parser.add_argument('--max_inner_iterations', '--generations', type=int, default=3,
                        help='Maximum inner iterations/evolution generations')
    parser.add_argument('--population_size', '--population-size', type=int, default=3,
                        help='Evolution algorithm population size')
    parser.add_argument('--offspring_count', type=int, default=1,
                        help='Number of offspring generated per generation')
    parser.add_argument('--code_variants', type=int, default=1,
                        help='Number of code variants generated per strategy')
    parser.add_argument('--time_limit', type=int, default=72000,
                        help='Total execution time limit (seconds)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--algorithm_dir', type=str, default='base_algorithm',
                        help='Algorithm template directory')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Specify run ID to analyze, None for latest run')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset name to use')
    parser.add_argument('--multi_dataset', action='store_true',
                        help='Evaluate strategies on multiple datasets')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated list of dataset IDs to use')
    parser.add_argument('--datasets_dir', type=str, default='datasets',
                        help='Datasets directory')

    return parser.parse_args()


def main():
    """Main function"""
    total_start_time = time.time()
    args = parse_args()
    base_config = {
        'enabled': True,
        'timeout': 60,
        'from_main': True
    }

    logger.info("Base configuration enabled")

    if args.multi_dataset:
        from dataset_manager import DatasetManager

        dataset_manager = DatasetManager(datasets_dir=args.datasets_dir)

        if args.datasets:
            dataset_ids = args.datasets.split(',')
            missing_datasets = []
            for dataset_id in dataset_ids:
                if not dataset_manager.get_dataset(dataset_id):
                    missing_datasets.append(dataset_id)

            if missing_datasets:
                logger.warning(
                    f"The following datasets were not found: {', '.join(missing_datasets)}")

        optimization_params.use_multi_dataset = True
        optimization_params.datasets = args.datasets

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir

    output_manager = OutputManager(base_dir=output_dir, run_id=run_id)

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Run parameters: {args}")
    output_manager.save_run_config(vars(args))

    try:
        if args.dataset:
            dataset_path = f"{args.dataset}/problem_data.json"
            if os.path.exists(dataset_path):
                args.data_path = dataset_path
                logger.info(f"Using dataset: {args.dataset}")
            else:
                logger.warning(
                    f"Dataset path does not exist: {dataset_path}, using default path")

        problem_data = load_problem_data(args.data_path)
        logger.info(f"Successfully loaded problem data: {args.data_path}")

        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        llm_config = config.get("llm", {})
        algorithm_loader = AlgorithmLoader(
            base_algorithm_dir=args.algorithm_dir)
        logger.info(
            f"Algorithm loading completed, {len(algorithm_loader.get_available_algorithms())} algorithm templates available")
        strategy_llm = StrategyLLMClient(
            config=llm_config,
            output_manager=output_manager,
            algorithm_loader=algorithm_loader
        )

        code_llm = CodeLLMClient(
            config=llm_config,
            output_manager=output_manager
        )

        analysis_llm = ProblemAnalysisLLMClient(
            config=llm_config,
            output_manager=output_manager
        )

        problem_params = {}
        if "parameters" in problem_data:
            problem_params = problem_data["parameters"]

        optimization_params = OptimizationParams(
            MAX_OUTER_ITERATIONS=args.max_outer_iterations,
            MAX_INNER_ITERATIONS=args.max_inner_iterations,
            POPULATION_SIZE=args.population_size,
            OFFSPRING_COUNT=args.offspring_count,
            CODE_VARIANTS=args.code_variants,
            LLM_MODEL=args.model
        )

        for key, value in problem_params.items():
            setattr(optimization_params, key, value)

        optimization_run = OptimizationRun(
            id=run_id,
            problem_type=problem_data.get(
                "problem_type", "multi_drone_delivery"),
            params=optimization_params.to_dict()
        )

        problem_analyzer = ProblemAnalyzer(
            problem_data,
            optimization_params,
            analysis_llm,
            output_manager
        )

        problem_info = problem_analyzer.analyze()
        logger.info(
            f"Problem analysis completed: {len(problem_info.get('hard_constraints', []))} hard constraints")

        mcts_tree = MCTSTree(
            problem_info.get("hard_constraints", []),
            optimization_params,
            output_manager
        )

        strategy_manager = TextStrategyManager(
            problem_info,
            strategy_llm,
            output_manager
        )

        evolutionary_algorithm = EvolutionaryAlgorithm(
            strategy_manager,
            problem_info,
            optimization_params,
            output_manager
        )

        mcts_evolution_integration = MCTSEvolutionIntegration(
            mcts_tree,
            evolutionary_algorithm,
            optimization_params,
            output_manager
        )

        code_generator = CodeGenerator(
            code_llm,
            problem_info,
            output_manager
        )
        code_generator.algorithm_loader = algorithm_loader
        code_generator.problem_data = problem_data
        code_generator.problem_info = problem_info

        solution_runner = SolutionRunner(
            output_manager
        )

        optimization_problem_def = {
            "execution_timeout": 30,
            "objective_function": "profit_objective",
            "objective_direction": "maximize"
        }

        strategy_evaluator = StrategyEvaluator(
            problem_data=problem_data,
            code_generator=code_generator,
            output_manager=output_manager,
            optimization_problem=optimization_problem_def
        )

        original_evaluate_strategy = strategy_evaluator.evaluate_strategy
        available_algorithms_count = len(
            algorithm_loader.get_available_algorithms())

        def enhanced_evaluate_strategy(strategy_id, force_regenerate=False, variant_count=1):
            """Intelligent evaluation method: select evaluation strategy based on algorithm framework count"""
            if available_algorithms_count >= 2:
                logger.info(
                    f"Using multi-algorithm framework to evaluate strategy: {strategy_id} (available algorithms: {available_algorithms_count})")
                return strategy_evaluator.evaluate_strategy_with_algorithms(strategy_id)
            else:
                logger.info(
                    f"Using single algorithm framework to evaluate strategy: {strategy_id} (available algorithms: {available_algorithms_count})")
                return original_evaluate_strategy(strategy_id, force_regenerate, variant_count)

        if available_algorithms_count >= 2:
            strategy_evaluator.evaluate_strategy = enhanced_evaluate_strategy
            logger.info(
                f"Multi-algorithm evaluation mode enabled (available algorithms: {available_algorithms_count})")
        else:
            logger.info(
                f"Using single algorithm evaluation mode (available algorithms: {available_algorithms_count})")

        strategy_evaluator.mcts_evolution = mcts_evolution_integration
        mcts_evolution_integration.strategy_evaluator = strategy_evaluator

        code_generator.strategy_manager = strategy_manager

        logger.info("Inter-component references established")

        framework_controller = FrameworkController(
            mcts_evolution_integration,
            code_generator,
            solution_runner,
            strategy_evaluator,
            optimization_params,
            optimization_run,
            output_manager,
            base_config
        )
        mcts_evolution_integration.framework_controller = framework_controller
        logger.info("Starting optimization process")
        start_time = time.time()

        try:
            best_result = framework_controller.run_optimization(
                time_limit=args.time_limit
            )

            execution_time = time.time() - start_time
            logger.info(
                f"Optimization process completed, time taken: {execution_time:.2f} seconds")

            output_manager.save_best_result(best_result)
            logger.info(f"Best result saved")

            optimization_run.mark_completed()
            output_manager.save_optimization_run(optimization_run.to_dict())

        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            output_manager.log_error("main", "execution_error", str(e))

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        output_manager.log_error("main", "execution_error", str(e))
    finally:
        total_execution_time = time.time() - total_start_time
        logger.info(
            f"Total program execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")

        if 'output_manager' in locals():
            try:
                time_report = {
                    "total_execution_time_seconds": total_execution_time,
                    "total_execution_time_minutes": total_execution_time / 60,
                    "timestamp": datetime.now().isoformat()
                }
                time_report_file = os.path.join(
                    output_dir, run_id, "execution_time_report.json")
                with open(time_report_file, 'w', encoding='utf-8') as f:
                    json.dump(time_report, f, indent=2)
                print(f"Execution time report saved to: {time_report_file}")
            except Exception as e:
                print(f"Error saving execution time report: {e}")


if __name__ == "__main__":
    main()
