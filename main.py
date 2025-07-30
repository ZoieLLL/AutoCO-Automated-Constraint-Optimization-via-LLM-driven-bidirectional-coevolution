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
from base_strategy import BaseStrategy
from solution_validator import SolutionValidator
from llm_client import LLMClient, StrategyLLMClient, CodeLLMClient, ProblemAnalysisLLMClient
from optimization_problem import ProblemParameters, ConstraintValidator
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

logger = logging.getLogger(__name__)


def load_problem_data(data_path: str) -> Dict:

    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        logger.error(f"wrong: {e}")
        raise


def save_problem_data(problem_data: Dict, file_path: str) -> None:
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(problem_data, file, indent=2, ensure_ascii=False)
        logger.info(f"problem data in: {file_path}")
    except Exception as e:
        logger.error(f"wrong: {e}")
        raise


def load_config(config_path="config.json"):
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        logger.warning(f"{config_path} not found, using default settings.")
        return {}
    except Exception as e:
        logger.error(f"Wrong: {e}")
        return {}


def parse_args():

    parser = argparse.ArgumentParser(description='framework')

    parser.add_argument('--data_path', type=str, default='problem_data.json',
                        help='data_path')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='output_dir')
    parser.add_argument('--config_path', type=str, default='config.json',
                        help='config_path')
    parser.add_argument('--api_key', type=str, default=None,
                        help='api_key')
    parser.add_argument('--base_url', type=str, default=None,
                        help='base_url')
    parser.add_argument('--model', type=str, default=None,
                        help='LLM model name')
    parser.add_argument('--max_outer_iterations', type=int, default=9,
                        help='max_outer_iterations')
    parser.add_argument('--max_inner_iterations', type=int, default=5,
                        help='max_inner_iterations')
    parser.add_argument('--population_size', type=int, default=3,
                        help='population_size')
    parser.add_argument('--code_variants', type=int, default=1,
                        help='code_variants')
    parser.add_argument('--time_limit', type=int, default=7200,
                        help='execution time limit in seconds')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode')
    parser.add_argument('--algorithm_dir', type=str, default='base_algorithm',
                        help='algorithm_dir')
    parser.add_argument('--generate_plots', action='store_true',
                        help='enerate_plots')
    parser.add_argument('--run_id', type=str, default=None,
                        help='run_id')
    parser.add_argument('--resume', action='store_true',
                        help='resume mode')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='checkpoint_dir')

    return parser.parse_args()


def main():

    total_start_time = time.time()

    args = parse_args()

    if args.generate_plots:
        try:
            print(f"plot{args.output_dir}figure...")
            from visualization import OptimizationVisualizer

            visualizer = OptimizationVisualizer(
                output_base_dir=args.output_dir,
                run_id=args.run_id
            )

            visualizer.generate_all_plots()
            print(f"plot : {visualizer.figures_dir}")
            return
        except Exception as e:
            print(f"wrong: {e}")
            traceback.print_exc()
            return

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume:
        if args.checkpoint_dir:

            run_id = os.path.basename(args.checkpoint_dir)
            output_dir = os.path.dirname(args.checkpoint_dir)
        else:

            latest_run = None
            latest_time = 0
            for d in os.listdir(args.output_dir):
                if d.startswith("run_"):
                    run_path = os.path.join(args.output_dir, d)
                    checkpoint_file = os.path.join(run_path, "checkpoint.json")
                    if os.path.exists(checkpoint_file):
                        mtime = os.path.getmtime(checkpoint_file)
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_run = d

            if latest_run:
                run_id = latest_run
                output_dir = args.output_dir
            else:
                print("restatr ")
                run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                output_dir = args.output_dir
    else:

        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = args.output_dir

    output_manager = OutputManager(base_dir=output_dir, run_id=run_id)

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"args: {args}")
    output_manager.save_run_config(vars(args))

    try:
        problem_data = load_problem_data(args.data_path)

        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        llm_config = config.get("llm", {})

        algorithm_loader = AlgorithmLoader(
            base_algorithm_dir=args.algorithm_dir)

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

        base_strategy = BaseStrategy(
            problem_data, optimization_params.to_dict())

        solution_validator = SolutionValidator(
            problem_data,
            optimization_params.to_dict(),
            output_manager,
            base_strategy
        )

        problem_analyzer = ProblemAnalyzer(
            problem_data,
            optimization_params,
            analysis_llm,
            output_manager
        )

        problem_info = problem_analyzer.analyze()
        logger.info(
            f"finish: {len(problem_info.get('hard_constraints', []))} hard constraints")

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

        def enhanced_evaluate_strategy(strategy_id, force_regenerate=False, variant_count=1):

            logger.info(f" {strategy_id}")
            return strategy_evaluator.evaluate_strategy_with_algorithms(strategy_id)

        strategy_evaluator.evaluate_strategy = enhanced_evaluate_strategy

        strategy_evaluator.mcts_evolution = mcts_evolution_integration
        mcts_evolution_integration.strategy_evaluator = strategy_evaluator

        code_generator.strategy_manager = strategy_manager

        framework_controller = FrameworkController(
            mcts_evolution_integration,
            code_generator,
            solution_runner,
            strategy_evaluator,
            optimization_params,
            optimization_run,
            output_manager
        )
        mcts_evolution_integration.framework_controller = framework_controller
        start_time = time.time()

        try:

            best_result = framework_controller.run_optimization(
                time_limit=args.time_limit
            )

            execution_time = time.time() - start_time
            logger.info(f"finished，time: {execution_time:.2f} ")

            output_manager.save_best_result(best_result)
            logger.info(f"saved")

            optimization_run.mark_completed()
            output_manager.save_optimization_run(optimization_run.to_dict())

            framework_controller.save_checkpoint()

        except Exception as e:
            logger.error(f"wrong: {e}", exc_info=True)
            output_manager.log_error("main", "execution_error", str(e))

            try:
                framework_controller.save_checkpoint()
            except:
                pass

    except Exception as e:
        logger.error(f"wrong: {e}", exc_info=True)
        output_manager.log_error("main", "execution_error", str(e))
    finally:

        total_execution_time = time.time() - total_start_time
        print("\n" + "="*80)
        print(f"{total_execution_time:.2f} ({total_execution_time/60:.2f})")
        print("="*80)
        logger.info(
            f"{total_execution_time:.2f} {total_execution_time/60:.2f}")

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
                print(f"finished: {time_report_file}")
            except Exception as e:
                print(f"wrong: {e}")


if __name__ == "__main__":
    main()
