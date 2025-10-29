import os
import time
import argparse
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import traceback
from models import OptimizationParams, OptimizationRun
from output_manager import OutputManager
from base_strategy import BaseStrategy
from solution_validator import SolutionValidator
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_problem_from_directory(problem_dir: str) -> Tuple[Dict, str, str]:
    """
    Load complete problem definition from a problem directory

    Args:
        problem_dir: Problem directory path

    Returns:
        Tuple[Dict, str, str]: (problem_data, instances_file_path, algorithm_file_path)
    """
    if not os.path.isdir(problem_dir):
        raise ValueError(f"Problem directory does not exist: {problem_dir}")

    # Find required files
    problem_data_path = os.path.join(problem_dir, "problem_data.json")
    algorithm_path = os.path.join(problem_dir, "PointSelectionAlgorithm.py")

    # Find instances file (could be .pkl or .json)
    instances_path = None
    for filename in os.listdir(problem_dir):
        if filename.startswith("instances") and (filename.endswith(".pkl") or filename.endswith(".json")):
            instances_path = os.path.join(problem_dir, filename)
            break

    # Check if required files exist
    if not os.path.exists(problem_data_path):
        raise ValueError(
            f"problem_data.json file does not exist: {problem_data_path}")

    if not os.path.exists(algorithm_path):
        raise ValueError(
            f"PointSelectionAlgorithm.py file does not exist: {algorithm_path}")

    if not instances_path:
        raise ValueError(
            f"instances file does not exist (should be .pkl or .json format): {problem_dir}")

    # Load problem data
    problem_data = load_problem_data(problem_data_path)

    return problem_data, instances_path, algorithm_path


def load_problem_data(data_path: str) -> Dict:
    """
    Load problem data

    Args:
        data_path: Data file path

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
            f"Configuration file {config_path} does not exist, will use command line parameters or environment variables")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        return {}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Optimization Framework')

    parser.add_argument('--problem_dir', type=str, default=None,
                        help='Problem directory path (containing problem_data.json, instances.pkl, PointSelectionAlgorithm.py)')
    parser.add_argument('--data_path', type=str, default='problem_data.json',
                        help='Problem data file path (backward compatibility, used when --problem_dir is not specified)')
    parser.add_argument('--pkl_data_path', type=str, default=None,
                        help='Multi-instance pkl data file path')
    parser.add_argument('--problem_config_path', type=str, default=None,
                        help='Problem configuration file path')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--config_path', type=str, default='config.json',
                        help='Configuration file path')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Default API key (can be overridden by specific component configurations)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='Default API base URL')
    parser.add_argument('--model', type=str, default=None,
                        help='Default LLM model name')
    parser.add_argument('--max_outer_iterations', type=int, default=8,
                        help='Maximum number of outer iterations')
    parser.add_argument('--max_inner_iterations', type=int, default=3,
                        help='Maximum number of inner iterations')
    parser.add_argument('--population_size', type=int, default=3,
                        help='Evolutionary algorithm population size')  # Initial population size
    parser.add_argument('--offspring_count', type=int, default=1,
                        help='Number of offspring generated per generation')
    parser.add_argument('--code_variants', type=int, default=1,
                        help='Number of code variants generated for each strategy')
    parser.add_argument('--time_limit', type=int, default=72000,  # Total runtime 20min
                        help='Total execution time limit (seconds)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--algorithm_dir', type=str, default='base_algorithm',
                        help='Algorithm template directory')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Specify run ID to analyze, use the latest run when None')
    parser.add_argument('--resume', action='store_true',
                        help='Resume run from checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Specify checkpoint directory for resume')
    parser.add_argument('--multi_dataset', action='store_true',
                        help='Evaluate strategies on multiple datasets')
    parser.add_argument('--datasets', type=str, default=None,
                        help='List of dataset IDs to use, comma-separated')
    parser.add_argument('--datasets_dir', type=str, default='datasets',
                        help='Datasets directory')
    parser.add_argument('--disable-repair', action='store_true',
                        help='Disable repair mechanism (repair mechanism enabled by default)')  # Default enabled, to disable: python main.py --disable-repair
    parser.add_argument('--max-repair-attempts', type=int, default=1,
                        help='Maximum repair attempts per strategy (default: 1)')
    return parser.parse_args()


def main():
    """Main function"""
    total_start_time = time.time()
    # Parse command line arguments
    args = parse_args()

    # Configure repair mechanism - enabled by default unless explicitly disabled by user
    repair_enabled = not getattr(args, 'disable_repair', False)

    # Build repair mechanism
    repair_config = {
        'enabled': repair_enabled,
        'max_repair_attempts': args.max_repair_attempts,
        'repair_timeout': 60,
        'from_main': True
    }



    # Log configuration information
    logger.info(
        f"Repair mechanism: {'enabled' if repair_config['enabled'] else 'disabled'} (enabled by default)")
    if repair_config['enabled']:
        logger.info(
            f"Maximum repair attempts: {repair_config['max_repair_attempts']}")
    else:
        logger.info(
            "Repair mechanism disabled by user (using --disable-repair parameter)")

    # Handle multi-dataset evaluation
    if args.multi_dataset:
        from dataset_manager import DatasetManager

        # Create dataset manager
        dataset_manager = DatasetManager(datasets_dir=args.datasets_dir)

        # Ensure datasets exist
        if args.datasets:
            dataset_ids = args.datasets.split(',')
            missing_datasets = []
            for dataset_id in dataset_ids:
                if not dataset_manager.get_dataset(dataset_id):
                    missing_datasets.append(dataset_id)

            if missing_datasets:
                logger.warning(
                    f"The following datasets were not found: {', '.join(missing_datasets)}")

        # Pass multi-dataset parameters to optimization parameters
        optimization_params.use_multi_dataset = True
        optimization_params.datasets = args.datasets
    # Set debug mode
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    # Ensure output directory exists

    os.makedirs(args.output_dir, exist_ok=True)
    # Handle resume run situation
    if args.resume:
        if args.checkpoint_dir:
            # Use specified checkpoint directory
            run_id = os.path.basename(args.checkpoint_dir)
            output_dir = os.path.dirname(args.checkpoint_dir)
        else:
            # Find the latest checkpoint
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
                print("No recoverable checkpoint found, will create a new run")
                run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                output_dir = args.output_dir
    else:
        # Create a new run
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = args.output_dir

    # Create output manager
    output_manager = OutputManager(base_dir=output_dir, run_id=run_id)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Log parameters
    logger.info(f"Run parameters: {args}")
    output_manager.save_run_config(vars(args))

    try:
        # New unified problem loading logic
        evaluation_data = None  # Initialize evaluation data

        if args.problem_dir:
            # Use new problem directory structure
            logger.info(f"Using problem directory: {args.problem_dir}")
            problem_data, instances_path, algorithm_path = load_problem_from_directory(
                args.problem_dir)

            # Separate the use of two types of data:
            # 1. problem_data for problem analysis and LLM interaction (from problem_data.json)
            # 2. evaluation_data for code execution evaluation (from instances.pkl)

            evaluation_data = None  # Data for code execution

            # Determine loading method based on instances file type
            if instances_path.endswith('.pkl'):
                # Multi-instance pkl file
                from dataset_manager import DatasetManager
                dataset_manager = DatasetManager()

                dataset_id = dataset_manager.load_multi_instance_dataset_new(
                    instances_path, problem_data
                )

                if dataset_id:
                    # Get complete dataset containing instance data for code execution evaluation
                    evaluation_data = dataset_manager.get_dataset(dataset_id)

                    logger.info(
                        f"Successfully loaded multi-instance dataset: {dataset_id}")
                    logger.info(
                        f"  Number of instances: {evaluation_data.get('instance_count', 0)}")
                    logger.info(f"  Evaluation mode: Multi-instance average")

                    # problem_data remains as original problem description (for LLM analysis)
                    logger.info(
                        "Problem analysis will use description from problem_data.json")
                    logger.info(
                        "Code execution evaluation will use instance data from instances.pkl")
                else:
                    logger.error("Failed to load multi-instance dataset")
                    evaluation_data = problem_data  # Both are the same when falling back
            else:
                # Single instance json file
                with open(instances_path, 'r', encoding='utf-8') as f:
                    instance_data = json.load(f)
                evaluation_data = problem_data.copy()
                evaluation_data["instance_data"] = instance_data
                evaluation_data["is_multi_instance"] = False
                logger.info(
                    f"Successfully loaded single instance data: {instances_path}")

        elif args.pkl_data_path:
            # Backward compatibility: use old multi-instance pkl data file method
            logger.info("Loading pkl data using backward compatibility mode")
            from dataset_manager import DatasetManager
            dataset_manager = DatasetManager()

            dataset_id = None

            if os.path.isfile(args.pkl_data_path) and args.pkl_data_path.endswith('.pkl'):
                # If it's a single pkl file
                dataset_id = dataset_manager.load_multi_instance_dataset(
                    args.pkl_data_path,
                    args.problem_config_path or args.data_path
                )
            elif os.path.isdir(args.pkl_data_path):
                # If it's a directory, try to load the first pkl file found
                pkl_files = [f for f in os.listdir(
                    args.pkl_data_path) if f.endswith('.pkl')]
                if pkl_files:
                    # Prioritize C1_25 pkl files (smaller test set)
                    preferred_files = [f for f in pkl_files if 'C1_25' in f]
                    if preferred_files:
                        pkl_file = preferred_files[0]
                    else:
                        pkl_file = pkl_files[0]

                    pkl_path = os.path.join(args.pkl_data_path, pkl_file)
                    logger.info(
                        f"Found pkl file in directory {args.pkl_data_path}: {pkl_file}")

                    dataset_id = dataset_manager.load_multi_instance_dataset(
                        pkl_path,
                        args.problem_config_path or args.data_path
                    )
                else:
                    logger.error(
                        f"No pkl file found in directory {args.pkl_data_path}")

            if dataset_id:
                evaluation_data = dataset_manager.get_dataset(dataset_id)
                logger.info(
                    f"Successfully loaded multi-instance dataset: {dataset_id}")
                logger.info(
                    f"  Number of instances: {evaluation_data.get('instance_count', 0)}")
                logger.info(f"  Evaluation mode: Multi-instance average")

                # problem_data for problem analysis, loaded from config file
                problem_data = load_problem_data(args.data_path)
            else:
                logger.error(
                    "Failed to load multi-instance dataset, falling back to single instance mode")
                problem_data = load_problem_data(args.data_path)
                evaluation_data = problem_data  # Both are the same for single instance
        else:
            # Use traditional single instance data file
            problem_data = load_problem_data(args.data_path)
            evaluation_data = problem_data  # Both are the same for single instance
            logger.info(f"Successfully loaded problem data: {args.data_path}")

        # Load configuration
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # Create LLM client
        llm_config = config.get("llm", {})
        # Create algorithm loader
        if args.problem_dir:
            # If using problem directory, try to load algorithm file from directory
            _, _, algorithm_path = load_problem_from_directory(
                args.problem_dir)
            algorithm_loader = AlgorithmLoader(
                single_algorithm_file=algorithm_path)
        else:
            # Use traditional algorithm directory method
            algorithm_loader = AlgorithmLoader(
                base_algorithm_dir=args.algorithm_dir)
        logger.info(
            f"Algorithm loading complete, {len(algorithm_loader.get_available_algorithms())} algorithm templates available")
        # Strategy LLM client
        strategy_llm = StrategyLLMClient(
            config=llm_config,
            output_manager=output_manager,
            algorithm_loader=algorithm_loader
        )

        # Code LLM client
        code_llm = CodeLLMClient(
            config=llm_config,
            output_manager=output_manager
        )

        # Problem analysis LLM client
        analysis_llm = ProblemAnalysisLLMClient(
            config=llm_config,
            output_manager=output_manager
        )

        # Extract parameters from problem data
        problem_params = {}
        if "parameters" in problem_data:
            problem_params = problem_data["parameters"]

        # Set optimization parameters
        optimization_params = OptimizationParams(
            MAX_OUTER_ITERATIONS=args.max_outer_iterations,
            MAX_INNER_ITERATIONS=args.max_inner_iterations,
            POPULATION_SIZE=args.population_size,
            OFFSPRING_COUNT=args.offspring_count,
            CODE_VARIANTS=args.code_variants,
            LLM_MODEL=args.model
        )

        # Update specific problem parameters
        for key, value in problem_params.items():
            setattr(optimization_params, key, value)

        # Create optimization run record
        optimization_run = OptimizationRun(
            id=run_id,
            problem_type=problem_data.get(
                "problem_type", "multi_drone_delivery"),
            params=optimization_params.to_dict()
        )

        # Create base strategy
        base_strategy = BaseStrategy(
            problem_data, optimization_params.to_dict())

        # Create solution validator
        solution_validator = SolutionValidator(
            problem_data,
            optimization_params.to_dict(),
            output_manager,
            base_strategy
        )

        # Create problem analyzer
        problem_analyzer = ProblemAnalyzer(
            problem_data,
            optimization_params,
            analysis_llm,
            output_manager
        )

        # Analyze problem, extract constraint information
        problem_info = problem_analyzer.analyze()
        logger.info(
            f"Problem analysis complete: {len(problem_info.get('hard_constraints', []))} hard constraints")

        # Create Monte Carlo Tree
        mcts_tree = MCTSTree(
            # Only pass in hard constraints list
            problem_info.get("hard_constraints", []),
            optimization_params,
            output_manager
        )

        # Create text strategy manager
        strategy_manager = TextStrategyManager(
            problem_info,
            strategy_llm,
            output_manager,
            algorithm_loader
        )

        # Create evolutionary algorithm
        evolutionary_algorithm = EvolutionaryAlgorithm(
            strategy_manager,
            problem_info,
            optimization_params,
            output_manager
        )

        # Create MCTS and evolutionary algorithm integration module
        mcts_evolution_integration = MCTSEvolutionIntegration(
            mcts_tree,
            evolutionary_algorithm,
            optimization_params,
            output_manager
        )

        # Create code generator (and set algorithm loader)
        code_generator = CodeGenerator(
            code_llm,
            problem_info,
            output_manager
        )
        code_generator.algorithm_loader = algorithm_loader  # Set algorithm loader
        code_generator.problem_data = problem_data  # Keep original data
        # Pass analyzed problem information to code generator
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
            problem_data=evaluation_data if evaluation_data else problem_data,
            code_generator=code_generator,
            output_manager=output_manager,
            optimization_problem=optimization_problem_def
        )

        original_evaluate_strategy = strategy_evaluator.evaluate_strategy
        available_algorithms_count = len(
            algorithm_loader.get_available_algorithms())

        def enhanced_evaluate_strategy(strategy_id, force_regenerate=False, variant_count=1):
            if available_algorithms_count >= 2:
                logger.info(
                    f"Evaluating strategy with multi-algorithm framework: {strategy_id} (Available algorithms: {available_algorithms_count})")
                return strategy_evaluator.evaluate_strategy_with_algorithms(strategy_id)
            else:
                logger.info(
                    f"Evaluating strategy with single-algorithm framework: {strategy_id} (Available algorithms: {available_algorithms_count})")
                return original_evaluate_strategy(strategy_id, force_regenerate, variant_count)

        if available_algorithms_count >= 2:
            strategy_evaluator.evaluate_strategy = enhanced_evaluate_strategy
            logger.info(
                f"Multi-algorithm evaluation mode enabled (Available algorithms: {available_algorithms_count})")
        else:
            logger.info(
                f"Single-algorithm evaluation mode enabled (Available algorithms: {available_algorithms_count})")

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
            output_manager,
            repair_config
        )
        mcts_evolution_integration.framework_controller = framework_controller
        logger.info("start optimization process")
        start_time = time.time()

        try:
            best_result = framework_controller.run_optimization(
                time_limit=args.time_limit
            )
            execution_time = time.time() - start_time
            logger.info(
                f"Optimization process completed, {execution_time:.2f} s")
            output_manager.save_best_result(best_result)
            logger.info(f"Save best result")
            optimization_run.mark_completed()
            output_manager.save_optimization_run(optimization_run.to_dict())
            framework_controller.save_checkpoint()

        except Exception as e:
            logger.error(f"Wrong: {e}", exc_info=True)
            output_manager.log_error("main", "execution_error", str(e))
            try:
                framework_controller.save_checkpoint()
            except:
                pass

    except Exception as e:
        logger.error(f"Wrong: {e}", exc_info=True)
        output_manager.log_error("main", "execution_error", str(e))
    finally:
        total_execution_time = time.time() - total_start_time
        print("\n" + "="*80)
        print(
            f"total_execution_time: {total_execution_time:.2f}s ({total_execution_time/60:.2f}min)")
        print("="*80)
        logger.info(
            f"total_execution_time: {total_execution_time:.2f}s ({total_execution_time/60:.2f}min)")

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
                print(f"time report saved: {time_report_file}")
            except Exception as e:
                print(f"Wrong: {e}")


if __name__ == "__main__":
    main()
    # conda activate llm4uav && python main.py --config_path config.json
    # python main.py --resume --checkpoint_dir outputs/run_20250519_135755
    # python main.py --problem_dir=problems/vrptw_fuel --max_outer_iterations=5 --max_inner_iterations=7 --population_size=3
