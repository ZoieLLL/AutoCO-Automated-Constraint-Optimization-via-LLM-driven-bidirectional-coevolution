import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import traceback
from colorama import Fore, Style, init
import re
import uuid
import time
import importlib
import sys
import traceback
import tempfile
import math
import random
import subprocess
import numpy as np
init()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            val = float(obj)
            if val == float('inf'):
                return "Infinity"
            elif val == float('-inf'):
                return "-Infinity"
            elif val != val:
                return "NaN"
            return val
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, float):
            if obj == float('inf'):
                return "Infinity"
            elif obj == float('-inf'):
                return "-Infinity"
            elif obj != obj:
                return "NaN"
            return obj
        return super().default(obj)


class OutputManager:
    """Output manager: manages storage and organization of all project output files"""

    def __init__(self, base_dir="outputs", run_id=None):
        self.base_dir = base_dir
        self.run_id = run_id

        if run_id:
            self.base_dir = os.path.join(base_dir, run_id)

        self.subdirs = {
            'strategies': {
                'upper': 'strategies/upper',
                'lower': 'strategies/lower'
            },
            'strategiesword': {
                'upper': 'strategiesword/upper',
                'lower': 'strategiesword/lower'
            },
            'codes': 'codes',
            'raw_codes': 'raw_codes',
            'finalcodes': 'finalcodes',
            'logs': {
                'strategy': {
                    'upper': 'logs/strategy/upper',
                    'lower': 'logs/strategy/lower'
                },
                'code': {
                    'upper': 'logs/code/upper',
                    'lower': 'logs/code/lower'
                },
                'error': 'logs/error',
                'fix': {
                    'upper': 'logs/fix/upper',
                    'lower': 'logs/fix/lower'
                },
                'code_check': 'logs/code_check'
            },
            'results': 'results',
            'history': 'history',
        }
        self.setup_directories()

    def save_optimization_run(self, optimization_run_data: Dict) -> None:
        try:
            output_dir = os.path.join(self.base_dir, "optimization_runs")
            os.makedirs(output_dir, exist_ok=True)

            run_id = optimization_run_data.get(
                "id", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            filename = f"optimization_run_{run_id}.json"
            file_path = os.path.join(output_dir, filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(optimization_run_data, f,
                          ensure_ascii=False, indent=2)
        except Exception as e:
            self.log_error(
                "output_manager", "save_optimization_run_error", f"save optimization run failed: {str(e)}")

    def save_strategy_text(self, strategy: str, level: str, iteration: int) -> str:
        try:
            filename = f"strategy_{level}_{iteration}.txt"
            filepath = os.path.join(
                self.base_dir,
                self.subdirs['strategiesword'][level],
                filename
            )

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(strategy)

            return filepath

        except Exception as e:
            print(
                f"{Fore.RED}⚠️ save strategy text failed: {str(e)}{Style.RESET_ALL}")
            return None

    def read_strategy_text(self, level: str, iteration: int) -> str:
        try:
            filename = f"strategy_{level}_{iteration}.txt"
            filepath = os.path.join(
                self.base_dir,
                self.subdirs['strategiesword'][level],
                filename
            )

            if not os.path.exists(filepath):
                print(f"strategy text file does not exist: {filepath}")
                return None

            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()

        except Exception as e:
            print(f"read strategy text failed: {str(e)}")
            return None

    def setup_directories(self) -> None:
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            for dir_name, structure in self.subdirs.items():
                if isinstance(structure, dict):
                    for subtype, substructure in structure.items():
                        if isinstance(substructure, dict):
                            for level, path in substructure.items():
                                full_path = os.path.join(self.base_dir, path)
                                os.makedirs(full_path, exist_ok=True)

                        else:
                            full_path = os.path.join(
                                self.base_dir, substructure)
                            os.makedirs(full_path, exist_ok=True)

                else:
                    full_path = os.path.join(self.base_dir, structure)
                    os.makedirs(full_path, exist_ok=True)
        except Exception as e:
            raise

    def _ensure_directory(self, path: str) -> bool:
        """Ensure the directory exists and is writable"""
        try:
            os.makedirs(path, exist_ok=True)
            return os.access(path, os.W_OK)
        except Exception as e:
            print(f"create directory failed: {str(e)}")
            return False

    def log_info(self, module: str, action: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{module}] [{action}] {message}"
        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "info.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        print(f"{Fore.GREEN}[INFO] {log_entry}{Style.RESET_ALL}")

    def log_warning(self, module: str, action: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{module}] [{action}] {message}"

        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "warning.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        print(f"{Fore.YELLOW}[WARNING] {log_entry}{Style.RESET_ALL}")

    def log_error(self, module: str, action: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{module}] [{action}] {message}"
        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "error.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def log_debug(self, module: str, action: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{module}] [{action}] {message}"
        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "debug.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def save_run_config(self, config: Dict) -> None:
        try:
            config_dir = os.path.join(self.base_dir, "config")
            os.makedirs(config_dir, exist_ok=True)

            config_file = os.path.join(config_dir, "run_config.json")
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log_error("output_manager", "save_config_error",
                           f"save run config failed: {str(e)}")

    def save_final_result(self, result: Dict) -> None:
        try:
            results_dir = os.path.join(self.base_dir, self.subdirs["results"])
            os.makedirs(results_dir, exist_ok=True)

            result_file = os.path.join(results_dir, "final_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self.log_info("output_manager", "save_final_result",
                          f"final result saved to: {result_file}")
        except Exception as e:
            self.log_error("output_manager", "save_result_error",
                           f"save final result failed: {str(e)}")

    def save_strategy(self, strategy: Dict, method: str) -> None:
        try:
            strategy_dir = os.path.join(self.base_dir, "strategies", method)
            os.makedirs(strategy_dir, exist_ok=True)

            strategy_file = os.path.join(
                strategy_dir, f"{strategy['id']}.json")
            with open(strategy_file, "w", encoding="utf-8") as f:
                json.dump(strategy, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log_error("output_manager",
                           "save_strategy_error", f"save strategy failed: {str(e)}")

    def update_strategy(self, strategy: Dict) -> None:
        try:
            method = strategy.get("method", "unknown")
            strategy_dir = os.path.join(self.base_dir, "strategies", method)
            strategy_file = os.path.join(
                strategy_dir, f"{strategy['id']}.json")
            if not os.path.exists(strategy_file):
                strategy_dir = os.path.join(
                    self.base_dir, "strategies", "default")
                os.makedirs(strategy_dir, exist_ok=True)
                strategy_file = os.path.join(
                    strategy_dir, f"{strategy['id']}.json")

            with open(strategy_file, "w", encoding="utf-8") as f:
                json.dump(strategy, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log_error("output_manager",
                           "update_strategy_error", f"update strategy failed: {str(e)}")

    def save_generated_code(self, code: str, code_id: str) -> str:
        try:
            code_dir = os.path.join(self.base_dir, self.subdirs["codes"])
            os.makedirs(code_dir, exist_ok=True)

            file_path = os.path.join(code_dir, f"{code_id}.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            return file_path
        except Exception as e:
            self.log_error("output_manager", "save_code_error",
                           f"save generated code failed: {str(e)}")
            return ""

    def save_problem_analysis(self, analysis: Dict) -> None:
        try:
            analysis_dir = os.path.join(self.base_dir, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            file_path = os.path.join(analysis_dir, "problem_analysis.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            self.log_info("output_manager", "save_analysis",
                          f"problem analysis saved to: {file_path}")
        except Exception as e:
            self.log_error("output_manager", "save_analysis_error",
                           f"save problem analysis failed: {str(e)}")

    def save_optimization_result(self, result: Dict) -> None:

        try:
            results_dir = os.path.join(
                self.base_dir, self.subdirs.get("results", "results"))
            os.makedirs(results_dir, exist_ok=True)

            result_file = os.path.join(results_dir, "optimization_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self.log_info("output_manager", "save_optimization_result",
                          f"optimization result saved to: {result_file}")
        except Exception as e:
            self.log_error(
                "output_manager", "save_optimization_result_error", f"save optimization result failed: {str(e)}")

    def save_best_result(self, result: Dict) -> None:
        try:
            results_dir = os.path.join(
                self.base_dir, self.subdirs.get("results", "results"))
            os.makedirs(results_dir, exist_ok=True)

            best_result_file = os.path.join(results_dir, "best_result.json")
            with open(best_result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self.log_info("output_manager", "save_best_result",
                          f"best result saved to: {best_result_file}")
        except Exception as e:
            self.log_error("output_manager",
                           "save_best_result_error", f"save best result failed: {str(e)}")

    def log_validation_result(self, validation_result: Dict) -> None:
        try:
            validation_dir = os.path.join(self.base_dir, "validation")
            os.makedirs(validation_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(
                validation_dir, f"validation_{timestamp}.json")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(validation_result, f, indent=2, ensure_ascii=False)
            feasible = validation_result.get("feasible", False)
            objective = validation_result.get("total_objective", 0)

            self.log_info(
                "solution_validator", "validation_result",
                f"validation result: {'feasible' if feasible else 'infeasible'}, objective: {objective}"
            )
        except Exception as e:
            self.log_error("output_manager",
                           "log_validation_error", f"log validation error: {str(e)}")

    def log_relaxed_validation_result(self, validation_result: Dict) -> None:
        try:
            validation_dir = os.path.join(self.base_dir, "validation")
            os.makedirs(validation_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(
                validation_dir, f"relaxed_validation_{timestamp}.json")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(validation_result, f, indent=2, ensure_ascii=False)
            self.log_info(
                "solution_validator", "relaxed_validation_result",
                f"relaxed validation result: {'feasible' if validation_result.get('feasible', False) else 'infeasible'}"
            )
        except Exception as e:
            self.log_error(
                "output_manager", "log_relaxed_validation_error", f"log relaxed validation error: {str(e)}")

    def save_population_stats(self, stats: Dict, filename: str = None) -> None:
        try:
            stats_dir = os.path.join(self.base_dir, "stats")
            os.makedirs(stats_dir, exist_ok=True)
            if filename is None:
                outer_iteration = stats.get("outer_iteration", 0)
                generation = stats.get("generation", 0)
                filename = f"population_stats_{outer_iteration}_{generation}.json"

            file_path = os.path.join(stats_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2,
                          ensure_ascii=False, cls=NumpyEncoder)

        except Exception as e:
            self.log_error(
                "output_manager", "save_population_stats_error", f"save population stats failed: {str(e)}")

    def save_iteration_stats(self, stats: Dict) -> None:
        try:
            stats_dir = os.path.join(self.base_dir, "stats", "iteration")
            os.makedirs(stats_dir, exist_ok=True)
            iteration = stats.get("iteration", 0)
            file_path = os.path.join(stats_dir, f"iteration_{iteration}.json")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.log_error(
                "output_manager", "save_iteration_stats_error", f"save iteration stats failed: {str(e)}")

    def save_evolution_result(self, result: Dict) -> None:
        try:
            output_dir = os.path.join(self.base_dir, "evolution_results")
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evolution_result_{timestamp}.json"
            file_path = os.path.join(output_dir, filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log_error(
                "output_manager", "save_evolution_result_error", f"save Evolution result failed: {str(e)}")

    def save_json(self, data: Dict, file_path: str) -> None:
        try:
            full_path = os.path.join(self.base_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.log_debug("output_manager", "save_json",
                           f"JSON data saved to: {full_path}")

        except Exception as e:
            self.log_error("output_manager", "save_json_error",
                           f"save JSON file failed: {str(e)}")
