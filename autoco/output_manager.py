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
    def __init__(self, base_dir="output", run_id=None, verbose=False):
        self.base_dir = base_dir
        self.run_id = run_id
        self.verbose = verbose

        if run_id:
            self.base_dir = os.path.join(base_dir, run_id)

        self.subdirs = {
            'results': 'results',
            'history': 'history',
        }
        self.setup_directories()
        self.code_dir = os.path.join(
            tempfile.gettempdir(), f"autoco_codes_{run_id or 'default'}")
        os.makedirs(self.code_dir, exist_ok=True)
       



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

    def log_info(self, module: str, action: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{module}] [{action}] {message}"

        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "info.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        if self.verbose:
            print(f"{Fore.GREEN}[INFO] {log_entry}{Style.RESET_ALL}")

    def log_warning(self, module: str, action: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{module}] [{action}] {message}"
        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "warning.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        if self.verbose:
            print(f"{Fore.YELLOW}[WARNING] {log_entry}{Style.RESET_ALL}")

    def log_error(self, module: str, action: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{module}] [{action}] {message}"
        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "error.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        print(f"{Fore.RED}[ERROR] {log_entry}{Style.RESET_ALL}")


    def save_run_config(self, config: Dict) -> None:
        try:
            config_dir = os.path.join(self.base_dir, "config")
            os.makedirs(config_dir, exist_ok=True)

            config_file = os.path.join(config_dir, "run_config.json")
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            self.log_info("output_manager", "save_config",
                          f"save_config: {config_file}")
        except Exception as e:
            self.log_error("output_manager", "save_config_error",
                           f"save_config_error: {str(e)}")

    def save_final_result(self, result: Dict) -> None:
        try:
            results_dir = os.path.join(self.base_dir, self.subdirs["results"])
            os.makedirs(results_dir, exist_ok=True)

            result_file = os.path.join(results_dir, "final_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self.log_info("output_manager", "save_final_result",
                          f"save_final_result: {result_file}")
        except Exception as e:
            self.log_error("output_manager", "save_result_error",
                           f"save_result_error: {str(e)}")

    def append_strategy(self, strategy: Dict) -> None:
        try:
            history_dir = os.path.join(self.base_dir, self.subdirs['history'])
            os.makedirs(history_dir, exist_ok=True)

            registry_file = os.path.join(history_dir, 'strategies.jsonl')
            with open(registry_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(strategy, ensure_ascii=False,
                                   cls=NumpyEncoder) + '\n')
        except Exception as e:
            self.log_error("output_manager", "append_strategy_error",
                           f"append strategy error: {str(e)}")

    def save_strategy(self, strategy: Dict, method: str) -> None:
        self.append_strategy(strategy)

    def update_strategy(self, strategy: Dict) -> None:
        self.append_strategy(strategy)

    def save_generated_code(self, code: str, code_id: str) -> str:
        try:
            code_dir = self.code_dir
            os.makedirs(code_dir, exist_ok=True)

            file_path = os.path.join(code_dir, f"{code_id}.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            return file_path
        except Exception as e:
            self.log_error("output_manager", "save_code_error",
                           f"save_code_error: {str(e)}")
            return ""

    def save_problem_analysis(self, analysis: Dict) -> None:
        try:
            analysis_dir = os.path.join(self.base_dir, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            file_path = os.path.join(analysis_dir, "problem_analysis.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            self.log_info("output_manager", "save_analysis",
                          f"save_analysis: {file_path}")
        except Exception as e:
            self.log_error("output_manager", "save_analysis_error",
                           f"save_analysis_error: {str(e)}")

    def save_optimization_result(self, result: Dict) -> None:
        try:
            results_dir = os.path.join(
                self.base_dir, self.subdirs.get("results", "results"))
            os.makedirs(results_dir, exist_ok=True)

            result_file = os.path.join(results_dir, "optimization_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self.log_info("output_manager", "save_optimization_result",
                          f"save_optimization_result: {result_file}")
        except Exception as e:
            self.log_error(
                "output_manager", "save_optimization_result_error", f"save_optimization_result_error: {str(e)}")

    def save_best_result(self, result: Dict) -> None:
        try:
            results_dir = os.path.join(
                self.base_dir, self.subdirs.get("results", "results"))
            os.makedirs(results_dir, exist_ok=True)

            best_result_file = os.path.join(results_dir, "best_result.json")
            with open(best_result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self.log_info("output_manager", "save_best_result",
                          f"save_best_result: {best_result_file}")
        except Exception as e:
            self.log_error("output_manager",
                           "save_best_result_error", f"save_best_result_error: {str(e)}")

    def save_mcts_result(self, result: Dict) -> None:
        try:
            output_dir = os.path.join(self.base_dir, "mcts_results")
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcts_result_{timestamp}.json"
            file_path = os.path.join(output_dir, filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self.log_info("output_manager", "save_mcts_result",
                          f"save_mcts_result: {file_path}")
        except Exception as e:
            self.log_error(
                "output_manager", "save_mcts_result_error", f"save_mcts_result_error: {str(e)}")


