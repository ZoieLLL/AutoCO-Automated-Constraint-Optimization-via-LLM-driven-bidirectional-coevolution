from output_manager import OutputManager
from typing import Dict, Any, Tuple, Optional, Union
import re
import traceback
import subprocess
import tempfile
import uuid
import time
import os
import sys
import json
import numpy as np


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


class SolutionRunner:

    def __init__(self, output_manager: OutputManager, problem_data: Dict = None):

        self.output_manager = output_manager
        self.problem_data = problem_data
        self.execution_timeout = 60
        self.max_memory_mb = 1024
        self.temp_dir = os.path.join(output_manager.base_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

    def run_solution_code(self, code_file: str, problem_data: Dict,
                          timeout: int = None) -> Dict:
        timeout = timeout or self.execution_timeout

        self.output_manager.log_info(
            "solution_runner", "run_start",
            f"run_start: {code_file}, timeout: {timeout}seconds"
        )

        start_time = time.time()

        try:

            result = self._run_in_subprocess(code_file, problem_data, timeout)
            simplified_result = self._simplify_result(result)
            execution_time = time.time() - start_time
            simplified_result["execution_time"] = execution_time
            simplified_result["execution_success"] = True

            self.output_manager.log_info(
                "solution_runner", "solution_result",
                f"solution_result: {json.dumps(simplified_result, indent=2, cls=NumpyEncoder, ensure_ascii=False)}"
            )

            return simplified_result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"execution_error: {str(e)}"
            error_result = {
                "execution_success": False,
                "execution_time": execution_time,
                "error": error_msg,
                "traceback": traceback.format_exc()
            }

            self.output_manager.log_info(
                "solution_runner", "solution_result",
                f"solution_result: {json.dumps(error_result, indent=2, cls=NumpyEncoder, ensure_ascii=False)}"
            )
            return error_result

    def _run_in_subprocess(self, code_file: str, problem_data: Dict, timeout: int) -> Dict:
        import tempfile
        import subprocess
        import json
        import os
        import sys
        import re
        import time
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w', encoding='utf-8') as f:
            json.dump(problem_data, f, cls=NumpyEncoder,
                      ensure_ascii=False, indent=2)
            problem_data_file = f.name
        runner_script = self._create_runner_script()

        try:
            cmd = [
                sys.executable,
                runner_script,
                code_file,
                problem_data_file
            ]

            cmd_str = " ".join([str(arg) for arg in cmd])
            self.output_manager.log_info(
                "solution_runner", "subprocess_command",
                f"subprocess_command: {cmd_str}"
            )
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                self.output_manager.log_warning(
                    "solution_runner", "execution_timeout",
                    f"execution_timeout: {timeout}秒"
                )

            self.output_manager.log_info(
                "solution_runner", "raw_output",
                f"raw_output: {stdout}"
            )

            if stderr:
                self.output_manager.log_warning(
                    "solution_runner", "stderr_output",
                    f"stderr_output: {stderr}"
                )

            result = {}
            try:

                json_pattern = r'({[\s\S]*})'
                matches = re.findall(json_pattern, stdout)

                if matches:
                    longest_match = max(matches, key=len)
                    try:
                        result = json.loads(longest_match)
                    except json.JSONDecodeError:
                        cleaned_json = re.sub(
                            r'[\x00-\x1F]+', '', longest_match)
                        result = json.loads(cleaned_json)
                else:
                    fitness_match = re.search(
                        r'fitness:\s*(-?\d+\.\d+)', stdout)
                    if fitness_match:
                        fitness = float(fitness_match.group(1))
                        result = {
                            "profit_objective": fitness,
                            "objective": fitness,
                            "success": True
                        }
                    else:
                        result = {"success": False}
                result["stdout"] = stdout
                result["stderr"] = stderr

            except Exception as e:
                result = {
                    'success': False,
                    'stdout': stdout,
                    'stderr': stderr,
                    'error': f'JSON error: {str(e)}'
                }

            return result

        except Exception as e:
            self.output_manager.log_error(
                "solution_runner", "execution_error",
                f"execution_error: {str(e)}"
            )
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        finally:
            try:
                os.unlink(problem_data_file)
                os.unlink(runner_script)
            except:
                pass

    def _create_runner_script(self) -> str:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as f:
            f.write("""import sys
import os
import json
import traceback
import importlib.util
import time
import io
import re
import numpy as np
import random
import math
import itertools
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional

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
            elif val != val:  # NaN check
                return "NaN"
            return val
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, float):
            if obj == float('inf'):
                return "Infinity"
            elif obj == float('-inf'):
                return "-Infinity"
            elif obj != obj:  # NaN check
                return "NaN"
            return obj
        return super().default(obj)

def run_solution_code(code_file, problem_data_file):
    try:
        with open(problem_data_file, 'r', encoding='utf-8') as f:
            problem_data = json.load(f)
        
        parameters = problem_data.get('parameters', {})
        
        module_name = os.path.basename(code_file).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, code_file)
        solution_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution_module)

        if not hasattr(solution_module, 'Solution'):
            print(f"NO Class Solution")
            return {"success": False, "error": "NO Class Solution"}

        solution_class = solution_module.Solution
        solution_instance = solution_class(problem_data, parameters,[])
        
        print("Start executing solve() method...")
        start_time = time.time()
        solution_result = solution_instance.solve()
        execution_time = time.time() - start_time
        print(f"solve() method executed, took: {execution_time:.4f} seconds")

        result = {}
        
        if isinstance(solution_result, dict):
            result = solution_result

            if "success" not in result:
                result["success"] = True
                
            if "execution_time" not in result:
                result["execution_time"] = execution_time
                

            if "violation_analysis" in result:
                violation_analysis = result["violation_analysis"]
                if violation_analysis:
                    result["has_violations"] = True
                    result["violation_types"] = list(violation_analysis.keys())
                    result["total_violation_types"] = len(violation_analysis)
                    total_degree = sum(
                        v.get('total_violation_degree', 0) 
                        for v in violation_analysis.values() 
                        if isinstance(v, dict) and v.get('violated', False)
                    )
                    result["total_violation_degree"] = total_degree
                else:
                    result["has_violations"] = False
                    result["violation_types"] = []
                    result["total_violation_types"] = 0
                    result["total_violation_degree"] = 0
        else:
            result = {
                "success": True,
                "solution": solution_result,
                "execution_time": execution_time
            }
        
        if "total_cost" not in result and "objective" not in result:
            output = sys.stdout.getvalue() if hasattr(sys.stdout, "getvalue") else ""
            fitness_match = re.search(r'total_cost:\s*(-?\d+\.\d+)', output)
            if fitness_match:
                fitness = float(fitness_match.group(1))
                result["total_cost"] = fitness
                result["objective"] = -fitness  

            if "statistics" not in result:
                result["statistics"] = {
                    "execution_time": execution_time,
                    "iterations": 0  
                }

                iter_match = re.search(r'iterations:\s*(\d+)', output)
                if iter_match:
                    result["statistics"]["iterations"] = int(iter_match.group(1))
        
        return result
            
    except Exception as e:
        print(f"Wrong: {str(e)}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback_str
        }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)

    original_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        code_file = sys.argv[1]
        problem_data_file = sys.argv[2]
        result = run_solution_code(code_file, problem_data_file)

        if not isinstance(result, dict):
            result = {"success": False, "error": "Non-dict result from solution code"}

        result["stdout"] = captured_output.getvalue()

        sys.stdout = original_stdout
        
        print(json.dumps(result, ensure_ascii=False, cls=NumpyEncoder))
        
    except Exception as e:

        sys.stdout = original_stdout

        print(json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "stdout": captured_output.getvalue()
        }, ensure_ascii=False, cls=NumpyEncoder))
""")
        return f.name

    def _extract_simple_data(self, data):
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self._extract_simple_data(value)
            return result
        elif isinstance(data, list):
            return [self._extract_simple_data(item) for item in data]
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)

    def _simplify_result(self, result):
        simplified = {
            'success': True,
            'profit_objective': None,
            'solution': {},
            'evaluation': {},
            'statistics': {},
            'violation_analysis': {}
        }

        stdout = result.get('stdout', '')
        match = re.search(r'fitness:\s*(-?\d+\.\d+)', stdout)
        if match:
            profit_value = float(match.group(1))
            simplified['profit_objective'] = profit_value

        for field in ['profit_objective', 'objective']:
            if field in result and result[field] is not None:
                simplified['profit_objective'] = result[field]
                break

        if simplified['profit_objective'] is None and 'evaluation' in result:
            if isinstance(result['evaluation'], dict) and 'objective' in result['evaluation']:
                simplified['profit_objective'] = result['evaluation']['objective']

        if simplified['profit_objective'] is None and 'statistics' in result:
            if isinstance(result['statistics'], dict) and 'best_objective' in result['statistics']:
                simplified['profit_objective'] = result['statistics']['best_objective']

        if 'violation_analysis' in result:
            simplified['violation_analysis'] = result['violation_analysis']

        for field in ['has_violations', 'violation_types', 'total_violation_types', 'total_violation_degree']:
            if field in result:
                simplified[field] = result[field]
        if 'metrics' in result and isinstance(result['metrics'], dict):
            metrics = result['metrics']
            if not simplified.get('violation_analysis') and metrics.get('has_violations'):
                simplified['violation_analysis'] = {
                    'constraint_violations': {
                        'violated': True,
                        'violations': metrics.get('violations', []),
                        'constraint_violations': metrics.get('constraint_violations', 0)
                    }
                }
                simplified['has_violations'] = metrics.get(
                    'has_violations', False)
                simplified['violation_types'] = metrics.get('violations', [])
                simplified['total_violation_types'] = len(
                    metrics.get('violations', []))

        for key in ['solution', 'evaluation', 'statistics']:
            if key in result and isinstance(result[key], dict):
                simplified[key] = self._extract_simple_data(result[key])

        if 'solution' in result and isinstance(result['solution'], dict):

            simplified['solution'] = result['solution']
        return simplified
