from typing import Dict, List, Any, Optional
import json
import os
import copy
from datetime import datetime
from repair_agent import RepairAgent
from output_manager import OutputManager


class RepairController:
    def __init__(self, repair_agent: RepairAgent,
                 output_manager: OutputManager,
                 config: Dict):
        self.repair_agent = repair_agent
        self.output_manager = output_manager
        self.config = config
        self.max_repair_attempts = config.get('max_repair_attempts', 3)
        self.repair_history = {}

    def should_trigger_repair(self, evaluation_result: Dict, strategy_id: str) -> bool:
        if not self.config.get('enabled', False):
            self.output_manager.log_debug(
                "repair_controller", "repair_disabled",
                f"Repair mechanism not enabled, skipping strategy {strategy_id}"
            )
            return False
        violation_analysis = evaluation_result.get('violation_analysis', {})
        has_violations = violation_analysis.get('has_violations', False)
        if not has_violations:
            self.output_manager.log_debug(
                "repair_controller", "no_violations",
                f"Strategy {strategy_id} has no constraint violations"
            )
            return False

        repair_count = self.repair_history.get(strategy_id, 0)
        if repair_count >= self.max_repair_attempts:
            self.output_manager.log_info(
                "repair_controller", "repair_limit_reached",
                f"Strategy {strategy_id} has reached maximum repair attempts {self.max_repair_attempts}, will be marked as repair completed"
            )
            return False
        return True

    def execute_repair_cycle(self, strategy_id: str, strategy_data: Dict,
                             violation_analysis: Dict) -> Optional[Dict]:
        self.repair_history[strategy_id] = self.repair_history.get(
            strategy_id, 0) + 1

        repaired_data = self.repair_agent.repair_strategy_and_code(
            strategy_data, violation_analysis
        )

        if repaired_data:
            updated_strategy = copy.deepcopy(strategy_data)
            for key, value in repaired_data.items():
                if value is not None:
                    updated_strategy[key] = value
            if 'relaxation_strategy' in repaired_data:
                relaxation_factors = {}
                for item in repaired_data['relaxation_strategy']:
                    if isinstance(item, dict) and 'constraint' in item and 'relaxation' in item:
                        relaxation_factors[item['constraint']
                                           ] = item['relaxation']
                updated_strategy['relaxation_factors'] = relaxation_factors

                if 'constraint_order' not in repaired_data:
                    constraint_order = []
                    for item in repaired_data['relaxation_strategy']:
                        if isinstance(item, dict) and 'constraint' in item:
                            constraint_order.append(item['constraint'])
                    updated_strategy['constraint_order'] = constraint_order
            if 'generated_code' in repaired_data and repaired_data['generated_code']:
                updated_strategy['code_snippet'] = repaired_data['generated_code']
            debug_info = self.repair_agent.debug_repair_process(
                strategy_id, updated_strategy)
            if debug_info.get('issues'):
                self.output_manager.log_warning(
                    "repair_controller", "repair_debug_issues",
                    f"Issues found during repair process: {', '.join(debug_info['issues'])}"
                )

            self.update_strategy_file(strategy_id, updated_strategy)
            repaired_code_file_path = self.repair_agent.generate_repaired_complete_code_file(
                strategy_id, updated_strategy
            )

            if repaired_code_file_path:
                updated_strategy['repaired_code_file_path'] = repaired_code_file_path
            else:
                self.output_manager.log_warning(
                    "repair_controller", "repaired_code_file_failed",
                    f"Failed to generate complete code file after repairing strategy {strategy_id}"
                )
            return updated_strategy
        else:
            self.output_manager.log_warning(
                "repair_controller", "repair_failed",
                f"Strategy {strategy_id} repair failed"
            )
            return None

    def update_strategy_file(self, strategy_id: str, updated_strategy: Dict):
        try:
            strategy_file_path = self.find_strategy_file(strategy_id)
            if strategy_file_path:
                with open(strategy_file_path, 'r', encoding='utf-8') as f:
                    strategy_file = json.load(f)
                if 'relaxation_strategy' in updated_strategy:
                    relaxation_factors = {}
                    for item in updated_strategy['relaxation_strategy']:
                        if isinstance(item, dict) and 'constraint' in item and 'relaxation' in item:
                            relaxation_factors[item['constraint']
                                               ] = item['relaxation']
                    strategy_file['relaxation_factors'] = relaxation_factors
                if 'text' in updated_strategy:
                    strategy_file['text'] = updated_strategy['text']
                if 'generated_code' in updated_strategy and updated_strategy['generated_code']:
                    strategy_file['code_snippet'] = updated_strategy['generated_code']
                if 'constraint_order' in updated_strategy:
                    strategy_file['constraint_order'] = updated_strategy['constraint_order']
                elif 'relaxation_strategy' in updated_strategy:
                    constraint_order = []
                    for item in updated_strategy['relaxation_strategy']:
                        if isinstance(item, dict) and 'constraint' in item:
                            constraint_order.append(item['constraint'])
                    strategy_file['constraint_order'] = constraint_order
                strategy_file['repair_info'] = {
                    'repaired': True,
                    'repair_time': datetime.now().isoformat(),
                    'repair_count': self.repair_history.get(strategy_id, 0)
                }
                if 'repaired_code_file_path' in updated_strategy:
                    strategy_file['repaired_code_file_path'] = updated_strategy['repaired_code_file_path']

                with open(strategy_file_path, 'w', encoding='utf-8') as f:
                    json.dump(strategy_file, f, indent=2, ensure_ascii=False)

                updated_fields = []
                if 'relaxation_strategy' in updated_strategy:
                    updated_fields.append('relaxation_factors')
                if 'text' in updated_strategy:
                    updated_fields.append('text')
                if 'generated_code' in updated_strategy and updated_strategy['generated_code']:
                    updated_fields.append('code_snippet')
                if 'constraint_order' in updated_strategy:
                    updated_fields.append('constraint_order')

                self.output_manager.log_debug(
                    "repair_controller", "fields_updated",
                    f"Strategy {strategy_id} updated fields: {updated_fields}"
                )
                self._update_in_memory_strategy(strategy_id, strategy_file)

            else:
                self.output_manager.log_warning(
                    "repair_controller", "file_not_found",
                    f"Strategy file not found: {strategy_id}"
                )

        except Exception as e:
            self.output_manager.log_error(
                "repair_controller", "file_update_failed",
                f"Failed to update strategy file: {str(e)}"
            )

    def find_strategy_file(self, strategy_id: str) -> Optional[str]:
        for root, dirs, files in os.walk(self.output_manager.base_dir):
            for file in files:
                if file.endswith('.json') and strategy_id in file:
                    return os.path.join(root, file)
        return None
    def _update_in_memory_strategy(self, strategy_id: str, updated_strategy_data: Dict):

        try:
            if hasattr(self, 'code_generator') and hasattr(self.code_generator, 'problem_info'):
                if 'strategies' in self.code_generator.problem_info:
                    if strategy_id in self.code_generator.problem_info['strategies']:
                        memory_strategy = self.code_generator.problem_info['strategies'][strategy_id]
                        if 'code_snippet' in updated_strategy_data:
                            memory_strategy['code_snippet'] = updated_strategy_data['code_snippet']
                        if 'relaxation_factors' in updated_strategy_data:
                            memory_strategy['relaxation_factors'] = updated_strategy_data['relaxation_factors']
                        if 'text' in updated_strategy_data:
                            memory_strategy['text'] = updated_strategy_data['text']
                        if 'constraint_order' in updated_strategy_data:
                            memory_strategy['constraint_order'] = updated_strategy_data['constraint_order']
                        if 'repair_info' in updated_strategy_data:
                            memory_strategy['repair_info'] = updated_strategy_data['repair_info']
                    else:
                        self.code_generator.problem_info['strategies'][strategy_id] = updated_strategy_data
                else:
                    self.code_generator.problem_info['strategies'] = {
                        strategy_id: updated_strategy_data}
            else:
                self.output_manager.log_warning(
                    "repair_controller", "no_code_generator",
                    f"Cannot update strategy {strategy_id} in memory because code_generator or problem_info is missing"
                )
        except Exception as e:
            self.output_manager.log_error(
                "repair_controller", "memory_update_failed",
                f"Failed to update strategy in memory: {str(e)}"
            )

    def set_code_generator(self, code_generator):
        self.code_generator = code_generator

    def is_repair_completed(self, strategy_id: str) -> bool:

        repair_count = self.repair_history.get(strategy_id, 0)
        return repair_count >= self.max_repair_attempts
