from openai import OpenAI
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from output_manager import OutputManager
from colorama import Fore, Style, init
import json
import time
import os
import re
import traceback
import random
import copy
init()


class LLMClient:

    def __init__(self,
                 config: Dict = None,
                 client_type: str = "default",
                 output_manager: Optional[OutputManager] = None):
        self.output_manager = output_manager or OutputManager(
            base_dir="outputs")
        self.config = config or self._load_default_config()
        self.client_type = client_type
        default_config = self.config.get("default", {})
        self.max_retries = default_config.get("max_retries", 5)
        self.retry_delay = default_config.get("retry_delay", 10)
        self.retry_backoff = default_config.get("retry_backoff", 2)
        self.providers = self.config.get("providers", [])
        if not self.providers:
            raise ValueError("API provider configuration not found")

        type_config = self.config.get(client_type, {})
        self.temperature = type_config.get("temperature", 0.7)
        self.max_tokens = type_config.get("max_tokens", 4096)
        self.current_provider_index = 0
        self.current_api_key_index = -1  # -1 means using the main API key
        self._activate_current_provider()
        self.log_dir = os.path.join("logs", "api_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.last_call_time = 0
        self.min_interval = 10

        if self.output_manager:
            self.output_manager.log_info(
                "llm_client", "initialization",
                f"Initialized LLM client, type: {client_type}, provider: {self.provider_name}, model: {self.model}"
            )

    def _load_default_config(self) -> Dict:

        try:
            config_path = "config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                if "default" in config:
                    self.min_interval = config["default"].get(
                        "min_call_interval", 10)
                return config.get("llm", {})
        except Exception as e:
            self.output_manager.log_error(
                "llm_client", "config_load_error",
                f"config_load_error: {str(e)}"
            )
            return {}

    def _activate_current_provider(self):

        if self.current_provider_index >= len(self.providers):
            raise ValueError("All API providers failed")

        provider = self.providers[self.current_provider_index]

        self.provider_name = provider.get(
            "name", f"provider_{self.current_provider_index}")
        self.base_url = provider.get("base_url")

        if self.current_api_key_index < 0:
            self.api_key = provider.get("api_key")
        else:
            backup_keys = provider.get("backup_api_keys", [])
            if self.current_api_key_index >= len(backup_keys):
                raise ValueError(
                    f"provider {self.provider_name} backup_api_keys all failed")
            self.api_key = backup_keys[self.current_api_key_index]

        models = provider.get("models", {})
        self.model = models.get(self.client_type, models.get(
            "default", "deepseek/deepseek-v3"))

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        self.default_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 1.0,
            "stream": False
        }

    def _switch_to_next_api_key(self):

        self.current_api_key_index += 1

        try:

            self._activate_current_provider()

            self.output_manager.log_warning(
                "llm_client", "api_key_switched",
                f"switch to {self.provider_name} backup api keys {self.current_api_key_index+1}"
            )
            return True

        except ValueError:
            return self._switch_to_next_provider()

    def _switch_to_next_provider(self):
        self.current_provider_index += 1
        self.current_api_key_index = -1

        try:
            self._activate_current_provider()

            self.output_manager.log_warning(
                "llm_client", "provider_switched",
                f"switch_to_next_provider: {self.provider_name}"
            )
            return True

        except ValueError:

            self.output_manager.log_error(
                "llm_client", "all_providers_failed",
                "all_providers_failed"
            )
            return False

    def _wait_for_rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_call_time

        if elapsed < self.min_interval and self.last_call_time > 0:
            wait_time = self.min_interval - elapsed
            time.sleep(wait_time)

        self.last_call_time = time.time()

    def chat_completion(self, messages: List[Dict[str, str]], params: Dict[str, Any] = None, stream: bool = False):

        self._wait_for_rate_limit()
        merged_params = {**self.default_params, **(params or {})}
        merged_params["stream"] = stream

        request_params = {
            "model": self.model,
            "messages": messages,
            **merged_params
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        request_log_file = os.path.join(
            self.log_dir, f"llm_request_{timestamp}.txt")

        try:
            with open(request_log_file, 'w', encoding='utf-8') as f:
                f.write(f"Model: {self.model}\n")
                f.write(f"Provider: {self.provider_name}\n")
                f.write(
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(
                    f"Params: {json.dumps(merged_params, ensure_ascii=False)}\n\n")
                f.write("=== MESSAGES ===\n\n")
                for idx, msg in enumerate(messages):
                    f.write(
                        f"[{idx}] {msg['role'].upper()}:\n{msg['content']}\n\n")
        except Exception as e:
            print(f"wrong: {str(e)}")

        start_time = time.time()
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                if attempt > 0 and self.output_manager:
                    self.output_manager.log_warning(
                        "llm_client", "retry_attempt",
                        f"retry_attempt {attempt+1}/{self.max_retries} "
                    )

                response = self.client.chat.completions.create(
                    **request_params)

                if stream:

                    collected_content = []

                    for chunk in response:
                        if chunk.choices and len(chunk.choices) > 0:
                            if chunk.choices[0].delta.content:
                                content_piece = chunk.choices[0].delta.content
                                collected_content.append(content_piece)

                    full_content = "".join(collected_content)
                    response_log_file = os.path.join(
                        self.log_dir, f"llm_response_stream_{timestamp}.txt")
                    try:
                        with open(response_log_file, 'w', encoding='utf-8') as f:
                            f.write("=== REQUEST DETAILS ===\n\n")
                            f.write(f"Request file: {request_log_file}\n")
                            f.write(f"Provider: {self.provider_name}\n")
                            f.write(f"Model: {self.model}\n\n")
                            f.write("=== STREAM RESPONSE ===\n\n")
                            f.write(full_content)
                    except Exception as e:
                        print(f"Error saving stream response log: {str(e)}")

                    if self.output_manager:
                        response_time = time.time() - start_time
                    return full_content
                else:
                    content = response.choices[0].message.content
                    if self.output_manager:
                        response_time = time.time() - start_time

                    response_log_file = os.path.join(
                        self.log_dir, f"llm_response_{timestamp}.txt")
                    try:
                        with open(response_log_file, 'w', encoding='utf-8') as f:
                            f.write("=== REQUEST DETAILS ===\n\n")
                            f.write(f"Request file: {request_log_file}\n")
                            f.write(f"Provider: {self.provider_name}\n")
                            f.write(f"Model: {self.model}\n\n")
                            for idx, msg in enumerate(messages):
                                f.write(
                                    f"[{idx}] {msg['role'].upper()}:\n{msg['content']}\n\n")
                            f.write("=== RESPONSE ===\n\n")
                            f.write(content)
                            f.write("\n\n=== RESPONSE METADATA ===\n\n")
                            f.write(f"Model: {response.model}\n")
                            f.write(f"Created: {response.created}\n")
                            f.write(
                                f"Response time: {time.time() - start_time:.2f} seconds\n")
                            f.write(
                                f"Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}\n")
                    except Exception as e:
                        print(f"Error saving response log: {str(e)}")

                    print(
                        f"\nFull LLM response saved to file:\n {response_log_file}\n")

                    return content

            except Exception as e:
                last_exception = e
                error_msg = str(e)
                traceback_str = traceback.format_exc()

                if self.output_manager:
                    self.output_manager.log_error(
                        "llm_client", "api_error",
                        f"api_error (attempt {attempt+1}/{self.max_retries}): {error_msg}"
                    )
                error_log_file = os.path.join(
                    self.log_dir, f"llm_error_{timestamp}_{attempt}.txt")
                try:
                    with open(error_log_file, 'w', encoding='utf-8') as f:
                        f.write("=== REQUEST DETAILS ===\n\n")
                        f.write(f"Request file: {request_log_file}\n")
                        f.write(f"Provider: {self.provider_name}\n")
                        f.write(f"Model: {self.model}\n\n")
                        for idx, msg in enumerate(messages):
                            f.write(
                                f"[{idx}] {msg['role'].upper()}:\n{msg['content']}\n\n")
                        f.write(f"Error: {error_msg}\n\n")
                        f.write(f"Traceback:\n{traceback_str}\n")
                except Exception as log_error:
                    print(f"Error saving error log: {str(log_error)}")
                if any(keyword in error_msg.lower() for keyword in
                       ["rate_limit", "capacity", "quota", "error code: 400", "api error"]):
                    if self.output_manager:
                        self.output_manager.log_warning(
                            "llm_client", "api_limit_detected",
                            f"API error detected: {error_msg}, attempting to switch API key/provider"
                        )

                    if not self._switch_to_next_api_key():
                        self.output_manager.log_error(
                            "llm_client", "no_more_providers",
                            "No more available API keys or providers, terminating retries"
                        )
                        break
                    request_params["model"] = self.model
                    continue
                else:

                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * \
                            (self.retry_backoff ** attempt)

                        if self.output_manager:
                            self.output_manager.log_info(
                                "llm_client", "retry_delay",
                                f"Waiting for {delay:.2f} seconds before retrying..."
                            )
                        time.sleep(delay)

        error_message = f"All API call attempts failed: {str(last_exception)}"
        if self.output_manager:
            self.output_manager.log_error(
                "llm_client", "api_all_retries_failed", error_message)

        raise Exception(error_message)

    def extract_content(self, response: str) -> dict:
        result = {
            "thinking": "",
            "code": "",
            "raw": response
        }

        thinking_match = re.search(
            r'<think>(.*?)</think>', response, re.DOTALL)
        if thinking_match:
            result["thinking"] = thinking_match.group(1).strip()

        code_match = re.search(r'```python(.*?)```', response, re.DOTALL)
        if code_match:
            result["code"] = code_match.group(1).strip()
        else:
            general_code_match = re.search(r'```(.*?)```', response, re.DOTALL)
            if general_code_match:
                result["code"] = general_code_match.group(1).strip()

        if not result["code"]:
            code_lines = []
            in_code_block = False
            for line in response.split('\n'):
                line_stripped = line.strip()
                if (line_stripped.startswith('def ') or
                    line_stripped.startswith('class ') or
                    line_stripped.startswith('import ') or
                        line_stripped.startswith('from ')):
                    in_code_block = True
                    code_lines.append(line)
                elif in_code_block:
                    code_lines.append(line)

            if code_lines:
                result["code"] = '\n'.join(code_lines)

        return result

    def generate_text(self,
                      prompt: str,
                      system_message: str = None,
                      params: Dict[str, Any] = None) -> str:

        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        return self.chat_completion(messages, params)

    def generate_code(self,
                      prompt: str,
                      language: str = "python",
                      context: str = None,
                      params: Dict[str, Any] = None) -> str:
        code_params = {
            "temperature": 0.2,
            "max_tokens": 4000,
            "top_p": 1.0,
            "response_format": {"type": "text"}
        }

        if params:
            code_params.update(params)

        system_message = (
            f"You are an expert {language} programmer. "
            "Generate clean, efficient, and well-commented code based on the requirements. "
            "Focus on correctness and readability."
        )

        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nRequirements:\n{prompt}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": full_prompt}
        ]
        return self.chat_completion(messages, code_params)

    def _build_format_instructions(self) -> str:

        return """
    Please strictly follow the format requirements below for your answer:

    1. If you need to think about the problem, put your thought process inside <think></think> tags
    2. Code or final output must be placed between ```python and ```
    3. Do not provide any content outside the above markers

    For example:

    <think>
    Here is my thinking process...
    Analyzed the characteristics of the problem...
    </think>

    ```python
    # This is the actual output code or content
    def solution():
        return "Final result"
    ```
    """

    def call(self, system_prompt: str, user_prompt: str, temperature: float = None) -> str:
        format_instructions = self._build_format_instructions()
        enhanced_user_prompt = f"{user_prompt}\n\n{format_instructions}"

        temp = temperature if temperature is not None else self.default_params["temperature"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_user_prompt}
        ]

        params = {**self.default_params, "temperature": temp}

        return self.chat_completion(messages, params)

    def call_json(self, system_prompt: str, user_prompt: str, temperature: float = None) -> Dict:

        format_instructions = self._build_format_instructions()
        enhanced_user_prompt = f"{user_prompt}\n\n{format_instructions}\n\nPlease ensure your response is a valid JSON object and make sure it is enclosed within ```python and ``` tags."

        response = self.call(system_prompt, enhanced_user_prompt, temperature)
        if hasattr(response, '__next__') or hasattr(response, '__iter__'):
            collected_response = "".join([chunk for chunk in response])
            response = collected_response

        response_cleaned = re.sub(r'<think>[\s\S]*?</think>', '', response)
        code_block_patterns = [
            r'```(?:python|json)?\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'`\s*([\s\S]*?)\s*`'
        ]

        for pattern in code_block_patterns:
            matches = re.findall(pattern, response_cleaned)
            for match in matches:
                cleaned_match = match.strip()
                try:
                    return json.loads(cleaned_match)
                except json.JSONDecodeError:
                    pass

        try:
            return json.loads(response_cleaned)
        except json.JSONDecodeError:
            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            matches = re.findall(json_pattern, response_cleaned)

            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            result = {}
            pattern = r'["\']?([\w\s]+)["\']?\s*[:：]\s*["\']?([\w\s\.]+)["\']?'
            matches = re.findall(pattern, response_cleaned)

            if matches:
                for key, value in matches:
                    key = key.strip()
                    value = value.strip()
                    if value.isdigit():
                        value = int(value)
                    elif re.match(r'^-?\d+(\.\d+)?$', value):
                        value = float(value)
                    result[key] = value

                if self.output_manager:
                    self.output_manager.log_warning(
                        "llm_client", "json_extraction",
                        "Extracted key-value pairs from text instead of full JSON"
                    )
                return result
            if self.output_manager:
                self.output_manager.log_error(
                    "llm_client", "json_parse_error",
                    "Failed to parse the JSON content, returning the original text."
                )

                if hasattr(self, 'log_dir'):
                    debug_file = os.path.join(
                        self.log_dir, f"json_parse_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    try:
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write("=== ORIGINAL RESPONSE ===\n\n")
                            f.write(response)
                            f.write("\n\n=== CLEANED RESPONSE ===\n\n")
                            f.write(response_cleaned)
                            f.write("\n\n=== EXTRACTION ATTEMPTS ===\n\n")
                            f.write("Code block extraction attempts:\n")
                            for pattern in code_block_patterns:
                                matches = re.findall(pattern, response_cleaned)
                                for i, match in enumerate(matches):
                                    f.write(f"Pattern {pattern}, Match {i}:\n")
                                    f.write(f"{match[:200]}...\n\n")

                    except Exception as e:
                        pass
            return {"raw_response": response}


class StrategyLLMClient(LLMClient):
    def __init__(self, config: Dict = None, output_manager: Optional[OutputManager] = None, algorithm_loader=None):
        super().__init__(config, "strategy", output_manager)
        self.algorithm_loader = algorithm_loader
        self.default_params["temperature"] = 0.8

    def generate_initial_strategy(self, problem_info: Dict,
                                  strategy_count: int = 5,
                                  best_code_snippet: str = None,
                                  problem_data: Dict = None) -> List[str]:
        system_prompt = """
        You are a strategy expert for combinatorial optimization problems. Your task is to create text strategies to solve the given optimization problem.
        The text strategy should describe how to handle hard constraints in the problem, including the order of constraint processing and relaxation factors for each constraint.
        
        Please follow the following format to return the result:

        ```json
        {
            "strategies": [
                {
                    "id": "Strategy1",
                    "steps": [
                        {"constraint": "constraint_name1", "relaxation": 1.9},
                        {"constraint": "constraint_name2", "relaxation": 3.0},
                        {"constraint": "constraint_name3", "relaxation": 2.1}
                    ]
                },
                {
                    "id": "Strategy2",
                    "steps": [
                        {"constraint": "constraint_name1", "relaxation": 4.1},
                        {"constraint": "constraint_name2", "relaxation": 2.9},
                        {"constraint": "constraint_name3", "relaxation": 5.0}
                    ]
                }
            ]
        }
        ```
        
        Please ensure that the generated strategies are practical and innovative. Each strategy must clearly specify the constraint processing order and relaxation factors.
        Please use the <think> tag to record your thinking process, analyze statistical data, and explore directions before generating strategies.
        """

        mcts_hints = problem_info.get("mcts_exploration_hints", [])
        mcts_stats = problem_info.get("mcts_statistics", {})
        mcts_prompt = ""
        if mcts_stats:
            mcts_prompt += "\n\nMonte Carlo Tree Search statistical analysis results:\n"
            mcts_prompt += f"- Total tree search nodes: {mcts_stats.get('total_nodes', 0)}\n"
            mcts_prompt += f"- Best reward value: {mcts_stats.get('best_reward', 0)}\n\n"

            constraint_freq = mcts_stats.get("constraint_frequency", {})
            if constraint_freq:
                mcts_prompt += "Constraint usage frequency statistics:\n"
                for constraint, freq in constraint_freq.items():
                    mcts_prompt += f"- {constraint}: {freq} times\n"
                mcts_prompt += "\n"

            relaxation_stats = mcts_stats.get("relaxation_stats", {})
            if relaxation_stats:
                mcts_prompt += "Constraint relaxation factor statistics:\n"
                for constraint, stats in relaxation_stats.items():
                    weighted_avg = stats.get("weighted_average", "unknown")
                    mcts_prompt += f"- {constraint} weighted average: {weighted_avg}\n"
                mcts_prompt += "\n"
        if mcts_hints:
            exploitation_paths = [h for h in mcts_hints if h.get(
                "path_type", h.get("type", "")) == "exploitation"]
            exploration_paths = [h for h in mcts_hints if h.get(
                "path_type", h.get("type", "")) == "exploration"]

            print(
                f"Processing MCTS hints: Found {len(exploitation_paths)} exploitation paths, {len(exploration_paths)} exploration paths")

            if exploitation_paths:
                mcts_prompt += "\nVerified high-reward paths (reliable solutions):\n"
                for i, hint in enumerate(exploitation_paths):
                    mcts_prompt += f"Path {i+1}:\n"
                    constraint_order = hint.get("constraint_order", [])
                    if constraint_order:
                        mcts_prompt += f"- Constraint processing order: {', '.join(constraint_order)}\n"
                    relaxation_factors = hint.get("relaxation_factors", {})
                    if relaxation_factors:
                        mcts_prompt += "- Relaxation factors:\n"
                        for constraint, factor in relaxation_factors.items():
                            mcts_prompt += f"  * {constraint}: {factor}\n"

                    avg_reward = hint.get("reward", 0)
                    visits = hint.get("visits", 0)
                    mcts_prompt += f"- Average reward: {avg_reward}\n"
                    mcts_prompt += f"- Visit count: {visits}\n\n"

            if exploration_paths:
                mcts_prompt += "\nPaths with exploratory potential (innovative directions):\n"
                for i, hint in enumerate(exploration_paths):
                    mcts_prompt += f"Direction {i+1}:\n"
                    constraint_order = hint.get("constraint_order", [])
                    if constraint_order:
                        mcts_prompt += f"- Constraint processing order: {', '.join(constraint_order)}\n"
                    relaxation_factors = hint.get("relaxation_factors", {})
                    if relaxation_factors:
                        mcts_prompt += "- Relaxation factors:\n"
                        for constraint, factor in relaxation_factors.items():
                            mcts_prompt += f"  * {constraint}: {factor}\n"
                    avg_reward = hint.get("reward", 0)
                    visits = hint.get("visits", 0)
                    mcts_prompt += f"- Average reward: {avg_reward}\n"
                    mcts_prompt += f"- Visit count: {visits} (lower visit count indicates exploratory potential)\n\n"

            if not exploitation_paths and not exploration_paths:
                mcts_prompt += "Potential exploratory directions identified by MCTS:\n"
                for i, hint in enumerate(mcts_hints):
                    mcts_prompt += f"Direction {i+1}:\n"
                    constraint_order = hint.get("constraint_order", [])
                    if constraint_order:
                        mcts_prompt += f"- Constraint processing order: {', '.join(constraint_order)}\n"
                    relaxation_factors = hint.get("relaxation_factors", {})
                    if relaxation_factors:
                        mcts_prompt += "- Relaxation factors:\n"
                        for constraint, factor in relaxation_factors.items():
                            mcts_prompt += f"  * {constraint}: {factor}\n"
                    avg_reward = hint.get("reward", 0)
                    mcts_prompt += f"- Average reward: {avg_reward}\n\n"

        if mcts_stats or mcts_hints:
            mcts_prompt += """
        Please analyze the above statistical data and exploration directions, paying special attention to constraint usage frequency patterns and relaxation factor patterns.
        Do not simply copy the provided exploration directions, but create new, diverse strategy combinations based on statistical analysis.
        """
        user_prompt = f"""
        Please generate {strategy_count} different text strategies for the following combinatorial optimization problem, using JSON format:
        
        Problem type: {problem_info.get('problem_type', 'general_optimization')}
        
        Hard constraints:
        {json.dumps(problem_info.get('hard_constraints', []), indent=2, ensure_ascii=False)}
        
        Constraint information:
        {json.dumps(problem_info.get('constraint_info', {}), indent=2, ensure_ascii=False)}
        
        Possible relaxation factors:
        {json.dumps(problem_info.get('relaxation_info', {}), indent=2, ensure_ascii=False)}
        
        Objective functions:
        {json.dumps(problem_info.get('objective_functions', []), indent=2, ensure_ascii=False)}
        {mcts_prompt}
        
        Return only the strategies in JSON format, without any explanations or additional text.
        """
        response_obj = self.call1(system_prompt, user_prompt)

        response = ""
        try:
            if hasattr(response_obj, '__iter__') and not isinstance(response_obj, str):
                response_parts = []
                part_count = 0
                for part in response_obj:
                    part_count += 1
                    if part:
                        response_parts.append(
                            part if isinstance(part, str) else str(part))

                response = "".join(response_parts)
                print(
                    f"A total of {part_count} parts were collected, with a combined total length of {len(response)}.")

                if response:
                    print(f"LLM Response:\n{response[:300]}...")
                else:
                    print("The merged response is empty.")
            elif isinstance(response_obj, str):
                response = response_obj
            else:
                response = str(response_obj)
        except Exception as e:
            print(f"response_processing_error: {str(e)}")
            import traceback
            traceback.print_exc()
            response = ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            "logs", "api_logs", f"llm_debug_response_{timestamp}.txt")
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=== RESPONSE TYPE ===\n")
                f.write(str(type(response_obj)))
                f.write("\n\n=== FULL RESPONSE CONTENT ===\n")
                f.write(response)
        except Exception as e:
            print(f"save_response_log_error: {str(e)}")

        hard_constraints = problem_info.get("hard_constraints", [])
        relaxation_info = problem_info.get("relaxation_info", {})

        default_strategies = []
        for i in range(strategy_count):
            template = f"Strategy: default_strategy_{i+1}\n"
            for j, constraint in enumerate(hard_constraints, 1):
                relaxation_options = relaxation_info.get(constraint, {}).get(
                    "possible_relaxations", [0.9, 1.0, 1.1])
                relaxation = random.choice(
                    relaxation_options) if relaxation_options else 1.0
                template += f"{j}. {constraint} Relaxation: {relaxation}\n"
            default_strategies.append(template)

        if not response.strip():
            print("The response is empty, returning default strategies.")
            return default_strategies

        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*?"strategies"\s*:\s*\[[\s\S]*?\]\s*\}'
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    if "strategies" in match:
                        data = json.loads(match)
                        if "strategies" in data and isinstance(data["strategies"], list):
                            strategies = []
                            for strategy in data["strategies"]:
                                strategy_id = strategy.get(
                                    "id", f"Strategy{len(strategies)+1}")
                                steps = strategy.get("steps", [])

                                strategy_text = f"Strategy: {strategy_id}\n"
                                for i, step in enumerate(steps, 1):
                                    constraint = step.get(
                                        "constraint", "unknown")
                                    relaxation = step.get("relaxation", 1.0)
                                    strategy_text += f"{i}. {constraint} Relaxation: {relaxation}\n"

                                strategies.append(strategy_text)
                            return strategies[:strategy_count]
                except json.JSONDecodeError:
                    continue
        return default_strategies

    def cross_strategies(self, parent_strategies: List[str],
                         problem_info: Dict,
                         parent_fitness: List[float] = None,
                         parent_details: List[Dict] = None,
                         system_prompt: str = None,
                         problem_data: Dict = None) -> str:
        import os
        import json
        if len(parent_strategies) < 2:
            return "Crossover cannot proceed; at least two parent strategies are required."

        parent_info = ""
        for i, (strategy, fitness) in enumerate(zip(parent_strategies, parent_fitness or [0, 0])):
            parent_info += f"Parent Strategy {i+1} (Fitness: {fitness}):\n{strategy}\n\n"
            if parent_details and i < len(parent_details) and 'code_snippet' in parent_details[i]:
                parent_info += f"Parent {i+1} Code Snippet:\n```python\n{parent_details[i]['code_snippet']}\n```\n\n"

        code_template = ""
        try:
            if hasattr(self, 'algorithm_loader'):
                available_algorithms = self.algorithm_loader.get_available_algorithms()
                if "PointSelectionAlgorithm" in available_algorithms:
                    code_template = self.algorithm_loader.get_template(
                        "PointSelectionAlgorithm")
                elif available_algorithms:
                    code_template = self.algorithm_loader.get_template(
                        available_algorithms[0])
            elif hasattr(self, 'llm_client') and hasattr(self.llm_client, 'algorithm_loader'):
                available_algorithms = self.llm_client.algorithm_loader.get_available_algorithms()
                if "PointSelectionAlgorithm" in available_algorithms:
                    code_template = self.llm_client.algorithm_loader.get_template(
                        "PointSelectionAlgorithm")
                elif available_algorithms:
                    code_template = self.llm_client.algorithm_loader.get_template(
                        available_algorithms[0])
        except Exception as e:
            if hasattr(self, 'output_manager'):
                self.output_manager.log_warning(
                    "text_strategy_manager", "template_error",
                    f"Failed to retrieve algorithm template during crossover: {str(e)}"
                )
        if not code_template:
            try:
                import os
                template_path = os.path.join(os.path.dirname(
                    __file__), "base_algorithm", "PointSelectionAlgorithm.py")
                if os.path.exists(template_path):
                    with open(template_path, 'r', encoding='utf-8') as f:
                        code_template = f.read()
            except Exception as e:
                pass

        code_info_prompt = ""
        if code_template:
            try:
                from code_analyzer import extract_code_information, get_function_signature_by_name, extract_llm_fill_area, format_code_information_for_prompt

                code_info = extract_code_information(code_template)
                select_next_point_signature = get_function_signature_by_name(
                    code_template, "select_next_point")
                llm_fill_area = extract_llm_fill_area(code_template)

                code_info_prompt = format_code_information_for_prompt(
                    code_info)

                if select_next_point_signature:
                    code_info_prompt += f"\nFunction Signature:\n```python\n{select_next_point_signature}\n```\n"

            except Exception as e:
                if hasattr(self, 'output_manager'):
                    self.output_manager.log_warning(
                        "text_strategy_manager", "code_analysis_error",
                        f"code_analysis_error: {str(e)}"
                    )

        if system_prompt is None:
            system_prompt = """
            You are an optimization algorithm expert. You need to cross two parent strategies to generate a new child strategy.
            
            Please carefully analyze the advantages and disadvantages of both parents, take the best from each, and create a new individual that combines their strengths.
            
            
            Please ensure that the generated child has innovative qualities and can solve the problem more effectively.
            """

        problem_data = {}
        problem_data_path = os.path.join(
            os.path.dirname(__file__), "problem_data.json")
        if os.path.exists(problem_data_path):
            try:
                with open(problem_data_path, 'r', encoding='utf-8') as f:
                    problem_data = json.load(f)
            except Exception as e:
                self.output_manager.log_warning(
                    "text_strategy_manager", "problem_data_error",
                    f"problem_data_error: {str(e)}"
                )

        limited_problem_data = self._limit_problem_data_points(
            problem_data, max_points=5)

        problem_data_json = json.dumps(
            limited_problem_data, indent=2) if limited_problem_data else "{}"

        user_prompt = f"""
        Parent Information:
        {parent_info}
    

        You must clearly follow the selected crossover strategy.
        
        Generate a child individual that includes three parts:
        1. Constraint relaxation strategy
        2. Algorithm design approach
        3. Code snippet
        
        Note: Please reference the function interfaces from the parent code to ensure your generated code has the same function signatures.
        If you need to introduce new functions or parameters, you must implement their complete functionality and definitions in your code snippet.
        
        Please use the following format:
        
        {{Constraint processing order and relaxation factors:
        1.constraint1:relaxation_factor1,
        2.constraint2:relaxation_factor2
        ...
        }}
        
        Algorithm design:
        ```json
        [Detailed algorithm design description]
        ```
        
        Code snippet:
        ```python
        [Python code implementation]
        ```
        Ensure that the constraint description and code are separate. The code section must be directly executable Python code without any Chinese comments or explanatory text.
        
        """
        return self.call(system_prompt, user_prompt)

    def mutate_strategy(self, strategy: str,
                        mutation_strength: float = 0.3,
                        problem_info: Dict = None,
                        strategy_fitness: float = None,
                        strategy_code: str = None,
                        system_prompt: str = None,
                        problem_data: Dict = None) -> str:

        import os
        import json

        if system_prompt is None:
            system_prompt = f"""
            You are an optimization algorithm expert. You need to mutate a parent strategy to generate a new child strategy.
            
            Please mutate the parent strategy appropriately based on the mutation strength {mutation_strength}. The higher the mutation strength, the greater the difference between the new strategy and the original strategy should be.
            
            The mutation should focus on any one or more of the following aspects:
                - Adjusting the constraint processing order
                - Modifying relaxation factor values
                - Improving algorithm design approach
                - Optimizing code implementation
            Please ensure the generated strategy is innovative and can solve the problem more effectively.
            """
        code_part = ""
        if strategy_code:
            code_part = f"Code snippet:\n```python\n{strategy_code}\n```\n"

        user_prompt = f"""
        Parent strategy (Fitness: {strategy_fitness or 'unknown'}):
        {strategy}
        
        {code_part}
        
        Note that your code cannot be identical to the parent individual's code.
        For all parameters and functions used, ensure their naming is consistent with the base algorithm framework. If you need to introduce new functions or parameters, you must implement their complete functionality and definitions in your code snippet.
        
        The generated mutated individual should include:
        1. Innovative constraint processing order and relaxation factors
        2. Improved algorithm design approach
        3. Code implementation reflecting mutation innovations
        
        Please use the following format to return the mutated individual:
        
        {{Constraint processing order and relaxation factors:
        1.constraint1:relaxation_factor1,
        2.constraint2:relaxation_factor2
        ...
        }}
        
        Algorithm design:
        ```json
        [Detailed algorithm design description]
        ```
        
        Code snippet:
        ```python
        [Python code implementation]
        ```
        Ensure that the constraint description and code are separate. The code section must be directly executable Python code without any non-English comments or explanatory text.
        """

        temperature = 0.5 + mutation_strength * 0.5
        return self.call(system_prompt, user_prompt, temperature)

    def generate_strategy_from_mcts(self, mcts_info: Dict,
                                    problem_info: Dict,
                                    outer_iteration: int = 0,
                                    use_llm: bool = True) -> str:

        import os
        import json

        problem_info = problem_info or {}
        problem_data = {}
        problem_data_path = os.path.join(
            os.path.dirname(__file__), "problem_data.json")
        if os.path.exists(problem_data_path):
            try:
                with open(problem_data_path, 'r', encoding='utf-8') as f:
                    problem_data = json.load(f)
            except Exception as e:
                if hasattr(self, 'output_manager'):
                    self.output_manager.log_warning(
                        "llm_client", "problem_data_error",
                        f"problem_data_error: {str(e)}"
                    )
        best_paths = mcts_info.get("best_paths", [])
        if not best_paths:
            return "# Error: MCTS failed to find a valid path"
        if not use_llm:
            return "# Error: LLM generation is not enabled, please implement manually"
        system_prompt = """You are an optimization strategy expert.
        Your task is to create an efficient strategy for combinatorial optimization problems based on the results of Monte Carlo Tree Search (MCTS).
        Please provide constraint processing strategies, detailed algorithm designs, and corresponding code implementations.
        Please follow the format below to return your answer:
        code must be compatible with the provided template functions, keeping the function names and parameters consistent, but the implementation logic should be optimized according to the constraint processing order and relaxation factors.
        
        Attention: Only implement the part marked as "LLM Filling Area" in the template, do not modify any other code."""

        problem_data_info = ""
        if problem_data:
            area_info = {}
            rectangles_info = []
            distance_costs_info = []

            simplified_data = {
                "problem_type": problem_data.get("problem_type", "safety_layout_problem"),
                "data_structure": {
                    "area": area_info,
                    "rectangles_sample": rectangles_info,
                    "distance_costs_sample": distance_costs_info
                },
                "parameters": problem_data.get("parameters", {})
            }

            problem_data_info = f"""
        data_structure:
        ```json
        {json.dumps(simplified_data, indent=2, ensure_ascii=False)}
        ``` """
        code_template = ""
        try:
            if hasattr(self, 'algorithm_loader'):
                available_algorithms = self.algorithm_loader.get_available_algorithms()
                if available_algorithms:
                    default_algo = "PointSelectionAlgorithm" if "PointSelectionAlgorithm" in available_algorithms else available_algorithms[
                        0]
                    code_template = self.algorithm_loader.get_template(
                        default_algo)
            if not code_template:
                template_path = os.path.join(os.path.dirname(
                    __file__), "base_algorithm", "PointSelectionAlgorithm.py")
                if os.path.exists(template_path):
                    with open(template_path, 'r', encoding='utf-8') as f:
                        code_template = f.read()
                        if hasattr(self, 'output_manager'):
                            self.output_manager.log_info(
                                "llm_client", "template_loaded",
                                f"Load templates directly from the file: {template_path}"
                            )
            if not code_template and "code_template" in problem_info:
                code_template = problem_info["code_template"]

            elif not code_template and "strategies" in problem_info and problem_info["strategies"]:
                best_strategy_id = None
                best_fitness = float('-inf')

                for strategy_id, strategy in problem_info["strategies"].items():
                    if strategy.get("fitness", float('-inf')) > best_fitness:
                        best_fitness = strategy.get("fitness", float('-inf'))
                        best_strategy_id = strategy_id
                if best_strategy_id and "code_snippet" in problem_info["strategies"][best_strategy_id]:
                    code_template = problem_info["strategies"][best_strategy_id]["code_snippet"]

            if not code_template:
                code_template = """
    import math
    from typing import Dict, List, Tuple

    class Solution:
        def __init__(self, problem_data: Dict, params: Dict = None):
            self.problem_data = problem_data
            self.params = params or {}
            self.constraint_order = [] 
            self.relaxation_factors = {}  
            
        def solve(self) -> Dict:

            delivery_points = self.problem_data.get("delivery_points", [])
            depot = self.problem_data.get("depot", {"x": 0, "y": 0})

            max_drones = self.params.get("MAX_DRONES", 3)
            max_payload = self.params.get("MAX_PAYLOAD", 50.0)
            max_battery = self.params.get("MAX_BATTERY", 1000.0)
            
            # ============ LLM Fill Area - Start ============
            # Implement an algorithm based on constraint processing order and relaxation factors

            # ============ LLM Fill Area - End ============

            solution = {
                "drone_routes": {},
                "unassigned_points": []
            }
            
            return solution
        """

        except Exception as e:
            code_template = """
    import math
    from typing import Dict, List, Tuple

    class Solution:
        def __init__(self, problem_data: Dict, params: Dict = None):
            self.problem_data = problem_data
            self.params = params or {}
            self.constraint_order = []  
            self.relaxation_factors = {} 
            
        def solve(self) -> Dict:
            solution = {
                "drone_routes": {},
                "unassigned_points": []
            }

            # ============ LLM Fill Area - Start ============
            # Implement an algorithm based on constraint processing order and relaxation factors

            # ============ LLM Fill Area - End ============

            return solution
    """
            if hasattr(self, 'output_manager'):
                self.output_manager.log_warning(
                    "strategy_llm_client", "template_error",
                    f"template_error: {str(e)}"
                )
        constraints_info = ""
        for constraint in problem_info.get("hard_constraints", []):
            constraint_desc = problem_info.get("constraint_info", {}).get(
                constraint, {}).get("description", "")
            constraints_info += f"- {constraint}: {constraint_desc}\n"

        parameters_info = ""
        if problem_info.get("parameters"):
            for name, value in problem_info.get("parameters", {}).items():
                if isinstance(value, dict):
                    parameters_info += f"- {name}: {value.get('value', 'N/A')} - {value.get('description', '')}\n"
                else:
                    parameters_info += f"- {name}: {value}\n"

        path_info = best_paths[0]
        constraint_order = path_info.get("constraint_order", [])
        relaxation_factors = path_info.get("relaxation_factors", {})

        user_prompt = f"""Please generate code implementation based on the following constraint order and relaxation factors for the given optimization problem:
        
        Problem description:
        {problem_info.get("description", "Optimization Problem")}
        Data situation:
        {problem_data_info}
        
        Constraint processing order and relaxation factors:
        """

        for i, constraint in enumerate(constraint_order):
            factor = relaxation_factors.get(constraint, 1.0)
            user_prompt += f"{i+1}. {constraint}: {factor}\n"

        user_prompt += f"""
        Constraints:
        {constraints_info}
        
        Problem parameters:
        {parameters_info}
        
        MCTS statistics:
        - Total nodes: {mcts_info.get("statistics", {}).get("total_nodes", 0)}
        - Maximum depth: {mcts_info.get("statistics", {}).get("max_depth", 0)}
        - Constraint frequency: {json.dumps(mcts_info.get("statistics", {}).get("constraint_frequency", {}), indent=2, ensure_ascii=False)}
        - Relaxation factor statistics: {json.dumps(mcts_info.get("statistics", {}).get("relaxation_stats", {}), indent=2, ensure_ascii=False)}
        
        Template function:
        ```python
    {code_template}
        ```
        
        Important instructions:
                1. You only need to fill in the code between "# ============ LLM Fill Area - Start" and "# ============ LLM Fill Area - End", do not modify other parts.
                2. Please ensure you include the complete function declaration (def select_next_point...), not just the function body code.
                3. Ensure the returned code can directly replace all content within the LLM fill area, and correctly handle constraint order and apply relaxation factors.
                4. All parameters and functions used should ensure their naming is consistent with the base algorithm framework. If you need to introduce new functions or parameters, you must implement their complete functionality and definitions in your code snippet.

        Please return in the following format (use English symbols):
        

        Code snippet:
        ```python
        # Use English comments, avoid using Chinese characters
        # The following code will be placed in the LLM Fill Area
        [Your Python code implementation]
        ```

        """

        import re

        response = self.call(system_prompt, user_prompt)
        code_pattern = r"```python\s*(.*?)\s*```"
        code_matches = re.findall(code_pattern, response, re.DOTALL)

        if code_matches:
            return code_matches[0]
        else:
            return response

    def _build_format_instructions(self) -> str:
        return """
    Please strictly follow these formatting requirements for your answer:

    1. Code or final output content must be placed between ```python and ```
    2. Strictly follow the template requirements, do not add content elsewhere
    """

    def call(self, system_prompt: str, user_prompt: str, temperature: float = None) -> str:
        format_instructions = self._build_format_instructions()
        enhanced_user_prompt = f"{user_prompt}\n\n{format_instructions}"

        temp = temperature if temperature is not None else self.default_params["temperature"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_user_prompt}
        ]

        params = {**self.default_params, "temperature": temp}

        return self.chat_completion(messages, params)

    def call1(self, system_prompt: str, user_prompt: str, temperature: float = None) -> str:
        enhanced_user_prompt = f"{user_prompt}\n"

        temp = temperature if temperature is not None else self.default_params["temperature"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_user_prompt}
        ]

        params = {**self.default_params, "temperature": temp}

        return self.chat_completion(messages, params)

    def call_json(self, system_prompt: str, user_prompt: str, temperature: float = None) -> Dict:
        format_instructions = self._build_format_instructions()
        enhanced_user_prompt = f"{user_prompt}\n\n{format_instructions}\n\nMake sure your response is a valid JSON object and is surrounded by ```python and ``` tags."

        response = self.call(system_prompt, enhanced_user_prompt, temperature)
        if hasattr(response, '__next__') or hasattr(response, '__iter__'):
            collected_response = "".join([chunk for chunk in response])
            response = collected_response
        response_cleaned = re.sub(r'<think>[\s\S]*?</think>', '', response)

        code_block_patterns = [
            r'```(?:python|json)?\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'`\s*([\s\S]*?)\s*`'
        ]
        for pattern in code_block_patterns:
            matches = re.findall(pattern, response_cleaned)
            for match in matches:
                cleaned_match = match.strip()
                try:
                    return json.loads(cleaned_match)
                except json.JSONDecodeError:
                    pass

        try:
            return json.loads(response_cleaned)
        except json.JSONDecodeError:
            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            matches = re.findall(json_pattern, response_cleaned)

            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            result = {}
            pattern = r'["\']?([\w\s]+)["\']?\s*[:：]\s*["\']?([\w\s\.]+)["\']?'
            matches = re.findall(pattern, response_cleaned)

            if matches:
                for key, value in matches:
                    key = key.strip()
                    value = value.strip()
                    if value.isdigit():
                        value = int(value)
                    elif re.match(r'^-?\d+(\.\d+)?$', value):
                        value = float(value)
                    result[key] = value

                if self.output_manager:
                    self.output_manager.log_warning(
                        "llm_client", "json_extraction",
                        "Extracted key-value pairs from text instead of full JSON"
                    )
                return result
            if self.output_manager:
                self.output_manager.log_error(
                    "llm_client", "json_parse_error",
                    "Failed to parse JSON content from LLM response, returning original text"
                )
                if hasattr(self, 'log_dir'):
                    debug_file = os.path.join(
                        self.log_dir, f"json_parse_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    try:
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write("=== ORIGINAL RESPONSE ===\n\n")
                            f.write(response)
                            f.write("\n\n=== CLEANED RESPONSE ===\n\n")
                            f.write(response_cleaned)
                            f.write("\n\n=== EXTRACTION ATTEMPTS ===\n\n")
                            f.write("Code block extraction attempts:\n")
                            for pattern in code_block_patterns:
                                matches = re.findall(pattern, response_cleaned)
                                for i, match in enumerate(matches):
                                    f.write(f"Pattern {pattern}, Match {i}:\n")
                                    f.write(f"{match[:200]}...\n\n")

                    except Exception as e:
                        pass

            return {"raw_response": response}

    def _limit_problem_data_points(self, problem_data: Dict, max_points: int = 5) -> Dict:
        if not problem_data:
            return {}
        limited_data = copy.deepcopy(problem_data)

        if "data" in limited_data and isinstance(limited_data["data"], list):
            for i, data_item in enumerate(limited_data["data"]):
                if isinstance(data_item, dict):
                    original_item = problem_data.get("data", [])[i] if i < len(
                        problem_data.get("data", [])) else {}
                    self._limit_points_in_dict(
                        data_item, max_points, original_item)

        for field_name in ["parameters", "constraints", "objectives"]:
            if field_name in limited_data and isinstance(limited_data[field_name], dict):
                original_field = problem_data.get(field_name, {})
                self._limit_points_in_dict(
                    limited_data[field_name], max_points, original_field)

        return limited_data

    def _limit_points_in_dict(self, data_dict: Dict, max_points: int, original_dict: Dict = None):
        if original_dict is None:
            original_dict = data_dict

        for field_name, field_value in list(data_dict.items()):
            if field_name in ["description", "type", "name", "id"]:
                continue

            if field_name.endswith("_limitation_note"):
                continue

            if isinstance(field_value, list) and len(field_value) > 0:
                original_value = original_dict.get(field_name, [])
                original_count = len(original_value)

                if self._is_object_list(field_value):
                    if len(field_value) > max_points:
                        data_dict[field_name] = field_value[:max_points]
                        note_key = f"_{field_name}_limitation_note"
                        data_dict[note_key] = f"Only the first {max_points} {field_name} are shown here, the actual data contains {original_count} items."

                elif self._is_matrix(field_value):
                    original_rows = len(original_value)
                    original_cols = len(original_value[0]) if original_value and len(
                        original_value) > 0 else 0

                    if len(field_value) > max_points:
                        data_dict[field_name] = field_value[:max_points]

                    for i, row in enumerate(data_dict[field_name]):
                        if isinstance(row, list) and len(row) > max_points:
                            data_dict[field_name][i] = row[:max_points]

                    if original_rows > max_points or original_cols > max_points:
                        note_key = f"_{field_name}_limitation_note"
                        data_dict[field_name +
                                  "_limitation_note"] = f"Only the first {max_points}x{max_points} of the {field_name} matrix are shown here, the actual matrix size is {original_rows}x{original_cols}."

                elif self._is_large_array(field_value, threshold=max_points * 2):
                    if len(field_value) > max_points:
                        data_dict[field_name] = field_value[:max_points]
                        note_key = f"_{field_name}_limitation_note"
                        data_dict[note_key] = f"To save tokens, only the first {max_points} {field_name} elements are shown here, the actual array contains {original_count} elements."

    def _is_object_list(self, data_list: list) -> bool:
        if not data_list:
            return False
        sample_size = min(3, len(data_list))
        dict_count = sum(
            1 for item in data_list[:sample_size] if isinstance(item, dict))
        return dict_count > 0

    def _is_matrix(self, data_list: list) -> bool:
        if not data_list:
            return False
        if not all(isinstance(item, list) for item in data_list):
            return False
        first_row = data_list[0]
        return len(first_row) > 0 and isinstance(first_row[0], (int, float))

    def _is_large_array(self, data_list: list, threshold: int = 10) -> bool:
        if not data_list or len(data_list) <= threshold:
            return False
        sample_size = min(3, len(data_list))
        simple_types = (int, float, str, bool)
        simple_count = sum(
            1 for item in data_list[:sample_size] if isinstance(item, simple_types))
        return simple_count == sample_size


class CodeLLMClient(LLMClient):

    def __init__(self, config: Dict = None, output_manager: Optional[OutputManager] = None):
        super().__init__(config, "code", output_manager)
        self.default_params["temperature"] = 0.2

    def generate_solution_code(self, strategy: str, problem_info: Dict,
                               code_paradigm: str = "general") -> str:

        system_prompt = f"""
        You are a Python expert specializing in combinatorial optimization problems. Your task is to generate Python solution code for {problem_info.get('problem_type', 'general optimization problem')} based on the given text strategy.
        You need to design code to implement the described strategy, ensuring the code can effectively solve the problem.
        
        Code requirements:
        1. The code should be complete and executable
        2. The code must follow the constraint processing order and relaxation factors in the text strategy
        3. The code must use the data structures provided in the problem information
        4. The code should use readable variable names and sufficient comments
        5. The code should ultimately return a solution to the optimization problem
        
        Return the code in a Python code block.
        """

        user_prompt = f"""
        Please generate solution code based on the following text strategy and problem information:
        
        Text strategy:
        {strategy}
        
        Problem type: {problem_info.get('problem_type', 'general optimization problem')}
        
        Problem data structures:
        {json.dumps(problem_info.get('data_structures', {}), indent=2, ensure_ascii=False)}
        
        Hard constraints:
        {json.dumps(problem_info.get('hard_constraints', []), indent=2, ensure_ascii=False)}
        
        Constraint information:
        {json.dumps(problem_info.get('constraint_info', {}), indent=2, ensure_ascii=False)}
        
        Objective functions:
        {json.dumps(problem_info.get('objective_functions', []), indent=2, ensure_ascii=False)}
        
        Please generate complete Python code that implements the solution process for this strategy.
        """

        response = self.call(system_prompt, user_prompt)
        extracted = self.extract_content(response)
        code = extracted["code"]
        self.output_manager.save_solution_code(
            code, f"{code_paradigm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        return extracted["code"]

    def _extract_code_from_response(self, response: str) -> str:
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response

        return code

    def fix_code_issues(self, code: str, error_message: str,
                        problem_info: Dict) -> str:
        system_prompt = """
        You are a Python code repair expert. Your task is to fix errors in the given code.
        
        Repair requirements:
        1. Carefully analyze the error message
        2. Locate and fix the problems in the code
        3. Maintain the original structure and functionality of the code
        4. Only modify the necessary parts
        5. Return the complete fixed code
        
        Return the complete fixed code in a Python code block.
        """

        user_prompt = f"""
        Please fix the errors in the following code:
        
        Original code:
        ```python
        {code}
        ```
        
        Error message:
        {error_message}
        
        Problem information:
        {json.dumps({k: problem_info[k] for k in problem_info if k != 'data_structures'}, indent=2, ensure_ascii=False)}
        
        Please return the complete fixed code.
        """
        response = self.call(system_prompt, user_prompt, temperature=0.1)
        fixed_code = self._extract_code_from_response(response)
        self.output_manager.save_fixed_code(fixed_code)
        return fixed_code

    def _build_format_instructions(self) -> str:
        return """
    Please strictly follow the following format for your answer:

    1. If you need to think about the problem, put your thought process inside <think></think> tags
    2. Code or final output content must be placed between ```python and ```
    3. Do not provide other content outside the above tags

    Example:

    <think>
    Here is my thinking process...
    Analyzing the characteristics of the problem...
    </think>

    ```python
    # This is the actual output code or content
    def solution():
        return "final result"
    ```
    """

    def call(self, system_prompt: str, user_prompt: str, temperature: float = None) -> str:
        format_instructions = self._build_format_instructions()
        enhanced_user_prompt = f"{user_prompt}\n\n{format_instructions}"

        temp = temperature if temperature is not None else self.default_params["temperature"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_user_prompt}
        ]

        params = {**self.default_params, "temperature": temp}

        return self.chat_completion(messages, params)

    def call_json(self, system_prompt: str, user_prompt: str, temperature: float = None) -> Dict:
        format_instructions = self._build_format_instructions()
        enhanced_user_prompt = f"{user_prompt}\n\n{format_instructions}\n\nMake sure your response is a valid JSON object and is surrounded by ```python and ``` tags."

        response = self.call(system_prompt, enhanced_user_prompt, temperature)
        if hasattr(response, '__next__') or hasattr(response, '__iter__'):
            collected_response = "".join([chunk for chunk in response])
            response = collected_response

        response_cleaned = re.sub(r'<think>[\s\S]*?</think>', '', response)

        code_block_patterns = [
            r'```(?:python|json)?\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'`\s*([\s\S]*?)\s*`'
        ]

        for pattern in code_block_patterns:
            matches = re.findall(pattern, response_cleaned)
            for match in matches:
                cleaned_match = match.strip()
                try:
                    return json.loads(cleaned_match)
                except json.JSONDecodeError:
                    pass

        try:
            return json.loads(response_cleaned)
        except json.JSONDecodeError:
            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            matches = re.findall(json_pattern, response_cleaned)

            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            result = {}
            pattern = r'["\']?([\w\s]+)["\']?\s*[:：]\s*["\']?([\w\s\.]+)["\']?'
            matches = re.findall(pattern, response_cleaned)

            if matches:
                for key, value in matches:
                    key = key.strip()
                    value = value.strip()
                    if value.isdigit():
                        value = int(value)
                    elif re.match(r'^-?\d+(\.\d+)?$', value):
                        value = float(value)
                    result[key] = value

                if self.output_manager:
                    self.output_manager.log_warning(
                        "llm_client", "json_extraction",
                        "Extracted key-value pairs from text instead of full JSON"
                    )
                return result
            if self.output_manager:
                self.output_manager.log_error(
                    "llm_client", "json_parse_error",
                    "Failed to parse JSON content from LLM response, returning original text"
                )
                if hasattr(self, 'log_dir'):
                    debug_file = os.path.join(
                        self.log_dir, f"json_parse_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    try:
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write("=== ORIGINAL RESPONSE ===\n\n")
                            f.write(response)
                            f.write("\n\n=== CLEANED RESPONSE ===\n\n")
                            f.write(response_cleaned)
                            f.write("\n\n=== EXTRACTION ATTEMPTS ===\n\n")
                            f.write("Code block extraction attempts:\n")
                            for pattern in code_block_patterns:
                                matches = re.findall(pattern, response_cleaned)
                                for i, match in enumerate(matches):
                                    f.write(f"Pattern {pattern}, Match {i}:\n")
                                    f.write(f"{match[:200]}...\n\n")

                    except Exception as e:
                        pass

            return {"raw_response": response}


class ProblemAnalysisLLMClient(LLMClient):

    def __init__(self, config: Dict = None, output_manager: Optional[OutputManager] = None):
        super().__init__(config, "analysis", output_manager)

    def _build_format_instructions(self) -> str:
        return """
    Please strictly follow the format below for your response:

    1. If you need to think about the problem, put your thinking process within <think></think> tags
    2. Code or final output content must be placed between ```python and ``` tags
    3. Do not provide any other content outside the above markers

    For example:

    <think>
    Here is my thinking process...
    Analyzing the characteristics of the problem...
    </think>

    ```python
    # This is the actual output code or content
    def solution():
        return "final result"
    ```
    """

    def call(self, system_prompt: str, user_prompt: str, temperature: float = None) -> str:
        format_instructions = self._build_format_instructions()
        enhanced_user_prompt = f"{user_prompt}\n\n{format_instructions}"

        temp = temperature if temperature is not None else self.default_params["temperature"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_user_prompt}
        ]

        params = {**self.default_params, "temperature": temp}

        return self.chat_completion(messages, params)

    def call_json(self, system_prompt: str, user_prompt: str, temperature: float = None) -> Dict:
        format_instructions = self._build_format_instructions()
        enhanced_user_prompt = f"{user_prompt}\n\n{format_instructions}\n\nPlease ensure your response is a valid JSON object without adding any other explanations or markers."

        response = self.call(system_prompt, enhanced_user_prompt, temperature)
        if hasattr(response, '__next__') or hasattr(response, '__iter__'):
            collected_response = "".join([chunk for chunk in response])
            response = collected_response

        original_response = response
        response = re.sub(r'<think>[\s\S]*?</think>', '', response)
        response = re.sub(r'```(?:python|json)?', '', response)
        response = re.sub(r'```', '', response)
        response = re.sub(r'(?m)^\s*//.*?$', '', response)
        response = re.sub(r'(?m)^\s*#.*?$', '', response)
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_content = response[start_idx:end_idx+1]
            try:
                result = json.loads(json_content)
                return result
            except json.JSONDecodeError:
                pass

        clean_json = self._aggressive_json_clean(response)
        try:
            return json.loads(clean_json)
        except json.JSONDecodeError:
            pass

        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        matches = re.findall(json_pattern, response)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        if self.output_manager:
            self.output_manager.log_error(
                "llm_client", "json_parse_error",
                "Failed to parse JSON content from LLM response, returning original text"
            )

        if hasattr(self, 'log_dir'):
            debug_file = os.path.join(
                self.log_dir, f"json_parse_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            try:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write("=== ORIGINAL RESPONSE ===\n\n")
                    f.write(original_response)
                    f.write("\n\n=== CLEANED RESPONSE ===\n\n")
                    f.write(response)
                    f.write("\n\n=== EXTRACTED JSON CANDIDATES ===\n\n")
                    for match in matches:
                        f.write(f"{match[:200]}...\n\n")
                    f.write("\n\n=== START/END INDICES ===\n")
                    f.write(
                        f"Start index: {start_idx}, End index: {end_idx}\n")
            except Exception:
                pass

        if original_response.find('"hard_constraints"') != -1 and original_response.find('"soft_constraints"') != -1:
            try:
                hard_constraints = self._extract_list_items(
                    original_response, "hard_constraints")
                soft_constraints = self._extract_list_items(
                    original_response, "soft_constraints")
                objectives = self._extract_list_items(
                    original_response, "objectives")

                result = {
                    "hard_constraints": hard_constraints,
                    "soft_constraints": soft_constraints,
                    "objectives": objectives,
                    "constraint_importance": {}
                }

                constraint_importance = {}
                importance_pattern = r'"([^"]+)":\s*(\d+)'
                for match in re.finditer(importance_pattern, original_response):
                    constraint_name, importance = match.groups()
                    if constraint_name and importance:
                        constraint_importance[constraint_name] = int(
                            importance)

                if constraint_importance:
                    result["constraint_importance"] = constraint_importance

                return result
            except Exception:
                pass
        return {"raw_response": original_response}

    def _extract_list_items(self, text, key):
        pattern = fr'"{key}":\s*\[(.*?)\]'
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return []

        items_text = match.group(1)
        items = []

        item_pattern = r'"([^"]*)"'
        for item_match in re.finditer(item_pattern, items_text):
            items.append(item_match.group(1))

        return items

    def _aggressive_json_clean(self, text):
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return "{}"

        potential_json = text[start_idx:end_idx+1]
        potential_json = re.sub(
            r'//.*?$', '', potential_json, flags=re.MULTILINE)
        potential_json = re.sub(
            r'/\*.*?\*/', '', potential_json, flags=re.DOTALL)

        potential_json = potential_json.replace("'", '"')
        potential_json = ''.join(c for c in potential_json if c.isprintable())

        return potential_json

    def call_json4suggest(self, system_prompt: str, user_prompt: str, temperature: float = None) -> Dict:
        format_instructions = self._build_format_instructions()
        enhanced_user_prompt = f"{user_prompt}\n\n{format_instructions}\n\nPlease ensure your response is a valid JSON object and make sure it is enclosed within ```python and ``` tags."

        response = self.call(system_prompt, enhanced_user_prompt, temperature)

        # If the response is a generator, collect the complete content
        if hasattr(response, '__next__') or hasattr(response, '__iter__'):
            collected_response = "".join([chunk for chunk in response])
            response = collected_response

        response_cleaned = re.sub(r'<think>[\s\S]*?</think>', '', response)

        code_block_patterns = [
            r'```(?:python|json)?\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'`\s*([\s\S]*?)\s*`'
        ]

        for pattern in code_block_patterns:
            matches = re.findall(pattern, response_cleaned)
            for match in matches:
                cleaned_match = match.strip()
                try:
                    return json.loads(cleaned_match)
                except json.JSONDecodeError:
                    pass

        try:
            return json.loads(response_cleaned)
        except json.JSONDecodeError:
            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            matches = re.findall(json_pattern, response_cleaned)

            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

            result = {}
            pattern = r'["\']?([\w\s]+)["\']?\s*[:：]\s*["\']?([\w\s\.]+)["\']?'
            matches = re.findall(pattern, response_cleaned)

            if matches:
                for key, value in matches:
                    key = key.strip()
                    value = value.strip()
                    # 转换数值
                    if value.isdigit():
                        value = int(value)
                    elif re.match(r'^-?\d+(\.\d+)?$', value):
                        value = float(value)
                    result[key] = value

                if self.output_manager:
                    self.output_manager.log_warning(
                        "llm_client", "json_extraction",
                        "Extracted key-value pairs from the text, rather than a complete JSON."
                    )
                return result

            if self.output_manager:
                self.output_manager.log_error(
                    "llm_client", "json_parse_error",
                    "Failed to parse LLM response JSON content, returning original text"
                )

                if hasattr(self, 'log_dir'):
                    debug_file = os.path.join(
                        self.log_dir, f"json_parse_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    try:
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write("=== ORIGINAL RESPONSE ===\n\n")
                            f.write(response)
                            f.write("\n\n=== CLEANED RESPONSE ===\n\n")
                            f.write(response_cleaned)
                            f.write("\n\n=== EXTRACTION ATTEMPTS ===\n\n")
                            f.write("Code block extraction attempts:\n")
                            for pattern in code_block_patterns:
                                matches = re.findall(pattern, response_cleaned)
                                for i, match in enumerate(matches):
                                    f.write(f"Pattern {pattern}, Match {i}:\n")
                                    f.write(f"{match[:200]}...\n\n")

                    except Exception as e:
                        pass

            return {"raw_response": response}

    def _limit_problem_data_points(self, problem_data: Dict, max_points: int = 5) -> Dict:
        if not problem_data:
            return {}

        import copy
        limited_data = copy.deepcopy(problem_data)
        if "data" in limited_data and isinstance(limited_data["data"], list):
            for i, data_item in enumerate(limited_data["data"]):
                if isinstance(data_item, dict):
                    original_item = problem_data.get("data", [])[i] if i < len(
                        problem_data.get("data", [])) else {}
                    self._limit_points_in_dict(
                        data_item, max_points, original_item)

        return limited_data

    def _limit_points_in_dict(self, data_dict: Dict, max_points: int, original_dict: Dict = None):
        if original_dict is None:
            original_dict = data_dict

        for field_name, field_value in list(data_dict.items()):
            if field_name in ["description", "type", "name", "id"] or field_name.endswith("_limitation_note"):
                continue

            if isinstance(field_value, list) and len(field_value) > 0:
                original_value = original_dict.get(field_name, [])
                original_count = len(original_value)

                if self._is_object_list(field_value):
                    if len(field_value) > max_points:
                        data_dict[field_name] = field_value[:max_points]
                        note_key = f"_{field_name}_limitation_note"
                        data_dict[
                            note_key] = f"To save tokens, only the beginning is shown here: {max_points} {field_name} items, actual data contains {original_count} items"

                elif self._is_matrix(field_value):
                    original_rows = len(original_value)
                    original_cols = len(original_value[0]) if original_value and len(
                        original_value) > 0 else 0

                    if len(field_value) > max_points:
                        data_dict[field_name] = field_value[:max_points]

                    for i, row in enumerate(data_dict[field_name]):
                        if isinstance(row, list) and len(row) > max_points:
                            data_dict[field_name][i] = row[:max_points]

                    if original_rows > max_points or original_cols > max_points:
                        note_key = f"_{field_name}_limitation_note"
                        data_dict[note_key] = f"To save tokens, only the beginning is shown here: {max_points} {field_name} matrix elements, actual matrix size is {original_rows}x{original_cols}"

    def _is_object_list(self, data_list: list) -> bool:
        if not data_list or not isinstance(data_list[0], dict):
            return False
        first_item = data_list[0]
        object_indicators = ['id', 'x', 'y', 'demand',
                             'length', 'width', 'height', 'center']
        return any(key in first_item for key in object_indicators)

    def _is_matrix(self, data_list: list) -> bool:
        if not data_list:
            return False
        return isinstance(data_list[0], list)
