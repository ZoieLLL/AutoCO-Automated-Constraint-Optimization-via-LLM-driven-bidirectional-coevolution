import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any
import copy


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


class DatasetManager:
    def __init__(self, datasets_dir="datasets", default_dataset_path="problem_data.json"):
        self.datasets_dir = datasets_dir
        self.default_dataset_path = default_dataset_path
        self.datasets = {}

        os.makedirs(datasets_dir, exist_ok=True)
        self.load_default_dataset()

    def load_default_dataset(self):
        try:
            with open(self.default_dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                self.datasets["default"] = dataset
        except Exception as e:
            print(f"Failed to load default dataset: {e}")

    def load_all_datasets(self):
        if "default" not in self.datasets:
            self.load_default_dataset()

        for file in os.listdir(self.datasets_dir):
            if file.endswith('.json'):
                dataset_id = file[:-5]
                if dataset_id not in self.datasets:
                    try:
                        with open(os.path.join(self.datasets_dir, file), 'r', encoding='utf-8') as f:
                            dataset = json.load(f)
                            self.datasets[dataset_id] = dataset
                    except Exception as e:
                        print(f"Failed to load dataset {file}: {e}")

    def get_dataset(self, dataset_id="default"):
        if dataset_id not in self.datasets:
            if dataset_id == "default":
                self.load_default_dataset()
            else:
                dataset_path = os.path.join(
                    self.datasets_dir, f"{dataset_id}.json")
                if os.path.exists(dataset_path):
                    try:
                        with open(dataset_path, 'r', encoding='utf-8') as f:
                            self.datasets[dataset_id] = json.load(f)
                    except Exception as e:
                        print(f"Failed to load dataset {dataset_id}: {e}")
                        return None
                else:
                    print(f"Dataset {dataset_id} does not exist")
                    return None

        return self.datasets.get(dataset_id)

    def get_all_dataset_ids(self):
        return list(self.datasets.keys())

    def add_dataset(self, dataset_id, dataset_data):
        self.datasets[dataset_id] = dataset_data
        dataset_path = os.path.join(self.datasets_dir, f"{dataset_id}.json")
        try:
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_data, f, indent=2,
                          ensure_ascii=False, cls=NumpyEncoder)
        except Exception as e:
            print(f"Failed to save dataset {dataset_id}: {e}")

    def create_dataset_variant(self, base_id="default", new_id=None, modifications=None):
        if base_id not in self.datasets:
            if not self.get_dataset(base_id):
                return None
        base_dataset = self.datasets[base_id]
        new_dataset = copy.deepcopy(base_dataset)

        if modifications:
            for path, value in modifications.items():
                parts = path.split('.')
                target = new_dataset
                for i, part in enumerate(parts[:-1]):
                    if part.isdigit():
                        part = int(part)
                    if i < len(parts) - 2:
                        target = target[part]
                    else:
                        last_part = parts[-1]
                        if last_part.isdigit():
                            last_part = int(last_part)
                        target[part][last_part] = value

        if new_id is None:
            import uuid
            new_id = f"variant_{str(uuid.uuid4())[:8]}"
        self.add_dataset(new_id, new_dataset)

        return new_id

    def load_multi_instance_dataset(self, pkl_file_path, problem_config_path=None):

        if not os.path.exists(pkl_file_path):
            print(f"PKL file does not exist: {pkl_file_path}")
            return None

        try:
            with open(pkl_file_path, 'rb') as f:
                instances = pickle.load(f)

            if not isinstance(instances, list):
                return None

            if problem_config_path and os.path.exists(problem_config_path):
                with open(problem_config_path, 'r', encoding='utf-8') as f:
                    problem_config = json.load(f)
            else:
                problem_config = self.datasets.get("default", {})
            dataset_id = f"multi_instance_{os.path.basename(pkl_file_path).replace('.pkl', '')}"

            multi_instance_dataset = {
                "is_multi_instance": True,
                "instances": instances,
                "instance_count": len(instances),
                "problem_config": problem_config,
                "pkl_file_path": pkl_file_path,
                "evaluation_mode": "multi_instance_average"
            }

            if problem_config:
                multi_instance_dataset.update({
                    "problem_type": problem_config.get("problem_type", "unknown"),
                    "description": problem_config.get("description", ""),
                    "detailed_description": problem_config.get("detailed_description", {}),
                    "solution_format": problem_config.get("solution_format", {}),
                    "parameters": problem_config.get("parameters", {})
                })

            self.datasets[dataset_id] = multi_instance_dataset

            return dataset_id

        except Exception as e:
            print(f"Failed to load multi-instance dataset: {e}")
            return None

    def load_multi_instance_dataset_new(self, pkl_file_path, problem_data):

        if not os.path.exists(pkl_file_path):
            return None

        try:
            with open(pkl_file_path, 'rb') as f:
                instances = pickle.load(f)

            if not isinstance(instances, list):
                return None

            dataset_id = f"multi_instance_{os.path.basename(pkl_file_path).replace('.pkl', '')}"
            multi_instance_dataset = {
                "is_multi_instance": True,
                "instances": instances,
                "instance_count": len(instances),
                "pkl_file_path": pkl_file_path,
                "evaluation_mode": "multi_instance_average"
            }

            multi_instance_dataset.update(problem_data)
            self.datasets[dataset_id] = multi_instance_dataset
            return dataset_id

        except Exception as e:
            print(f"Failed to load multi-instance dataset: {e}")
            return None

    def is_multi_instance_dataset(self, dataset_id):

        dataset = self.get_dataset(dataset_id)
        return dataset and dataset.get("is_multi_instance", False)

    def get_instance_count(self, dataset_id):

        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return 0
        if dataset.get("is_multi_instance", False):
            return dataset.get("instance_count", 0)
        else:
            return 1

    def get_instance_data(self, dataset_id, instance_index=0):

        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None

        if dataset.get("is_multi_instance", False):
            instances = dataset.get("instances", [])
            if 0 <= instance_index < len(instances):
                instance_data = instances[instance_index]
                problem_config = dataset.get("problem_config", {})
                complete_instance = copy.deepcopy(problem_config)
                if isinstance(instance_data, dict):
                    complete_instance.update(instance_data)
                else:
                    complete_instance["instance_data"] = instance_data

                return complete_instance
            else:
                return None
        else:
            return dataset

    def get_all_instances_data(self, dataset_id):
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return []

        if dataset.get("is_multi_instance", False):
            instance_count = self.get_instance_count(dataset_id)
            return [self.get_instance_data(dataset_id, i) for i in range(instance_count)]
        else:
            return [dataset]
