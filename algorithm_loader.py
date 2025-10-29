import os


class AlgorithmLoader:

    def __init__(self, base_algorithm_dir="base_algorithm"):

        self.base_algorithm_dir = base_algorithm_dir
        self.algorithm_templates = {}
        self.load_all_templates()

    def load_all_templates(self):

        if not os.path.exists(self.base_algorithm_dir):
            raise ValueError(
                f"Algorithm template directory does not exist: {self.base_algorithm_dir}")

        for file in os.listdir(self.base_algorithm_dir):
            if file.endswith(".py"):
                algorithm_name = file[:-3]
                file_path = os.path.join(self.base_algorithm_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.algorithm_templates[algorithm_name] = content

        if not self.algorithm_templates:
            raise ValueError(
                f"NO algorithm template,please check: {self.base_algorithm_dir}")

    def get_template(self, algorithm_name):

        if algorithm_name not in self.algorithm_templates:
            raise ValueError(f"NO algorithm template: {algorithm_name}")
        return self.algorithm_templates[algorithm_name]

    def get_available_algorithms(self):

        return list(self.algorithm_templates.keys())
