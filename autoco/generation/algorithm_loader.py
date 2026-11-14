import os


class AlgorithmLoader:

    def __init__(self, base_algorithm_dir="base_algorithm", single_algorithm_file=None):
        """Initialize algorithm loader"""
        self.base_algorithm_dir = base_algorithm_dir
        self.single_algorithm_file = single_algorithm_file
        self.algorithm_templates = {}

        if single_algorithm_file:
            self.load_single_template()
        else:
            self.load_all_templates()

    def load_single_template(self):
        """Load a single algorithm template file"""
        if not os.path.exists(self.single_algorithm_file):
            raise ValueError(
                f"Algorithm template file does not exist: {self.single_algorithm_file}")

        algorithm_name = os.path.basename(
            self.single_algorithm_file)[:-3]  
        with open(self.single_algorithm_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.algorithm_templates[algorithm_name] = content

    def load_all_templates(self):
        """Load all available algorithm templates"""
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
                f"No algorithm templates found, please check the directory: {self.base_algorithm_dir}")

    def get_available_algorithms(self):
        """Get all available algorithm names"""
        return list(self.algorithm_templates.keys())

    def get_template(self, algorithm_name):
        """Get the content of a specific algorithm template"""
        return self.algorithm_templates.get(algorithm_name)
