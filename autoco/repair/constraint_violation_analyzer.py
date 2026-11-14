
from typing import Dict, List, Tuple, Any, Optional, Set
import json
import time
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass
from autoco.output_manager import OutputManager
import numpy as np


@dataclass
class ConstraintViolation:

    constraint_name: str
    constraint_type: str  
    violation_degree: float  
    violation_ratio: float  
    description: str  
    suggested_relaxation: float  
    timestamp: str


@dataclass
class ViolationPattern:

    pattern_id: str
    constraints_involved: List[str]
    frequency: int
    avg_violation_degree: float
    common_causes: List[str]
    suggested_strategy: str


class ConstraintViolationAnalyzer:

    def __init__(self, problem_data: Dict, output_manager: OutputManager):

        self.problem_data = problem_data
        self.output_manager = output_manager

        self.violation_history: List[Dict] = []

        self.constraint_stats = defaultdict(lambda: {
            'total_violations': 0,
            'total_degree': 0.0,
            'avg_degree': 0.0,
            'max_degree': 0.0,
            'violation_ratios': []
        })

        self.violation_patterns: List[ViolationPattern] = []

       

        
