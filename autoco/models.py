from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime


@dataclass
class OptimizationParams:
    MAX_ITERATIONS: int = 5
    MAX_DRONES: int = 3
    FIXED_COST: float = 100.0
    MAX_PAYLOAD: float = 50.0
    MAX_BATTERY: float = 1000.0
    DRONE_SPEED: float = 1.0
    BATTERY_PER_KM: float = 10.0
    ENERGY_CONSUMPTION_RATIO: float = 0.1
    WAITING_TIME_PENALTY: float = 0.1
    DEMAND_REWARD_RATIO: float = 7.0
    time_limit: float = 600.0


    MAX_OUTER_ITERATIONS: int = 9
    MAX_INNER_ITERATIONS: int = 5
    MAX_STAGNATION_COUNT: int = 45
    POPULATION_SIZE: int = 3
    OFFSPRING_COUNT: int = 1
    TOURNAMENT_SIZE: int = 3
    CROSSOVER_RATE: float = 0.8
    MUTATION_STRENGTH: float = 0.4
    ELITE_COUNT: int = 2
    MCTS_C: float = 1.414
    MCTS_SIMULATIONS: int = 2
    MCTS_TIME_LIMIT: float = 50.0
    MCTS_TRIGGER_STAGNATION: int = 2
    MCTS_INTERVAL: int = 3
    CODE_VARIANTS: int = 3

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class TextStrategy:
    id: str 
    text: str 
    constraint_order: List[str] = field(default_factory=list) 
    relaxation_factors: Dict[str, float] = field(
        default_factory=dict)  
    algorithm_design: str = "" 
    code_snippet: str = "" 
    fitness: float = 0.0  
    created_at: datetime = field(default_factory=datetime.now) 
    parent_ids: List[str] = field(default_factory=list)  
    generation: int = 0 
    method: str = "initial"  
    evaluated: bool = False 
    outer_iteration: int = 0  
    metadata: Dict[str, Any] = field(default_factory=dict)
    solution: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "constraint_order": self.constraint_order,
            "relaxation_factors": self.relaxation_factors,
            "algorithm_design": self.algorithm_design,
            "code_snippet": self.code_snippet,
            "fitness": self.fitness,
            "created_at": self.created_at.isoformat(),
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "method": self.method,
            "evaluated": self.evaluated,
            "solution": self.solution,
            "outer_iteration": self.outer_iteration
        }

    def get_path_representation(self) -> List[Union[str, float]]:
        path = []
        for constraint in self.constraint_order:
            path.append(constraint)
            path.append(self.relaxation_factors.get(constraint, 1.0))     
        return path


@dataclass
class Solution:
    id: str
    strategy_id: str 
    code_id: str  
    solution_data: Dict  
    is_feasible: bool = False  
    objective_value: float = 0.0  
    constraint_violations: Dict[str, float] = field(
        default_factory=dict) 
    computation_time: float = 0.0  
    created_at: datetime = field(default_factory=datetime.now)  
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "code_id": self.code_id,
            "solution_data": self.solution_data,
            "is_feasible": self.is_feasible,
            "objective_value": self.objective_value,
            "constraint_violations": self.constraint_violations,
            "computation_time": self.computation_time,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class CodeImplementation:
    id: str 
    strategy_id: str 
    code: str  
    paradigm: str 
    language: str = "python"
    is_valid: bool = False 
    error_message: str = "" 
    created_at: datetime = field(default_factory=datetime.now)  
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "code": self.code,
            "paradigm": self.paradigm,
            "language": self.language,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class MCTSNode:
    id: str 
    node_type: str  
    value: Union[str, float] 
    parent_id: Optional[str] = None  
    depth: int = 0 
    visits: int = 0 
    total_reward: float = 0.0  
    children_ids: List[str] = field(default_factory=list) 
    is_expanded: bool = False


@dataclass
class OptimizationRun:
    id: str 
    problem_type: str  
    start_time: datetime = field(default_factory=datetime.now) 
    end_time: Optional[datetime] = None  
    best_strategy_id: Optional[str] = None  
    best_solution_id: Optional[str] = None  
    best_objective: float = 0.0 
    params: Dict = field(default_factory=dict)  
    iterations_completed: int = 0  

    def to_dict(self) -> Dict:
        result = {
            "id": self.id,
            "problem_type": self.problem_type,
            "start_time": self.start_time.isoformat(),
            "best_strategy_id": self.best_strategy_id,
            "best_solution_id": self.best_solution_id,
            "best_objective": self.best_objective,
            "params": self.params,
            "iterations_completed": self.iterations_completed
        }
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        return result

    def mark_completed(self) -> None:
        self.end_time = datetime.now()

    def update_best(self, strategy_id: str, solution_id: str, objective: float) -> None:
        self.best_strategy_id = strategy_id
        self.best_solution_id = solution_id
        self.best_objective = objective
