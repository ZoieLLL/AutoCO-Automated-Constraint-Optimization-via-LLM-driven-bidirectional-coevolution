

from typing import Dict, List, Tuple, Any, Optional, Set, Union
import random
import math
import uuid
import time
from models import MCTSNode, OptimizationParams
from output_manager import OutputManager


class MCTSTree:

    def __init__(self, constraints: List[str], params, output_manager, problem_info=None):

        self.constraints = constraints
        self.params = params
        self.output_manager = output_manager
        self.problem_info = problem_info or {}

        self.nodes: Dict[str, MCTSNode] = {}

        self.best_path: List[Union[str, float]] = []
        self.best_reward: float = float('-inf')

        self.root_id = self._create_root_node()

    def set_problem_info(self, problem_info: Dict) -> None:

        self.problem_info = problem_info

    def _create_root_node(self) -> str:

        root_id = str(uuid.uuid4())
        root_node = MCTSNode(
            id=root_id,
            node_type="root",
            value="root",
            depth=0
        )
        self.nodes[root_id] = root_node
        return root_id

    def reset(self) -> None:

        self.nodes = {}
        self.best_path = []
        self.best_reward = float('-inf')
        self.root_id = self._create_root_node()
        self.output_manager.log_info("mcts", "reset", "MCTSreset")

    def run_search(self, simulation_count: int,
                   evaluator=None,
                   time_limit: float = 60.0) -> Dict:

        if simulation_count <= 0:

            smulation_count = 1
        start_time = time.time()

        for i in range(simulation_count):

            if time.time() - start_time > time_limit:

                break

            selected_node_id = self._selection(self.root_id)

            if self.nodes[selected_node_id].is_expanded:

                continue

            path = self._get_path_to_node(selected_node_id)

            if len(path) < 2 * len(self.constraints) and not self.nodes[selected_node_id].is_expanded:
                expanded_node_ids = self._expansion(selected_node_id)

                if expanded_node_ids:

                    selected_node_id = random.choice(expanded_node_ids)

            path = self._get_path_to_node(selected_node_id)
            final_path, reward = self._simulation(path, evaluator)

            self._backpropagation(selected_node_id, reward)

            if reward > self.best_reward:
                self.best_path = final_path
                self.best_reward = reward

                self.output_manager.log_info(
                    "mcts", "new_best_path",
                    f"new_best_path:{self._format_path(final_path)}"
                )

        search_time = time.time() - start_time
        result = {
            "best_path": self.best_path,
            "best_reward": self.best_reward,
            "simulations_completed": min(i + 1, simulation_count),
            "search_time": search_time,
            "path_text": self._format_path(self.best_path),
            "statistics": self._get_statistics()
        }

        self.output_manager.save_mcts_result(result)

        return result

    def _selection(self, node_id: str) -> str:

        current_id = node_id

        while self.nodes[current_id].is_expanded and self.nodes[current_id].children_ids:

            ucb_values = []
            for child_id in self.nodes[current_id].children_ids:
                child = self.nodes[child_id]

                if child.visits == 0:
                    ucb_values.append((child_id, float('inf')))
                    continue

                exploitation = child.total_reward / child.visits
                exploration = self.params.MCTS_C * \
                    math.sqrt(
                        2 * math.log(self.nodes[current_id].visits) / child.visits)
                ucb = exploitation + exploration

                ucb_values.append((child_id, ucb))

            current_id = max(ucb_values, key=lambda x: x[1])[0]

        return current_id

    def _expansion(self, node_id: str) -> List[str]:

        node = self.nodes[node_id]
        path = self._get_path_to_node(node_id)

        existing_constraints = [
            p for p in path if isinstance(p, str) and p != "root"]

        new_node_ids = []

        if node.node_type == "root" or node.node_type == "relaxation":

            remaining_constraints = [
                c for c in self.constraints if c not in existing_constraints]

            if not remaining_constraints:
                node.is_expanded = True
                return []

            for constraint in remaining_constraints:
                child_id = str(uuid.uuid4())
                child = MCTSNode(
                    id=child_id,
                    node_type="constraint",
                    value=constraint,
                    parent_id=node_id,
                    depth=node.depth + 1
                )

                self.nodes[child_id] = child
                node.children_ids.append(child_id)
                new_node_ids.append(child_id)

        elif node.node_type == "constraint":
            constraint = node.value
            relaxation_info = self.problem_info.get(
                "relaxation_info", {}).get(constraint, {})

            possible_relaxations = relaxation_info.get(
                "possible_relaxations", [0.9, 0.95, 1.0, 1.05, 1.1])

            for relaxation in possible_relaxations:
                child_id = str(uuid.uuid4())
                child = MCTSNode(
                    id=child_id,
                    node_type="relaxation",
                    value=relaxation,
                    parent_id=node_id,
                    depth=node.depth + 1
                )

                self.nodes[child_id] = child
                node.children_ids.append(child_id)
                new_node_ids.append(child_id)

        node.is_expanded = True

        return new_node_ids

    def _simulation(self, path: List[Union[str, float]],
                    evaluator=None) -> Tuple[List[Union[str, float]], float]:

        sim_path = path.copy()

        existing_constraints = [
            p for p in sim_path if isinstance(p, str) and p != "root"]

        remaining_constraints = [
            c for c in self.constraints if c not in existing_constraints]
        random.shuffle(remaining_constraints)

        for constraint in remaining_constraints:
            sim_path.append(constraint)

            relaxation_info = self.problem_info.get(
                "relaxation_info", {}).get(constraint, {})
            possible_relaxations = relaxation_info.get(
                "possible_relaxations", [0.9, 0.95, 1.0, 1.05, 1.1])
            relaxation = random.choice(possible_relaxations)

            sim_path.append(relaxation)

        reward = 0.0
        if evaluator is not None:
            constraint_order = []
            relaxation_factors = {}

            for i in range(0, len(sim_path) - 1, 2):
                if i + 1 < len(sim_path) and isinstance(sim_path[i], str) and sim_path[i] != "root":
                    constraint = sim_path[i]
                    relaxation = sim_path[i + 1]
                    constraint_order.append(constraint)
                    relaxation_factors[constraint] = relaxation

            reward = evaluator(constraint_order, relaxation_factors)
        else:

            reward = len(set(existing_constraints)) / len(self.constraints)

        return sim_path, reward

    def _backpropagation(self, node_id: str, reward: float) -> None:

        current_id = node_id

        while current_id is not None:
            node = self.nodes[current_id]
            node.visits += 1
            node.total_reward += reward
            current_id = node.parent_id

    def _get_path_to_node(self, node_id: str) -> List[Union[str, float]]:

        path = []
        current_id = node_id

        while current_id is not None:
            current_node = self.nodes[current_id]
            path.append(current_node.value)
            current_id = current_node.parent_id

        path.reverse()

        return path

    def _get_statistics(self) -> Dict:

        visit_histogram = {}
        for node in self.nodes.values():
            if node.visits > 0:
                visit_histogram[node.visits] = visit_histogram.get(
                    node.visits, 0) + 1

        constraint_frequency = {}
        for node in self.nodes.values():
            if node.node_type == "constraint" and node.visits > 0:
                constraint_frequency[node.value] = constraint_frequency.get(
                    node.value, 0) + node.visits

        relaxation_stats = {}
        for node in self.nodes.values():
            if node.node_type == "relaxation" and node.visits > 0:
                parent = self.nodes.get(node.parent_id)
                if parent and parent.node_type == "constraint":
                    constraint = parent.value
                    if constraint not in relaxation_stats:
                        relaxation_stats[constraint] = {
                            "sum": 0, "count": 0, "weighted_sum": 0}

                    relaxation_stats[constraint]["sum"] += node.value
                    relaxation_stats[constraint]["count"] += 1
                    relaxation_stats[constraint]["weighted_sum"] += node.value * node.visits

        for constraint, stats in relaxation_stats.items():
            if stats["count"] > 0:
                stats["average"] = stats["sum"] / stats["count"]
                stats["weighted_average"] = stats["weighted_sum"] / \
                    (stats["count"] * stats["count"])
                del stats["sum"]
                del stats["weighted_sum"]

        return {
            "total_nodes": len(self.nodes),
            "max_depth": max(node.depth for node in self.nodes.values()),
            "visit_histogram": visit_histogram,
            "constraint_frequency": constraint_frequency,
            "relaxation_stats": relaxation_stats
        }
        return best_paths

    def get_best_paths1(self, top_n: int = 3, exploration_weight: float = 0.5) -> List[Dict]:

        leaf_nodes = []
        for node_id, node in self.nodes.items():
            if not node.children_ids and node.visits > 0:
                path = self._get_path_to_node(node_id)

                constraints_in_path = [
                    p for p in path if isinstance(p, str) and p != "root"]
                if len(set(constraints_in_path)) == len(self.constraints):

                    exploitation = node.total_reward / max(1, node.visits)
                    parent_id = node.parent_id
                    parent_visits = self.nodes[parent_id].visits if parent_id in self.nodes else 1
                    exploration = exploration_weight * \
                        math.sqrt(math.log(parent_visits) /
                                  max(1, node.visits))
                    ucb_value = exploitation + exploration

                    leaf_nodes.append((
                        node_id,
                        exploitation,
                        ucb_value,
                        node.visits
                    ))

        exploitation_paths = []
        exploration_paths = []

        for i, (node_id, exploitation, _, visits) in enumerate(
            sorted(leaf_nodes, key=lambda x: x[1], reverse=True)[:top_n]
        ):
            path = self._get_path_to_node(node_id)
            path_info = {
                "path": path,
                "average_reward": exploitation,
                "visits": visits,
                "type": "exploitation",
                "rank": i + 1
            }
            exploitation_paths.append(path_info)

        low_visited_nodes = [n for n in leaf_nodes if n[3]
                             < max(n[3] for n in leaf_nodes) / 2]
        if low_visited_nodes:
            for i, (node_id, exploitation, ucb, visits) in enumerate(
                sorted(low_visited_nodes,
                       key=lambda x: x[2], reverse=True)[:top_n]
            ):
                path = self._get_path_to_node(node_id)

                if not any(self._paths_are_similar(path, p["path"]) for p in exploitation_paths):
                    path_info = {
                        "path": path,
                        "average_reward": exploitation,
                        "ucb_value": ucb,
                        "visits": visits,
                        "type": "exploration",
                        "rank": i + 1
                    }
                    exploration_paths.append(path_info)

        combined_paths = exploitation_paths + exploration_paths
        return combined_paths[:top_n]

    def _paths_are_similar1(self, path1, path2, similarity_threshold=0.7):
        """
        Determine whether two paths are similar
        """
        constraints1 = [p for i, p in enumerate(
            path1) if isinstance(p, str) and i > 0]
        factors1 = [p for p in path1 if isinstance(p, (int, float))]

        constraints2 = [p for i, p in enumerate(
            path2) if isinstance(p, str) and i > 0]
        factors2 = [p for p in path2 if isinstance(p, (int, float))]

        constraint_similarity = len(set(constraints1) & set(
            constraints2)) / max(1, len(set(constraints1) | set(constraints2)))

        return constraint_similarity > similarity_threshold

    def get_best_paths(self, top_n: int = 3, exploration_weight: float = 0.5) -> List[Dict]:
        """
        Get the best paths, including optimal paths and paths with exploration potential

        Args:
            top_n: Number of paths to return
            exploration_weight: Exploration weight parameter in UCB formula, 0 means pure exploitation, greater than 0 increases exploration tendency

        Returns:
            List[Dict]: List of best paths and paths with exploration potential
        """
        import math

        leaf_nodes = []
        for node_id, node in self.nodes.items():
            if not node.children_ids and node.visits > 0:
                path = self._get_path_to_node(node_id)

                constraints_in_path = [
                    p for p in path if isinstance(p, str) and p != "root"]
                if len(set(constraints_in_path)) == len(self.constraints):
                    exploitation = node.total_reward / max(1, node.visits)
                    parent_id = node.parent_id
                    parent_visits = self.nodes[parent_id].visits if parent_id in self.nodes else 1
                    exploration = exploration_weight * \
                        math.sqrt(math.log(parent_visits) /
                                  max(1, node.visits))
                    ucb_value = exploitation + exploration

                    leaf_nodes.append((
                        node_id,
                        exploitation,
                        ucb_value,
                        node.visits
                    ))

        if not leaf_nodes:
            return []

        exploitation_paths = []
        exploration_paths = []

        sorted_by_reward = sorted(leaf_nodes, key=lambda x: x[1], reverse=True)
        exploitation_count = min(
            top_n // 2 + 1, len(sorted_by_reward))

        for i, (node_id, exploitation, _, visits) in enumerate(sorted_by_reward[:exploitation_count]):
            path = self._get_path_to_node(node_id)
            path_info = {
                "path": path,
                "average_reward": exploitation,
                "visits": visits,
                "type": "exploitation",
                "path_type": "exploitation",
                "rank": i + 1
            }
            exploitation_paths.append(path_info)

        sorted_by_visits = sorted(leaf_nodes, key=lambda x: x[3])
        potential_exploration_paths = []

        low_visit_cutoff = min(len(sorted_by_visits),
                               max(3, len(sorted_by_visits) // 2))
        candidates = sorted_by_visits[:low_visit_cutoff]

        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)

            for i, (node_id, exploitation, ucb, visits) in enumerate(candidates):
                path = self._get_path_to_node(node_id)

                is_duplicate = False
                for exp_path in exploitation_paths:
                    if self._simplified_path_similarity(path, exp_path["path"]) > 0.7:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    path_info = {
                        "path": path,
                        "average_reward": exploitation,
                        "ucb_value": ucb,
                        "visits": visits,
                        "type": "exploration",
                        "path_type": "exploration",
                        "rank": i + 1
                    }
                    potential_exploration_paths.append(path_info)

                    if len(potential_exploration_paths) >= top_n - len(exploitation_paths):
                        break

        exploration_paths = potential_exploration_paths[:top_n - len(
            exploitation_paths)]

        combined_paths = exploitation_paths + exploration_paths

        self.output_manager.log_info(
            "mcts", "path_extraction",
            f"Extracted {len(exploitation_paths)} exploitation paths and {len(exploration_paths)} exploration paths"
        )

        self.output_manager.log_info(
            "mcts", "path_extraction_detail",
            f"Total leaf nodes: {len(leaf_nodes)}, Low-visit candidate nodes: {len(candidates)}, "
            f"Found before exploration paths: {len(potential_exploration_paths)}, Added to results: {len(exploration_paths)}"
        )

        return combined_paths

    def _simplified_path_similarity(self, path1, path2):
        """
        Calculate the similarity between two paths, considering constraint order and relaxation factors

        Args:
            path1: First path
            path2: Second path

        Returns:
            float: Similarity between 0-1, where 1 means completely identical
        """
        pairs1 = []
        pairs2 = []

        idx = 1 if path1[0] == "root" else 0
        while idx < len(path1) - 1:
            if isinstance(path1[idx], str) and path1[idx] != "root" and isinstance(path1[idx+1], (int, float)):
                pairs1.append((path1[idx], path1[idx+1]))
            idx += 2

        idx = 1 if path2[0] == "root" else 0
        while idx < len(path2) - 1:
            if isinstance(path2[idx], str) and path2[idx] != "root" and isinstance(path2[idx+1], (int, float)):
                pairs2.append((path2[idx], path2[idx+1]))
            idx += 2

        if not pairs1 or not pairs2:
            return 0.0

        constraints1 = [p[0] for p in pairs1]
        constraints2 = [p[0] for p in pairs2]

        if set(constraints1) != set(constraints2):
            return 0.0

        order_similarity = 0.0
        if len(constraints1) == len(constraints2):
            same_pos = sum(1 for i in range(len(constraints1))
                           if constraints1[i] == constraints2[i])
            order_similarity = same_pos / len(constraints1)

        factor_similarity = 0.0
        common_constraints = set(constraints1) & set(constraints2)
        if common_constraints:
            factors1 = {pairs1[i][0]: pairs1[i][1] for i in range(len(pairs1))}
            factors2 = {pairs2[i][0]: pairs2[i][1] for i in range(len(pairs2))}

            factor_diffs = []
            for constraint in common_constraints:
                if constraint in factors1 and constraint in factors2:
                    max_factor = max(
                        abs(factors1[constraint]), abs(factors2[constraint]))
                    if max_factor > 0:
                        diff = abs(factors1[constraint] -
                                   factors2[constraint]) / max_factor
                        factor_diffs.append(1.0 - min(diff, 1.0))

            if factor_diffs:
                factor_similarity = sum(factor_diffs) / len(factor_diffs)

        combined_similarity = 0.6 * order_similarity + 0.4 * factor_similarity

        return combined_similarity

    def _format_path(self, path: List[Union[str, float]]) -> str:
        """
        Format path into a readable string

        Args:
            path: Path

        Returns:
            str: Formatted path string
        """
        if not path or len(path) < 2:
            return "Empty path"

        result = []

        idx = 1 if path[0] == "root" else 0

        while idx < len(path) - 1:
            constraint = path[idx]
            relaxation = path[idx + 1]

            if isinstance(constraint, str) and isinstance(relaxation, (int, float)):
                result.append(f"{constraint} relaxed to {relaxation}")

            idx += 2

        return "; ".join(result)

    def get_best_strategy(self) -> Dict:
        """
        Get the best strategy

        Returns:
            Dict: Strategy information
        """
        paths = self.get_best_paths(1)

        if not paths:
            return {
                "path": [],
                "formatted_path": "No strategy found",
                "constraint_order": [],
                "relaxation_factors": {}
            }

        best_path = paths[0]["path"]

        constraint_order = []
        relaxation_factors = {}

        idx = 1 if best_path[0] == "root" else 0

        while idx < len(best_path) - 1:
            constraint = best_path[idx]
            relaxation = best_path[idx + 1]

            if isinstance(constraint, str) and isinstance(relaxation, (int, float)):
                constraint_order.append(constraint)
                relaxation_factors[constraint] = relaxation

            idx += 2

        return {
            "path": best_path,
            "formatted_path": self._format_path(best_path),
            "constraint_order": constraint_order,
            "relaxation_factors": relaxation_factors,
            "statistics": self._get_statistics()
        }

    def _evaluate_path(self, path: List[Any]) -> float:
        """
        Evaluate the reward of a path

        Args:
            path: Path

        Returns:
            float: Reward value
        """
        if self.evaluation_function:
            constraint_order, relaxation_factors = self._extract_strategy_from_path(
                path)

            if not constraint_order or not relaxation_factors:
                return float('-inf')

            path_str = str((constraint_order, relaxation_factors))
            path_hash = self._get_path_hash(path)

            if path_hash in self.evaluation_cache:
                return self.evaluation_cache[path_hash]

            try:
                strategy_id = path[-1] if isinstance(
                    path[-1], str) and len(path[-1]) > 10 else None

                if hasattr(self, 'text_strategy_manager') and hasattr(self, 'strategy_evaluator'):
                    if strategy_id is None or strategy_id not in self.text_strategy_manager.strategies:
                        strategy_text = "Strategy: MCTS exploration\n"
                        for i, constraint in enumerate(constraint_order):
                            strategy_text += f"{i+1}. {constraint} relaxed to {relaxation_factors[constraint]}\n"

                        strategy_id = self.text_strategy_manager._create_strategy(
                            strategy_text,
                            method="mcts",
                            generation=0,
                            parent_ids=[],
                            outer_iteration=getattr(self, 'outer_iteration', 0)
                        )

                        if len(path) > 0:
                            path[-1] = strategy_id
                        else:
                            path.append(strategy_id)

                reward = self.evaluation_function(path)

                self.evaluation_cache[path_hash] = reward
                return reward
            except Exception as e:
                self.output_manager.log_warning(
                    "mcts", "evaluation_error",
                    f"Path evaluation failed: {e}"
                )
                return float('-inf')

        return 0.0

    def update_statistics(self, constraint_order: List[str], relaxation_factors: Dict[str, float], reward: float) -> bool:
        """
        Update node statistics from external sources, allowing evolutionary algorithms to feedback results to MCTS

        Args:
            constraint_order: Constraint processing order
            relaxation_factors: Constraint relaxation factors
            reward: Reward value from strategy evaluation

        Returns:
            bool: Whether update was successful
        """
        try:
            path = ["root"]
            for constraint in constraint_order:
                path.append(constraint)
                relaxation = relaxation_factors.get(constraint, 1.0)
                path.append(relaxation)

            import uuid as uuid_module

            current_node_id = self.root_id
            current_depth = 0

            for i in range(1, len(path), 2):
                if i+1 >= len(path):
                    break

                constraint = path[i]
                relaxation = path[i+1]

                child_found = False
                if current_node_id in self.nodes:
                    current_node = self.nodes[current_node_id]

                    for child_id in current_node.children_ids:
                        child_node = self.nodes[child_id]
                        if child_node.value == constraint:
                            current_node_id = child_id
                            child_found = True
                            break

                    if not child_found:
                        new_constraint_id = str(uuid_module.uuid4())
                        new_constraint_node = MCTSNode(
                            id=new_constraint_id,
                            node_type="constraint",
                            value=constraint,
                            parent_id=current_node_id,
                            depth=current_depth + 1
                        )
                        self.nodes[new_constraint_id] = new_constraint_node
                        current_node.children_ids.append(new_constraint_id)
                        current_node_id = new_constraint_id

                    relaxation_node_id = None
                    current_node = self.nodes[current_node_id]
                    current_depth += 1

                    for child_id in current_node.children_ids:
                        child_node = self.nodes[child_id]
                        if abs(child_node.value - relaxation) < 0.001:
                            relaxation_node_id = child_id
                            break

                    if not relaxation_node_id:
                        new_relaxation_id = str(uuid_module.uuid4())
                        new_relaxation_node = MCTSNode(
                            id=new_relaxation_id,
                            node_type="relaxation",
                            value=relaxation,
                            parent_id=current_node_id,
                            depth=current_depth + 1
                        )
                        self.nodes[new_relaxation_id] = new_relaxation_node
                        current_node.children_ids.append(new_relaxation_id)
                        current_node_id = new_relaxation_id
                    else:
                        current_node_id = relaxation_node_id

                    current_depth += 1
                else:
                    return False

            if current_node_id in self.nodes:
                node = self.nodes[current_node_id]
                node.visits += 1
                node.total_reward += reward

                self._backpropagation(current_node_id, reward)

                self.output_manager.log_info(
                    "mcts", "external_statistics_update",
                    f"Updated MCTS statistics from external source, path: {' -> '.join(map(str, path))}, reward: {reward}"
                )
                return True

            return False
        except Exception as e:
            self.output_manager.log_error(
                "mcts", "statistics_update_failed",
                f"Failed to update MCTS statistics: {str(e)}"
            )
            return False
