"""
VRPTW (Vehicle Routing Problem with Time Windows) - example template.

A problem directory contains three files:
  - problem_data.json  : problem definition (constraints / objectives / parameters)
  - instances.json     : instance data (coordinates, demands, time windows, ...)
  - PointSelectionAlgorithm.py : this template

The framework keeps everything outside the two marker areas unchanged and only
asks the LLM to rewrite the code between
"# ============ LLM Fill Area - START/END ============".
"""
import math


class Solution:
    def __init__(self, problem_data, problem_params, relaxation_strategy):
        self.problem_data = problem_data
        self.relaxation_strategy = relaxation_strategy or []

        # The runner injects a single instance into "instance_data".
        instance = problem_data.get("instance_data", problem_data)
        self.coordinates = instance.get("coordinates", [[0, 0]])
        self.distance_matrix = instance.get("distance_matrix")
        self.demands = instance.get("demands", [0])
        self.vehicle_capacity = instance.get("vehicle_capacity", 10)
        self.time_windows = instance.get("time_windows", [[0, 10 ** 9]])
        self.service_time = instance.get("service_time", [0])
        self.max_vehicles = instance.get("max_vehicles", 3)
        self.depot = instance.get("depot", 0)
        self.n_customers = len(self.demands) - 1

    # ============ Utility Function Area - START ============
    # Helpers already implemented; the generated code may call them.

    def get_relaxation_factor(self, name, default=1.0):
        for item in self.relaxation_strategy:
            if item.get("constraint") == name:
                return item.get("relaxation", default)
        return default

    def calculate_distance(self, i, j):
        if self.distance_matrix is not None:
            return self.distance_matrix[i][j]
        p1, p2 = self.coordinates[i], self.coordinates[j]
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def parse_routes(self, route):
        routes, current = [], [self.depot]
        for node in route[1:]:
            current.append(node)
            if node == self.depot:
                routes.append(current)
                current = [self.depot]
        return routes

    def evaluate_solution(self, solution):
        route = solution.get("route", [])
        if not route:
            return {"objective": float("-inf"), "feasible": False, "violations": ["empty_route"]}

        routes = self.parse_routes(route)
        capacity_relax = self.get_relaxation_factor("capacity_constraint")
        fleet_relax = self.get_relaxation_factor("fleet_size_constraint")

        violations = []
        if any(sum(self.demands[n] for n in r if n != self.depot) >
               self.vehicle_capacity * capacity_relax for r in routes):
            violations.append("capacity_constraint")
        if len(routes) > self.max_vehicles * fleet_relax:
            violations.append("fleet_size_constraint")
        if sorted(n for n in route if n != self.depot) != list(range(1, self.n_customers + 1)):
            violations.append("visit_constraint")
        if not self._time_windows_ok(route):
            violations.append("time_window_constraint")

        distance = sum(self.calculate_distance(route[k], route[k + 1])
                       for k in range(len(route) - 1))
        feasible = not violations
        return {
            "objective": -distance if feasible else float("-inf"),
            "feasible": feasible,
            "violations": violations,
        }

    def _time_windows_ok(self, route):
        t = 0.0
        for k in range(len(route) - 1):
            t += self.calculate_distance(route[k], route[k + 1])
            if route[k + 1] == self.depot:
                t = 0.0
                continue
            t = max(t, self.time_windows[route[k + 1]][0])
            if t > self.time_windows[route[k + 1]][1]:
                return False
            t += self.service_time[route[k + 1]]
        return True
    # ============ Utility Function Area - END ============

    # ============ LLM Fill Area - START ============
    # The LLM rewrites this function. Keep the name and return a full route:
    # [0, customer, ..., 0, customer, ..., 0].
    def select_next_point(self):
        route, remaining = [self.depot], set(range(1, self.n_customers + 1))
        current, load = self.depot, 0
        capacity = self.vehicle_capacity * self.get_relaxation_factor("capacity_constraint")
        while remaining:
            nxt = min(
                (p for p in remaining if load + self.demands[p] <= capacity),
                key=lambda p: self.calculate_distance(current, p),
                default=None,
            )
            if nxt is None:
                route.append(self.depot)
                current, load = self.depot, 0
                continue
            route.append(nxt)
            remaining.remove(nxt)
            load += self.demands[nxt]
            current = nxt
        if route[-1] != self.depot:
            route.append(self.depot)
        return route
    # ============ LLM Fill Area - END ============

    def solve(self):
        route = self.select_next_point()
        evaluation = self.evaluate_solution({"route": route})
        return {
            "solution": {"route": route},
            "objective": evaluation["objective"],
            "success": evaluation["feasible"],
            "metrics": evaluation,
        }
