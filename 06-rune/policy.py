from __future__ import annotations

from math import inf

from rune_scheduler.api.v1 import (
    AttemptPhase,
    DispatchAction,
    DispatchDecision,
    Observation,
    Placement,
    PublicScenarioModel,
    WorkerSpec,
    WorkerState,
)


class Policy:
    def __init__(self, model: PublicScenarioModel) -> None:
        self._types = {task_type.id: task_type for task_type in model.task_types}
        self._specs = {worker.id: worker for worker in model.workers}
        self._max_batch = model.safety_limits.max_batch_size
        self._speed = {
            worker.id: {
                item.task_type: item.ratio.numerator / item.ratio.denominator
                for item in worker.duration_multipliers
            }
            for worker in model.workers
        }
        self._children: dict[str, set[str]] = {name: set() for name in self._types}
        for rule in model.derivations:
            self._children.setdefault(rule.parent_type, set()).add(rule.successor_type)
        for template in model.workflow_templates:
            by_node = {node.id: node.task_type for node in template.nodes}
            for node in template.nodes:
                for parent in node.depends_on:
                    parent_type = by_node.get(parent)
                    if parent_type is not None:
                        self._children.setdefault(parent_type, set()).add(node.task_type)
        self._criticality = self._build_criticality()
        self._mean_runtime = {
            name: (item.runtime_work.min + item.runtime_work.max) / 2
            for name, item in self._types.items()
        }

    def _build_criticality(self) -> dict[str, float]:
        memo: dict[str, float] = {}
        visiting: set[str] = set()

        def depth(task_type: str) -> float:
            if task_type in memo:
                return memo[task_type]
            if task_type in visiting:
                return 0.0
            visiting.add(task_type)
            task = self._types.get(task_type)
            own = 0.0 if task is None else self._mean(task.runtime_work) + task.setup_work
            tail = max(
                (depth(child) for child in self._children.get(task_type, ())),
                default=0.0,
            )
            visiting.remove(task_type)
            memo[task_type] = own + 0.65 * tail
            return memo[task_type]

        for task_type in self._types:
            depth(task_type)
        return memo

    @staticmethod
    def _mean(value) -> float:
        return (value.min + value.max) / 2

    def _worker_loads(
        self, observation: Observation, worker_specs: dict[str, WorkerSpec]
    ) -> tuple[dict[str, float], dict[str, float]]:
        load: dict[str, float] = {worker_id: 0.0 for worker_id in worker_specs}
        longest: dict[str, float] = {worker_id: 0.0 for worker_id in worker_specs}
        attempts_by_worker: dict[str, list] = {worker_id: [] for worker_id in worker_specs}
        for attempt in observation.active_attempts:
            attempts_by_worker.setdefault(attempt.worker_id, []).append(attempt)

        groups = {group.reference: group for group in observation.setup_groups}
        for worker_id, attempts in attempts_by_worker.items():
            spec = worker_specs.get(worker_id)
            if spec is None:
                continue
            capacity = max(1, spec.cpu_capacity)
            state = next(
                (w for w in observation.workers if w.worker_id == worker_id), None
            )
            slowdown = 1.0
            if state is not None and state.progress_factor.denominator:
                slowdown = state.progress_factor.denominator / max(1, state.progress_factor.numerator)

            seen_groups = set()
            for attempt in attempts:
                task = self._types.get(attempt.task_type)
                if task is None:
                    continue
                ratio = self._speed.get(worker_id, {}).get(attempt.task_type, 1.0)
                runtime_service = (
                    attempt.service_units
                    if attempt.phase == AttemptPhase.RUNNING
                    else 0
                )
                remaining = max(
                    0.0, self._mean_runtime[attempt.task_type] - runtime_service
                )
                duration = remaining * ratio * slowdown
                if (
                    attempt.phase == AttemptPhase.AWAITING_SETUP
                    and attempt.setup_group is not None
                ):
                    ref = attempt.setup_group
                    if ref not in seen_groups:
                        seen_groups.add(ref)
                        group = groups.get(ref)
                        setup_remaining = max(
                            0.0,
                            (task.setup_work if group is None else group.setup_work)
                            - (0 if group is None else group.service_units),
                        )
                        members = 1 if group is None else max(1, len(group.members))
                        setup_duration = setup_remaining * ratio * slowdown
                        duration += setup_duration
                        load[worker_id] += (
                            setup_duration * task.cpu_demand * members / capacity
                        )
                load[worker_id] += remaining * ratio * slowdown * task.cpu_demand / capacity
                longest[worker_id] = max(longest[worker_id], duration)
        return load, longest

    def choose_placements(self, observation: Observation) -> DispatchDecision:
        if not observation.ready_tasks:
            return DispatchDecision(DispatchAction.DEFER, ())

        specs = self._specs
        states: dict[str, WorkerState] = {
            worker.worker_id: worker for worker in observation.workers
        }
        remaining_mem = {
            worker_id: specs[worker_id].memory_capacity - state.memory_reserved
            for worker_id, state in states.items()
            if worker_id in specs
        }
        cpu_demand = {worker_id: state.cpu_demand for worker_id, state in states.items()}
        load, longest = self._worker_loads(observation, specs)
        placed: list[Placement] = []
        batch_type_counts: dict[tuple[str, str], int] = {}

        pending = list(observation.ready_tasks)
        while pending and len(placed) < self._max_batch:
            candidates = []
            feasible_count: dict[str, int] = {}
            for task in pending:
                task_type = self._types.get(task.task_type)
                if task_type is None:
                    continue
                feasible = []
                required = set(task_type.required_capabilities)
                for worker_id, state in states.items():
                    spec = specs.get(worker_id)
                    if (
                        spec is not None
                        and state.available
                        and required.issubset(spec.capabilities)
                        and remaining_mem[worker_id] >= task_type.memory_reservation
                        and (
                            cpu_demand[worker_id] == 0
                            or cpu_demand[worker_id] + task_type.cpu_demand
                            <= spec.cpu_capacity
                        )
                    ):
                        feasible.append(worker_id)
                feasible_count[task.task_id] = len(feasible)
                if feasible:
                    candidates.append(task)

            if not candidates:
                break

            # Preserve workers with rare capabilities for tasks that need them.
            task = max(
                candidates,
                key=lambda item: (
                    self._criticality.get(item.task_type, 0.0),
                    -feasible_count.get(item.task_id, 0),
                    -item.failed_attempts,
                ),
            )
            task_type = self._types[task.task_type]
            # Compare dispatching now with waiting for a compatible busy worker.
            # This estimate uses only currently observed work, never future tasks.
            best_finish = inf
            for wid, ws in states.items():
                sp = specs[wid]
                if (
                    not ws.available
                    or not set(task_type.required_capabilities).issubset(
                        sp.capabilities
                    )
                    or sp.memory_capacity < task_type.memory_reservation
                ):
                    continue
                d = (
                    task_type.setup_work + self._mean_runtime[task.task_type]
                ) * self._speed.get(wid, {}).get(task.task_type, 1.0)
                best_finish = min(best_finish, load[wid] + d)
            choices = []
            for worker_id, state in states.items():
                spec = specs.get(worker_id)
                if (
                    spec is None
                    or not state.available
                    or not set(task_type.required_capabilities).issubset(spec.capabilities)
                    or remaining_mem[worker_id] < task_type.memory_reservation
                    or (
                        cpu_demand[worker_id] > 0
                        and cpu_demand[worker_id] + task_type.cpu_demand
                        > spec.cpu_capacity
                    )
                ):
                    continue

                ratio = self._speed.get(worker_id, {}).get(task.task_type, 1.0)
                count = batch_type_counts.get((worker_id, task.task_type), 0)
                setup = 0.0 if count else float(task_type.setup_work)
                runtime = self._mean_runtime[task.task_type]
                duration = (setup + runtime) * ratio
                if (
                    (task_type.setup_work + runtime) * ratio > 1.0 * best_finish
                    and (observation.active_attempts or placed)
                ):
                    continue
                demand_after = cpu_demand[worker_id] + task_type.cpu_demand
                capacity = max(1, spec.cpu_capacity)
                overload = max(1.0, demand_after / capacity)
                predicted_load = load[worker_id] + duration * task_type.cpu_demand / capacity
                predicted_span = max(longest[worker_id], duration * overload)

                outage = spec.outage_prior
                reliability = 1.0
                if outage is not None:
                    uptime = max(1.0, self._mean(outage.uptime))
                    downtime = self._mean(outage.duration)
                    reliability += min(0.35, downtime / (4.0 * uptime) + duration / (12.0 * uptime))
                score = (predicted_load + 0.12 * predicted_span) * reliability
                choices.append((score, predicted_load, worker_id, duration))

            if not choices:
                pending.remove(task)
                continue

            _, _, worker_id, duration = min(choices)
            placed.append(Placement(task.task_id, worker_id))
            pending.remove(task)
            remaining_mem[worker_id] -= task_type.memory_reservation
            cpu_demand[worker_id] += task_type.cpu_demand
            load[worker_id] += duration * task_type.cpu_demand / max(1, specs[worker_id].cpu_capacity)
            longest[worker_id] = max(longest[worker_id], duration)
            key = (worker_id, task.task_type)
            batch_type_counts[key] = batch_type_counts.get(key, 0) + 1

        if placed:
            return DispatchDecision(DispatchAction.DISPATCH, tuple(placed))
        return DispatchDecision(DispatchAction.DEFER, ())
