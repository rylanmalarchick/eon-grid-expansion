from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from time import perf_counter
from typing import Any

import gurobipy as gp
import pandapower as pp
from gurobipy import GRB

from eon.instances.candidate_lines import CandidateLine
from eon.instances.distribution_feeders import slack_bus_index
from eon.instances.scenarios import Scenario
from eon.metrics.objective import ObjectiveWeights, ScenarioMetrics, aggregate_metrics


@dataclass(frozen=True, slots=True)
class ExpansionProblemConfig:
    max_new_lines: int = 3
    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05
    base_voltage_pu: float = 1.0
    n_scenarios_aggregated: int = 3
    time_limit_s: float = 60.0
    mip_gap: float = 0.01
    flow_big_m_mva: float = 25.0
    voltage_big_m_pu: float = 0.35
    weights: ObjectiveWeights = ObjectiveWeights()


@dataclass(frozen=True, slots=True)
class ExpansionResult:
    objective_value: float
    runtime_s: float
    mip_gap: float | None
    best_bound: float | None
    termination_status: str
    selected_candidates: tuple[str, ...]
    build_decisions: dict[str, float]
    scenario_metrics: list[ScenarioMetrics]
    bus_voltage_pu: dict[str, dict[int, float]]
    line_loading_proxy_mva: dict[str, dict[str, float]]
    aggregate_metrics: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scenario_metrics"] = [asdict(metric) for metric in self.scenario_metrics]
        return payload


@dataclass(frozen=True, slots=True)
class _LineSpec:
    name: str
    from_bus: int
    to_bus: int
    r_pu: float
    x_pu: float
    capacity_mva: float
    build_cost: float
    is_candidate: bool


@dataclass(frozen=True, slots=True)
class _OptimizationResult:
    backend: str
    status: str
    runtime_s: float
    objective_value: float
    best_bound: float | None
    mip_gap: float | None
    variable_values: dict[str, float]

    @property
    def has_solution(self) -> bool:
        return bool(self.variable_values)


def solve_lindistflow_expansion(
    net: pp.pandapowerNet,
    scenarios: list[Scenario],
    candidates: list[CandidateLine],
    config: ExpansionProblemConfig | None = None,
    *,
    fixed_builds: dict[str, int] | None = None,
) -> ExpansionResult:
    cfg = config or ExpansionProblemConfig()
    fixed_builds = fixed_builds or {}
    slack_bus = slack_bus_index(net)
    buses = [int(bus) for bus in net.bus.index]
    base_kv_by_bus = {int(bus): float(vn_kv) for bus, vn_kv in net.bus["vn_kv"].items()}
    existing_lines = _existing_line_specs(net, base_kv_by_bus)
    candidate_specs = [
        _candidate_to_spec(candidate, net.sn_mva, base_kv_by_bus) for candidate in candidates
    ]
    all_lines = existing_lines + candidate_specs

    load_p = _aggregate_bus_table(net.load, "p_mw", buses)
    load_q = _aggregate_bus_table(net.load, "q_mvar", buses)
    gen_p = _aggregate_generation(net, buses)

    model = gp.Model("lindistflow_expansion")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = cfg.time_limit_s
    model.Params.MIPGap = cfg.mip_gap

    build_vars: dict[str, gp.Var] = {}
    for line in candidate_specs:
        build_vars[line.name] = model.addVar(vtype=GRB.BINARY, name=f"build[{line.name}]")
        if line.name in fixed_builds:
            model.addConstr(
                build_vars[line.name] == int(fixed_builds[line.name]),
                name=f"fix[{line.name}]",
            )
    if build_vars:
        model.addConstr(
            gp.quicksum(build_vars.values()) <= cfg.max_new_lines,
            name="candidate_budget",
        )

    p_flow: dict[tuple[str, str], gp.Var] = {}
    q_flow: dict[tuple[str, str], gp.Var] = {}
    p_abs: dict[tuple[str, str], gp.Var] = {}
    q_abs: dict[tuple[str, str], gp.Var] = {}
    overload: dict[tuple[str, str], gp.Var] = {}
    v_bus: dict[tuple[str, int], gp.Var] = {}
    v_low: dict[tuple[str, int], gp.Var] = {}
    v_high: dict[tuple[str, int], gp.Var] = {}
    curtail: dict[tuple[str, int], gp.Var] = {}
    import_p: dict[str, gp.Var] = {}
    import_q: dict[str, gp.Var] = {}
    incident: dict[int, list[_LineSpec]] = defaultdict(list)

    for line in all_lines:
        incident[line.from_bus].append(line)
        incident[line.to_bus].append(line)

    for scenario in scenarios:
        import_p[scenario.name] = model.addVar(lb=0.0, name=f"grid_import[{scenario.name}]")
        import_q[scenario.name] = model.addVar(lb=0.0, name=f"grid_import_q[{scenario.name}]")
        for bus in buses:
            v_bus[(scenario.name, bus)] = model.addVar(
                lb=cfg.voltage_min_pu - cfg.voltage_big_m_pu,
                ub=cfg.voltage_max_pu + cfg.voltage_big_m_pu,
                name=f"v[{scenario.name},{bus}]",
            )
            v_low[(scenario.name, bus)] = model.addVar(lb=0.0, name=f"v_low[{scenario.name},{bus}]")
            v_high[(scenario.name, bus)] = model.addVar(
                lb=0.0,
                name=f"v_high[{scenario.name},{bus}]",
            )
            curtail[(scenario.name, bus)] = model.addVar(
                lb=0.0,
                ub=gen_p[bus] * scenario.generation_scale,
                name=f"curtail[{scenario.name},{bus}]",
            )
            model.addConstr(
                v_bus[(scenario.name, bus)] + v_low[(scenario.name, bus)] >= cfg.voltage_min_pu,
                name=f"voltage_min[{scenario.name},{bus}]",
            )
            model.addConstr(
                v_bus[(scenario.name, bus)] - v_high[(scenario.name, bus)] <= cfg.voltage_max_pu,
                name=f"voltage_max[{scenario.name},{bus}]",
            )
        model.addConstr(
            v_bus[(scenario.name, slack_bus)] == cfg.base_voltage_pu,
            name=f"slack_v[{scenario.name}]",
        )

        for line in all_lines:
            key = (scenario.name, line.name)
            cap = max(line.capacity_mva * scenario.line_capacity_scale, 1e-3)
            p_flow[key] = model.addVar(
                lb=-cfg.flow_big_m_mva,
                ub=cfg.flow_big_m_mva,
                name=f"p[{scenario.name},{line.name}]",
            )
            q_flow[key] = model.addVar(
                lb=-cfg.flow_big_m_mva,
                ub=cfg.flow_big_m_mva,
                name=f"q[{scenario.name},{line.name}]",
            )
            p_abs[key] = model.addVar(lb=0.0, name=f"p_abs[{scenario.name},{line.name}]")
            q_abs[key] = model.addVar(lb=0.0, name=f"q_abs[{scenario.name},{line.name}]")
            overload[key] = model.addVar(lb=0.0, name=f"overload[{scenario.name},{line.name}]")
            z = 1.0 if not line.is_candidate else build_vars[line.name]
            cap = min(cap, cfg.flow_big_m_mva)
            model.addConstr(
                p_abs[key] >= p_flow[key],
                name=f"p_abs_pos[{scenario.name},{line.name}]",
            )
            model.addConstr(
                p_abs[key] >= -p_flow[key],
                name=f"p_abs_neg[{scenario.name},{line.name}]",
            )
            model.addConstr(
                q_abs[key] >= q_flow[key],
                name=f"q_abs_pos[{scenario.name},{line.name}]",
            )
            model.addConstr(
                q_abs[key] >= -q_flow[key],
                name=f"q_abs_neg[{scenario.name},{line.name}]",
            )
            model.addConstr(
                p_abs[key] <= cfg.flow_big_m_mva,
                name=f"p_abs_cap[{scenario.name},{line.name}]",
            )
            model.addConstr(
                q_abs[key] <= cfg.flow_big_m_mva,
                name=f"q_abs_cap[{scenario.name},{line.name}]",
            )
            model.addConstr(
                p_abs[key] + q_abs[key] <= cap * z + overload[key],
                name=f"thermal[{scenario.name},{line.name}]",
            )
            voltage_drop = (
                v_bus[(scenario.name, line.to_bus)] - v_bus[(scenario.name, line.from_bus)]
            )
            flow_drop = 2.0 * (line.r_pu * p_flow[key] + line.x_pu * q_flow[key])
            model.addConstr(
                voltage_drop + flow_drop <= cfg.voltage_big_m_pu * (1.0 - z),
                name=f"vdrop_upper[{scenario.name},{line.name}]",
            )
            model.addConstr(
                voltage_drop + flow_drop >= -cfg.voltage_big_m_pu * (1.0 - z),
                name=f"vdrop_lower[{scenario.name},{line.name}]",
            )

        for bus in buses:
            p_balance: list[Any] = []
            q_balance: list[Any] = []
            for line in incident[bus]:
                key = (scenario.name, line.name)
                if line.from_bus == bus:
                    p_balance.append(p_flow[key])
                    q_balance.append(q_flow[key])
                else:
                    p_balance.append(-p_flow[key])
                    q_balance.append(-q_flow[key])
            supply: Any = gen_p[bus] * scenario.generation_scale - curtail[(scenario.name, bus)]
            demand_p = load_p[bus] * scenario.load_scale
            demand_q: Any = load_q[bus] * scenario.load_scale
            if bus == slack_bus:
                supply = supply + import_p[scenario.name]
                demand_q = demand_q - import_q[scenario.name]
            model.addConstr(
                gp.quicksum(p_balance) + supply - demand_p == 0.0,
                name=f"p_balance[{scenario.name},{bus}]",
            )
            model.addConstr(
                gp.quicksum(q_balance) - demand_q == 0.0,
                name=f"q_balance[{scenario.name},{bus}]",
            )

    objective_terms = []
    for line in candidate_specs:
        objective_terms.append(cfg.weights.build_cost * line.build_cost * build_vars[line.name])
    for scenario in scenarios:
        probability = scenario.probability
        for bus in buses:
            objective_terms.append(
                probability
                * cfg.weights.voltage_violation
                * (v_low[(scenario.name, bus)] + v_high[(scenario.name, bus)])
            )
            objective_terms.append(
                probability * cfg.weights.curtailment * curtail[(scenario.name, bus)]
            )
        for line in all_lines:
            key = (scenario.name, line.name)
            objective_terms.append(probability * cfg.weights.thermal_violation * overload[key])
            objective_terms.append(probability * cfg.weights.loss_proxy * line.r_pu * p_abs[key])
    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)

    optimization = _optimize_expansion_model(
        model,
        time_limit_s=cfg.time_limit_s,
        mip_gap=cfg.mip_gap,
    )

    termination_status = optimization.status
    objective_value = optimization.objective_value
    best_bound = optimization.best_bound
    mip_gap = optimization.mip_gap
    build_decisions = (
        {
            name: round(optimization.variable_values[var.VarName], 6)
            for name, var in build_vars.items()
        }
        if optimization.has_solution
        else {}
    )
    selected_candidates = tuple(
        sorted(name for name, value in build_decisions.items() if value >= 0.5)
    )

    scenario_metrics: list[ScenarioMetrics] = []
    bus_voltage_pu: dict[str, dict[int, float]] = {}
    line_loading_proxy_mva: dict[str, dict[str, float]] = {}
    if optimization.has_solution:
        for scenario in scenarios:
            voltage_map = {
                bus: optimization.variable_values[v_bus[(scenario.name, bus)].VarName]
                for bus in buses
            }
            loading_map: dict[str, float] = {}
            thermal = 0.0
            voltage_violation = 0.0
            curtailment = 0.0
            loss_proxy = 0.0
            for bus in buses:
                voltage_violation += (
                    optimization.variable_values[v_low[(scenario.name, bus)].VarName]
                    + optimization.variable_values[v_high[(scenario.name, bus)].VarName]
                )
                curtailment += optimization.variable_values[curtail[(scenario.name, bus)].VarName]
            for line in all_lines:
                key = (scenario.name, line.name)
                loading = (
                    optimization.variable_values[p_abs[key].VarName]
                    + optimization.variable_values[q_abs[key].VarName]
                )
                loading_map[line.name] = loading
                thermal += optimization.variable_values[overload[key].VarName]
                loss_proxy += line.r_pu * optimization.variable_values[p_abs[key].VarName]
            scenario_metrics.append(
                ScenarioMetrics(
                    scenario=scenario.name,
                    probability=scenario.probability,
                    thermal_overlimit_mw=thermal,
                    voltage_violation_pu=voltage_violation,
                    curtailment_mw=curtailment,
                    loss_proxy_mw=loss_proxy,
                )
            )
            bus_voltage_pu[scenario.name] = voltage_map
            line_loading_proxy_mva[scenario.name] = loading_map

    return ExpansionResult(
        objective_value=objective_value,
        runtime_s=optimization.runtime_s,
        mip_gap=mip_gap,
        best_bound=best_bound,
        termination_status=termination_status,
        selected_candidates=selected_candidates,
        build_decisions=build_decisions,
        scenario_metrics=scenario_metrics,
        bus_voltage_pu=bus_voltage_pu,
        line_loading_proxy_mva=line_loading_proxy_mva,
        aggregate_metrics=aggregate_metrics(scenario_metrics),
        metadata={
            "bus_count": len(buses),
            "existing_line_count": len(existing_lines),
            "candidate_count": len(candidate_specs),
            "slack_bus": slack_bus,
            "solver_backend": optimization.backend,
        },
    )


def _existing_line_specs(
    net: pp.pandapowerNet,
    base_kv_by_bus: dict[int, float],
) -> list[_LineSpec]:
    line_specs: list[_LineSpec] = []
    for line_index, row in net.line.iterrows():
        from_bus = int(row.from_bus)
        to_bus = int(row.to_bus)
        base_kv = max(base_kv_by_bus[from_bus], base_kv_by_bus[to_bus])
        z_base = (base_kv**2) / max(net.sn_mva, 1e-3)
        r_pu = float(row.r_ohm_per_km * row.length_km / z_base)
        x_pu = float(row.x_ohm_per_km * row.length_km / z_base)
        capacity = float(sqrt(3.0) * base_kv * row.max_i_ka)
        line_specs.append(
            _LineSpec(
                name=f"line_{line_index}",
                from_bus=from_bus,
                to_bus=to_bus,
                r_pu=r_pu,
                x_pu=x_pu,
                capacity_mva=max(capacity, 1e-3),
                build_cost=0.0,
                is_candidate=False,
            )
        )
    for trafo_index, row in net.trafo.iterrows():
        from_bus = int(row.hv_bus)
        to_bus = int(row.lv_bus)
        base_kv = max(base_kv_by_bus[from_bus], base_kv_by_bus[to_bus])
        z_base = (base_kv**2) / max(net.sn_mva, 1e-3)
        r_pu = float((row.vkr_percent / 100.0) / max(row.sn_mva, 1e-6) * net.sn_mva)
        x_sq = max((row.vk_percent / 100.0) ** 2 - (row.vkr_percent / 100.0) ** 2, 0.0)
        x_pu = float(sqrt(x_sq) / max(row.sn_mva, 1e-6) * net.sn_mva)
        capacity = float(row.sn_mva)
        line_specs.append(
            _LineSpec(
                name=f"trafo_{trafo_index}",
                from_bus=from_bus,
                to_bus=to_bus,
                r_pu=r_pu / max(z_base, 1e-6),
                x_pu=x_pu / max(z_base, 1e-6),
                capacity_mva=max(capacity, 1e-3),
                build_cost=0.0,
                is_candidate=False,
            )
        )
    return line_specs


def _candidate_to_spec(
    candidate: CandidateLine,
    sn_mva: float,
    base_kv_by_bus: dict[int, float],
) -> _LineSpec:
    base_kv = max(base_kv_by_bus[candidate.from_bus], base_kv_by_bus[candidate.to_bus])
    z_base = (base_kv**2) / max(sn_mva, 1e-3)
    r_pu = candidate.r_ohm_per_km * candidate.length_km / z_base
    x_pu = candidate.x_ohm_per_km * candidate.length_km / z_base
    capacity = sqrt(3.0) * base_kv * candidate.max_i_ka
    return _LineSpec(
        name=candidate.name,
        from_bus=candidate.from_bus,
        to_bus=candidate.to_bus,
        r_pu=float(r_pu),
        x_pu=float(x_pu),
        capacity_mva=float(max(capacity, 1e-3)),
        build_cost=candidate.build_cost,
        is_candidate=True,
    )


def _aggregate_bus_table(table: Any, value_col: str, buses: list[int]) -> dict[int, float]:
    values = {bus: 0.0 for bus in buses}
    if table is None or not len(table):
        return values
    for _, row in table.iterrows():
        values[int(row.bus)] += float(row[value_col])
    return values


def _aggregate_generation(net: pp.pandapowerNet, buses: list[int]) -> dict[int, float]:
    values = {bus: 0.0 for bus in buses}
    if len(net.gen):
        for _, row in net.gen.iterrows():
            if "slack" in net.gen.columns and bool(row.slack):
                continue
            values[int(row.bus)] += float(row.p_mw)
    if len(net.sgen):
        for _, row in net.sgen.iterrows():
            values[int(row.bus)] += float(row.p_mw)
    return values


def _optimize_expansion_model(
    model: gp.Model,
    *,
    time_limit_s: float,
    mip_gap: float,
) -> _OptimizationResult:
    start = perf_counter()
    try:
        model.optimize()
    except gp.GurobiError as exc:
        if not _is_size_limited_license_error(exc):
            raise
        return _optimize_with_highs(
            model,
            time_limit_s=time_limit_s,
            mip_gap=mip_gap,
        )

    runtime_s = perf_counter() - start
    variable_values = (
        {var.VarName: float(var.X) for var in model.getVars()} if model.SolCount else {}
    )
    return _OptimizationResult(
        backend="gurobi",
        status=_status_name(model.Status),
        runtime_s=runtime_s,
        objective_value=float(model.ObjVal) if model.SolCount else float("inf"),
        best_bound=(
            float(model.ObjBound)
            if model.SolCount or model.Status == GRB.TIME_LIMIT
            else None
        ),
        mip_gap=float(model.MIPGap) if model.SolCount and model.IsMIP else None,
        variable_values=variable_values,
    )


def _optimize_with_highs(
    model: gp.Model,
    *,
    time_limit_s: float,
    mip_gap: float,
) -> _OptimizationResult:
    try:
        from highspy import Highs, HighsStatus
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Model exceeded the size-limited Gurobi license, but highspy is not installed."
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as handle:
        model_path = handle.name
    try:
        model.write(model_path)
        highs = Highs()
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("time_limit", time_limit_s)
        highs.setOptionValue("mip_rel_gap", mip_gap)
        highs.setOptionValue("mip_heuristic_effort", 0.0)
        read_status = highs.readModel(model_path)
        if read_status != HighsStatus.kOk:
            raise RuntimeError(f"HiGHS failed to read exported model: {read_status}")

        start = perf_counter()
        run_status = highs.run()
        runtime_s = perf_counter() - start
        if run_status not in {HighsStatus.kOk, HighsStatus.kWarning}:
            raise RuntimeError(f"HiGHS failed while solving exported model: {run_status}")

        info = highs.getInfo()
        variable_values: dict[str, float] = {}
        if isfinite(float(info.objective_function_value)):
            solution = highs.getSolution()
            for column in range(highs.getNumCol()):
                _, name = highs.getColName(column)
                variable_values[name] = float(solution.col_value[column])

        return _OptimizationResult(
            backend="highs",
            status=_highs_status_name(highs),
            runtime_s=runtime_s,
            objective_value=(
                float(info.objective_function_value) if variable_values else float("inf")
            ),
            best_bound=(
                float(info.mip_dual_bound) if isfinite(float(info.mip_dual_bound)) else None
            ),
            mip_gap=(
                float(info.mip_gap)
                if variable_values and isfinite(float(info.mip_gap))
                else None
            ),
            variable_values=variable_values,
        )
    finally:
        os.unlink(model_path)


def _is_size_limited_license_error(exc: gp.GurobiError) -> bool:
    return "Model too large for size-limited license" in str(exc)


def _status_name(status_code: int) -> str:
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return status_map.get(status_code, str(status_code))


def _highs_status_name(highs: Any) -> str:
    status_text = highs.modelStatusToString(highs.getModelStatus())
    status_map = {
        "Optimal": "OPTIMAL",
        "Time limit": "TIME_LIMIT",
        "Infeasible": "INFEASIBLE",
        "Unbounded": "UNBOUNDED",
        "Unbounded or infeasible": "INF_OR_UNBD",
    }
    return status_map.get(status_text, status_text.upper().replace(" ", "_"))
