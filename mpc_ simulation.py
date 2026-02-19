"""
Tactical MPC
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import IntEnum
from scipy import stats
import itertools
import json
import time
import os



# ─────────────────────────────────────────────────────────────────────
# Constants & Enums
# ─────────────────────────────────────────────────────────────────────

class Model(IntEnum):
    NANO = 0
    SMALL = 1
    MEDIUM = 2


MODEL_NAMES = {Model.NANO: "nano", Model.SMALL: "small", Model.MEDIUM: "medium"}

# Power consumption (Watts) — from Jetson Orin Nano benchmarking (representative)
POWER = {
    Model.NANO:   3.5,   # W
    Model.SMALL:  5.8,   # W
    Model.MEDIUM: 9.2,   # W
}


# ─────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class NIGParams:
    """Normal-Inverse-Gamma parameters from the evidential output layer."""
    gamma: float    # predicted mean (scene complexity)
    nu: float       # precision of mean estimate (confidence)
    alpha: float    # shape of inverse-gamma over σ²
    beta: float     # scale of inverse-gamma over σ²

    @property
    def prediction(self) -> float:
        """E[μ] = γ"""
        return self.gamma

    @property
    def aleatoric_uncertainty(self) -> float:
        """E[σ²] = β / (α - 1)"""
        if self.alpha <= 1.0:
            return float('inf')
        return self.beta / (self.alpha - 1.0)

    @property
    def epistemic_uncertainty(self) -> float:
        """Var[μ] = β / (ν(α - 1))"""
        if self.alpha <= 1.0 or self.nu <= 0:
            return float('inf')
        return self.beta / (self.nu * (self.alpha - 1.0))


@dataclass
class ClassProbabilities:
    """Discrete difficulty class probabilities derived from Student-t CDF."""
    p_easy: float
    p_moderate: float
    p_hard: float


@dataclass
class LSTMForecast:
    """Complete forecast for one horizon step from the Evidential LSTM."""
    nig: NIGParams
    class_probs: ClassProbabilities
    epistemic_unc: float


@dataclass
class MPCConfig:
    """Configuration parameters for the Tactical MPC."""
    # Loss matrix
    lambda_under: float = 1.0       # cost of under-provisioning (missing detections)
    lambda_over_base: float = 0.3   # base cost of over-provisioning (wasting energy)
    kappa: float = 2.0              # budget slack sensitivity for λ_over modulation

    # Switching penalty
    w_switch: float = 0.15          # base switching penalty
    eta: float = 5.0                # sensitivity to epistemic uncertainty

    # Horizon
    N: int = 5                      # MPC planning horizon (steps)
    delta_t: float = 1.0            # timestep duration (seconds)

    # Difficulty thresholds (for Student-t CDF conversion)
    t1: float = 0.25                # easy/moderate boundary
    t2: float = 0.55                # moderate/hard boundary


@dataclass
class StrategicPlannerState:
    """State maintained by the Strategic Planner."""
    E_remaining: float              # total energy remaining in budget (Joules)
    E_budget_total: float           # full budget at reset (Joules)
    E_consumed_hour: float = 0.0    # energy consumed in current hour (Joules)
    current_hour: int = 0           # current hour index (0-23)
    current_day: int = 0            # current day index (0-6, Mon=0)
    demand_profile: np.ndarray = field(default=None)  # 7×24 demand table
    alpha_ema: float = 0.1          # EMA adaptation rate

    def __post_init__(self):
        if self.demand_profile is None:
            self.demand_profile = np.full((7, 24), 0.5)


# ─────────────────────────────────────────────────────────────────────
# Synthetic Evidential LSTM Simulator
# ─────────────────────────────────────────────────────────────────────

class SyntheticEvidentialLSTM:
    """
    Simulates the Evidential LSTM for a real-time surveillance scenario.

    Generates realistic scene complexity patterns with temporal correlations,
    producing NIG parameters as if output by the evidential LSTM layer.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._time_step = 0
        self._base_complexity = 0.3   # current underlying complexity

    def _get_time_of_day_factor(self, hour: int) -> float:
        """
        Surveillance complexity profile: peaks during rush hours,
        low during night.
        """
        # Morning rush 7-9, evening rush 17-19, quiet 0-5
        profile = np.array([
            0.05, 0.05, 0.05, 0.08, 0.10, 0.15,  # 0-5 AM (quiet)
            0.25, 0.70, 0.85, 0.60, 0.50, 0.45,  # 6-11 AM (morning rush)
            0.50, 0.55, 0.45, 0.50, 0.65, 0.80,  # 12-5 PM (afternoon)
            0.75, 0.55, 0.35, 0.20, 0.10, 0.05,  # 6-11 PM (evening)
        ])
        return profile[hour % 24]

    def _evolve_complexity(self, hour: int) -> float:
        """Evolve scene complexity with temporal correlations and noise."""
        target = self._get_time_of_day_factor(hour)
        # Smooth drift toward target + stochastic jumps
        drift = 0.1 * (target - self._base_complexity)
        noise = self.rng.normal(0, 0.05)
        jump = 0.3 * (self.rng.random() < 0.02)  # rare sudden events
        self._base_complexity = np.clip(
            self._base_complexity + drift + noise + jump, 0.0, 1.0
        )
        return self._base_complexity

    def generate_nig_params(self, complexity: float) -> NIGParams:
        """
        Generate NIG parameters simulating the evidential output layer.

        Higher complexity → slightly higher epistemic uncertainty (less training data
        for complex scenes). Aleatoric uncertainty reflects scene stochasticity.
        """
        gamma = complexity + self.rng.normal(0, 0.03)  # predicted mean
        gamma = np.clip(gamma, 0.0, 1.0)

        # Confidence inversely related to complexity volatility
        nu = max(2.0, 15.0 - 8.0 * complexity + self.rng.normal(0, 1.0))
        alpha = max(1.5, 5.0 - 2.0 * complexity + self.rng.normal(0, 0.3))
        beta = max(0.01, 0.05 + 0.1 * complexity + self.rng.normal(0, 0.01))

        return NIGParams(gamma=gamma, nu=nu, alpha=alpha, beta=beta)

    def forecast(self, hour: int, horizon: int) -> List[LSTMForecast]:
        """
        Generate forecasts over a horizon, simulating autoregressive LSTM output.

        Args:
            hour: Current hour of day (0-23)
            horizon: Number of future steps to forecast

        Returns:
            List of LSTMForecast for each horizon step
        """
        forecasts = []
        temp_complexity = self._base_complexity

        for k in range(horizon):
            # Future complexity drifts with increasing uncertainty
            target = self._get_time_of_day_factor(hour)
            drift = 0.08 * (target - temp_complexity)
            noise = self.rng.normal(0, 0.03 * (1 + 0.3 * k))  # grows with horizon
            temp_complexity = np.clip(temp_complexity + drift + noise, 0.0, 1.0)

            nig = self.generate_nig_params(temp_complexity)
            # Epistemic uncertainty grows with forecast horizon
            nig = NIGParams(
                gamma=nig.gamma,
                nu=max(1.5, nig.nu - 0.5 * k),   # confidence decays
                alpha=nig.alpha,
                beta=nig.beta
            )

            class_probs = nig_to_class_probabilities(nig)
            ep_unc = nig.epistemic_uncertainty

            forecasts.append(LSTMForecast(
                nig=nig, class_probs=class_probs, epistemic_unc=ep_unc
            ))

        return forecasts

    def step(self, hour: int) -> float:
        """Advance one timestep and return current scene complexity."""
        self._time_step += 1
        return self._evolve_complexity(hour)


# ─────────────────────────────────────────────────────────────────────
# Student-t CDF Conversion  (Section 4.2.1.2)
# ─────────────────────────────────────────────────────────────────────

def nig_to_class_probabilities(
    nig: NIGParams,
    t1: float = 0.25,
    t2: float = 0.55
) -> ClassProbabilities:
    """
    Convert NIG parameters to discrete class probabilities using
    the Student-t predictive marginal CDF.

    The predictive distribution is Student-t with:
        location = γ
        scale    = sqrt(β(1+ν) / (να))
        df       = 2α
    """
    df = 2.0 * nig.alpha
    scale = np.sqrt(nig.beta * (1.0 + nig.nu) / (nig.nu * nig.alpha))

    # Avoid degenerate cases
    if scale < 1e-8:
        scale = 1e-8
    if df < 1.0:
        df = 1.0

    rv = stats.t(df=df, loc=nig.gamma, scale=scale)

    p_easy = rv.cdf(t1)
    p_moderate = rv.cdf(t2) - rv.cdf(t1)
    p_hard = 1.0 - rv.cdf(t2)

    # Ensure valid probabilities
    total = p_easy + p_moderate + p_hard
    if total < 1e-10:
        return ClassProbabilities(1 / 3, 1 / 3, 1 / 3)

    return ClassProbabilities(
        p_easy=p_easy / total,
        p_moderate=p_moderate / total,
        p_hard=p_hard / total
    )


# ─────────────────────────────────────────────────────────────────────
# Strategic Planner Simulator  (Section 4.2.3)
# ─────────────────────────────────────────────────────────────────────

class StrategicPlanner:
    """
    Simulates the Strategic Planner for long-horizon budget allocation.

    Maintains a weekly demand profile and computes the budget floor F(t)
    to determine how much energy the MPC may spend in the current hour.
    """

    def __init__(
        self,
        total_budget_joules: float = 500_000.0,  # ~24h at moderate load
        hours_in_period: int = 24,
        seed: int = 123
    ):
        self.rng = np.random.RandomState(seed)
        self.hours_in_period = hours_in_period

        # Initialize demand profile from a synthetic surveillance pattern
        self.state = StrategicPlannerState(
            E_remaining=total_budget_joules,
            E_budget_total=total_budget_joules,
            demand_profile=self._init_demand_profile()
        )

    def _init_demand_profile(self) -> np.ndarray:
        """
        Initialize D[day][hour] from a realistic surveillance demand pattern.
        Values in [0, 1] represent expected complexity (mAP gap magnitude).
        """
        profile = np.zeros((7, 24))
        base_hourly = np.array([
            0.05, 0.05, 0.05, 0.08, 0.10, 0.15,
            0.25, 0.70, 0.85, 0.60, 0.50, 0.45,
            0.50, 0.55, 0.45, 0.50, 0.65, 0.80,
            0.75, 0.55, 0.35, 0.20, 0.10, 0.05,
        ])
        for day in range(7):
            day_scale = 1.0 if day < 5 else 0.6  # weekends are quieter
            noise = self.rng.normal(0, 0.03, 24)
            profile[day] = np.clip(base_hourly * day_scale + noise, 0.0, 1.0)
        return profile

    def expected_power(self, demand: float) -> float:
        """
        P̄(h) = P(nano) + d × (P(medium) - P(nano))
        Linear interpolation from nano to medium power based on demand.
        """
        return POWER[Model.NANO] + demand * (POWER[Model.MEDIUM] - POWER[Model.NANO])

    def compute_budget_floor(self, current_hour: int, current_day: int) -> float:
        """
        F(t) = Σ_{h=t+1}^{T} P̄(h) × Δt_hour

        Energy that must be reserved for future hours until budget reset.
        """
        floor = 0.0
        for h in range(current_hour + 1, self.hours_in_period):
            demand = self.state.demand_profile[current_day % 7][h]
            floor += self.expected_power(demand) * 3600.0  # Δt_hour = 3600s
        return floor

    def compute_available_energy(self, current_hour: int, current_day: int) -> float:
        """
        E_available(t) = E_remaining(t) - F(t)
        """
        floor = self.compute_budget_floor(current_hour, current_day)
        return max(0.0, self.state.E_remaining - floor)

    def check_degraded_mode(
        self, E_available: float, E_consumed: float, t_remaining: float
    ) -> bool:
        """
        Check minimum guarantee condition (Section 4.2.3.3):
        E_available - E_consumed ≤ P(nano) × t_remaining → degraded mode
        """
        return (E_available - E_consumed) <= POWER[Model.NANO] * t_remaining

    def end_of_hour_update(
        self, E_consumed: float, predicted_complexity: float,
        current_hour: int, current_day: int
    ):
        """Update state at hour boundary."""
        # Update remaining energy
        self.state.E_remaining -= E_consumed
        self.state.E_consumed_hour = 0.0

        # Online EMA update of demand profile
        alpha = self.state.alpha_ema
        old = self.state.demand_profile[current_day % 7][current_hour]
        self.state.demand_profile[current_day % 7][current_hour] = (
            (1 - alpha) * old + alpha * predicted_complexity
        )

    def reset_budget(self):
        """Reset budget at period boundary."""
        self.state.E_remaining = self.state.E_budget_total
        self.state.E_consumed_hour = 0.0


# ─────────────────────────────────────────────────────────────────────
# Tactical MPC  (Section 4.2.2)
# ─────────────────────────────────────────────────────────────────────

class TacticalMPC:
    """
    Tactical Model Predictive Controller for real-time model selection.

    Solves a finite-horizon optimization over YOLO11 variants (nano, small, medium)
    minimizing Bayes risk + uncertainty-weighted switching penalty subject to
    energy constraints from the Strategic Planner.
    """

    def __init__(self, config: MPCConfig = None):
        self.config = config or MPCConfig()
        self.current_model: Model = Model.NANO  # initial model
        self.E_consumed: float = 0.0             # energy consumed this hour (J)
        self.step_count: int = 0
        self.history: List[Dict] = []

    # ── Detection Loss (Bayes Risk)  ──────────────────────────────────

    def _detection_loss(
        self, model: Model, probs: ClassProbabilities, lambda_over: float
    ) -> float:
        """
        L_det(k, m) = Σ_j λ(m|ω_j) × P(ω_j | x̂_k)

        From the loss matrix (Table 8):
            nano:   λ_under × (1 - P(easy))
            small:  λ_over  × P(easy) + λ_under × P(hard)
            medium: λ_over  × (1 - P(hard))
        """
        lu = self.config.lambda_under
        lo = lambda_over

        if model == Model.NANO:
            return lu * (1.0 - probs.p_easy)
        elif model == Model.SMALL:
            return lo * probs.p_easy + lu * probs.p_hard
        else:  # MEDIUM
            return lo * (1.0 - probs.p_hard)

    # ── Switching Loss ────────────────────────────────────────────────

    def _switching_loss(
        self, model_k: Model, model_prev: Model, epistemic_unc: float
    ) -> float:
        """
        L_switch(k) = w_switch × (1 + η × σ²_ep,k) × 𝟙[m_k ≠ m_{k-1}]
        """
        if model_k == model_prev:
            return 0.0
        return self.config.w_switch * (1.0 + self.config.eta * epistemic_unc)

    # ── Budget-Aware λ_over Modulation ────────────────────────────────

    def _compute_lambda_over(self, E_available: float) -> float:
        """
        s(t) = (E_available - E_consumed) / E_available
        λ_over(t) = λ_over,base × (1 + κ(1 - s(t)))

        When budget is plentiful (s≈1): λ_over ≈ λ_over,base
        When budget is scarce  (s≈0): λ_over ≈ λ_over,base × (1 + κ)
        """
        if E_available <= 0:
            return self.config.lambda_over_base * (1.0 + self.config.kappa)

        slack = max(0.0, (E_available - self.E_consumed)) / E_available
        slack = np.clip(slack, 0.0, 1.0)
        return self.config.lambda_over_base * (1.0 + self.config.kappa * (1.0 - slack))

    # ── Horizon Cost for a Model Sequence ─────────────────────────────

    def _evaluate_sequence(
        self,
        sequence: List[Model],
        forecasts: List[LSTMForecast],
        lambda_over: float,
        prev_model: Model
    ) -> float:
        """Compute total cost J(m) for a candidate model sequence."""
        cost = 0.0
        m_prev = prev_model
        for k, (m_k, fc) in enumerate(zip(sequence, forecasts)):
            det = self._detection_loss(m_k, fc.class_probs, lambda_over)
            sw = self._switching_loss(m_k, m_prev, fc.epistemic_unc)
            cost += det + sw
            m_prev = m_k
        return cost

    def _check_energy_feasible(
        self, sequence: List[Model], E_available: float
    ) -> bool:
        """
        Check constraint: E_consumed + Σ_k P(m_k) · Δt ≤ E_available
        """
        planned_energy = sum(
            POWER[m] * self.config.delta_t for m in sequence
        )
        return (self.E_consumed + planned_energy) <= E_available


    @staticmethod
    def _is_adjacent(m1: Model, m2: Model) -> bool:
        """
        Check if transition from m1 to m2 is allowed.
        Only single-step transitions permitted:
            nano ↔ small ↔ medium
        No direct nano ↔ medium jumps.
        """
        return abs(int(m1) - int(m2)) <= 1

    def _is_sequence_valid(self, sequence: List[Model], start_model: Model) -> bool:
        """
        Validate that an entire sequence respects the adjacency constraint.
        Each consecutive pair (including start_model → sequence[0]) must be adjacent.
        """
        prev = start_model
        for m in sequence:
            if not self._is_adjacent(prev, m):
                return False
            prev = m
        return True

    # ── MPC Optimization (Exhaustive for small horizon) ───────────────

    def optimize(
        self,
        forecasts: List[LSTMForecast],
        E_available: float,
        degraded: bool = False
    ) -> Tuple[Model, Dict]:
        """
        Solve the MPC optimization problem.

        Args:
            forecasts: List of LSTMForecast from the Evidential LSTM
            E_available: Energy budget from the Strategic Planner (Joules)
            degraded: If True, force nano-only operation

        Returns:
            (selected_model, info_dict)

        """
        N = min(self.config.N, len(forecasts))

        # Degraded mode — nano-only operation unconditionally (Section 4.2.3.3)
        # "the MPC optimization is bypassed, and the nano model is selected
        #  unconditionally until the next budget reset."
        if degraded:
            return Model.NANO, {"mode": "degraded", "cost": 0.0}

        # Compute budget-aware λ_over
        lambda_over = self._compute_lambda_over(E_available)

        best_cost = float('inf')
        best_sequence = [Model.NANO] * N
        models = list(Model)

        # Exhaustive search over 3^N sequences (3^5 = 243 — very fast)
        for seq in itertools.product(models, repeat=N):
            seq = list(seq)

            # Adjacency constraint: no skipping tiers (nano↔small↔medium only)
            if not self._is_sequence_valid(seq, self.current_model):
                continue

            # Energy feasibility check
            if not self._check_energy_feasible(seq, E_available):
                continue

            cost = self._evaluate_sequence(
                seq, forecasts[:N], lambda_over, self.current_model
            )
            if cost < best_cost:
                best_cost = cost
                best_sequence = seq

        # Execute first action (receding horizon)
        selected = best_sequence[0]

        info = {
            "mode": "normal",
            "cost": best_cost,
            "lambda_over": lambda_over,
            "budget_slack": max(0, (E_available - self.E_consumed) / max(E_available, 1e-6)),
            "planned_sequence": [MODEL_NAMES[m] for m in best_sequence],
            "forecasted_probs": [
                {
                    "p_easy": f"{fc.class_probs.p_easy:.3f}",
                    "p_mod": f"{fc.class_probs.p_moderate:.3f}",
                    "p_hard": f"{fc.class_probs.p_hard:.3f}",
                    "ep_unc": f"{fc.epistemic_unc:.4f}",
                }
                for fc in forecasts[:N]
            ],
        }

        return selected, info

    def execute(
        self, selected_model: Model, forecasts: List[LSTMForecast]
    ) -> Dict:
        """
        Execute the selected model and update internal state.

        Returns a record of this step for logging.
        """
        prev_model = self.current_model
        self.current_model = selected_model

        # Update energy consumed
        energy_used = POWER[selected_model] * self.config.delta_t
        self.E_consumed += energy_used
        self.step_count += 1

        record = {
            "step": self.step_count,
            "model": MODEL_NAMES[selected_model],
            "power_W": POWER[selected_model],
            "energy_J": energy_used,
            "E_consumed_total": self.E_consumed,
            "switched": selected_model != prev_model,
            "complexity_pred": forecasts[0].nig.prediction if forecasts else None,
            "epistemic_unc": forecasts[0].epistemic_unc if forecasts else None,
        }
        self.history.append(record)
        return record

    def reset_hour(self):
        """Reset hourly consumption counter."""
        consumed = self.E_consumed
        self.E_consumed = 0.0
        return consumed


# ─────────────────────────────────────────────────────────────────────
# Simulation Runner
# ─────────────────────────────────────────────────────────────────────

def run_simulation(
    duration_hours: int = 24,
    steps_per_second: float = 1.0,
    mpc_config: MPCConfig = None,
    total_budget_joules: float = 500_000.0,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Run a full ECHO simulation for a surveillance deployment scenario.

    Args:
        duration_hours: Length of simulation in hours
        steps_per_second: MPC control cycles per second
        mpc_config: MPC configuration (defaults used if None)
        total_budget_joules: Total energy budget for the period
        seed: Random seed for reproducibility
        verbose: Print hourly summaries

    Returns:
        Dictionary with full simulation results
    """
    config = mpc_config or MPCConfig()
    lstm = SyntheticEvidentialLSTM(seed=seed)
    planner = StrategicPlanner(
        total_budget_joules=total_budget_joules,
        hours_in_period=duration_hours,
        seed=seed + 1
    )
    mpc = TacticalMPC(config=config)

    steps_per_hour = int(3600 * steps_per_second)
    current_day = 0  # Monday

    all_records = []
    hourly_summaries = []

    if verbose:
        print("=" * 80)
        print("Tactical MPC")
        print(f"Duration: {duration_hours}h | Budget: {total_budget_joules/1000:.0f} kJ "
              f"| MPC Horizon: {config.N} | Δt: {config.delta_t}s")
        print("=" * 80)

    for hour in range(duration_hours):
        # Strategic Planner computes available energy for this hour
        E_available = planner.compute_available_energy(hour, current_day)
        t_remaining_hour = 3600.0  # full hour

        # Check degraded mode
        degraded = planner.check_degraded_mode(
            E_available, mpc.E_consumed, t_remaining_hour
        )

        hour_models = {Model.NANO: 0, Model.SMALL: 0, Model.MEDIUM: 0}
        hour_switches = 0
        hour_complexities = []

        for step in range(steps_per_hour):
            # LSTM evolves scene and generates forecast
            current_complexity = lstm.step(hour)
            forecasts = lstm.forecast(hour, config.N)
            hour_complexities.append(current_complexity)

            # Update degraded check intra-hour
            t_remaining = (steps_per_hour - step) * config.delta_t
            degraded = planner.check_degraded_mode(
                E_available, mpc.E_consumed, t_remaining
            )

            # MPC optimizes and selects model
            selected, info = mpc.optimize(forecasts, E_available, degraded)
            record = mpc.execute(selected, forecasts)
            record["hour"] = hour
            record["E_available"] = E_available
            record["degraded"] = degraded
            all_records.append(record)

            hour_models[selected] += 1
            if record["switched"]:
                hour_switches += 1

        # End of hour
        E_consumed_hour = mpc.reset_hour()
        avg_complexity = np.mean(hour_complexities) if hour_complexities else 0.0
        planner.end_of_hour_update(E_consumed_hour, avg_complexity, hour, current_day)

        summary = {
            "hour": hour,
            "E_available": E_available,
            "E_consumed": E_consumed_hour,
            "E_remaining": planner.state.E_remaining,
            "avg_complexity": avg_complexity,
            "model_counts": {MODEL_NAMES[m]: c for m, c in hour_models.items()},
            "switches": hour_switches,
            "degraded_steps": sum(1 for r in all_records[-steps_per_hour:]
                                 if r.get("degraded")),
        }
        hourly_summaries.append(summary)

        if verbose:
            dominant = max(hour_models, key=hour_models.get)
            budget_pct = (planner.state.E_remaining / total_budget_joules) * 100
            print(
                f"  Hour {hour:2d} | "
                f"Complexity: {avg_complexity:.3f} | "
                f"Dominant: {MODEL_NAMES[dominant]:6s} "
                f"(N:{hour_models[Model.NANO]:4d} S:{hour_models[Model.SMALL]:4d} "
                f"M:{hour_models[Model.MEDIUM]:4d}) | "
                f"Switches: {hour_switches:3d} | "
                f"E_consumed: {E_consumed_hour/1000:.1f}kJ | "
                f"Budget: {budget_pct:.1f}%"
                f"{' [DEGRADED]' if summary['degraded_steps'] > 0 else ''}"
            )

    # Final statistics
    total_energy = sum(s["E_consumed"] for s in hourly_summaries)
    total_switches = sum(s["switches"] for s in hourly_summaries)
    model_totals = {m: 0 for m in MODEL_NAMES.values()}
    for s in hourly_summaries:
        for m, c in s["model_counts"].items():
            model_totals[m] += c

    results = {
        "config": {
            "duration_hours": duration_hours,
            "total_budget_J": total_budget_joules,
            "mpc_horizon": config.N,
            "lambda_under": config.lambda_under,
            "lambda_over_base": config.lambda_over_base,
            "kappa": config.kappa,
            "w_switch": config.w_switch,
            "eta": config.eta,
        },
        "summary": {
            "total_energy_J": total_energy,
            "budget_utilization": total_energy / total_budget_joules,
            "total_switches": total_switches,
            "model_distribution": model_totals,
            "total_steps": len(all_records),
        },
        "hourly": hourly_summaries,
        "records": all_records,
    }

    if verbose:
        print("=" * 80)
        print("SIMULATION COMPLETE")
        print(f"  Total Energy:  {total_energy/1000:.1f} kJ "
              f"({total_energy/total_budget_joules*100:.1f}% of budget)")
        print(f"  Model Usage:   {model_totals}")
        print(f"  Total Switches: {total_switches}")
        print("=" * 80)

    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Default configuration matching the paper's design
    config = MPCConfig(
        lambda_under=1.0,
        lambda_over_base=0.3,
        kappa=2.0,
        w_switch=0.15,
        eta=5.0,
        N=5,
        delta_t=1.0,
        t1=0.25,
        t2=0.55,
    )

    results = run_simulation(
        duration_hours=24,
        steps_per_second=1.0,     # 1 MPC cycle per second
        mpc_config=config,
        total_budget_joules=500_000.0,  # ~500 kJ for 24h
        seed=42,
        verbose=True,
    )

    # Save results
    # Convert records to JSON-serializable format
    output = {
        "config": results["config"],
        "summary": results["summary"],
        "hourly": results["hourly"],
    }
file_path = os.path.join(os.path.dirname(__file__), "echo_results.json")
with open(file_path, "w") as f:
    json.dump(output, f, indent=2, default=str)