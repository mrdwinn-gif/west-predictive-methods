"""Active Inference / Free Energy model + multi-agent POMDP.

Extracted from a Gemini methods export ("Incomplete Transmission
Evaluation", partial export).

Theory (from the source, after Karl Friston's Free Energy Principle):
an agent holds preferred states ``P*(s)`` (its intent) and minimises
variational free energy ``F = E_Q[ln Q(s) - ln P(o, s)]``; it selects
policies by minimising expected free energy ``G(pi)``, which trades
epistemic value (information seeking / ambiguity resolution) against
pragmatic value (goal seeking / staying close to preferred outcomes
``C = P*(o)``). Policy posteriors follow a softmax,
``P(pi) = softmax(-gamma * G)``.

Two classes are preserved from the source:
* ``AdversarialActiveInference`` -- a single agent choosing between
  three policies (infiltrate / probe / egress) under a fixed
  observation model ``A = P(o|s)``.
* ``MultiAgentActiveInferencePOMDP`` -- a defender/adversary pair in
  which the defender's action directly warps the adversary's
  A-matrix (passive / jamming / spoofing), shifting the adversary's
  ``G`` and hence its policy distribution; the defender picks the
  action maximising expected preference fulfilment minus action cost.

This is a textbook-style simulation of the published Active Inference
formalism. It contains no operational, targeting, or intelligence
content.
"""

import numpy as np


class AdversarialActiveInference:
    def __init__(self, n_states: int, n_obs: int, n_policies: int):
        self.S = n_states
        self.O = n_obs
        self.K = n_policies

        # A matrix: P(o | s) of shape (O, S)
        self.A = np.eye(self.O, self.S)

        # C vector: ln P*(o) - Log preferences over observation states
        # Preferences: 0 = Safe/Covert (High), 1 = Ambiguous, 2 = Compromised/Targeted (Extremely Low)
        self.ln_C = np.array([2.0, 0.0, -10.0])

        # Gamma: Policy Precision parameter
        self.gamma = 1.0

    def compute_expected_free_energy(self, qs_pi: np.ndarray, qo_pi: np.ndarray) -> float:
        """
        Calculates G(pi) = Pragmatic Cost (Risk) + Epistemic Ambiguity.
        G(pi) = - E_Q[ln P*(o)] - E_Q[H(A)]
        """
        # 1. Pragmatic Cost: KL-divergence against target preferences
        # Negative dot product: - sum( Q(o|pi) * ln P*(o) )
        pragmatic_risk = -np.sum(qo_pi * self.ln_C)

        # 2. Epistemic Cost / Ambiguity: Expected conditional entropy H(A)
        # H(A_j) = - sum_i A[i, j] * ln(A[i, j])
        eps = 1e-12
        entropy_A = -np.sum(self.A * np.log(self.A + eps), axis=0)
        epistemic_ambiguity = np.sum(qs_pi * entropy_A)

        # Total G(pi) to minimize
        return float(pragmatic_risk + epistemic_ambiguity)

    def evaluate_policy_posteriors(self, projected_qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            projected_qs: Shape (K, S) - Projected state distribution under each candidate policy.
        Returns:
            G: Array of shape (K,) with expected free energy per policy.
            P_pi: Softmax policy distribution of shape (K,).
        """
        G = np.zeros(self.K)
        for k in range(self.K):
            qs_k = projected_qs[k]
            # Project observation distribution: Q(o | pi) = A * Q(s | pi)
            qo_k = self.A @ qs_k
            G[k] = self.compute_expected_free_energy(qs_k, qo_k)

        # Policy selection via softmax: P(pi) = softmax(-gamma * G)
        scaled_neg_G = -self.gamma * (G - np.min(G))  # Numerical stability shift
        exp_G = np.exp(scaled_neg_G)
        P_pi = exp_G / np.sum(exp_G)

        return G, P_pi


class MultiAgentActiveInferencePOMDP:
    def __init__(self):
        # State Space S: [0: Infiltrating/Unchecked, 1: Ambiguous Sector, 2: Under Defensive Lock]
        self.n_states = 3
        # Adversary Observations O: [0: Path Clear, 1: Sensor Degraded, 2: Threat Alarm Detected]
        self.n_obs = 3

        # Adversary Policies K:
        # Policy 0: Direct Assault / Sustained Infiltration
        # Policy 1: Covert Perimeter Reconnaissance
        # Policy 2: Egress / Emergency Abort
        self.n_adv_policies = 3

        # Defender Actions M:
        # 0: Passive Reconnaissance (Monitor only)
        # 1: RF / Sensor Denial (Inject high entropy noise)
        # 2: Active Deception / Honeypot Trapping (Spoof 'Clear' while forcing State 2)
        self.n_def_actions = 3

        # Adversary Preferences: ln C_adv
        # Strongly prefers 'Path Clear' (2.0), dislikes 'Degraded' (-1.0), detests 'Alarm' (-12.0)
        self.ln_C_adv = np.array([2.0, -1.0, -12.0])

        # Defender Preferences: ln C_def over Adversary Policies executed
        # Defender wants Adversary to pick Policy 2 (Abort: +5.0) or Policy 1 (Recon: +1.0); penalizes Policy 0 (Assault: -10.0)
        self.ln_C_def = np.array([-10.0, 1.0, 5.0])
        self.def_action_costs = np.array([0.0, 1.2, 2.0])  # Operational cost/exposure of deploying EW/Deception

        # Adversary Precision
        self.gamma_adv = 1.0

    def get_adversary_A_matrix(self, def_action: int) -> np.ndarray:
        """
        Dynamically returns the Adversary's A-matrix conditioned on Defender's action.
        Shape: (n_obs, n_states) where A[i, j] = P(o=i | s=j)
        """
        eps = 1e-6
        if def_action == 0:
            # Passive: High fidelity mapping
            A = np.array([
                [0.90, 0.10, 0.05],   # Clear
                [0.08, 0.80, 0.15],   # Degraded
                [0.02, 0.10, 0.80]    # Alarm
            ])
        elif def_action == 1:
            # EW / Jamming: Smashes state mapping into maximum entropy (blackout)
            A = np.array([
                [0.20, 0.20, 0.20],
                [0.60, 0.60, 0.60],   # Heavy noise across all states
                [0.20, 0.20, 0.20]
            ])
        elif def_action == 2:
            # Active Deception / Spoofing: Inverts threat telemetry
            # Target sees "Clear" even when walking into Defensive Lock (s=2)
            A = np.array([
                [0.85, 0.70, 0.75],   # Spoofed Clear
                [0.10, 0.20, 0.15],
                [0.05, 0.10, 0.10]    # Suppressed Alarms
            ])
        else:
            raise ValueError("Invalid defensive action.")

        return A / np.sum(A, axis=0, keepdims=True)

    def evaluate_adversary_policies(self, def_action: int,
                                    projected_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates Adversary's Expected Free Energy G and policy posteriors P(pi)
        given a specific defensive posture.

        Args:
            def_action: Defender's chosen tactical action.
            projected_states: Shape (n_policies, n_states) - Adversary's projected states.
        """
        A = self.get_adversary_A_matrix(def_action)
        eps = 1e-12

        # Precompute conditional entropy H(A) = - sum_i A[i, j] * ln A[i, j]
        H_A = -np.sum(A * np.log(A + eps), axis=0)  # Shape (n_states,)

        G = np.zeros(self.n_adv_policies)
        for k in range(self.n_adv_policies):
            qs_k = projected_states[k]
            qo_k = A @ qs_k  # Projected observation distribution

            # Pragmatic Risk: - sum( qo_k * ln C_adv )
            pragmatic_risk = -np.sum(qo_k * self.ln_C_adv)

            # Epistemic Ambiguity: sum( qs_k * H(A) )
            epistemic_ambiguity = np.sum(qs_k * H_A)

            G[k] = pragmatic_risk + epistemic_ambiguity

        # Softmax over negative Expected Free Energy
        shifted_G = -self.gamma_adv * (G - np.min(G))
        exp_G = np.exp(shifted_G)
        P_pi_adv = exp_G / np.sum(exp_G)

        return G, P_pi_adv

    def compute_defender_optimal_policy(self, baseline_projected_states: np.ndarray) -> dict:
        """
        Solves the meta-game: Defender evaluates the adversarial policy shift
        induced by each defensive action and selects the optimal counter-measure.
        """
        defender_payoffs = np.zeros(self.n_def_actions)
        detailed_log = []

        for a_def in range(self.n_def_actions):
            # Alter adversary's belief projection under defensive action
            # Note: Active deception (a_def=2) channels adversary into locked states
            states = baseline_projected_states.copy()
            if a_def == 1:
                # Jamming forces higher state uncertainty
                states[0] = np.array([0.40, 0.40, 0.20])
            elif a_def == 2:
                # Deception allows adversary to push deeper into physical target zones (s=2)
                states[0] = np.array([0.10, 0.10, 0.80])

            G_adv, P_pi_adv = self.evaluate_adversary_policies(a_def, states)

            # Defender Utility = Expected Prior Fulfillment - Tactical Cost
            # Defender goal: Maximize E_P(pi)[ln C_def] - Action_Cost
            expected_strategic_utility = np.sum(P_pi_adv * self.ln_C_def)
            net_utility = expected_strategic_utility - self.def_action_costs[a_def]
            defender_payoffs[a_def] = net_utility

            detailed_log.append({
                "def_action": a_def,
                "G_adv": G_adv,
                "P_pi_adv": P_pi_adv,
                "net_utility": net_utility
            })

        best_def_action = int(np.argmax(defender_payoffs))
        return {
            "optimal_def_action": best_def_action,
            "defender_payoffs": defender_payoffs,
            "detailed_log": detailed_log
        }


# =====================================================================
# Execution Scenario: Injection of Counter-Surveillance Telemetry
# =====================================================================
def _run_single_agent_demo():
    # 3 States: [Unobserved, Ambiguous, Tracked]
    # 3 Observations: [Clean, Signal Noise, Locked by Interdiction]
    # 3 Policies: [0: Infiltrate, 1: Probe/Recon, 2: Emergency Egress]
    engine = AdversarialActiveInference(n_states=3, n_obs=3, n_policies=3)

    # Baseline Scenario: Target believes infiltration path is secure
    projected_qs_baseline = np.array([
        [0.85, 0.10, 0.05],  # Policy 0: Leads to Unobserved state
        [0.40, 0.50, 0.10],  # Policy 1: High ambiguity state
        [0.10, 0.30, 0.60]   # Policy 2: Egress has high friction
    ])

    G_base, P_pi_base = engine.evaluate_policy_posteriors(projected_qs_baseline)
    print("--- BASELINE ADVERSARIAL STANCE ---")
    for idx, (g, p) in enumerate(zip(G_base, P_pi_base)):
        print(f"Policy {idx} | G: {g:6.2f} | Execution Probability: {p * 100:5.1f}%")

    # Interdiction Event: Defense alters state transitions; infiltration now guarantees tracking
    projected_qs_interdicted = np.array([
        [0.05, 0.15, 0.80],  # Policy 0: Infiltration now leads directly to Tracked state
        [0.10, 0.60, 0.30],  # Policy 1: Recon exposes intermediate risk
        [0.70, 0.20, 0.10]   # Policy 2: Egress now guarantees return to Unobserved state
    ])

    G_shift, P_pi_shift = engine.evaluate_policy_posteriors(projected_qs_interdicted)
    print("\n--- POST-INTERDICTION ADVERSARIAL SHIFT ---")
    for idx, (g, p) in enumerate(zip(G_shift, P_pi_shift)):
        print(f"Policy {idx} | G: {g:6.2f} | Execution Probability: {p * 100:5.1f}%")


def _run_multi_agent_demo():
    game = MultiAgentActiveInferencePOMDP()

    # Baseline: Target believes Assault (Policy 0) is unobstructed (High State 0 density)
    # Policy 0: Infiltrate [State 0: 0.80, State 1: 0.15, State 2: 0.05]
    # Policy 1: Recon       [State 0: 0.30, State 1: 0.60, State 2: 0.10]
    # Policy 2: Abort       [State 0: 0.10, State 1: 0.20, State 2: 0.70]
    baseline_states = np.array([
        [0.80, 0.15, 0.05],
        [0.30, 0.60, 0.10],
        [0.10, 0.20, 0.70]
    ])

    results = game.compute_defender_optimal_policy(baseline_states)

    action_names = ["Passive Monitor", "Sensor Denial (Jamming)", "Active Deception (Honeypot)"]
    policy_names = ["Policy 0 (Assault)", "Policy 1 (Recon)", "Policy 2 (Abort)"]

    print("=" * 80)
    print("MULTI-AGENT ACTIVE INFERENCE: DEFENSIVE POMDP EQUILIBRIUM")
    print("=" * 80)

    for log in results["detailed_log"]:
        act_idx = log["def_action"]
        print(f"\n[+] DEFENSIVE MOVE: {action_names[act_idx]}")
        print(f"    Expected Net Defender Utility: {log['net_utility']:.3f}")
        print("    Adversary Policy Distribution:")
        for p_idx, p_prob in enumerate(log["P_pi_adv"]):
            print(f"      - {policy_names[p_idx]:<20} | G: {log['G_adv'][p_idx]:6.2f} | P(pi): {p_prob * 100:5.1f}%")

    best = results["optimal_def_action"]
    print("\n" + "=" * 80)
    print(f"STRATEGIC DIRECTIVE: Deploy [{action_names[best].upper()}]")
    print("Operational Mechanism: Warps adversary A-matrix to induce optimal systemic divergence.")
    print("=" * 80)


if __name__ == "__main__":
    _run_single_agent_demo()
    print()
    _run_multi_agent_demo()
