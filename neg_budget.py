import numpy as np
import networkx as nx
import copy
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Bernoulli
except ImportError:
    raise ImportError("Please install PyTorch to run the PPO baseline: pip install torch")

warnings.filterwarnings("ignore")


# ============================================================
# 1. Negative externality environment
# ============================================================
class NegativeExternalityDDERMAB:
    def __init__(self, N, M, P, alpha, beta,
                 p0, pa, pz, q0, qa, qz, cost, gamma, seed=None):
        self.N = N
        self.M = M
        self.P = P.astype(np.float64)
        self.alpha = alpha
        self.beta = beta
        self.p0, self.pa, self.pz = p0, pa, pz
        self.q0, self.qa, self.qz = q0, qa, qz
        self.cost = cost
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.s = (self.rng.rand(self.N) < 0.3).astype(np.float32)
        self.z = np.zeros(self.N, dtype=np.float32)
        return self.s.copy(), self.z.copy()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        reward = np.sum(self.s - self.cost * actions)

        injection = self.beta * actions
        self.z = self.alpha * (self.P @ self.z) + injection
        self.z = np.clip(self.z, 0, None)

        p_good = np.clip(self.p0 + self.pa * actions - self.pz * self.z, 0, 1)
        p_bad = np.clip(self.q0 - self.qa * actions + self.qz * self.z, 0, 1)

        mask_bad = (self.s == 0)
        mask_good = (self.s == 1)

        self.s[mask_bad] = (
            self.rng.rand(mask_bad.sum()) < p_good[mask_bad]
        ).astype(np.float32)

        self.s[mask_good] = 1 - (
            self.rng.rand(mask_good.sum()) < p_bad[mask_good]
        ).astype(np.float32)

        return (self.s.copy(), self.z.copy()), reward


# ============================================================
# 2. Single-arm DP utilities
# ============================================================
def solve_single_arm_value_negative(env, z_grid):
    n_z = len(z_grid)
    V = np.zeros((2, n_z))
    Q = np.zeros((2, n_z, 2))

    for _ in range(5000):
        V_old = V.copy()

        for s in [0, 1]:
            for iz, z in enumerate(z_grid):
                for a in [0, 1]:
                    if s == 0:
                        p_g = np.clip(env.p0 + env.pa * a - env.pz * z, 0, 1)
                        nxt_probs = [1 - p_g, p_g]
                    else:
                        p_b = np.clip(env.q0 - env.qa * a + env.qz * z, 0, 1)
                        nxt_probs = [p_b, 1 - p_b]

                    exp_V = nxt_probs[0] * V[0, iz] + nxt_probs[1] * V[1, iz]
                    Q[s, iz, a] = (s - env.cost * a) + env.gamma * exp_V

                V[s, iz] = np.max(Q[s, iz, :])

        if np.max(np.abs(V - V_old)) < 1e-6:
            break

    advantage = Q[:, :, 1] - Q[:, :, 0]
    return V, advantage


def compute_nu_analytical_negative(env, V, z_grid):
    n_z = len(z_grid)
    nu = np.zeros((2, n_z))

    for s in [0, 1]:
        for iz, z in enumerate(z_grid):
            if s == 0:
                dp_dz = -env.pz
            else:
                dp_dz = -env.qz

            delta_V = V[1, iz] - V[0, iz]
            nu[s, iz] = env.gamma * dp_dz * delta_V

    return nu


# ============================================================
# 3. Policies
# ============================================================
class MyopicPolicy:
    def act(self, s, z, env):
        actions = np.zeros(env.N, dtype=int)
        actions[np.argsort(s)[-env.M:]] = 1
        return actions


class TrueWhittlePolicy:
    def __init__(self, env, z_grid):
        env_zero = copy.deepcopy(env)
        env_zero.alpha = 0.0
        _, advantage = solve_single_arm_value_negative(env_zero, z_grid)
        self.whittle_index = advantage[:, 0]

    def act(self, s, z, env):
        scores = self.whittle_index[s.astype(int)]
        actions = np.zeros(env.N, dtype=int)
        actions[np.argsort(scores)[-env.M:]] = 1
        return actions


class GRETAPolicy:
    def __init__(self, env, z_grid):
        self.env = env
        self.z_grid = z_grid

        env_zero = copy.deepcopy(env)
        env_zero.alpha = 0.0

        self.V, self.advantage = solve_single_arm_value_negative(env_zero, z_grid)

    def _lookup(self, table, s_vals, z_vals):
        idx = np.clip(np.searchsorted(self.z_grid, z_vals), 1, len(self.z_grid) - 1)
        z_l, z_r = self.z_grid[idx - 1], self.z_grid[idx]
        frac = (z_vals - z_l) / (z_r - z_l + 1e-12)

        return (
            (1 - frac) * table[s_vals.astype(int), idx - 1]
            +
            frac * table[s_vals.astype(int), idx]
        )

    def act(self, s, z, env):
        scores = np.zeros(env.N)
        v_current = self._lookup(self.V, s, z)

        for u in range(env.N):
            w_pull = self._lookup(
                self.advantage,
                np.array([s[u]]),
                np.array([z[u]])
            )[0]

            # Kept unchanged from the original implementation.
            z_new = z + env.beta * env.P[:, u]

            v_next = self._lookup(self.V, s, z_new)
            w_msg_total = np.sum(v_next - v_current) - (v_next[u] - v_current[u])
            scores[u] = w_pull + w_msg_total

        actions = np.zeros(env.N, dtype=int)
        actions[np.argsort(scores)[-env.M:]] = 1
        return actions


class PAIPolicyNegative:
    def __init__(self, env, z_grid):
        self.env = env
        self.z_grid = z_grid

        V, advantage = solve_single_arm_value_negative(env, z_grid)
        self.advantage = advantage
        self.nu_table = compute_nu_analytical_negative(env, V, z_grid)

        I = np.eye(env.N)
        self.resolvent = np.linalg.inv(I - env.gamma * env.alpha * env.P)

    def _lookup(self, table, s_vals, z_vals):
        idx = np.clip(np.searchsorted(self.z_grid, z_vals), 1, len(self.z_grid) - 1)
        z_l, z_r = self.z_grid[idx - 1], self.z_grid[idx]
        frac = (z_vals - z_l) / (z_r - z_l + 1e-12)

        return (
            (1 - frac) * table[s_vals.astype(int), idx - 1]
            +
            frac * table[s_vals.astype(int), idx]
        )

    def act(self, s, z, env):
        delta = self._lookup(self.advantage, s, z)
        nu = self._lookup(self.nu_table, s, z)
        psi = env.beta * (self.resolvent.T @ nu)

        scores = delta + psi

        actions = np.zeros(env.N, dtype=int)
        actions[np.argsort(scores)[-env.M:]] = 1
        return actions


# ============================================================
# 4. PPO baseline
# ============================================================
class SharedActorCritic(nn.Module):
    def __init__(self, state_dim=2):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class PPOPolicy_AAAI25:
    """
    PPO RMAB baseline with a shared actor-critic network across arms.
    At test time, the policy activates the M arms with the largest
    predicted activation probabilities.
    """

    def __init__(self, env, num_train_episodes=30, max_steps=100):
        self.env = env
        self.device = torch.device("cpu")
        self.model = SharedActorCritic(state_dim=2).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-3)
        self.train_ppo(num_train_episodes, max_steps)

    def train_ppo(self, episodes, max_steps):
        gamma = self.env.gamma
        clip_ratio = 0.2
        ppo_epochs = 4

        train_env = copy.deepcopy(self.env)

        for ep in range(episodes):
            s, z = train_env.reset()

            states_buf = []
            actions_buf = []
            rewards_buf = []
            values_buf = []
            logprobs_buf = []

            for step in range(max_steps):
                state_tensor = torch.tensor(
                    np.stack([s, z], axis=1),
                    dtype=torch.float32
                ).to(self.device)

                with torch.no_grad():
                    probs, values = self.model(state_tensor)
                    probs = probs.squeeze(-1)
                    probs = torch.clamp(probs, 1e-5, 1.0 - 1e-5)
                    values = values.squeeze(-1)

                probs_np = probs.cpu().numpy()

                actions = np.zeros(train_env.N, dtype=np.float32)
                actions[np.argsort(probs_np)[-train_env.M:]] = 1.0

                (s_next, z_next), _ = train_env.step(actions)

                indiv_rewards = s - train_env.cost * actions

                states_buf.append(state_tensor)
                actions_buf.append(torch.tensor(actions, dtype=torch.float32).to(self.device))
                rewards_buf.append(torch.tensor(indiv_rewards, dtype=torch.float32).to(self.device))
                values_buf.append(values)

                dist = Bernoulli(probs)
                logp = dist.log_prob(
                    torch.tensor(actions, dtype=torch.float32).to(self.device)
                )
                logprobs_buf.append(logp)

                s, z = s_next, z_next

            states_t = torch.cat(states_buf)
            actions_t = torch.cat(actions_buf)
            old_logp_t = torch.cat(logprobs_buf)

            rewards_mat = torch.stack(rewards_buf)
            values_mat = torch.stack(values_buf)

            returns_mat = torch.zeros_like(rewards_mat)

            state_last = torch.tensor(
                np.stack([s, z], axis=1),
                dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                _, last_val = self.model(state_last)
                running_ret = last_val.squeeze(-1)

            for t in reversed(range(max_steps)):
                running_ret = rewards_mat[t] + gamma * running_ret
                returns_mat[t] = running_ret

            adv_mat = returns_mat - values_mat
            adv_mat = (adv_mat - adv_mat.mean()) / (adv_mat.std() + 1e-8)

            returns_t = returns_mat.view(-1)
            adv_t = adv_mat.view(-1)

            for _ in range(ppo_epochs):
                probs_new, values_new = self.model(states_t)
                probs_new = probs_new.squeeze(-1)
                probs_new = torch.clamp(probs_new, 1e-5, 1.0 - 1e-5)
                values_new = values_new.squeeze(-1)

                dist_new = Bernoulli(probs_new)
                logp_new = dist_new.log_prob(actions_t)

                ratio = torch.exp(logp_new - old_logp_t)
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_t

                loss_pi = -(torch.min(ratio * adv_t, clip_adv)).mean()
                loss_v = nn.MSELoss()(values_new, returns_t)
                loss = loss_pi + 0.5 * loss_v

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def act(self, s, z, env):
        state_tensor = torch.tensor(
            np.stack([s, z], axis=1),
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            probs, _ = self.model(state_tensor)
            probs = probs.squeeze(-1).numpy()

        actions = np.zeros(env.N, dtype=int)
        actions[np.argsort(probs)[-env.M:]] = 1
        return actions


# ============================================================
# 5. Simulation
# ============================================================
def run_simulation(env, policy, T=500, seed=42):
    env.rng.seed(seed)
    s, z = env.reset()

    rewards = []

    for _ in range(T):
        actions = policy.act(s, z, env)
        (s, z), r = env.step(actions)
        rewards.append(r)

    return np.sum(env.gamma ** np.arange(T) * np.array(rewards))


# ============================================================
# 6. Main experiment with budget sweep
# ============================================================
def run_negative_experiment():
    N = 50
    T = 1000
    gamma = 0.99
    num_seeds = 10

    budget_list = [2, 4, 6, 8, 10]

    p0, pa, pz = 0.1, 0.15, 0.3
    q0, qa, qz = 0.15, 0.1, 0.3
    cost = 0.1

    alpha_list = [0.6]
    beta_list = [0.2]

    G = nx.barabasi_albert_graph(N, 2, seed=42)
    A = nx.adjacency_matrix(G).toarray().astype(float)

    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = A / row_sums

    policy_names = [
        "Myopic",
        "PPO-R",
        "WhittleIndex",
        "GRETA",
        "PAI (Ours)"
    ]

    results_db = {}

    print("=" * 70)
    print("Budget sweep under negative externalities")
    print("=" * 70)

    for M in budget_list:
        print(f"\nBudget M = {M}")

        for beta in beta_list:
            for alpha in alpha_list:
                print(f"[Setting] Alpha = {alpha}, Beta = {beta}")

                env_base = NegativeExternalityDDERMAB(
                    N, M, P, alpha, beta,
                    p0, pa, pz, q0, qa, qz, cost, gamma
                )

                z_max = beta / (1 - alpha + 1e-6) * 3
                z_grid = np.linspace(0, z_max, 50)

                policies = {
                    "Myopic": MyopicPolicy(),
                    "PPO-R": PPOPolicy_AAAI25(env_base, num_train_episodes=30),
                    "WhittleIndex": TrueWhittlePolicy(env_base, z_grid),
                    "GRETA": GRETAPolicy(env_base, z_grid),
                    "PAI (Ours)": PAIPolicyNegative(env_base, z_grid)
                }

                results_db[(M, alpha, beta)] = {}

                for name in policy_names:
                    seed_returns = [
                        run_simulation(
                            NegativeExternalityDDERMAB(
                                N, M, P, alpha, beta,
                                p0, pa, pz, q0, qa, qz, cost, gamma
                            ),
                            policies[name],
                            T=T,
                            seed=s
                        )
                        for s in range(num_seeds)
                    ]

                    mean_ret = np.mean(seed_returns)
                    std_ret = np.std(seed_returns)

                    results_db[(M, alpha, beta)][name] = {
                        "mean": mean_ret,
                        "std": std_ret
                    }

                    print(f"   => {name:15s}: {mean_ret:7.2f}  ± {std_ret:5.2f}")

    return results_db


if __name__ == "__main__":
    run_negative_experiment()