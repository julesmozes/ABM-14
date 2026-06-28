import numpy as np
import networkx as nx

from collections import defaultdict
from random import Random

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        # Fallback no-op decorator that supports both forms:
        # - @njit
        # - @njit(...)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Used as `@njit` without arguments; return the function itself.
            return args[0]

        def decorator(func):
            return func

        return decorator

from mesa import Agent, Model
from mesa.agent import AgentSet
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid, PropertyLayer

# Fixed design / literature constants
V_SEE = 1 
RHO = 0.88  
N_PATCHES = 6
K_MAX = 1.0  
METABOLISM = 0.02 
TRAVEL_COST_DEFAULT = 0.05
C_BIRTH_DEFAULT = 2.0
INITIAL_CAPITAL = 1.0
LAMBDA_MIN = 1.0
LAMBDA_MAX = 5.0
LOSS_AVERSION_DEFAULT = 2.25


class FishingGrid(MultiGrid):
    """MultiGrid that keeps the model RNG when the space has no agents."""

    def __init__(self, *args, random: Random | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._random = random

    @property
    def agents(self) -> AgentSet:
        agents = []
        for entry in self:
            if not entry:
                continue
            if not isinstance(entry, list):
                entry = [entry]
            for agent in entry:
                agents.append(agent)
        rng = agents[0].random if agents else self._random
        return AgentSet(agents, random=rng)


@njit
def chebyshev(pos, target, width, height, torus=True):
    """Chebyshev distance with optional torus wrapping."""
    dx = pos[0] - target[0]
    if dx < 0:
        dx = -dx
    dy = pos[1] - target[1]
    if dy < 0:
        dy = -dy
    if torus:
        if dx > width - dx:
            dx = width - dx
        if dy > height - dy:
            dy = height - dy
    if dx > dy:
        return dx
    return dy


@njit
def prospect_value(delta, rho, loss_aversion):
    """Tversky-Kahneman value v(δ) with shared curvature ρ and loss aversion λ."""
    if delta >= 0.0:
        if rho == 1.0:
            return delta
        return delta**rho
    d = -delta
    if rho == 1.0:
        loss = -d
    else:
        loss = -(d**rho)
    return loss_aversion * loss


@njit
def logistic(x):
    """Numerically stable logistic σ(x)."""
    if x >= 20.0:
        return 1.0
    if x <= -20.0:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))


@njit
def scrounge_probability(v_produce, v_scrounge, beta):
    """P(scrounge) = σ(β · (V_scrounge - V_produce))."""
    return logistic(beta * (v_scrounge - v_produce))


@njit
def _gini_coefficient_numba(arr):
    n = arr.shape[0]
    if n < 2:
        return 0.0
    # simple insertion sort for small n; np.sort support is not guaranteed in nopython mode
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    total = 0.0
    for i in range(n):
        total += arr[i]
    if total <= 0.0:
        return 0.0
    weights = np.empty(n, dtype=np.float64)
    for i in range(n):
        weights[i] = 2.0 * (i + 1) - n - 1.0
    weighted = 0.0
    for i in range(n):
        weighted += weights[i] * arr[i]
    return weighted / (n * total)


def gini_coefficient(values):
    """Gini coefficient for a list of non-negative wealth values."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    # _gini_coefficient_numba may return a numpy scalar/array depending on
    # whether numba is available; coerce to a Python float robustly.
    res = _gini_coefficient_numba(arr.copy())
    return _as_scalar(res, 0.0)


def _as_scalar(x, default=0.0):
    """Coerce numeric-like values (including 0-dim numpy) to Python float.

    If conversion fails or x is an array with more than one element, return
    `default` to keep DataCollector robust at runtime.
    """
    try:
        return float(x)
    except Exception:
        try:
            arr = np.asarray(x)
            if arr.size == 1:
                return float(arr.item())
        except Exception:
            pass
    return default


def report_wealth_gini(m):
    """Robust model reporter returning a Python float for Wealth Gini."""
    if not m.agents:
        return 0.0
    caps = np.array([a.capital for a in m.agents], dtype=np.float64)
    return _as_scalar(gini_coefficient(caps), 0.0)


@njit
def _expected_harvest_numba(density_at_cell, q, f, n_scr, as_scrounge, joiner_self, self_scrounge):
    pool = q * density_at_cell
    if as_scrounge:
        n_joiners = n_scr
        if joiner_self or self_scrounge:
            n_joiners += 1
        if n_joiners < 1:
            n_joiners = 1
        return (1.0 - f) * pool / n_joiners
    if n_scr > 0:
        return f * pool
    return pool


@njit
def _best_opportunity_numba(densities, distances, q, f, c, n_scr, as_scrounge, joiner_self, self_scrounge):
    mu_stay = _expected_harvest_numba(densities[0], q, f, n_scr, as_scrounge, joiner_self, self_scrounge)
    best = mu_stay
    for i in range(1, densities.shape[0]):
        mu = _expected_harvest_numba(densities[i], q, f, n_scr, as_scrounge, joiner_self, self_scrounge) - c * distances[i]
        if mu > best:
            best = mu
    return best, mu_stay


@njit
def _compute_utilities_numba(densities, distances, q, f, c, beta, lam, rho, n_scr, as_scr):
    utilities = np.empty(densities.shape[0], dtype=np.float64)
    mu_stay = _expected_harvest_numba(densities[0], q, f, n_scr, as_scr, False, as_scr)
    for i in range(densities.shape[0]):
        mu = _expected_harvest_numba(densities[i], q, f, n_scr, as_scr, False, as_scr) - c * distances[i]
        utilities[i] = prospect_value(mu - mu_stay, rho, lam)
    return utilities


@njit
def _logit_probs_numba(utilities, beta):
    max_val = utilities[0]
    for i in range(1, utilities.shape[0]):
        if utilities[i] > max_val:
            max_val = utilities[i]
    probs = np.empty(utilities.shape[0], dtype=np.float64)
    total = 0.0
    for i in range(utilities.shape[0]):
        probs[i] = np.exp(beta * (utilities[i] - max_val))
        total += probs[i]
    if total == 0.0:
        for i in range(utilities.shape[0]):
            probs[i] = 1.0 / utilities.shape[0]
        return probs
    for i in range(utilities.shape[0]):
        probs[i] /= total
    return probs


def mutate_lambda(lam, rng, sigma):
    """Gaussian mutation for per-agent loss aversion λ."""
    return float(np.clip(lam + rng.normal(0.0, sigma), LAMBDA_MIN, LAMBDA_MAX))


class Boat(Agent):
    """A fishing vessel with capital and evolving loss aversion λ."""

    def __init__(self, model, capital, loss_aversion):
        super().__init__(model)
        self.capital = capital
        self.loss_aversion = loss_aversion
        self.scrounge = False
        self.prev_scrounge = False
        self.role_decided = False
        self._target = None

    def _local_scrounger_count(self):
        """Neighbours within comm range likely to scrounge (for expected joiner load)."""
        m = self.model
        n = 0
        for neighbor in m.grid.iter_neighbors(self.pos, moore=True, radius=m.v):
            if neighbor is self:
                continue
            if neighbor.role_decided:
                if neighbor.scrounge:
                    n += 1
            elif neighbor.prev_scrounge:
                n += 1
        return n

    def _expected_harvest(self, density_at_cell, as_scrounge, joiner_self=False):
        """Expected capital from one patch event (PS split if joiners present)."""
        m = self.model
        pool = m.q * density_at_cell
        n_scr = self._local_scrounger_count()
        f = m.finders_share
        if as_scrounge:
            n_joiners = max(1, n_scr + (1 if joiner_self or self.scrounge else 0))
            return (1.0 - f) * pool / n_joiners
        if n_scr > 0:
            return f * pool
        return pool

    def _best_opportunity(self, cells, as_scrounge, joiner_self=False):
        """Best net payoff over cells using PS-consistent expected harvest."""
        m = self.model
        density = m.density.data
        n_scr = self._local_scrounger_count()
        if self.pos in cells:
            choices = [self.pos] + [cell for cell in cells if cell != self.pos]
        else:
            choices = [self.pos] + list(cells)

        n = len(choices)
        densities = np.empty(n, dtype=np.float64)
        distances = np.empty(n, dtype=np.float64)
        densities[0] = density[self.pos]
        distances[0] = 0.0
        for i, cell in enumerate(choices[1:], start=1):
            densities[i] = density[cell]
            distances[i] = chebyshev(self.pos, cell, m.width, m.height, m.grid.torus)

        return _best_opportunity_numba(
            densities,
            distances,
            m.q,
            m.finders_share,
            m.c,
            n_scr,
            as_scrounge,
            joiner_self,
            self.scrounge,
        )

    def _producing_neighbors(self):
        """Neighbours within comm range that broadcast private finds (producers)."""
        m = self.model
        producers = []
        for neighbor in m.grid.iter_neighbors(self.pos, moore=True, radius=m.v):
            if neighbor is self:
                continue
            if neighbor.role_decided:
                if not neighbor.scrounge:
                    producers.append(neighbor)
            elif not neighbor.prev_scrounge:
                producers.append(neighbor)
        return producers

    def decide_role(self):
        """Prospect-theoretic produce vs scrounge (producer-scrounger game)."""
        m = self.model
        lam = self.loss_aversion
        perceived = set(self.perceived_cells())
        perceived.add(self.pos)
        best_produce, mu_stay = self._best_opportunity(perceived, as_scrounge=False)
        v_produce = prospect_value(best_produce - mu_stay, RHO, lam)

        scrounge_cells = set()
        for producer in self._producing_neighbors():
            scrounge_cells |= set(producer.perceived_cells())

        if scrounge_cells:
            best_scrounge, _ = self._best_opportunity(
                scrounge_cells, as_scrounge=True, joiner_self=True
            )
            v_scrounge = prospect_value(best_scrounge - mu_stay, RHO, lam)
        else:
            v_scrounge = -1e6

        p_scrounge = scrounge_probability(v_produce, v_scrounge, m.beta)
        self.scrounge = m.rng.random() < p_scrounge
        self.role_decided = True

    def perceived_cells(self):
        """Cells the boat privately senses (radius V_SEE)."""
        return self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=V_SEE
        )

    def _cell_utilities(self, cells):
        m = self.model
        density = m.density.data
        as_scr = self.scrounge
        lam = self.loss_aversion
        if self.pos in cells:
            choices = [self.pos] + [cell for cell in cells if cell != self.pos]
        else:
            choices = [self.pos] + list(cells)

        n = len(choices)
        densities = np.empty(n, dtype=np.float64)
        distances = np.empty(n, dtype=np.float64)
        densities[0] = density[self.pos]
        distances[0] = 0.0
        for i, cell in enumerate(choices[1:], start=1):
            densities[i] = density[cell]
            distances[i] = chebyshev(self.pos, cell, m.width, m.height, m.grid.torus)

        return _compute_utilities_numba(
            densities,
            distances,
            m.q,
            m.finders_share,
            m.c,
            m.beta,
            lam,
            RHO,
            self._local_scrounger_count(),
            as_scr,
        )

    def _logit_probs(self, utilities):
        m = self.model
        return _logit_probs_numba(utilities, m.beta)

    def choose_and_move(self):
        """Pick destination, move, pay travel; harvest deferred to patch settlement."""
        m = self.model
        perceived = set(self.perceived_cells())
        perceived.add(self.pos)
        if self.scrounge:
            reach = perceived | m._neighbor_broadcast_cells(self)
            choices = list(reach)
        else:
            choices = list(perceived)

        utilities = self._cell_utilities(choices)
        probs = self._logit_probs(utilities)
        self._target = choices[m.rng.choice(len(choices), p=probs)]

        dist = chebyshev(self.pos, self._target, m.width, m.height, m.grid.torus)
        if self._target != self.pos:
            m.grid.move_agent(self, self._target)
        self.capital -= m.c * dist


class FishingModel(Model):
    """fishing fleet on a regenerating resource lattice."""

    def __init__(
        self,
        width=100,
        height=100,
        n_agents=80,
        v=3,
        r=0.1,
        q=0.3,
        c=TRAVEL_COST_DEFAULT,
        beta=2.0,
        C_birth=C_BIRTH_DEFAULT,
        sigma=0.5,
        finders_share=0.5,
        patch_scale=0.1,
        rng=42,
    ):
        super().__init__(rng=rng)
        self.width = width
        self.height = height
        self.v = v
        self.r = r
        self.K = K_MAX
        self.q = q
        self.c = c
        self.beta = beta
        self.metabolism = METABOLISM
        self.C_birth = C_birth
        self.sigma = sigma
        self.finders_share = finders_share
        self.patch_scale = patch_scale

        density_layer = PropertyLayer("fish density", width, height, 0.0)
        capacity_layer = PropertyLayer("capacity", width, height, K_MAX)
        self.grid = FishingGrid(
            width,
            height,
            torus=True,
            property_layers=[density_layer, capacity_layer],
            random=self.random,
        )
        self.density = self.grid.properties["fish density"]
        self.capacity = self.grid.properties["capacity"]
        sigma_cells = max(self.patch_scale * min(width, height), 1.0)
        xs, ys = np.mgrid[0:width, 0:height]
        field = np.zeros((width, height))
        for _ in range(N_PATCHES):
            cx = int(self.rng.integers(width))
            cy = int(self.rng.integers(height))
            dx = np.minimum(np.abs(xs - cx), width - np.abs(xs - cx))
            dy = np.minimum(np.abs(ys - cy), height - np.abs(ys - cy))
            field += np.exp(-(dx**2 + dy**2) / (2.0 * sigma_cells**2))
        field /= field.max()
        cap = 0.1 * K_MAX + 0.9 * K_MAX * field
        self.capacity.data[:] = cap
        self.density.data[:] = cap * self.rng.uniform(0.7, 1.0, (width, height))

        for _ in range(n_agents):
            pos = (self.rng.integers(width), self.rng.integers(height))
            boat = Boat(
                self,
                capital=INITIAL_CAPITAL,
                loss_aversion=float(self.rng.uniform(LAMBDA_MIN, LAMBDA_MAX)),
            )
            self.grid.place_agent(boat, pos)

        self._broadcast_map = {}
        self.deaths_this_step = 0
        self.network_mean_degree = 0.0
        self.network_lcc_fraction = 0.0
        self.network_components = 0
        self.role_switch_rate = 0.0

        self.datacollector = DataCollector(
            model_reporters={
                "Boats": lambda m: int(len(m.agents)),
                "Bankruptcies": lambda m: int(m.deaths_this_step),
                "Mean fish density": lambda m: _as_scalar(m.density.data.mean(), 0.0),
                "Mean lambda": lambda m: _as_scalar(np.mean([a.loss_aversion for a in m.agents]), 0.0)
                if m.agents
                else 0.0,
                "Std lambda": lambda m: _as_scalar(np.std([a.loss_aversion for a in m.agents]), 0.0)
                if len(m.agents) > 1
                else 0.0,
                "Producer rate": lambda m: _as_scalar(np.mean([not a.scrounge for a in m.agents]), 0.0)
                if m.agents
                else 0.0,
                "Scrounge rate": lambda m: _as_scalar(np.mean([a.scrounge for a in m.agents]), 0.0)
                if m.agents
                else 0.0,
                "Role switch rate": lambda m: _as_scalar(m.role_switch_rate, 0.0),
                "Network mean degree": lambda m: _as_scalar(m.network_mean_degree, 0.0),
                "Network LCC fraction": lambda m: _as_scalar(m.network_lcc_fraction, 0.0),
                "Network components": lambda m: int(m.network_components),
                "Wealth Gini": report_wealth_gini,
            }
        )
        self.datacollector.collect(self)

    def _regenerate(self):
        d = self.density.data
        k = self.capacity.data
        np.clip(d + self.r * d * (1.0 - d / k), 0.0, k, out=d)

    def _build_broadcast_map(self):
        self._broadcast_map = {}
        for agent in self.agents:
            if agent.scrounge:
                self._broadcast_map[agent] = set()
            else:
                self._broadcast_map[agent] = set(agent.perceived_cells())

    def _neighbor_broadcast_cells(self, agent):
        extra = set()
        for neighbor in self.grid.iter_neighbors(agent.pos, moore=True, radius=self.v):
            extra |= self._broadcast_map.get(neighbor, set())
        return extra

    def _record_info_network(self):
        """Producer-scrounger information links within comm range."""
        agents = list(self.agents)
        n = len(agents)
        if n == 0:
            self.network_mean_degree = 0.0
            self.network_lcc_fraction = 0.0
            self.network_components = 0
            return

        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        index = {agent: i for i, agent in enumerate(agents)}

        for i, scrounger in enumerate(agents):
            if not scrounger.scrounge:
                continue
            for producer in self.grid.iter_neighbors(
                scrounger.pos, moore=True, radius=self.v
            ):
                if producer is scrounger or producer.scrounge:
                    continue
                j = index[producer]
                graph.add_edge(i, j)

        self.network_mean_degree = float(
            sum(dict(graph.degree()).values()) / n if n else 0.0
        )
        components = list(nx.connected_components(graph))
        self.network_components = len(components)
        if components:
            largest = max(len(c) for c in components)
            self.network_lcc_fraction = largest / n
        else:
            self.network_lcc_fraction = 0.0

    def _settle_patches(self):
        """Joint PS harvest: finder(s) keep f·q·D, joiners split (1-f)·q·D; deplete once."""
        density = self.density.data
        visits = defaultdict(list)
        for agent in self.agents:
            visits[agent._target].append(agent)

        f = self.finders_share
        for cell, boats in visits.items():
            pool = self.q * density[cell]
            producers = [b for b in boats if not b.scrounge]
            joiners = [b for b in boats if b.scrounge]

            if pool > 0:
                if joiners and producers:
                    share_f = f * pool / len(producers)
                    share_j = (1.0 - f) * pool / len(joiners)
                    for b in producers:
                        b.capital += share_f
                    for b in joiners:
                        b.capital += share_j
                else:
                    share = pool / len(boats)
                    for b in boats:
                        b.capital += share
                density[cell] *= 1.0 - self.q

            for b in boats:
                b.capital -= self.metabolism

    def _birth_and_death(self):
        to_remove = []
        to_reproduce = []
        for agent in self.agents:
            if agent.capital <= 0:
                to_remove.append(agent)
            elif agent.capital >= self.C_birth:
                to_reproduce.append(agent)

        self.deaths_this_step = len(to_remove)
        for agent in to_remove:
            self.grid.remove_agent(agent)
            agent.remove()

        for parent in to_reproduce:
            if parent not in self.agents:
                continue
            half = parent.capital / 2.0
            parent.capital = half
            child = Boat(
                self,
                capital=half,
                loss_aversion=mutate_lambda(parent.loss_aversion, self.rng, self.sigma),
            )
            self.grid.place_agent(child, parent.pos)

    def step(self):
        self._regenerate()

        for agent in self.agents.shuffle():
            agent.role_decided = False
            agent.decide_role()
        self._build_broadcast_map()
        self._record_info_network()

        for agent in self.agents.shuffle():
            agent.choose_and_move()
        self._settle_patches()

        agents = list(self.agents)
        if agents:
            self.role_switch_rate = float(
                np.mean([a.prev_scrounge != a.scrounge for a in agents])
            )
        else:
            self.role_switch_rate = 0.0

        for agent in agents:
            agent.prev_scrounge = agent.scrounge

        self._birth_and_death()
        self.datacollector.collect(self)
