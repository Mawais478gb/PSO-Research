import os
import zlib
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact


DATA_FILE = "cf3_5d_data.npz"
RESULTS_DIR = "results_cec2013"
FIGURES_DIR = "figures_cec2013"

DIM = 5
LOWER, UPPER = -5.0, 5.0
N_PARTICLES = 50
N_RUNS = 50
MAX_EVALUATIONS = 400_000
SUCCESS_RADIUS = 0.1
HIT_ACCURACY = 1e-4

SETTINGS = {
    "Overdamped": (1.4, 0.7),
    "Damped": (0.3215, 0.958058),
    "Divergent O": (0.03, 1.0),
    "Divergent A": (2.2, 0.7),
    "Divergent B": (1.0, 0.95),
    "Divergent C": (1.4, 0.85),
}
STABLE = ["Overdamped", "Damped"]
DIVERGENT = ["Divergent O", "Divergent A", "Divergent B", "Divergent C"]

STARTS = ["cube", "same_same", "same_small", "same_large"]
START_LABELS = {
    "cube": "cube, v=0",
    "same_same": "same point, same v",
    "same_small": "same point, small diff. v",
    "same_large": "same point, large diff. v",
}
REPETITIONS = [("main", "+", 1), ("new seed", "+", 2), ("opposite corner", "-", 1)]


def griewank(z):
    i = np.sqrt(np.arange(z.shape[-1]) + 1.0)
    return np.sum(z ** 2, axis=-1) / 4000.0 - np.prod(np.cos(z / i), axis=-1) + 1.0


K = np.arange(21)
A_K = 0.5 ** K
B_K = 3.0 ** K


def weierstrass(z):
    d = z.shape[-1]
    c = -d * np.sum(A_K * np.cos(2.0 * np.pi * B_K * 0.5))
    phase = np.mod(B_K * (z[..., None] + 0.5), 1.0)
    return np.sum(A_K * np.cos(2.0 * np.pi * phase), axis=(-1, -2)) + c


def ef8f2(z):
    y = z + 1.0
    a, b = y, np.roll(y, -1, axis=-1)
    f2 = 100.0 * (a ** 2 - b) ** 2 + (1.0 - a) ** 2
    return np.sum(1.0 + f2 ** 2 / 4000.0 - np.cos(f2), axis=-1)


class CompositionFunction3:
    def __init__(self, data_file):
        data = np.load(data_file)
        self.optima = data["O"]
        self.rotations = data["M"]
        self.lam = data["lam"]
        self.sigma = data["sigma"]
        self.f_max = data["fmax"]
        self.C = float(data["C"])
        self.bias = data["bias"]
        self.basic = [ef8f2, ef8f2, weierstrass, weierstrass, griewank, griewank]
        self.dim = self.optima.shape[1]

    def composition(self, x):
        n = len(self.optima)
        d2 = np.stack([np.sum((x - self.optima[i]) ** 2, axis=-1) for i in range(n)], axis=-1)
        w = np.exp(-d2 / (2.0 * self.dim * self.sigma ** 2))
        w_max = np.max(w, axis=-1, keepdims=True)
        w = np.where(w == w_max, w, w * (1.0 - w_max ** 10))
        s = np.sum(w, axis=-1, keepdims=True)
        w = np.where(s == 0.0, 1.0 / n, w / np.where(s == 0.0, 1.0, s))
        total = 0.0
        for i in range(n):
            z = ((x - self.optima[i]) / self.lam[i]) @ self.rotations[i]
            total = total + w[..., i] * (self.C * self.basic[i](z) / self.f_max[i] + self.bias[i])
        return total

    def __call__(self, x):
        x = np.asarray(x, float)
        outside = np.any((x < LOWER) | (x > UPPER), axis=-1)
        f = np.full(outside.shape, 1e10)
        if (~outside).any():
            f[~outside] = self.composition(x[~outside])
        return f


def far_corner(optima, offset=4.5):
    corners = offset * np.array(list(itertools.product([-1.0, 1.0], repeat=optima.shape[1])))
    distance = np.min(np.linalg.norm(corners[:, None, :] - optima[None], axis=2), axis=1)
    return corners[np.argmax(distance)], float(distance.max())


def case_seed(corner_sign, repetition, problem=16, n_particles=N_PARTICLES, n_runs=N_RUNS):
    key = f"P{problem}_N{n_particles}_c{corner_sign}_R{n_runs}_rep{repetition}"
    return zlib.crc32(key.encode())


def initial_swarm(start, c, n_runs, n_particles, dim, seed):
    rng = np.random.default_rng([seed, 0])
    u = rng.uniform(-0.5, 0.5, (n_runs, n_particles, dim))
    x = np.broadcast_to(c, (n_runs, n_particles, dim)).copy()
    if start == "cube":
        return c + u, np.zeros_like(u)
    if start == "same_zero":
        return x, np.zeros_like(u)
    if start == "same_same":
        return x, np.broadcast_to(u[:, :1, :], u.shape).copy()
    if start == "same_small":
        return x, u.copy()
    if start == "same_large":
        r = rng.uniform(0.0, 1.0, u.shape)
        return x, (LOWER - c) + r * (UPPER - LOWER)
    raise ValueError(start)


def swarm_spread(x):
    return np.sqrt(np.mean(np.sum((x - x.mean(1, keepdims=True)) ** 2, 2), 1))


def run_pso(f, mu, w, x, v, max_evaluations, optima, seed):
    rng = np.random.default_rng([seed, 1])
    n_runs, n_particles, dim = x.shape
    n_iterations = max_evaluations // n_particles - 1
    n_opt = len(optima)
    runs = np.arange(n_runs)
    region_best_f = np.full((n_runs, n_opt), np.inf)
    region_best_d = np.full((n_runs, n_opt), np.inf)
    region_hit = np.zeros((n_runs, n_opt), bool)
    o2 = np.sum(optima ** 2, axis=1)

    def record(x, fx):
        flat = x.reshape(-1, dim)
        d2 = np.maximum(np.sum(flat ** 2, 1)[:, None] + o2[None] - 2 * flat @ optima.T, 0)
        d2 = d2.reshape(n_runs, n_particles, n_opt)
        k = np.argmin(d2, axis=2)
        rr = np.broadcast_to(runs[:, None], (n_runs, n_particles))
        candidate = fx < region_best_f[rr, k]
        if candidate.any():
            r_c, k_c, f_c = rr[candidate], k[candidate], fx[candidate]
            d_c = np.sqrt(np.take_along_axis(d2, k[..., None], 2)[..., 0][candidate])
            key = r_c * n_opt + k_c
            order = np.lexsort((f_c, key))
            first = order[np.unique(key[order], return_index=True)[1]]
            region_best_f[r_c[first], k_c[first]] = f_c[first]
            region_best_d[r_c[first], k_c[first]] = d_c[first]
            hit = f_c < HIT_ACCURACY
            if hit.any():
                region_hit[r_c[hit], k_c[hit]] = True

    checkpoints = sorted(set(range(61)) | set(np.unique(np.geomspace(1, n_iterations, 120).astype(int)).tolist()))
    checkpoint_set = set(checkpoints)
    spread = []

    fx = f(x)
    p, fp = x.copy(), fx.copy()
    best = np.argmin(fp, 1)
    g, fg = p[runs, best].copy(), fp[runs, best].copy()
    record(x, fx)
    spread.append(swarm_spread(x))
    for t in range(1, n_iterations + 1):
        r1 = rng.random((n_runs, n_particles, dim))
        r2 = rng.random((n_runs, n_particles, dim))
        v = w * v + mu * r1 * (p - x) + mu * r2 * (g[:, None] - x)
        x = x + v
        outside = (x < LOWER) | (x > UPPER)
        if outside.any():
            x = np.clip(x, LOWER, UPPER)
            v = np.where(outside, -0.5 * v, v)
        fx = f(x)
        better = fx < fp
        if better.any():
            p[better] = x[better]
            fp[better] = fx[better]
            best = np.argmin(fp, 1)
            g, fg = p[runs, best].copy(), fp[runs, best].copy()
        record(x, fx)
        if t in checkpoint_set:
            spread.append(swarm_spread(x))
    reached = region_best_d <= SUCCESS_RADIUS
    return {
        "success": reached.any(1),
        "reached": reached,
        "hit_1e4": region_hit,
        "final_g": fg,
        "rec_iter": np.array(checkpoints[:len(spread)]),
        "spread": np.array(spread),
    }


def case_file(corner_sign, start, setting, repetition):
    corner = "p" if corner_sign == "+" else "m"
    return os.path.join(RESULTS_DIR, f"P16_N{N_PARTICLES}_c{corner}_{start}_{setting.replace(' ', '')}_rep{repetition}.npz")


def run_case(f, c_main, corner_sign, start, setting, repetition):
    path = case_file(corner_sign, start, setting, repetition)
    if os.path.exists(path):
        data = np.load(path)
        return {k: data[k] for k in ("success", "hit_1e4", "rec_iter", "spread")}
    mu, w = SETTINGS[setting]
    c = c_main if corner_sign == "+" else -c_main
    seed = case_seed(corner_sign, repetition)
    x, v = initial_swarm(start, c, N_RUNS, N_PARTICLES, DIM, seed)
    result = run_pso(f, mu, w, x, v, MAX_EVALUATIONS, f.optima, seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(path, **result)
    print(f"computed {os.path.basename(path)}: {int(result['success'].sum())}/{N_RUNS}", flush=True)
    return result


def check_same_point_facts(f, c):
    x, v = initial_swarm("same_zero", c, 3, N_PARTICLES, DIM, seed=7)
    result = run_pso(f, *SETTINGS["Divergent A"], x, v, N_PARTICLES * 200, f.optima, seed=7)
    print(f"same point, v = 0: largest spread in 200 iterations = {result['spread'].max():.1f}")
    for start, factor in (("same_small", 0.645), ("same_large", 6.45)):
        for setting in STABLE:
            mu, w = SETTINGS[setting]
            x, v = initial_swarm(start, c, 200, N_PARTICLES, DIM, seed=7)
            result = run_pso(f, mu, w, x, v, N_PARTICLES * 2, f.optima, seed=7)
            print(f"{START_LABELS[start]:26s} {setting:10s}: spread after one step "
                  f"{result['spread'][1].mean():.3f} (predicted {factor * w:.3f})")


def print_tables(summary, hits):
    print(f"\nMain run: runs (of {N_RUNS}) reaching a global optimum [hit to {HIT_ACCURACY:g}]")
    print(f"{'setting':12s}" + "".join(f"{START_LABELS[s]:>28s}" for s in STARTS))
    for setting in SETTINGS:
        cells = [f"{summary[('main', s, setting)]:>22d} [{hits[('main', s, setting)]:2d}]" for s in STARTS]
        print(f"{setting:12s}" + "".join(cells))
    print(f"\nRobustness: runs (of {N_RUNS}) reaching a global optimum")
    print(f"{'setting':12s} {'start':28s}" + "".join(f"{r[0]:>17s}" for r in REPETITIONS))
    for setting in SETTINGS:
        for s in STARTS:
            print(f"{setting:12s} {START_LABELS[s]:28s}" + "".join(f"{summary[(r[0], s, setting)]:>17d}" for r in REPETITIONS))


def statistics(summary):
    tests = []
    for name, _, _ in REPETITIONS:
        for start in ["cube", "same_same", "same_small"]:
            best_stable = max(summary[(name, start, s)] for s in STABLE)
            for setting in DIVERGENT:
                a = summary[(name, start, setting)]
                p = fisher_exact([[a, N_RUNS - a], [best_stable, N_RUNS - best_stable]]).pvalue
                tests.append((name, start, setting, a, best_stable, p))
    n = len(tests)
    better = sum(a > b and p < 0.05 for _, _, _, a, b, p in tests)
    bonferroni = sum(a > b and p < 0.05 / n for _, _, _, a, b, p in tests)
    worse = sum(a < b for _, _, _, a, b, _ in tests)
    print(f"\nDivergent vs better stable setting (Fisher exact test, {n} comparisons):")
    print(f"significantly better: {better}   after Bonferroni: {bonferroni}   worse: {worse}")
    for name, start, setting, a, b, p in tests:
        if not (a > b and p < 0.05):
            print(f"  not significant: {name:15s} {START_LABELS[start]:26s} {setting:12s} {a:2d} vs {b:2d}  p = {p:.2g}")
    print("\nEffect of the start condition (main run, against the cube start):")
    for setting in SETTINGS:
        a = summary[("main", "cube", setting)]
        for start in STARTS[1:]:
            b = summary[("main", start, setting)]
            p = fisher_exact([[a, N_RUNS - a], [b, N_RUNS - b]]).pvalue
            if p < 0.05:
                print(f"  {setting:12s}: cube {a:2d} -> {START_LABELS[start]:26s} {b:2d}   p = {p:.1e}")


def plot_results(summary, results):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    colors = dict(zip(STARTS, ["#3a3a38", "#6250d6", "#eb6834", "#1baf7a"]))
    positions = np.arange(len(SETTINGS))
    width = 0.2

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, (name, corner, _) in zip(axes, REPETITIONS):
        for i, start in enumerate(STARTS):
            values = [summary[(name, start, s)] for s in SETTINGS]
            ax.bar(positions + (i - 1.5) * width, values, width * 0.92, color=colors[start], label=START_LABELS[start])
        ax.set_xticks(positions, [s.replace("Divergent ", "Div ") for s in SETTINGS])
        ax.set_title(f"{name} (corner {'c' if corner == '+' else '-c'})", loc="left")
        ax.set_ylim(0, N_RUNS + 2)
    axes[0].set_ylabel(f"runs (of {N_RUNS}) reaching a global optimum")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_success.png"), dpi=200)

    for name, last, scale in (("fig2_spread_early.png", 60, "linear"), ("fig3_spread_whole_run.png", None, "log")):
        fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
        for ax, setting in zip(axes.ravel(), SETTINGS):
            for start in STARTS:
                r = results[("main", start, setting)]
                it, sp = r["rec_iter"], r["spread"].mean(1)
                if last is not None:
                    keep = it <= last
                    ax.plot(it[keep], sp[keep], color=colors[start], label=START_LABELS[start])
                else:
                    ax.plot(it + 1, np.maximum(sp, 1e-10), color=colors[start], label=START_LABELS[start])
            if scale == "log":
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_title(setting, loc="left")
        for ax in axes[1]:
            ax.set_xlabel("iteration" if scale == "linear" else "iteration + 1")
        for ax in axes[:, 0]:
            ax.set_ylabel("swarm spread")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(FIGURES_DIR, name), dpi=200)
    plt.show()


def main():
    f = CompositionFunction3(DATA_FILE)
    c, distance = far_corner(f.optima)
    print(f"start corner c = {c}, distance to the nearest global optimum = {distance:.1f}")
    check_same_point_facts(f, c)
    results, summary, hits = {}, {}, {}
    for name, corner_sign, repetition in REPETITIONS:
        for start in STARTS:
            for setting in SETTINGS:
                r = run_case(f, c, corner_sign, start, setting, repetition)
                results[(name, start, setting)] = r
                summary[(name, start, setting)] = int(r["success"].sum())
                hits[(name, start, setting)] = int(r["hit_1e4"].any(1).sum())
    print_tables(summary, hits)
    statistics(summary)
    plot_results(summary, results)


if __name__ == "__main__":
    main()
