# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.4.3",
#     "matplotlib==3.10.8",
#     "scikit-learn==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    return load_iris, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # k-means++: The Advantages of Careful Seeding

    **Arthur & Vassilvitskii, 2007** — [paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)

    Standard k-means picks its initial centroids **randomly**, which can lead to
    poor clusters and slow convergence. k-means++ fixes this with one simple idea:

    > Choose each new centroid with probability **proportional to its squared
    > distance** from the nearest existing centroid.

    That's the whole paper. This notebook lets you see why it matters.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Algorithm

    **Standard k-means initialization:** pick $k$ points uniformly at random.

    **k-means++ initialization:**

    1. Pick the first centroid $c_1$ uniformly at random from the data
    2. For each remaining centroid $c_i$:
       - For every point $x$, compute $D(x)$ = distance to the **nearest** existing centroid
       - Pick the next centroid with probability: $\quad P(x) = \dfrac{D(x)^2}{\sum_{x'} D(x')^2}$
    3. Run standard k-means from these starting centroids

    **Why it works:** Points far from all existing centroids are more likely to be
    chosen, so the seeds spread out across the data. The paper proves this gives an
    $O(\log k)$ approximation to the optimal clustering cost — **in expectation,
    before k-means even runs a single iteration**.
    """)
    return


@app.cell
def _(load_iris, np):
    # Load Iris and take the two most separable features (petal length & width)
    iris = load_iris()
    iris_data = iris.data[:, 2:4]  # petal length, petal width
    iris_labels = iris.target
    iris_feature_names = [iris.feature_names[2], iris.feature_names[3]]

    # Normalize for cleaner distance calculations
    iris_min = iris_data.min(axis=0)
    iris_max = iris_data.max(axis=0)
    X = (iris_data - iris_min) / (iris_max - iris_min)

    # True number of species
    k_true = len(np.unique(iris_labels))
    return X, k_true


@app.cell
def _(mo):
    mo.md(r"""
    ## Step-by-Step: How k-means++ Seeds

    Watch the algorithm pick centroids one at a time on the Iris dataset
    (petal length vs. petal width). The **shading** shows the probability of each
    region being picked next — darker = more likely.
    """)
    return


@app.cell
def _(k_true, mo):
    seed_slider = mo.ui.slider(start=0, stop=99, value=42, step=1, label="Random seed")
    k_slider = mo.ui.slider(start=2, stop=6, value=k_true, step=1, label="k (number of clusters)")
    mo.hstack([k_slider, seed_slider], justify="start", gap=2)
    return k_slider, seed_slider


@app.cell
def _(X, k_slider, np, plt, seed_slider):
    _k = k_slider.value
    _rng = np.random.default_rng(seed_slider.value)

    # --- k-means++ seeding, step by step ---
    _n = X.shape[0]
    _centroids = []
    _step_distances = []  # D(x)^2 at each step

    # Step 1: first centroid uniformly at random
    _idx = _rng.integers(_n)
    _centroids.append(X[_idx])

    for _step in range(1, _k):
        # Compute squared distance to nearest centroid
        _dists = np.array([np.min(np.sum((X - c) ** 2, axis=1)) for c in _centroids]).T
        if len(_centroids) > 1:
            _dists = np.min(
                np.column_stack([np.sum((X - c) ** 2, axis=1) for c in _centroids]),
                axis=1,
            )
        else:
            _dists = np.sum((X - _centroids[0]) ** 2, axis=1)

        _step_distances.append(_dists.copy())
        _probs = _dists / _dists.sum()
        _idx = _rng.choice(_n, p=_probs)
        _centroids.append(X[_idx])

    _centroids_arr = np.array(_centroids)

    # --- Plot each step ---
    _n_steps = len(_step_distances)
    _fig, _axes = plt.subplots(1, _n_steps, figsize=(4.5 * _n_steps, 4), squeeze=False)

    for _s in range(_n_steps):
        _ax = _axes[0][_s]
        _d2 = _step_distances[_s]
        _probs = _d2 / _d2.sum()

        # Scatter points, colored by selection probability
        _sc = _ax.scatter(
            X[:, 0], X[:, 1],
            c=_probs, cmap="YlOrRd", s=30, alpha=0.8, edgecolors="none",
        )

        # Show centroids chosen so far
        _chosen = _centroids_arr[: _s + 1]
        _ax.scatter(
            _chosen[:, 0], _chosen[:, 1],
            c="blue", marker="X", s=150, edgecolors="white", linewidths=1.5,
            zorder=5, label="Chosen centroids",
        )
        # Highlight the NEW centroid about to be chosen
        _new = _centroids_arr[_s + 1]
        _ax.scatter(
            _new[0], _new[1],
            c="lime", marker="X", s=200, edgecolors="black", linewidths=2,
            zorder=6, label="Next pick",
        )

        _ax.set_title(f"Step {_s + 1}: picking centroid {_s + 2}", fontsize=10)
        _ax.set_xlabel("Petal length (norm)")
        if _s == 0:
            _ax.set_ylabel("Petal width (norm)")
        _ax.legend(fontsize=7, loc="upper left")
        _fig.colorbar(_sc, ax=_ax, shrink=0.7, label="P(x)")

    _fig.suptitle("k-means++ Seeding: Probability of Being Chosen Next", fontsize=12, fontweight="bold")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Random vs. k-means++: Side by Side

    Now let's run full k-means (10 iterations) from both initialization
    strategies and compare the results.
    """)
    return


@app.cell
def _(X, k_slider, np, plt, seed_slider):
    _k = k_slider.value
    _n = X.shape[0]

    def _run_kmeans(data, k, centroids, n_iter=10):
        """Run k-means for n_iter iterations from given centroids."""
        c = centroids.copy()
        for _ in range(n_iter):
            # Assign
            dists = np.column_stack([np.sum((data - c[j]) ** 2, axis=1) for j in range(k)])
            labels = np.argmin(dists, axis=1)
            # Update
            for j in range(k):
                mask = labels == j
                if mask.any():
                    c[j] = data[mask].mean(axis=0)
        # Final cost (total within-cluster sum of squares)
        dists = np.column_stack([np.sum((data - c[j]) ** 2, axis=1) for j in range(k)])
        labels = np.argmin(dists, axis=1)
        cost = sum(np.sum((data[labels == j] - c[j]) ** 2) for j in range(k))
        return labels, c, cost

    def _kmeans_pp_init(data, k, rng):
        """k-means++ initialization."""
        n = data.shape[0]
        idx = rng.integers(n)
        centroids = [data[idx]]
        for _ in range(1, k):
            d2 = np.min(
                np.column_stack([np.sum((data - c) ** 2, axis=1) for c in centroids]),
                axis=1,
            )
            probs = d2 / d2.sum()
            idx = rng.choice(n, p=probs)
            centroids.append(data[idx])
        return np.array(centroids)

    _rng_rand = np.random.default_rng(seed_slider.value)
    _rng_pp = np.random.default_rng(seed_slider.value)

    # Random init
    _random_centroids = X[_rng_rand.choice(_n, size=_k, replace=False)]
    _labels_rand, _c_rand, _cost_rand = _run_kmeans(X, _k, _random_centroids)

    # k-means++ init
    _pp_centroids = _kmeans_pp_init(X, _k, _rng_pp)
    _labels_pp, _c_pp, _cost_pp = _run_kmeans(X, _k, _pp_centroids)

    # Plot
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    _colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    for _ax, _labels, _c, _cost, _title in [
        (_ax1, _labels_rand, _c_rand, _cost_rand, "Random Init"),
        (_ax2, _labels_pp, _c_pp, _cost_pp, "k-means++ Init"),
    ]:
        for _j in range(_k):
            _mask = _labels == _j
            _ax.scatter(X[_mask, 0], X[_mask, 1], c=_colors[_j], s=30, alpha=0.6, label=f"Cluster {_j+1}")
        _ax.scatter(_c[:, 0], _c[:, 1], c="black", marker="X", s=200, edgecolors="white", linewidths=2, zorder=5)
        _ax.set_title(f"{_title}\nCost: {_cost:.3f}", fontsize=11)
        _ax.set_xlabel("Petal length (norm)")
        _ax.set_ylabel("Petal width (norm)")
        _ax.legend(fontsize=7)

    _fig.suptitle(
        f"k-means (10 iterations)  |  k={_k}  |  seed={seed_slider.value}",
        fontsize=12, fontweight="bold",
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Try different seeds** to see cases where random initialization gets unlucky
    (high cost, weird cluster shapes) while k-means++ stays consistent.

    The cost number is the **within-cluster sum of squares** — lower is better.
    k-means++ won't always win on a single run, but it wins **on average** and
    avoids the worst-case scenarios that random init stumbles into.

    ---

    *Built with [marimo](https://marimo.io)*
    """)
    return


if __name__ == "__main__":
    app.run()
