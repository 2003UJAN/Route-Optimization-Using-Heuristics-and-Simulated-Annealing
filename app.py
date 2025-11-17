"""
Route Optimization Streamlit App
Everything in this single file (app.py).

Features:
- Generate synthetic city coordinates + package drop points
- Greedy (nearest neighbor) baseline
- Simulated Annealing optimizer
- Visualize optimized route vs. non-optimized path (matplotlib)
- Map-style scatter & path, distance and cost/time saved estimates

How to run:
1. Save this file as `app.py`.
2. Install dependencies: `pip install streamlit numpy pandas matplotlib scipy`.
3. Run: `streamlit run app.py`.

Optional: push to GitHub and deploy to Streamlit Cloud (or any other host supporting Streamlit).

Author: Generated for user request — self-contained single-file app.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

# ------------------------ Helpers ------------------------

def haversine(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return Euclidean distance (approx for small synthetic city) between two points.
    Using plain Euclidean because coordinates will be generated on a 2D plane. If you want
    geographic distances, replace with great-circle distance.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_distance_matrix(points: List[Tuple[float, float]]) -> np.ndarray:
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = haversine(points[i], points[j])
    return D


def total_route_distance(route: List[int], D: np.ndarray) -> float:
    # closed route (back to depot) -- if you don't want closed, remove last leg
    dist = 0.0
    for i in range(len(route) - 1):
        dist += D[route[i], route[i + 1]]
    # optionally return to start
    dist += D[route[-1], route[0]]
    return dist

# ------------------------ Route algorithms ------------------------

def greedy_nearest_neighbor(D: np.ndarray, start: int = 0) -> List[int]:
    n = D.shape[0]
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        # choose nearest unvisited
        nearest = min(unvisited, key=lambda x: D[current, x])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    return route


def random_route(n: int, start: int = 0) -> List[int]:
    route = list(range(n))
    random.shuffle(route)
    # rotate such that start is at index 0
    si = route.index(start)
    route = route[si:] + route[:si]
    return route


def two_opt_swap(route: List[int], i: int, k: int) -> List[int]:
    new_route = route[:i] + route[i:k + 1][::-1] + route[k + 1:]
    return new_route


def simulated_annealing(D: np.ndarray,
                        initial_route: List[int],
                        initial_temp: float = 1.0,
                        cooling_rate: float = 0.995,
                        iterations: int = 5000,
                        time_limit: float = None) -> Tuple[List[int], float]:
    """Simple simulated annealing implementation for TSP-like problem.

    Returns: (best_route, best_distance)
    """
    start_time = time.time()
    current_route = initial_route.copy()
    current_cost = total_route_distance(current_route, D)
    best_route = current_route.copy()
    best_cost = current_cost
    T = initial_temp

    for it in range(iterations):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        # pick two indices to reverse between (2-opt neighbor)
        i = random.randint(1, len(current_route) - 2)
        k = random.randint(i + 1, len(current_route) - 1)
        candidate = two_opt_swap(current_route, i, k)
        candidate_cost = total_route_distance(candidate, D)
        delta = candidate_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_route = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_route = current_route.copy()
        # cool down
        T *= cooling_rate
        if T < 1e-8:
            break
    return best_route, best_cost

# ------------------------ Visualization ------------------------

def plot_routes(points: List[Tuple[float, float]],
                route_a: List[int],
                route_b: List[int],
                title_a: str = "Route A",
                title_b: str = "Route B",
                figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    pts = np.array(points)
    ax.scatter(pts[:, 0], pts[:, 1], s=40)
    # label points
    for i, (x, y) in enumerate(points):
        ax.text(x, y, str(i), fontsize=9, verticalalignment='bottom', horizontalalignment='right')

    # plot route A
    ra = route_a + [route_a[0]]
    ax.plot(pts[ra, 0], pts[ra, 1], linestyle='-', linewidth=2, label=title_a)
    # plot route B
    rb = route_b + [route_b[0]]
    ax.plot(pts[rb, 0], pts[rb, 1], linestyle='--', linewidth=2, label=title_b)

    ax.set_title(f"{title_a} (solid) vs {title_b} (dashed)")
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    return fig

# ------------------------ Streamlit App ------------------------

st.set_page_config(page_title='Route Optimization — Greedy + Simulated Annealing', layout='wide')

st.title('🚚 Route Optimization — Greedy vs. Simulated Annealing')
st.markdown(
    """
    Single-file Streamlit app that generates synthetic city coordinates and package drop points,
    computes baseline routes (Greedy nearest neighbor and Random baseline) and optimizes using
    Simulated Annealing. Visualizes routes and estimates cost/time savings.
    """
)

# Sidebar controls
with st.sidebar:
    st.header('Experiment controls')
    n_points = st.slider('Number of drop points (including depot)', min_value=5, max_value=60, value=20)
    seed = st.number_input('Random seed (for reproducibility)', value=42, step=1)
    start_index = st.number_input('Depot / start index (0-based)', min_value=0, max_value=max(0, n_points - 1), value=0)
    method = st.selectbox('Optimization method', ['Simulated Annealing', 'Genetic Algorithm (not implemented)'])
    st.markdown('---')
    st.subheader('Simulated Annealing params')
    sa_temp = st.number_input('Initial temperature', value=1.0, format='%f')
    sa_cooling = st.number_input('Cooling rate (0.8 - 0.9999)', value=0.995, format='%f')
    sa_iters = st.number_input('Max iterations', value=5000, step=100)
    time_limit = st.number_input('Time limit (s) — 0 for none', value=0)
    st.markdown('---')
    st.subheader('Cost assumptions')
    fuel_cost_per_km = st.number_input('Fuel cost (currency unit) per unit distance', value=0.1, format='%f')
    avg_speed = st.number_input('Average speed (distance units per hour)', value=40.0, format='%f')
    st.markdown('---')
    st.button('Regenerate and run', key='run_button')

# Main
random.seed(seed)
np.random.seed(seed)

# Synthetic city generation: sample points from 0..100 square, but make depot near center
city_scale = 100
# depot at center-ish
depot = (city_scale * 0.5 + np.random.normal(scale=5), city_scale * 0.5 + np.random.normal(scale=5))
points = [depot]
# rest sample clustered + some noise so it's non-trivial
cluster_centers = [
    (20 + np.random.normal(scale=5), 20 + np.random.normal(scale=5)),
    (80 + np.random.normal(scale=5), 20 + np.random.normal(scale=5)),
    (20 + np.random.normal(scale=5), 80 + np.random.normal(scale=5)),
    (80 + np.random.normal(scale=5), 80 + np.random.normal(scale=5)),
]
for i in range(n_points - 1):
    cc = cluster_centers[i % len(cluster_centers)]
    x = np.random.normal(loc=cc[0], scale=8)
    y = np.random.normal(loc=cc[1], scale=8)
    points.append((x, y))

# compute distance matrix
D = compute_distance_matrix(points)

# Baseline routes
start_idx = int(start_index)
baseline_random = random_route(len(points), start=start_idx)
baseline_greedy = greedy_nearest_neighbor(D, start=start_idx)

# Initial solution for optimizer: use greedy
initial_route = baseline_greedy.copy()

# Run optimizer
if method == 'Simulated Annealing':
    sa_time_limit = None if time_limit <= 0 else float(time_limit)
    with st.spinner('Running Simulated Annealing...'):
        best_route, best_cost = simulated_annealing(D,
                                                   initial_route=initial_route,
                                                   initial_temp=float(sa_temp),
                                                   cooling_rate=float(sa_cooling),
                                                   iterations=int(sa_iters),
                                                   time_limit=sa_time_limit)
else:
    st.warning('Genetic Algorithm not implemented in this single-file demo. Using Simulated Annealing instead.')
    best_route, best_cost = simulated_annealing(D,
                                               initial_route=initial_route,
                                               initial_temp=float(sa_temp),
                                               cooling_rate=float(sa_cooling),
                                               iterations=int(sa_iters))

# Distances
dist_random = total_route_distance(baseline_random, D)
dist_greedy = total_route_distance(baseline_greedy, D)
dist_optimized = total_route_distance(best_route, D)

# Cost/time estimates
fuel_random = dist_random * fuel_cost_per_km
fuel_greedy = dist_greedy * fuel_cost_per_km
fuel_opt = dist_optimized * fuel_cost_per_km

# time in hours
time_random = dist_random / avg_speed
time_greedy = dist_greedy / avg_speed
time_opt = dist_optimized / avg_speed

# Show summary metrics
col1, col2, col3 = st.columns(3)
col1.metric('Random baseline distance', f"{dist_random:.2f}")
col2.metric('Greedy distance', f"{dist_greedy:.2f}", delta=f"{dist_random - dist_greedy:.2f}")
col3.metric('Optimized distance', f"{dist_optimized:.2f}", delta=f"{dist_greedy - dist_optimized:.2f}")

col1, col2 = st.columns(2)
col1.subheader('Estimated fuel cost')
col1.write(f"Random: {fuel_random:.2f} | Greedy: {fuel_greedy:.2f} | Optimized: {fuel_opt:.2f}")
col1.write(f"Fuel saved (Greedy -> Optimized): {fuel_greedy - fuel_opt:.2f}")

col2.subheader('Estimated time (hours)')
col2.write(f"Random: {time_random:.2f}h | Greedy: {time_greedy:.2f}h | Optimized: {time_opt:.2f}h")
col2.write(f"Time saved (Greedy -> Optimized): {time_greedy - time_opt:.2f}h")

# Visuals
st.subheader('Route visualization')
fig = plot_routes(points, baseline_greedy, best_route, title_a='Greedy (baseline)', title_b='Optimized (SA)')
st.pyplot(fig)

# Show alternative comparison with random baseline
st.subheader('Greedy vs Random baseline')
fig2 = plot_routes(points, baseline_greedy, baseline_random, title_a='Greedy (baseline)', title_b='Random baseline')
st.pyplot(fig2)

# Show route order as table
st.subheader('Route orders and distances')
route_df = pd.DataFrame({
    'Index': list(range(len(points))),
    'X': [p[0] for p in points],
    'Y': [p[1] for p in points]
})

st.markdown('**Greedy route order:**')
st.write(baseline_greedy)
st.markdown('**Optimized route order:**')
st.write(best_route)

st.markdown('**Points table**')
st.dataframe(route_df)

# Downloadable CSV of optimized route
st.markdown('---')
if st.button('Download optimized route as CSV'):
    out_df = route_df.copy()
    out_df['greedy_pos'] = pd.Series(index=baseline_greedy).sort_index().reset_index(drop=True)
    # safer: build order mapping
    greedy_order = {node: pos for pos, node in enumerate(baseline_greedy)}
    opt_order = {node: pos for pos, node in enumerate(best_route)}
    out_df['greedy_pos'] = out_df['Index'].map(greedy_order)
    out_df['opt_pos'] = out_df['Index'].map(opt_order)
    csv = out_df.to_csv(index=False)
    st.download_button('Click to download CSV', data=csv, file_name='optimized_route.csv', mime='text/csv')

st.markdown('---')
st.write('Notes:')
st.write('- This app uses Euclidean distances on synthetic 2D coordinates; for real road networks use routing APIs or graph representations.')
st.write('- Simulated Annealing here uses a simple 2-opt neighbor; for better results consider population-based genetic algorithms, local search hybrids, or using OR-Tools.')

# Short guide for GitHub + Streamlit deploy (instructions only)
with st.expander('How to put this on GitHub and deploy to Streamlit Cloud'):
    st.markdown(
        """
1. Create a new repository on GitHub and push this `app.py` to the root.
2. Add `requirements.txt` listing packages: streamlit, numpy, pandas, matplotlib, scipy
3. Sign in to Streamlit Cloud (https://share.streamlit.io), link your GitHub repo and select `app.py` to deploy.
4. Optionally, add a small `README.md` explaining parameters and usage.

This single-file structure is convenient for quick experiments and demos.
        """
    )

# End of app
