import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing

st.set_page_config(page_title="Route Optimizer", layout="wide")

# ---------------------------------------------------------
# Light / Dark Mode Toggle
# ---------------------------------------------------------
mode = st.sidebar.radio("Theme", ["Light", "Dark"])

if mode == "Dark":
    plt.style.use("dark_background")
else:
    plt.style.use("default")

# ---------------------------------------------------------
# Hidden synthetic city data
# ---------------------------------------------------------
def load_city_data(n):
    rng = np.random.default_rng(11)
    coords = rng.normal(loc=[12.93, 77.59], scale=[0.04, 0.03], size=(n, 2))
    return coords

# ---------------------------------------------------------
# Route distance
# ---------------------------------------------------------
def route_length(order, coords):
    d = 0
    for i in range(len(order) - 1):
        d += np.linalg.norm(coords[order[i]] - coords[order[i + 1]])
    return d

# ---------------------------------------------------------
# Simulated Annealing objective builder
# ---------------------------------------------------------
def build_sa_objective(coords):
    def objective(x):
        order = np.argsort(x)
        return route_length(order, coords)
    return objective

# ---------------------------------------------------------
# Genetic Algorithm (Corrected)
# ---------------------------------------------------------
def ga_optimize(coords, pop_size=40, generations=80, elite=4, mutation_rate=0.15):
    n = len(coords)

    def fitness(order):
        return route_length(order, coords)

    # Initial population as NumPy array
    population = np.array([np.random.permutation(n) for _ in range(pop_size)])

    for _ in range(generations):
        scores = np.array([fitness(p) for p in population])
        ranked = population[np.argsort(scores)]

        # Elitism
        new_pop = ranked[:elite].tolist()

        # Crossover
        while len(new_pop) < pop_size:
            p1 = ranked[np.random.randint(elite)]
            p2 = ranked[np.random.randint(elite)]
            cut = np.random.randint(1, n - 1)

            child = np.concatenate((p1[:cut], [i for i in p2 if i not in p1[:cut]]))
            new_pop.append(child)

        # Mutation
        for i in range(elite, pop_size):
            if np.random.rand() < mutation_rate:
                a, b = np.random.randint(0, n, 2)
                new_pop[i][a], new_pop[i][b] = new_pop[i][b], new_pop[i][a]

        population = np.array(new_pop)

    best = population[np.argmin([fitness(p) for p in population])]
    return best

# ---------------------------------------------------------
# Tutorial (Sidebar)
# ---------------------------------------------------------
st.sidebar.title("How to Use")
st.sidebar.markdown("""
1. Choose algorithm (Simulated Annealing or Genetic Algorithm).  
2. Select number of delivery stops.  
3. Click **Run Optimization**.  
4. View *Before vs After* route and cost saved.  
""")

# ---------------------------------------------------------
# Main Interface
# ---------------------------------------------------------
st.title("🚚 Route Optimization — Simulated Annealing + Genetic Algorithm")
st.markdown("Compare two classic optimization algorithms on a delivery route.")

algo = st.selectbox("Choose Optimization Method", ["Simulated Annealing", "Genetic Algorithm"])
n_points = st.slider("Number of Delivery Stops", 10, 40, 20)

if st.button("Run Optimization"):
    coords = load_city_data(n_points)

    base_order = np.arange(n_points)
    base_cost = route_length(base_order, coords)

    # Optimization
    if algo == "Simulated Annealing":
        bounds = [(-1, 1)] * n_points
        objective = build_sa_objective(coords)
        result = dual_annealing(objective, bounds, maxiter=200)
        final_order = np.argsort(result.x)

    else:
        final_order = ga_optimize(coords)

    final_cost = route_length(final_order, coords)
    saved = (base_cost - final_cost) / base_cost * 100

    col1, col2 = st.columns(2)

    # BEFORE
    with col1:
        st.subheader("📍 Before Optimization")
        fig1, ax1 = plt.subplots()
        ax1.plot(coords[base_order, 0], coords[base_order, 1], marker='o')
        ax1.set_title(f"Distance: {base_cost:.4f}")
        st.pyplot(fig1)

    # AFTER
    with col2:
        st.subheader("🚀 After Optimization")
        fig2, ax2 = plt.subplots()
        ax2.plot(coords[final_order, 0], coords[final_order, 1], marker='o')
        ax2.set_title(f"Optimized Distance: {final_cost:.4f}")
        st.pyplot(fig2)

    st.success(f"💰 **Cost Saved: {saved:.2f}%**")

