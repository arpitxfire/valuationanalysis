import numpy as np


def run_simulation(s0, mu, sigma, T, n_sims=10000, dt=1 / 252):
    n_steps = int(T * 252)
    shocks = np.random.normal(0, 1, (n_steps, n_sims))
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * shocks
    daily_growth = np.exp(drift + diffusion)
    path_matrix = np.zeros((n_steps + 1, n_sims))
    path_matrix[0] = s0
    path_matrix[1:] = s0 * np.cumprod(daily_growth, axis=0)
    low_band = np.percentile(path_matrix, 5, axis=1)
    high_band = np.percentile(path_matrix, 95, axis=1)
    return path_matrix, low_band, high_band
