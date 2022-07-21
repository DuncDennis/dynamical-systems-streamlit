"""Measures for the dynamical systems."""

from __future__ import annotations

from typing import Callable
import numpy as np

def largest_lyapunov_exponent(iterator_func: Callable[[np.ndarray], np.ndarray],
                              starting_point: np.ndarray,
                              deviation_scale: float = 1e-10,
                              N: int = int(1e5),
                              part_time_steps: int = 10,
                              dt: float = 1.0,
                              initial_perturbation: np.ndarray | None = None,
                              return_convergence: bool = False
                              ) -> float | np.ndarray:
    """Numerically calculate the largest lyapunov exponent given an iterator function.

    See: Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series analysis. Vol. 69.
    Oxford: Oxford university press, 2003.

    Args:
        iterator_func: Function to iterate the system to the next time step: x(i+1) = F(x(i))
        starting_point: The starting_point of the main trajectory.
        deviation_scale: The L2-norm of the initial perturbation.
        N: Number of renormalization steps.
        part_time_steps: Time steps between renormalization steps.
        dt: Size of time step.
        initial_perturbation:
            - If np.ndarray: The direction of the initial perturbation.
            - If None: The direction of the initial perturbation is assumed to be np.ones(..).
        return_convergence: If True, return the convergence of the largest LE; a numpy array of
                            the shape (N, ).

    Returns:
        The largest Lyapunov Exponent. If return_convergence is True: The convergence (np.ndarray),
        else just the float value.
    """

    x_dim = starting_point.size

    if initial_perturbation is None:
        initial_perturbation = np.ones(x_dim)

    initial_perturbation *= deviation_scale/np.linalg.norm(initial_perturbation)

    log_divergence = np.zeros(N)

    x = starting_point
    x_pert = starting_point + initial_perturbation

    for i_n in range(N):
        for i_t in range(part_time_steps):
            x = iterator_func(x)
            x_pert = iterator_func(x_pert)
        dx = x_pert - x
        norm_dx = np.linalg.norm(dx)
        log_divergence[i_n] = np.log(norm_dx / deviation_scale)
        x_pert = x + dx * (deviation_scale/norm_dx)

    if return_convergence:
        return np.cumsum(log_divergence) / (np.arange(1, N + 1) * dt * part_time_steps)
    else:
        return np.average(log_divergence)/(dt*part_time_steps)

