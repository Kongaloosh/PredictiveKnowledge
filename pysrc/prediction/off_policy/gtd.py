"""
Functions which perform the necessary operations for GTD(lambda) learning.

TODO:
    - add TIDBD-like operations
"""

import numpy as np

__author__ = "Alex Kearney"


def replace(traces, gamma, lmbda, phi, rho):
    """Updates traces with replacing
    Args:
        traces: the eligibility traces used to assign credit to previous states for current rewards or observations.
        gamma: the discount function for the current state.
        lmbda: the eligibility trace decay value. 
        phi: the binary feature vector representing the current state.
        rho: importance sampling ratio.
        
    Returns:
        tracs: the eligibility traces updated for the current state.
    
    """
    traces = rho[:, None] * (gamma * lmbda * traces * (phi.T == 0.) + (phi.T != 0.) * phi.T)
    return traces


def accumulate(traces, gamma, lmbda, phi, rho):
    """Updates traces without replacing
    Args: 
        traces: the eligibility traces to asign credit for current observations to previous states.
        gamma: the discounting value for the current update.
        lmbda: the amount by which we decay our eligibility traces.
        phi: binary feature vector representing the current state.
        rho: importance sampling ratio.
    Returns:
        traces: the eligibility traces to asign credit for current observations to previous states.
    """
    return rho[:,None] * (traces * (gamma * lmbda) + phi.T)


def calculate_temporal_difference_error(weights, cumulant, gamma_next, phi_next, phi):
    """Based on given inputs, calculates a TD error
    Args:
        weights: the learned weights of our model.
        cumulant: the currently observed cumulant signal which we are learning to predict.
        gamma_next: the discounting value for the current update.
        phi_next: binary feature vector representing the next state.
        phi: binary feature vector representing the current state.
    Returns:
        td_error: the temporal-difference error for the current observations.
    """
    return cumulant + gamma_next * np.inner(weights, phi_next) - np.inner(weights, phi_next)


def update_weights(td_error, traces, weights, gamma, lmbda, step_size, phi_next, h):
    """Updates the weights given a step-size, traces, and a TD error.
    Args:
        td_error: the temporal-difference error.
        traces: the eligibility traces to assign credit for current observations to previous states.
        weights: the learned weights of our model.
        gamma: the discounting value for the current update.
        lmbda: the amount by which we decay our eligibility traces.
        step_size: the amount by which weight updates are scaled.
        phi_next: binary feature vector representing the next state.
        h: bias correction term.
    Returns:
        weights: updated weights correcting for current TD error.
    """
    return weights + step_size * (td_error * traces - gamma * (1 - lmbda) * np.inner(traces, h) * phi_next.T)


def update_h_trace(h, td_error, step_size,traces,phi):
    """Updates the GTD bias correction term h
    Args:.
        h: bias correction term.
        td_error: the temporal-difference error.
        step_size: the amount by which weight updates are scaled.
        traces: the eligibility traces to assign credit for current observations to previous states
        phi: binary feature vector representing the current state.
    Returns:
        h: updated bias correction term.
    """
    return h + step_size * (td_error * traces - np.inner(h, phi) * phi.T)
