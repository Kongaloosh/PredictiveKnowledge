"""
Functions which perform the necessary operations for GTD(lambda) learning.
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
    return rho[:, None] * (traces * (gamma * lmbda) + phi.T)


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

    return cumulant + gamma_next.T * np.dot(weights, phi_next) - np.dot(weights, phi)


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
    return weights + step_size * (td_error.T * traces - gamma * (1 - lmbda) * np.sum(traces * h, axis=1)[:, None] * phi_next)


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
    return h + step_size * (td_error.T * traces - np.dot(h, phi)[:, None] * phi.T)


def update_meta_traces(h, step_size, td_error, traces):
    """Updates the meta-traces for TDBID; is an accumulating trace of recent weight updates.
    Args:
        phi (ndarray): the last feature vector; represents s_t
        td_error (float): the temporal-difference error for the current time-step.
    """
    h += step_size * td_error * traces
    return h


def update_meta_weights(phi, td_error, meta_weights, meta_trace, meta_step_size):
    """Updates the meta-weights for TIDBD; these are used to set the step-sizes.
    Args:
        phi (ndarray): the last feature vector; represents s_t
        td_error (float): the temporal-difference error for the current time-step.
    """
    meta_weights += phi * meta_step_size * td_error * meta_trace
    return meta_weights


def update_normalizer_accumulation(phi, td_error, tau, meta_weights, meta_trace, traces, meta_normalizer_trace):
    """Tracks the size of the meta-weight updates.
    Args:
        phi (ndarray): the last feature vector; represents s_t
        td_error (float): the temporal-difference error for the current time-step."""
    delta_phi = -phi
    update = np.abs(td_error * delta_phi * meta_trace)
    tracker = np.exp(meta_weights) * traces * delta_phi
    meta_normalizer_trace = np.maximum(
        np.abs(update),
        meta_normalizer_trace + (1. / tau) * tracker * (np.abs(update) - meta_normalizer_trace)
    )
    return meta_normalizer_trace


def get_effective_step_size(self, gamma, phi, phi_next):
    """Returns the effective step-size for a given time-step
    Args:
        phi (ndarray): the last feature vector; represents s_t
        phi_next (ndarray): feature vector for state s_{t+1}
        gamma (float): discount factor
    Returns:
        effective_step_size (float): the amount by which the error was reduced on a given example.
    """
    delta_phi = (gamma * phi_next - phi)
    return np.dot(-(np.exp(self.meta_weights) * self.eligibilityTrace), delta_phi)


def normalize_step_size(gamma, phi, phi_next, meta_weights):
    """Calculates the effective step-size and normalizes the current step-size by that amount.
    Args:
        gamma (float): discount factor
        phi (ndarray): feature vector for state s_t
        phi_next (ndarray): feature vector for state s_{t+1}"""
    effective_step_size = get_effective_step_size(gamma, phi, phi_next)
    m = np.maximum(effective_step_size, 1.)
    meta_weights /= np.log(m)
    return meta_weights


def tdbid(self, phi, phi_next, gamma, td_error):
    """Using the feature vector for s_t and the current TD error, performs TIDBD and updates step-sizes.
    Args:
        phi (ndarray): the last feature vector; represents s_t
        phi_next (ndarray): feature vector for state s_{t+1}
        gamma (float): discount factor
        td_error (float): the temporal-difference error for the current time-step.

    """
    self.update_normalizer_accumulation(phi, td_error)
    self.update_meta_weights(phi, td_error)
    self.normalize_step_size(gamma, phi, phi_next)
    self.calculate_step_size()
    self.update_meta_traces(phi, td_error)


def calculate_step_size(meta_weights):
    """Calculates the current alpha value using the meta-weights
    Returns:
         None
    """
    return np.exp(meta_weights)