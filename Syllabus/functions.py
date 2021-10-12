"""
Functions for APML syllabus file.

Functions: Bayes Theorem - bayes.
"""

import numpy as np


def scalar_bayes(likelihood, prior, marginal):
    """Scalar verson of bayes theorem.

    Args:
        likelihood (Float): Likelihood of observation.
        prior (Float): Prior belief before observation.
        marginal (Float): Evidence (probability of observation).
    Return:
        posteriorn (Float): Posterior belief of event given observation.
    """
    return((likelihood*prior) / marginal)
