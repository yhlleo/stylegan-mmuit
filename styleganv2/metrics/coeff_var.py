# Coefficient of variation, also known as relative standard deviation (RSD), 
# is a standardized measure of dispersion of a probability distribution or 
# frequency distribution. 
# It is often expressed as a percentage, and is defined as the ratio of the 
# standard deviation to the mean.

import numpy as np

def coefficient_variation_standard(distances):
    mu = np.mean(distances, axis=1)
    sigma = np.std(distances, axis=1)
    return sigma/(mu + 1e-6)

def coefficient_variation_normal(distances):
    mu = np.mean(distances, axis=1)
    unit_distances = distances - mu + 1.0
    sigma = np.std(unit_distances, axis=1)
    return sigma

def coefficient_variation_maxmin(distances, shortest_path):
    d_max = np.max(distances, axis=1)
    d_min = np.min(distances, axis=1)
    return (d_max - d_min)/(shortest_path + 1e-6)
