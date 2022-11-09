import numpy as np

class BackwardEvolutionMetrics:
    def __init__(self):
        # Backward evolution metrics along branches
        self.local_symmetry = np.array([])
        self.overshoot = np.array([])
        self.angular_deflection = np.array([])
        
        # Backward evolution metrics in bifurcation nodes
        self.length_mismatch = np.array([])
        self.ai_coefficients = np.array([])
        
    def compare_networks(self, network_original, network_back_forth):
        1