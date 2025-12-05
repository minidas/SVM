# Core Python
from functools import reduce
import os
import time

# Scientific Computing
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
import scipy.sparse.linalg

# Quantum Computing
from qiskit.quantum_info import Statevector, SparsePauliOp

# Optimization and Visualization
import cvxpy as cp
import matplotlib.pyplot as plt

class Nearest_Neighbour_1d:
    """
    Constructs a 1D nearest-neighbor quantum Hamiltonian with optional periodic boundary conditions.
    
    Parameters:
    -----------
    n : int
        Number of qubits
    Jx, Jy, Jz : float
        Coupling strengths for XX, YY, ZZ interactions
    hx, hy, hz : float
        Local field strengths
    pbc : bool
        Whether to use periodic boundary conditions
    verbose : bool
        Whether to print Hamiltonian information
    rand_field : list
        Random field values for each qubit
    """
    
    def __init__(self, n: int, Jx=0, Jy=0, Jz=0, hx=0, hy=0, hz=0, pbc=False, verbose=False, rand_field=[]):
        self.n = n
        self.Jx, self.Jy, self.Jz = Jx, Jy, Jz
        self.hx, self.hy, self.hz = hx, hy, hz

        # Build interaction terms
        self.xx_tuples = [('XX', [i, i + 1], Jx) for i in range(0, n-1)]
        self.yy_tuples = [('YY', [i, i + 1], Jy) for i in range(0, n-1)]
        self.zz_tuples = [('ZZ', [i, i + 1], Jz) for i in range(0, n-1)]

        # Handle random fields
        if len(rand_field) == 0:
            self.rand_field = [0]*n
        elif len(rand_field) >= n:
            self.rand_field = rand_field[:n]
        else:
            raise ValueError(f'Length of random field should be at least {n}!')

        # Build local field terms
        self.x_tuples = [('X', [i], (self.rand_field[i]+1)*hx) for i in range(0, n)] 
        self.y_tuples = [('Y', [i], (self.rand_field[i]+1)*hy) for i in range(0, n)] 
        self.z_tuples = [('Z', [i], (self.rand_field[i]+1)*hz) for i in range(0, n)] 

        # Add periodic boundary conditions
        if pbc and n > 2: 
            self.xx_tuples.append(('XX', [n-1, 0], Jx))
            self.yy_tuples.append(('YY', [n-1, 0], Jy))
            self.zz_tuples.append(('ZZ', [n-1, 0], Jz))

        # Construct the full Hamiltonian
        all_terms = [*self.xx_tuples, *self.yy_tuples, *self.zz_tuples, 
                    *self.x_tuples, *self.y_tuples, *self.z_tuples]
        self.ham = SparsePauliOp.from_sparse_list(all_terms, num_qubits=n).simplify()
        
        # Group terms for analysis
        self.xyz_group()
        self.par_group()
        
        if verbose: 
            print('The Hamiltonian: \n', self.ham)
            print('The xyz grouping: \n', self.ham_xyz)
            print('The parity grouping: \n', self.ham_par)

    def xyz_group(self):
        """
        Group Hamiltonian terms by Pauli operator types (X, Y, Z).
        
        This creates separate SparsePauliOp objects for each type of Pauli operator,
        making it easier to analyze the different components of the Hamiltonian.
        """
        # Extract X, Y, Z terms from the full Hamiltonian
        all_terms = self.ham.to_list()
        
        # Separate terms by Pauli operator type
        x_terms = []
        y_terms = []
        z_terms = []
        
        for pauli_string, coeff in all_terms:
            if 'X' in pauli_string and 'Y' not in pauli_string and 'Z' not in pauli_string:
                x_terms.append((pauli_string, coeff))
            elif 'Y' in pauli_string and 'X' not in pauli_string and 'Z' not in pauli_string:
                y_terms.append((pauli_string, coeff))
            elif 'Z' in pauli_string and 'X' not in pauli_string and 'Y' not in pauli_string:
                z_terms.append((pauli_string, coeff))
        
        # Create grouped Hamiltonians
        self.ham_x = SparsePauliOp.from_list(x_terms) if x_terms else SparsePauliOp('I' * self.n, coeffs=[0])
        self.ham_y = SparsePauliOp.from_list(y_terms) if y_terms else SparsePauliOp('I' * self.n, coeffs=[0])
        self.ham_z = SparsePauliOp.from_list(z_terms) if z_terms else SparsePauliOp('I' * self.n, coeffs=[0])
        
        # Store the grouped terms
        self.ham_xyz = {
            'X': self.ham_x,
            'Y': self.ham_y, 
            'Z': self.ham_z
        }
    
    def par_group(self):
        """
        Group Hamiltonian terms by parity (even/odd).
        
        This separates terms based on whether they preserve or flip certain symmetries,
        which is useful for analyzing conserved quantities and symmetries.
        """
        # Extract all terms from the full Hamiltonian
        all_terms = self.ham.to_list()
        
        # Separate terms by parity
        even_terms = []
        odd_terms = []
        
        for pauli_string, coeff in all_terms:
            # Count the number of non-identity Pauli operators
            num_paulis = sum(1 for p in pauli_string if p != 'I')
            
            # Even parity: even number of Pauli operators
            # Odd parity: odd number of Pauli operators
            if num_paulis % 2 == 0:
                even_terms.append((pauli_string, coeff))
            else:
                odd_terms.append((pauli_string, coeff))
        
        # Create grouped Hamiltonians
        self.ham_even = SparsePauliOp.from_list(even_terms) if even_terms else SparsePauliOp('I' * self.n, coeffs=[0])
        self.ham_odd = SparsePauliOp.from_list(odd_terms) if odd_terms else SparsePauliOp('I' * self.n, coeffs=[0])
        
        # Store the grouped terms
        self.ham_par = {
            'even': self.ham_even,
            'odd': self.ham_odd
        }