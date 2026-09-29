import numpy as np
import scipy.sparse as sp
from qiskit.quantum_info import SparsePauliOp

def pad_matrix_to_power_of_2(matrix):
    """
    Qiskit requires matrix dimensions to be exactly 2^n x 2^n.
    This function pads the matrix with 1s on the diagonal for the extra dimensions
    so that the matrix remains invertible for VQLS.
    """
    size = matrix.shape[0]
    num_qubits = int(np.ceil(np.log2(size)))
    new_size = 2**num_qubits
    
    if size == new_size:
        return matrix, num_qubits
        
    print(f"Padding matrix from {size}x{size} to {new_size}x{new_size}...")
    padded = sp.eye(new_size, dtype=complex).tolil()
    
    if sp.issparse(matrix):
        padded[:size, :size] = matrix
    else:
        padded[:size, :size] = sp.csr_matrix(matrix)
        
    return padded.tocsr(), num_qubits

def pad_vector_to_power_of_2(vector, new_size):
    """Pads the RHS vector with 0s to match the padded matrix dimension."""
    size = len(vector)
    if size == new_size:
        return vector
    padded = np.zeros(new_size, dtype=vector.dtype)
    padded[:size] = vector
    return padded

def decompose_to_paulis(matrix):
    """
    Decomposes a padded matrix into a sum of Pauli strings (SparsePauliOp).
    """
    padded_matrix, num_qubits = pad_matrix_to_power_of_2(matrix)
    
    print(f"Decomposing into Pauli strings ({num_qubits} qubits)... this may take a moment for large matrices.")
    # from_operator requires dense array representation
    pauli_op = SparsePauliOp.from_operator(padded_matrix.toarray())
    
    # Clean up near-zero coefficients (helps VQLS circuit depth)
    print(f"Original Pauli strings count: {len(pauli_op)}")
    pauli_op = pauli_op.simplify(atol=1e-6)
    print(f"Simplified Pauli strings count: {len(pauli_op)}")
    
    return pauli_op, num_qubits
