import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    eigenvector = np.random.uniform(0, 1, data.shape[0])
    
    for _ in range (num_steps):
        eigenvector = data.dot(eigenvector) / np.linalg.norm(data.dot(eigenvector))
    eigenvalue = float(eigenvector.dot(data.dot(eigenvector))/ (eigenvector.dot(eigenvector)))

    return eigenvalue, eigenvector
