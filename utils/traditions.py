import numpy as np
from scipy import sparse

def fd_A_dir(n):
    N2 = n**2
    A = sparse.lil_matrix((N2, N2))
    for i in range(1, n-1):
        for j in range(1, n-1):
            idx = i * n + j
            A[idx, idx] += 4
            A[idx, idx-1] = -1
            A[idx, idx+1] = -1
            A[idx, idx-n] = -1
            A[idx, idx+n] = -1
    # Homogeneous Dirichlet Boundary
    for i in range(0, n):
        idx = 0 * n + i
        A[idx, idx] = 1

        idx = (n-1) * n + i
        A[idx, idx] = 1

        idx = i * n
        A[idx, idx] = 1

        idx = i * n + n - 1
        A[idx, idx] = 1
    A = A.tocoo()
    return A

def fd_b_dir(f, h, order=2):
    n, _ = f.shape
    h2 = h**2
    b = np.zeros(n**2)

    for i in range(1, n-1):
        for j in range(1, n-1):
            idx = i * n + j
            b[idx] = f[i, j]*h2

    return b

def fd_A_neu(n, neus=['left', 'right'], diris=['top', 'bottom']):
    '''
    Generate linear system for 
        Top, Down Dirichlet Left, Right Neumann Boundary
    '''
    N2 = n**2
    A = sparse.lil_matrix((N2, N2))
    
    for i in range(1, n-1):
        for j in range(1, n-1):
            idx = i * n + j
            A[idx, idx] += 4
            A[idx, idx-1] = -1
            A[idx, idx+1] = -1
            A[idx, idx-n] = -1
            A[idx, idx+n] = -1
    
    # Neumann Boundary
    for k in neus:
        if k == 'top':
            for i in range(1, n-1):
                idx = 0 * n + i
                A[idx, idx] = 4
                A[idx, idx+n] = -2
                A[idx, idx-1] = A[idx, idx+1] = -1
                
        if k == 'bottom':
            for i in range(1, n-1):
                idx = (n-1) * n + i
                A[idx, idx] = 4
                A[idx, idx-n] = -2
                A[idx, idx-1] = A[idx, idx+1] = -1
                
        if k == 'left':
            for i in range(1, n-1):
                idx = i * n
                A[idx, idx] = 4
                A[idx, idx+1] = -2
                A[idx, idx-n] = A[idx, idx+n] = -1
                
        if k == 'right':
            for i in range(1, n-1):
                idx = i * n + n -1
                A[idx, idx] = 4
                A[idx, idx-1] = -2
                A[idx, idx-n] = A[idx, idx+n] = -1
                
    # Dirichlet Boundary
    for k in diris:
        if k == 'top':
            for i in range(0, n):
                idx = 0 * n + i
                A[idx, idx] = 1
                
        if k == 'bottom':
            for i in range(0, n):
                idx = (n-1) * n + i
                A[idx, idx] = 1
                
        if k == 'left':
            for i in range(0, n):
                idx = i * n
                A[idx, idx] = 1
            
        if k == 'right':
            for i in range(0, n):
                idx = i * n + n -1
                A[idx, idx] = 1
    A = A.tocoo()
    return A

def fv_A_dirichlet(n):
    n2 = n**2
    A = sparse.lil_matrix((n2, n2))
    # Interior points
    for i in range(1, n-1):
        for j in range(1, n-1):
            idx = i * n + j
            A[idx, idx] = -4
            A[idx, idx+1] = A[idx, idx-1] = A[idx, idx-n] = A[idx, idx+n] = 1
    
    # Boundary points
    for i in range(1, n-1):
        # Top
        idx = i
        A[idx, idx] = -6
        A[idx, idx+1] = A[idx, idx-1] = 1
        A[idx, idx+n] = 4/3
        
        # Bottom
        idx = (n-1) * n + i
        A[idx, idx] = -6
        A[idx, idx+1] = A[idx, idx-1] = 1
        A[idx, idx-n] = 4/3
        
        # Left
        idx = i * n
        A[idx, idx] = -6
        A[idx, idx+n] = A[idx, idx-n] = 1
        A[idx, idx+1] = 4/3
        
        # Right
        idx = i * n + n - 1
        A[idx, idx] = -6
        A[idx, idx+n] = A[idx, idx-n] = 1
        A[idx, idx-1] = 4/3
        
    # Four corners
    # Left top
    idx = 0
    A[idx, idx] = -8
    A[idx, idx+1] = A[idx, idx+n] = 4/3
    # Right Top
    idx = n-1
    A[idx, idx] = -8
    A[idx, idx-1] = A[idx, idx+n] = 4/3
    # Left Bottom
    idx = (n-1) * n
    A[idx, idx] = -8
    A[idx, idx+1] = A[idx, idx-n] = 4/3
    # Right Bottom
    idx = n2 - 1
    A[idx, idx] = -8
    A[idx, idx-1] = A[idx, idx-n] = 4/3
    
    A = A.tocoo()
    return A

def fv_A_neu(n):
    '''
    Left and Right are neumann, top and down are dirichlet
    '''
    n2 = n**2
    A = sparse.lil_matrix((n2, n2))
    # Interior points
    for i in range(1, n-1):
        for j in range(1, n-1):
            idx = i * n + j
            A[idx, idx] = -4
            A[idx, idx+1] = A[idx, idx-1] = A[idx, idx-n] = A[idx, idx+n] = 1
    
    # Boundary points
    for i in range(1, n-1):
        # Top
        idx = i
        A[idx, idx] = -6
        A[idx, idx+1] = A[idx, idx-1] = 1
        A[idx, idx+n] = 4/3
        
        # Bottom
        idx = (n-1) * n + i
        A[idx, idx] = -6
        A[idx, idx+1] = A[idx, idx-1] = 1
        A[idx, idx-n] = 4/3
        
        # Left
        idx = i * n
        A[idx, idx] = -3
        A[idx, idx+n] = A[idx, idx-n] = A[idx, idx+1] = 1
        
        # Right
        idx = i * n + n - 1
        A[idx, idx] = -3
        A[idx, idx+n] = A[idx, idx-n] = A[idx, idx-1] = 1
        
    # Four corners
    
    # Left top
    idx = 0
    A[idx, idx] = -5
    A[idx, idx+n] = 4/3
    A[idx, idx+1] = 1
    
    # Right Top
    idx = n-1
    A[idx, idx] = -5
    A[idx, idx+n] = 4/3
    A[idx, idx-1] = 1
    
    # Left Bottom
    idx = (n-1) * n
    A[idx, idx] = -5
    A[idx, idx-n] = 4/3
    A[idx, idx+1] = 1
    
    # Right Bottom
    idx = n2 - 1
    A[idx, idx] = -5
    A[idx, idx-n] = 4/3
    A[idx, idx-1] = 1
    
    A = A.tocoo()
    return A

def apply_neumann_bc(b, h, f, bcs={'left': 0, 'right': 0}):
    '''
    Apply Neumann boundary conditions on the left, right boundary.
    '''
    n = int(np.sqrt(len(b)))
    h2 = h**2
    for k in bcs.keys():
        g = bcs[k]
        if k == 'top':
            for i in range(1, n-1):
                idx = i
                b[idx] = h2 * f[0, i] - 2 * h * g

        if k == 'bottom':
            for i in range(1, n-1):
                idx = (n-1) * n + i
                b[idx] = h2 * f[-1, i] - 2 * h * g
        
        if k == 'left':
            for i in range(1, n-1):
                idx = i * n
                b[idx] = h2 * f[i, 0] - 2 * h * g

        if k == 'right':
            for i in range(1, n-1):
                idx = i * n + n - 1
                b[idx] = h2 * f[i, -1] - 2 * h * g
    return b

def apply_diri_bc(b, bcs={'top': 0, 'bottom': 0}):
    n = int(np.sqrt(len(b)))
    
    for k in bcs.keys():
        g = bcs[k]
        if k == 'top':
            for i in range(n):
                idx = 0 * n + i
                b[idx] = g
        if k == 'bottom':
            for i in range(n):
                idx = (n-1) * n + i
                b[idx] = g
        if k == 'left':
            for i in range(n):
                idx = i * n
                b[idx] = g
        if k == 'right':
            for i in range(n):
                idx = i * n + n - 1 
                b[idx] = g
    return b

def apply_dirichlet_bc_for_all(b, bc, g, order=2):
    '''
    bc --> True for soft, Flase for Hard
    g --> the value of boundary
    '''
    if g == 0:
        return b
    n = int(np.sqrt(len(b)))
    if bc:
        for i in range(0, n):
            idx = 0 * n + i
            b[idx] = g

            idx = (n-1) * n + i
            b[idx] = g

            idx = i * n
            b[idx] = g

            idx = i * n + n - 1
            b[idx] = g
    else:
        if order == 2:
            for i in range(0, n):
                idx = 0 * n + i
                b[idx] += g

                idx = (n-1) * n + i
                b[idx] += g

                idx = i * n
                b[idx] += g

                idx = i * n + n - 1
                b[idx] += g

        elif order == 4:
            for i in range(0, n):
                idx = 0 * n + i
                b[idx] += 6*g

                idx = (n-1) * n + i
                b[idx] += 6*g

                idx = i * n
                b[idx] += 6*g

                idx = i * n + n - 1
                b[idx] += 6*g
            b[0] -= g
            b[n-1] -= g
            b[(n-1)*n] -= g
            b[-1] -= g
    return b