import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

def coo2tensor(A):
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(torch.float32)

def np2torch(data_path, backward_type='jac', boundary_type='D', numerical_method='fd'):
    '''
    backward_type: To identify which iterative method to use.
        Jacobian, Gauess Seidel, CG.
    '''
    A_path = f'{data_path}{numerical_method}_A{boundary_type}'
    invM_path = f'{A_path}_{backward_type}_invM.npz'
    M_path = f'{A_path}_{backward_type}_M.npz'  

    A = sparse.load_npz(A_path+'.npz')
    invM = sparse.load_npz(invM_path)
    M = sparse.load_npz(M_path)
    return coo2tensor(A), coo2tensor(invM.tocoo()), coo2tensor(M.tocoo())

def gradient_descent(x, A, b):
    r = mmbv(A, x) - b
    Ar = mmbv(A, r)
    alpha = bvi(r, r)/bvi(r, Ar)
    y = x + alpha * r
    return y

def mmbv(A, y):
    """
    Sparse matrix multiply Batched vectors
    """
    y = torch.transpose(y, 0, 1)
    v = torch.sparse.mm(A, y)
    v = torch.transpose(v, 0, 1)
    return v

def bvi(x, y):
    """
    inner product of Batched vectors x and y
    """
    b, n = x.shape
    inner_values =  torch.bmm(x.view((b, 1, n)), y.view((b, n, 1))) 
    return inner_values.reshape(b, 1)

def energy(x, A, b):
    Ax = mmbv(A, x)
    bx = bvi(b, x)
    xAx = bvi(x, Ax)
    return (xAx/2 - bx).mean()

def mse_loss(x, A, b):
    Ax = mmbv(A, x)
    norms = torch.norm((Ax-b), p=2, dim=1, keepdim=True)
    return norms.mean()

def fd_pad_neu_bc(x, h=0, pad=(1, 1, 0, 0), g = 0):
    val = 2 * h * g
    x = F.pad(x, pad=pad, mode='reflect')
    if val == 0:
        return x
    if pad[0] == 1:
        x[..., :, 0] += val
    if pad[1] == 1:
        x[..., :, -1] += val
    if pad[2] == 1:
        x[..., 0, :] += val
    if pad[3] == 1:
        x[...,-1, :] += val
    return x

def fd_pad_diri_bc(x, pad=(0, 0, 1, 1), g = 0):
    x = F.pad(x, pad=pad, mode='constant', value=g)
    return x

def internal_conv(x):
    kernel = torch.tensor([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    kernel = kernel.type_as(x).view(1, 1, 3, 3).repeat(1, 1, 1, 1)
    rhs = F.conv2d(x, kernel)
    return rhs

def convRhs(numerical_method, a, n, dir_bcs=(1,1,1,1), neu_bcs=None, gn=0, gd=0, k=1):
    '''
    Generate rhs function.
    Params:
        numerical_method: fd, fv, (fem)
        a: Length of edges in the squared domain.
        n: Resolution of the mesh.
        dir_bcs: Which edge is dirichlet type boundary.(left, right, top, bottom).
            (1, 1, 1, 1) means all edges are dirichlet type.
        neu_bcs: Which edge is neumann type boundary.Same as dir_bcs.
        gd: The constant value of dirichlet type boundaries.
        gn: The constant value of neumann type boundaries.
        k: The constant parameters in lapalce equation.
    '''
    dir_pad = lambda x: x
    neu_pad = lambda x: x
    force = lambda x:x
    if numerical_method == 'fd':
        h = 2 * a / (n - 1)
        h2 = h**2
        
        force = lambda f: h2 * f/ (4 * k)
        if not dir_bcs is None:
            dir_pad = lambda x: fd_pad_diri_bc(x, dir_bcs, gd)
            force = lambda f: h2 * _genforce(f, dir_bcs, n)  / (4 * k)

        if not neu_bcs is None:
            neu_pad = lambda x: fd_pad_neu_bc(x, h, neu_bcs, gn)

        padder = lambda x: dir_pad(x)
        conver = lambda x, f: internal_conv(dir_pad(neu_pad(x))) + force(f)

        return padder, conver
    
    elif numerical_method == 'fv':
        h = 2 * a / n
        pass
        
    pass

def _genforce(f, dir_bcs, n):
    bs, c, _, _ = f.shape
    idx = torch.ones((bs, c, n-2, n-2))

    # pad idx with one for neumann boundary
    pad = tuple(int(i == 0) for i in dir_bcs)
    idx = F.pad(idx, pad, 'constant', 1)

    # pad idx with zero for dirichlet boundary
    pad = dir_bcs
    idx = F.pad(idx, pad, 'constant', 0)

    idx = torch.gt(idx, 0)
    nx = n - sum(dir_bcs[:2])
    ny = n - sum(dir_bcs[2:])
    return f[idx].reshape(bs, c, nx, ny)
    
if __name__ == '__main__':
    u = torch.rand(1, 1, 3, 5)
    f = torch.rand(1, 1, 5, 5)

    padder, conver = convRhs('fd', 1, 5, (1,1,0,0), (0, 0, 1, 1))
    u = padder(u)
    print(u.shape)
    conv = conver(u, f)
    print(conv.shape)
