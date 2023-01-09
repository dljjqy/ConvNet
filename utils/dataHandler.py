import scipy.sparse.linalg as sla
from traditions import *
from deltas import *
from pathlib import Path
from tqdm import tqdm


def _getXsFVM_fs(xs, ys, n, a, q=1):
    h = (2*a)/n
    l = -a + h/2
    fs = []
    for point in zip(xs, ys):
        idx = int((point[0] - l) // h)
        idy = int((point[1] - l) // h)
        f = np.zeros((n, n))
        f[idx, idy] = q
        fs.append(f)
    return fs

def _getQsFVM_fs(n, a, Qs, locs=[[0, 0]]):
    h = (2*a)/n
    l = -a + h/2
    f = np.zeros((1, n, n))
    for point in locs:
        idx = int((point[0] - l) // h)
        idy = int((point[1] - l) // h)
        f[0, idx, idy] = 1
    
    return Qs * f

def _getJacMatrix(dir, A):
    D = sparse.diags(A.diagonal())
    L = sparse.tril(A, -1)
    U = sparse.triu(A, 1)

    invM = sparse.diags(1/A.diagonal())
    M = L + U
    sparse.save_npz(dir+'_jac_invM', invM.tocoo())
    sparse.save_npz(dir+'_jac_M', M.tocoo())
    return True


def _getMatrix(dir, n):
    Ad = -fd_A_dir(n)
    p = dir + 'fd_AD'
    sparse.save_npz(p, Ad)
    _getJacMatrix(p, Ad)
    del Ad

    An = -fd_A_neu(n)
    p = dir + 'fd_AN'
    sparse.save_npz(p, An)
    _getJacMatrix(p, An)
    del An

    Ad = fv_A_dirichlet(n)
    p = dir + 'fv_AD'
    sparse.save_npz(p, Ad)
    _getJacMatrix(p, Ad)
    del Ad

    An = fv_A_neu(n)
    p = dir + 'fv_AN'
    sparse.save_npz(p, An)
    _getJacMatrix(p, An)
    del An
    return True

def _getFdata(dir, fs, valfs,numerical_method='fd'):
    np.save(f'{dir}{numerical_method}_F.npy', fs)
    np.save(f'{dir}{numerical_method}_ValF.npy', valfs)
    return True

def _getBdata(dir, a, n, fs, valfs, numerical_method='fd'):  
    if numerical_method == 'fd':
        h = 2*a/(n-1)
        B = fs.reshape(-1, n**2) * h**2
        valB = valfs.reshape(-1, n**2) * h**2 
    
    elif numerical_method == 'fv':  
        B = fs.reshape(-1, n**2)
        valB = valfs.reshape(-1, n**2)

    np.save(f'{dir}{numerical_method}_B.npy', B)
    np.save(f'{dir}{numerical_method}_ValB.npy', valB)
    del B, valB
    return True

def _genLocsData(dir, a=1, Q=1, n=129, train_N=1000, val_N=100):
    p = Path(dir)
    if not p.is_dir():
        p.mkdir(exist_ok=False)
    h = 2*a/(n-1)
    x = np.linspace(-a, a, n)
    y = np.linspace(-a, a, n)
    xx, yy = np.meshgrid(x, y)
    train_xs = np.random.uniform(-a+3*h, a-3*h, train_N)
    train_ys = np.random.uniform(-a+3*h, a-3*h, train_N)
    val_xs = np.random.uniform(-a+3*h, a-3*h, val_N)
    val_ys = np.random.uniform(-a+3*h, a-3*h, val_N)

    # FDM
    fd_train_fs = list(normal(xx, yy, h, point) \
        for point in zip(train_xs, train_ys))
    fd_val_fs = list(normal(xx, yy, h, point) \
        for point in zip(val_xs, val_ys))
    fd_train_fs = np.stack(fd_train_fs, axis=0)
    fd_val_fs = np.stack(fd_val_fs, axis=0)

    _getFdata(dir, fd_train_fs, fd_val_fs, 'fd')
    _getBdata(dir, a, n, fd_train_fs, fd_val_fs, 'fd')

    # FVM
    fv_train_fs = _getXsFVM_fs(train_xs, train_ys, n, a, Q)
    fv_val_fs = _getXsFVM_fs(val_xs, val_ys, n, a, Q)
    fv_train_fs = np.stack(fv_train_fs, axis=0)
    fv_val_fs = np.stack(fv_val_fs, axis=0)

    _getFdata(dir, fv_train_fs, fv_val_fs, 'fv')
    _getBdata(dir, a, n, fv_train_fs, fv_val_fs, 'fv')
    return True

def _genQsData(dir, a=1, minQ=1, maxQ=2, n=130, train_N=2500, val_N=10, four=False):
    p = Path(dir)
    if not p.is_dir():
        p.mkdir(exist_ok=False)

    train_Qs = np.random.uniform(minQ, maxQ, (train_N, 1, 1))
    val_Qs = np.linspace(minQ, maxQ, val_N).reshape((val_N, 1, 1))

    # FDM
    h = (2*a)/(n-1)
    x = np.linspace(-a, a, n)
    y = np.linspace(-a, a, n)
    xx, yy = np.meshgrid(x, y)
    
    fd_train_fs = train_Qs * normal(xx, yy, h) \
        if not four else train_Qs * normal_fourth(xx, yy, h, a)
    fd_val_fs = val_Qs * normal(xx, yy, h) \
        if not four else val_Qs * normal_fourth(xx, yy, h, a)
    _getFdata(dir, fd_train_fs, fd_val_fs, 'fd')
    _getBdata(dir, a, n, fd_train_fs, fd_val_fs, 'fd')

    # FVM   
    fv_train_fs = _getQsFVM_fs(n, a, train_Qs, [[0, 0]]) \
        if not four else _getQsFVM_fs(n, a, train_Qs, [[a/2, a/2], [a/2, -a/2], [-a/2, a/2], [-a/2, -a/2]])
    fv_val_fs = _getQsFVM_fs(n, a, val_Qs, [[0, 0]]) \
        if not four else _getQsFVM_fs(n, a, val_Qs, [[a/2, a/2], [a/2, -a/2], [-a/2, a/2], [-a/2, -a/2]])
    _getFdata(dir, fv_train_fs, fv_val_fs, 'fv')
    _getBdata(dir, a, n, fv_train_fs, fv_val_fs, 'fv')
    return True

def _genData(path, n):
    '''
    Generate all types data and matrix.
    '''
    p = Path(f'{path}/{n}/')
    m = int(100/7)
    if not p.is_dir(): p.mkdir()
    with tqdm(total=100, desc=f'n={n}Generating') as pbar:
    # Generate the linear system
        pbar.set_description('Matrix Generating')
        p = Path(f'{path}/{n}/mat/')
        if not p.is_dir(): p.mkdir(exist_ok=False)
        _getMatrix(f'{path}/{n}/mat/', n)
        pbar.update(m)

        pbar.set_description('Type One Generating')
        p = f'{path}/{n}/One/'
        _genQsData(p, 1, 1, 2, n, 2000, 10, False)
        pbar.update(m)

        pbar.set_description('Type Four Generating')
        p = f'{path}/{n}/Four/'
        _genQsData(p, 1, 1, 2, n, 2000, 10, True)
        pbar.update(m)

        pbar.set_description('Type BigOne Generating')
        p = f'{path}/{n}/BigOne/'
        _genQsData(p, 500, 10000, 15000, n, 2500, 50, False)
        pbar.update(m)

        pbar.set_description('Type BigFour Generating')
        p = f'{path}/{n}/BigFour/'
        _genQsData(p, 500, 10000, 15000, n, 2500, 50, True)
        pbar.update(m)

        pbar.set_description('Type Locs Generating')
        p = f'{path}/{n}/Locs/'
        _genLocsData(p, 1, 1, n, 2000, 100)
        pbar.update(m)

        pbar.set_description('Type BigLocs Generating')
        p = f'{path}/{n}/BigLocs/'
        _genLocsData(p, 500, 10000, n, 5000, 500)
        pbar.update(m)
    
    return True


def _genMixData(max_point_source_num=5, min_allow_h=6, sample_points=1000,
        minQ=1, maxQ=2, a=1, n=65, trainN=5000, valN=100, path='../data'):
    '''
    Generate mixed type data for finite difference:
        params:
            max_point_source_num: How many point sources mostly will be located.
            min_allow_h: The minimum times of h distance between any two source and edge.
            a: The length of the square domain.
            n: Resolution of the mesh.
            nums: How many of data.
            path: Save path.
        return: 
            None.
    This function will randomly generate data with the number of point sources under max_point_source_num and more than one.
    '''
    p = Path(f'{path}/{n}/mixed/')
    if not p.is_dir():
        p.mkdir(parents=True)
    
    h = 2 * a / (n - 1)
    x = np.linspace(-a, a, n)
    y = np.linspace(-a, a, n)
    xx, yy = np.meshgrid(x, y)
    gap = min_allow_h * h
    N = trainN + valN
    fd_fs = np.zeros((N, n, n))

    rng = np.random.default_rng(0)
    source_nums = rng.integers(low=1, high=max_point_source_num+1, size=N)
    xs = np.random.uniform(-a+min_allow_h*h, a-min_allow_h*h, sample_points)
    ys = np.random.uniform(-a+min_allow_h*h, a-min_allow_h*h, sample_points)
    Qs = np.random.uniform(minQ, maxQ, sample_points)

    for i, source_num in enumerate(source_nums): 
        coords = np.array([])
        qs = np.random.choice(Qs, size=source_num)
        while coords.shape[0] < source_num:
            new_coord = np.array([np.random.choice(xs, size=1), np.random.choice(ys, size=1)])
            if coords.size > 0:
                distances = np.linalg.norm(coords - new_coord, axis=1)
                if (distances >= gap).all():
                    coords = np.stack([*coords, new_coord], 0)
                    fd_fs[i] += qs[coords.shape[0]-1] * normal(xx, yy, h, new_coord.squeeze())
                else:
                    continue
            else:
                coords = np.array([new_coord])
                fd_fs[i] += qs[coords.shape[0]-1] * normal(xx, yy, h, new_coord.squeeze())
    fd_B = fd_fs.reshape(-1, n**2) * h**2

    fd_trainF = fd_fs[:trainN]
    fd_valF = fd_fs[trainN:]

    fd_trainB = fd_B[:trainN]
    fd_valB = fd_B[trainN:]
    np.save(f'{p}/fd_F', fd_trainF)
    np.save(f'{p}/fd_ValF.npy', fd_valF)
    np.save(f'{p}/fd_B', fd_trainB)
    np.save(f'{p}/fd_ValB.npy', fd_valB)


def gen_test_data(data_path='../data/', save_path='../test_data/'):
    save_path = Path(save_path)
    data_path = Path(data_path)
    
    methods = ['fd', 'fv']
    boundary_types = ['D', 'N']

    for size in data_path.iterdir():
        types = [f for f in size.iterdir() if not 'mat' in f'{f}']
        mat_path = f'{size}/mat/'

        for method in methods:
            for boundary_type in boundary_types:
                A = sparse.load_npz(f'{mat_path}{method}_A{boundary_type}.npz').tocsc()
                lu = sla.splu(A)
                for t in types:
                    p = Path(f'{save_path}/{size.stem}/{t.stem}')
                    if not p.is_dir():  p.mkdir(parents=True)
                    
                    B = np.load(f'{t}/{method}_B.npy')
                    valB = np.load(f'{t}/{method}_ValB.npy')
                    
                    X = np.zeros_like(B)
                    valX = np.zeros_like(valB)
                    for i in range(B.shape[0]):
                        X[i] = lu.solve(B[i])
                    np.save(f'{save_path}/{size.stem}/{t.stem}/{method}_X{boundary_type}.npy', X)                
                    
                    for i in range(valB.shape[0]):
                        valX[i] = lu.solve(valB[i])
                    np.save(f'{save_path}/{size.stem}/{t.stem}/{method}_valX{boundary_type}.npy', X)                

                    F = np.load(f'{t}/{method}_F.npy')
                    valF = np.load(f'{t}/{method}_ValF.npy')
                    np.save(f'{save_path}/{size.stem}/{t.stem}/{method}_F.npy', F)
                    np.save(f'{save_path}/{size.stem}/{t.stem}/{method}_valF.npy', valF)

        
    

    return True

if __name__ == '__main__':
    _genMixData()