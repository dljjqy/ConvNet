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
    Ad = fd_A_dir(n)
    p = dir + 'fd_AD'
    sparse.save_npz(p, Ad)
    _getJacMatrix(p, Ad)
    del Ad

    An = fd_A_neu(n)
    p = dir + 'fd_AN'
    sparse.save_npz(p, An)
    _getJacMatrix(p, An)
    del An
    return True

def _getFdata(dir, fs, valfs):
    np.save(f'{dir}_F.npy', fs)
    np.save(f'{dir}_ValF.npy', valfs)
    return True

def _getBdata(dir, a, n, fs, valfs):  
    h = 2*a/(n-1)
    B = fs.reshape(-1, n**2) * h**2
    valB = valfs.reshape(-1, n**2) * h**2 

    np.save(f'{dir}_B.npy', B)
    np.save(f'{dir}_ValB.npy', valB)
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


def _genMixData(max_point_source_num=10, gap=0.05, k=1,
        minQ=0.5, maxQ=2.5, a=1, n=128, trainN=5000, valN=100, path='../data'):
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
    N = trainN + valN
    fd_fs = np.zeros((N, n, n))

    rng = np.random.default_rng(0)
    source_nums = rng.integers(low=1, high=max_point_source_num+1, size=N)
    
    for i, source_num in enumerate(source_nums): 
        qs = np.random.uniform(minQ, maxQ, source_num)
        points = _gen_points(source_num, gap, a)
        for j, point in enumerate(points):
            fd_fs[i] -= qs[j] * normal(xx, yy, h, point)
        
    fd_B = fd_fs.reshape(-1, n**2) * h**2 / k
    fd_trainF = fd_fs[:trainN]
    fd_valF = fd_fs[trainN:]

    fd_trainB = fd_B[:trainN]
    fd_valB = fd_B[trainN:]
    np.save(f'{p}/fd_F', fd_trainF)
    np.save(f'{p}/fd_ValF.npy', fd_valF)
    np.save(f'{p}/fd_B', fd_trainB)
    np.save(f'{p}/fd_ValB.npy', fd_valB)

def _gen_points(n, min_dist, a=1):
    points = np.empty((0, 2))
    while len(points) < n :
        new_point = np.random.uniform(-1 + min_dist, 1 - min_dist, size=(2,))
        if len(points) == 0:
            points = np.append(points, [new_point], axis=0)
            continue
        dist = np.sqrt(np.sum((points - new_point) ** 2, axis=1))
        if np.min(dist) >= min_dist:
            points = np.append(points, [new_point], axis=0)
    return points
        

def gen_test_data(data_path='../data/', save_path='../test_data/'):
    save_path = Path(save_path)
    data_path = Path(data_path)
    
    boundary_types = ['D', 'N']

    for size in data_path.iterdir():
        types = [f for f in size.iterdir() if not 'mat' in f'{f}']
        mat_path = f'{size}/mat/'
    for boundary_type in boundary_types:
            A = sparse.load_npz(f'{mat_path}_A{boundary_type}.npz').tocsc()
            lu = sla.splu(A)
            for t in types:
                p = Path(f'{save_path}/{size.stem}/{t.stem}')
                if not p.is_dir():  p.mkdir(parents=True)
                
                B = np.load(f'{t}/_B.npy')
                valB = np.load(f'{t}/_ValB.npy')
                
                X = np.zeros_like(B)
                valX = np.zeros_like(valB)
                for i in range(B.shape[0]):
                    X[i] = lu.solve(B[i])
                np.save(f'{save_path}/{size.stem}/{t.stem}/_X{boundary_type}.npy', X)                
                
                for i in range(valB.shape[0]):
                    valX[i] = lu.solve(valB[i])
                np.save(f'{save_path}/{size.stem}/{t.stem}/_valX{boundary_type}.npy', valX)                

                F = np.load(f'{t}/_F.npy')
                valF = np.load(f'{t}/_ValF.npy')
                np.save(f'{save_path}/{size.stem}/{t.stem}/_F.npy', F)
                np.save(f'{save_path}/{size.stem}/{t.stem}/_valF.npy', valF)
    return True

def _solver(data_path, mat_path, n):
    B = np.load(f'{data_path}/fd_B.npy')
    valB = np.load(f'{data_path}/fd_ValB.npy')

    boundary_types = ['D', 'N']
    for boundary_type in boundary_types:
        A = sparse.load_npz(f'{mat_path}/fd_A{boundary_type}.npz').tocsc()
        lu = sla.splu(A)
        
        X = np.zeros((B.shape[0], n, n))
        valX = np.zeros((valB.shape[0], n, n))
        for i in range(B.shape[0]):
            X[i] = lu.solve(B[i]).reshape(n, n)

        np.save(f'{data_path}/fd_X{boundary_type}.npy', X)                
        
        for i in range(valB.shape[0]):
            valX[i] = lu.solve(valB[i]).reshape(n, n)
        np.save(f'{data_path}/fd_ValX{boundary_type}.npy', valX)

    

def genMixData(n):
    _genMixData(max_point_source_num=10, gap=0.1, k=1, minQ=0.5, maxQ=2.5, a=1, 
                n=n, trainN=5000, valN=100, path='../data/')
    mat_path = Path(f'../data/{n}/mat/')
    if not mat_path.is_dir(): 
        mat_path.mkdir(exist_ok=True)
        _getMatrix(f'{mat_path}/', n)  
    _solver(f'../data/{n}/mixed', f'../data/{n}/mat', n)


def gen_rectangles(a, n, block_nums, min_width, max_width, min_height, max_height, dist):
    rectangles = []
    while len(rectangles) < block_nums:
        width = np.random.uniform(min_width, max_width)
        height = np.random.uniform(min_height, max_height)
        x = np.random.uniform(-a + dist, a - width - dist)
        y = np.random.uniform(-a + dist, a - height - dist)
        overlaps = False
        for rect in rectangles:
            if (x < rect[0] + rect[2]) and (rect[0] < x + width) and (y < rect[1] + rect[3]) and (rect[1] < y + height):
                overlaps = True
                break
        if not overlaps:
            rectangles.append((x, y, width, height))

    grid = np.zeros((n, n))
    h = 2 * a / (n - 1)
    for rect in rectangles:
        x_start = int((rect[0] + a) / h)
        x_end = int((rect[0] + rect[2] + a) / h)
        y_start = int((rect[1] + a) / h)
        y_end = int((rect[1] + rect[3] + a) / h)
        grid[y_start:y_end, x_start:x_end] = 1
    return grid

def _gen_block_data(a, n, max_block_num, trainN, valN, k=1, 
                    path='../data/', min_width=0.05, max_width=0.5, min_height=0.05, max_height=0.5, dist=0.1):
    # Check folder exists?
    path = Path(path)/f'{n}'/'block'
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)

    N = trainN + valN
    h = 2 * a / (n - 1)
    h2 = h**2
    # Generate bolck numbers
    rng = np.random.default_rng(0)
    block_nums = rng.integers(low=1, high=max_block_num+1, size=N)
    
    # Generate force matrix and B
    fs = np.zeros((N, n , n))
    for i in range(len(block_nums)):
        fs[i] = gen_rectangles(a, n, block_nums[i], min_width, max_width, min_height, max_height, dist)
    B = fs.reshape(-1, n**2) * h2 / k

    fd_F = fs[:trainN]
    np.save(f'{path/"fd_F.npy"}', fd_F)

    fd_B = B[:trainN]
    np.save(f'{path/"fd_B.npy"}', fd_B)

    fd_ValF = fs[trainN:]
    np.save(f'{path/"fd_ValF.npy"}', fd_ValF)

    fd_ValB = B[trainN:]
    np.save(f'{path/"fd_ValB.npy"}', fd_ValB)
    return 


def gen_block_data(n):
    _gen_block_data(1, n, 10, 5000, 100, path='../data/')
    mat_path = Path(f'../data/{n}/mat/')
    if not mat_path.is_dir(): 
        mat_path.mkdir(exist_ok=True)
        _getMatrix(f'{mat_path}/', n)  
    _solver(f'../data/{n}/block', f'../data/{n}/mat', n) 
        

        
if __name__ == '__main__':
    for n in [64]:
        genMixData(n)
        gen_block_data(n)