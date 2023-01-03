import numpy as np
import pytorch_lightning as pl
import torch
from pl_modules.datamodule import LADataModule
from pl_modules.LAModel import LAModel
from models import model_names
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def gen_hyper_dict(gridSize, batch_size, net, features, data_type, boundary_type, layers,
            numerical_method='fd', backward_type='jac', lr=1e-3, max_epochs=100, ckpt=False, gpus=1,
            dm=LADataModule, pl_model=LAModel):
    '''
    gridSize: How big mesh. 33, 65, 129
    batch_size: batch size. 8, 16, 24, 32
    input_type: F type or M type.
    net: The architecture.UNet or Attention UNet.
    features: To control the parameters size of network. [16, 32]
    data_type: One or four point source, Big or small area. [bigOne, bigFour, One, Four]
    boundary_type: Dirichlet or mixed with neumann.[D, N]
    backward_type: The loss function used to train the network.[jac, mse, conv, cg, energy, real]
    lr:learning rate
    max_epochs: epochs
    ckpt: True for load parameters from ckpt
    '''
    exp_name = f'{numerical_method}_{backward_type}_{gridSize}_{net}{layers}_{features}_bs{batch_size}_{data_type}{boundary_type}'
    data_path = f'./data/{gridSize}/{data_type}/'
    mat_path = f'./data/{gridSize}/mat/'
    if ckpt:
        exp_name = 'resume_' + exp_name
    layers = list(2**i for i in range(int(np.log2(gridSize)) - 2))
    model = model_names[net](layers=layers, features=features, boundary_type=boundary_type, numerical_method=numerical_method)
    
    dc = {'trainer':{}, 'pl_model':{}, 'pl_dataModule':{}}
    dc['name'] = exp_name

    dc['trainer'] = {'max_epochs': max_epochs, 'precision': 32, 'check_val_every_n_epoch': 1, 'accelerator': 'cuda', 'devices': gpus}
    dc['trainer']['logger'] = TensorBoardLogger('./lightning_logs/', exp_name)
    dc['trainer']['callbacks'] = ModelCheckpoint(monitor= f'val_{backward_type}', mode='min', every_n_train_steps=0,
                                        every_n_epochs=1, train_time_interval=None, save_top_k=3, save_last=True,)
    
    n = gridSize
    a = 500 if 'big' in data_type else 1
    h = 2*a/(n-1) if numerical_method == 'fd' else 2*a/n
    dc['pl_model']['name'] = pl_model
    dc['pl_model']['args'] = [model, a, n, mat_path, lr, numerical_method, backward_type, boundary_type, gridSize//2]
    dc['ckpt'] = ckpt
    
    dc['pl_dataModule']['name'] = dm
    dc['pl_dataModule']['args'] = [data_path, batch_size, a, n, numerical_method]
    
    return dc

def main(kwargs):
    # Initilize the Data Module
    args = kwargs['pl_dataModule']['args']
    dm = kwargs['pl_dataModule']['name'](*args)

    # Initilize the model
    args = kwargs['pl_model']['args']
    pl_model = kwargs['pl_model']['name'](*args)
    
    # Initilize Pytorch lightning trainer
    pl_trainer = pl.Trainer(**kwargs['trainer'])

    if kwargs['ckpt'] is not None:
        ckpt = torch.load(kwargs['ckpt'])
        pl_model.load_state_dict(ckpt['state_dict'])   

    pl_trainer.fit(
        model=pl_model,
        datamodule=dm,)
    return True