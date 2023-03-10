from utils.runer import *
from itertools import product
from pathlib import Path

hyper_parameters_dict = {
"grid_sizes" : [128],
"batch_sizes" : [8],
"net" : ['varyUNet'],
"features" : [16],
"data_type": ['block'],
"boundary_type":['D', 'N'],
"backward_type": ['cg'],
"lr":[5e-4], 
"max_epochs":[150],
"ckpt": [None],
# "ckpt": ['./lightning_logs/fv_cg_65_UNet_32_bs32_OneD/version_1/checkpoints/last.ckpt']
}

log_dir = './lightning_logs/'

for parameter in product(*hyper_parameters_dict.values()):
    case = gen_hyper_dict(*parameter, gpus=1)
    path = Path(f"{log_dir}{case['name']}")
    if not path.exists():
        print(f"\nExperiment Name: {case['name']}\n")
        main(case)
    else:
        print(f"\nExperiment Name: {case['name']} has already done!\n")