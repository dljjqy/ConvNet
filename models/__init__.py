from .unet import UNet, myUNet
from .attunet import AttUNet
from .myunet import varyUNet
model_names = {
    'UNet': UNet, 
    'AttUNet': AttUNet, 
    'myUNet': myUNet,
    'varyUNet':varyUNet,
    }
