from .utils import *
import pytorch_lightning as pl

class LAModel(pl.LightningModule):
    def __init__(self, net, a, n, data_path='./data/', lr=1e-3, numerical_method='fd',
                backward_type='jac', boundary_type='D', cg_max_iter=20):
        '''
            All right side computation:
            dir --> output N-2 x N-2 --> pad G ---> get all loss value.
            mixed --> output N-2 x N --> pad ghost points and G ---> get all conv type loss value.
            mixed --> output N-2 x N --> pad G ---> get linear type loss.
        '''
        super().__init__()
        self.net = net
        self.lr = lr
        self.a = a
        self.n = n
        self.backward_type = backward_type
        self.cg_max_iters = cg_max_iter
        self.boundary_type = boundary_type
        
        if boundary_type == 'D':
            self.padder, self.conver = convRhs(numerical_method, a, n, (1, 1, 1, 1))
        elif boundary_type == 'N':
            self.padder, self.conver = convRhs(numerical_method, a, n, (1, 1, 0, 0), (0, 0, 1, 1))

        A, invM, M = np2torch(data_path, 'jac', boundary_type, numerical_method)
        self.register_buffer('A', A)
        self.register_buffer('invM', invM)
        self.register_buffer('M', M)


    def forward(self, x):
        y = self.net(x)
        return y

    def training_step(self, batch, batch_idx):
        x, b, f = batch
        u = self(x)
        y = torch.flatten(self.padder(u), 1, -1)

        with torch.no_grad():
            jac = self.rhs_jac(y, b)
            cg = self.rhs_cg(y, b, self.cg_max_iters)
            conv = self.conver(u, f)

        loss_values = {
            'mse' : mse_loss(y, self.A, b),
            'jac' : F.l1_loss(y, jac),
            'cg': F.l1_loss(y, cg),
            'energy' : energy(y, self.A, b),
            'conv': F.l1_loss(u, conv),}
        self.log_dict(loss_values)
        return {'loss' : loss_values[self.backward_type]}

    def validation_step(self, batch, batch_idx):
        x, b, f = batch
        u = self(x)
        y = torch.flatten(self.padder(u), 1, -1)

        jac = self.rhs_jac(y, b)
        cg = self.rhs_cg(y, b, self.cg_max_iters)            
        conv = self.conver(u, f)

        loss_values = {
            'val_mse' : mse_loss(y, self.A, b),
            'val_jac' : F.l1_loss(y, jac),
            'val_cg': F.l1_loss(y, cg),
            'val_energy' : energy(y, self.A, b),
            'val_conv': F.l1_loss(u, conv)}
            
        self.log_dict(loss_values)
        return loss_values
    
    def rhs_jac(self, x, b):
        Mx = mmbv(self.M, x)
        x_new = mmbv(self.invM, (b-Mx))
        return x_new

    def rhs_cg(self, x, b, max_iters=20):
        r = b - mmbv(self.A, x)
        p = r
        for _ in range(max_iters):
            rr = bvi(r, r)
            Ap = mmbv(self.A, p)
            alpha = rr / bvi(p, Ap)
            x = x + alpha * p
            r1 = r - alpha * Ap
            beta = bvi(r1, r1) / rr
            p = r1 + beta * p
            r = r1
        return x
                
    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [lr_scheduler]
