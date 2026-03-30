import math
import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn
import torch


class MLorc_AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, correct_bias=True, rank=4, p=0, torchapi=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        self.rank=rank
        self.p=p
        self.torchapi=torchapi
        super().__init__(params, defaults)



    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.grad = None

                if grad.dim() != 2:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                    
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["m_u"] = torch.zeros((p.data.shape[0], self.rank), dtype=p.data.dtype, device=p.data.device)
                    state["m_v"] = torch.zeros((self.rank, p.data.shape[1]), dtype=p.data.dtype, device=p.data.device)
                    state["m_s"] = torch.zeros((self.rank), dtype=p.data.dtype, device=p.data.device)
                    # Exponential moving average of squared gradient values
                    state["sq_u"] = torch.zeros((p.data.shape[0], self.rank), dtype=p.data.dtype, device=p.data.device)
                    state["sq_v"] = torch.zeros((self.rank, p.data.shape[1]), dtype=p.data.dtype, device=p.data.device)
                    state["sq_s"] = torch.zeros((self.rank), dtype=p.data.dtype, device=p.data.device)

                m_u, m_s, m_v, sq_u, sq_s, sq_v= state["m_u"], state["m_s"], state["m_v"], state["sq_u"], state["sq_s"], state["sq_v"]

                beta1, beta2 = group["betas"]

                state["step"] += 1

                m=beta1 * m_u @ torch.diag(m_s) @ m_v + (1-beta1) * grad

                sq=sq_u  @ torch.diag(sq_s) @ sq_v

                if (sq < 0).any():
                    neg_mask = sq < 0
                    neg_abs_mean = sq[neg_mask].abs().mean()
                    sq= torch.relu(sq)+ neg_mask.to(dtype=p.data.dtype) * neg_abs_mean


                sq=beta2 * sq + (1-beta2) * grad * grad

                m_u, m_s, m_v = randomized_svd(m, self.rank, p=self.p, torchapi=self.torchapi)
                sq_u, sq_s, sq_v = randomized_svd(sq, self.rank, p=self.p, torchapi=self.torchapi)

                state["m_u"] = m_u
                state["m_s"] = m_s
                state["m_v"] = m_v
                state["sq_u"] = sq_u
                state["sq_s"] = sq_s
                state["sq_v"] = sq_v


                denom = sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, m, denom)


                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss
        
class MLorc_Lion(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.95, 0.98), weight_decay=0.05, rank=4, p=0, torchapi=False):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        self.rank=rank
        super().__init__(params, defaults)



    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.dim() != 2:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                    
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values

                    state["m_u"] = torch.zeros((p.data.shape[0], self.rank), dtype=p.data.dtype, device=p.data.device)
                    state["m_v"] = torch.zeros((self.rank, p.data.shape[1]), dtype=p.data.dtype, device=p.data.device)
                    state["m_s"] = torch.zeros((self.rank), dtype=p.data.dtype, device=p.data.device)

                m_u, m_v, m_s= state["m_u"], state["m_v"], state["m_s"]
                beta1, beta2 = group["betas"]

                m=m_u @ torch.diag(m_s) @ m_v
                update=(beta1 * m + (1-beta1) * grad).sign_()

                state["step"] += 1
                step_size = group["lr"]
                p.data.add_(update, alpha=-step_size)

                m_=beta2 * m + (1-beta2) * grad
                m_u, m_s, m_v = randomized_svd(m_, self.rank)
                state["m_u"], state["m_v"], state["m_s"] = m_u, m_v, m_s
                
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


def project(full_rank_grad, proj_type, projector):
    if proj_type=="left":
        low_rank_grad = projector.T @ full_rank_grad
    elif proj_type=="right": 
        low_rank_grad =  full_rank_grad @ projector.T
    return low_rank_grad

def project_back(low_rank_grad, proj_type, projector):
    if proj_type=="left":
        full_rank_grad = projector @ low_rank_grad
    elif proj_type=="right": 
        full_rank_grad = low_rank_grad @ projector
    return full_rank_grad
    
class GaLore(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, correct_bias=True, rank=4, T=300):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        self.rank=rank
        self.T=T
        super().__init__(params, defaults)


    
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.dim() != 2:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")


                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    
                if "projector" not in state:
                    if grad.shape[0] >= grad.shape[1]:
                        state["proj_type"] = "right"
                        state["projector"] = torch.zeros((self.rank, p.data.shape[1]), dtype=p.data.dtype, device=p.data.device)
                    else:
                        state["proj_type"] = "left"
                        state["projector"] = torch.zeros((p.data.shape[0], self.rank), dtype=p.data.dtype, device=p.data.device)
                        
                if(state["step"]%self.T==0):
                    u, s, v=torch.linalg.svd(grad.float(), full_matrices=False)
                    if state["proj_type"] == "left":
                        state["projector"] = u[:, :self.rank].to(p.data.dtype)
                    elif state["proj_type"] == "right":
                        state["projector"] = v[:self.rank, :].to(p.data.dtype)

                grad = project(grad, state["proj_type"], state["projector"])
            
                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                norm_grad = torch.div(exp_avg, denom)
                norm_grad = project_back(norm_grad, state["proj_type"], state["projector"])

                p.data.add_(norm_grad, alpha=-step_size)


                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


def randomized_svd(A, rank, p=0, torchapi=False):
    
    m, n = A.shape
    device = A.device
    datatype = A.dtype
    
    if torchapi== False:
        random_matrix = torch.randn(size=(n, rank+p), device=device)
        Y = A @ random_matrix.to(datatype)
        Q, _ = torch.linalg.qr(Y.float())
        Q = Q.to(datatype)
        B = Q.T @ A
        U_hat, S, V = torch.linalg.svd(B.float(), full_matrices=False)

        U = (Q @ U_hat.to(datatype))[:, :rank]
        S = (S.to(datatype))[:rank]
        V = (V.to(datatype))[:rank, :]
        
    elif torchapi== True:
        U, S, V = torch.pca_lowrank(A.float(), q=rank+p, center=False, niter=1)
        U = (U.to(datatype))[:, :rank]
        S = (S.to(datatype))[:rank]
        V = (V.to(datatype).T)[:rank, :]
        
    return U, S, V

class low_rank_projector:
    def __init__(self, rank, proj_type='std'):
        self.rank = rank
        self.proj_type = proj_type
        self.ortho_matrix = None

    def project(self, full_rank_grad):
        if self.proj_type == 'right':
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)

        return full_rank_grad


    # svd decomposition
    def get_orthogonal_matrix_svd(self, grad, svd_lowrank=False):
        matrix = grad.data.clone()

        if matrix.dtype != torch.float: #torch.linalg.svd doesn't support half precision types such as torch.bfloat16
            float_data = False
            original_type = matrix.dtype
            original_device = matrix.device
            matrix = matrix.float()
        else :
            float_data = True

        if svd_lowrank :
            U, s, V = torch.svd_lowrank(matrix, q = self.rank+2, niter=1) #q a slightly overestimated rank of A
            Vh = V.t()
        else :
            U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)

        if self.proj_type =='right' :
            ortho_matrix = Vh[:self.rank, :]
        elif self.proj_type=='left' :
            ortho_matrix = U[:, :self.rank]

        if not(float_data) :
            ortho_matrix = ortho_matrix.to(device=original_device, dtype=original_type)

        self.ortho_matrix = ortho_matrix


    def power_iteration(self, matrix, init, intermediate_orthogonalization=False):
        if self.proj_type == 'right':
            U = matrix @ init.t()
            if intermediate_orthogonalization : # Not necessary for computing right singular vectors
                U = Gram_Schmidt(U)
            projection_map = matrix.t() @ U
            del U

            projection_map = Gram_Schmidt(projection_map)
            self.ortho_matrix = projection_map.t()

        elif self.proj_type == 'left':
            V = matrix.t() @ init
            if intermediate_orthogonalization : # Not necessary for computing left singular vectors
                V = Gram_Schmidt(V)
            projection_map = matrix @ V
            del V

            projection_map = Gram_Schmidt(projection_map)
            self.ortho_matrix = projection_map


def Gram_Schmidt(matrix):
    original_type = matrix.dtype #torch.linalg.qr doesn't support helf precision types such as torch.bfloat16
    matrix, _ = torch.linalg.qr(matrix.to(dtype=torch.float32))
    matrix = matrix.to(dtype=original_type)

    return matrix

class LDAdamW(torch.optim.Optimizer):
    def __init__(self,
        params,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.908,0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rank: int = 16,
        rho: float = 0.908,
        proj_type: str = 'std',
        proj_method: str = 'power_iteration',
        error_feedback: bool = True,
    ):

        #Sanity check
        if not isinstance(lr, (int, float)) or lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not isinstance(betas, tuple) or len(betas) != 2 or not all(isinstance(beta, (int, float)) for beta in betas) or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("Invalid betas: {}".format(betas))
        if not isinstance(eps, (int, float)) or eps <= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0.0:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError("Invalid rank value: {}".format(rank))
        if not isinstance(rho, (int, float)) or not(0.0 <= rho < 1.0):
            raise ValueError("Invalid rho value: {}".format(rho))
        if proj_type not in ['std', 'left', 'right', 'reverse_std']:
            raise ValueError("Invalid projection type: {}".format(proj_type))
        if not isinstance(proj_method, str) or proj_method not in ['svd', 'svd_lowrank', 'power_iteration']:
            raise ValueError("Invalid projection method: {}".format(proj_method))
        if not isinstance(error_feedback, bool):
            raise ValueError("Invalid error feedback value: {}".format(error_feedback))

        #Construct optimizer
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super(LDAdamW, self).__init__(params, defaults)

        #Default hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.error_feedback = error_feedback

        #Initialize optimizer states
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue

                st = self.state[p]

                #AdamW hyperparameters
                st['lr'] = group.get('lr', lr)
                st['beta1'] = group.get('beta1', betas[0])
                st['beta2'] = group.get('beta2', betas[1])
                st['weight_decay'] =  group.get('weight_decay', weight_decay)
                st['eps'] = group.get('eps', eps)

                if not group['enable_lowrank'] :
                    st['m'] = torch.zeros_like(p)
                    st['v'] = torch.zeros_like(p)
                    continue

                #LDAdamW hyperparameters
                st['rho'] = group.get('rho', rho)
                st['rank'] = group.get('rank', rank)

                layer_shape = p.shape
                proj_type = group.get('proj_type', proj_type)
                st['left_proj'], st['right_proj'] = False, False # Setting an optimizer state to a string value is not standard practice and is generally not recommended
                if proj_type =='right' or (layer_shape[0]>layer_shape[1] and proj_type =='std') or (layer_shape[0]<layer_shape[1] and proj_type =='reverse_std'):
                    st['right_proj'] = True
                    st['previous_projector']  = low_rank_projector(rank=st["rank"], proj_type='right')
                    lowdim_shape = (layer_shape[0], st['rank'])
                    if st['rank'] > layer_shape[1] :
                        raise ValueError("For right projection, rank cannot be greater than the number of columns in the weight matrix")
                elif proj_type=='left' or (layer_shape[0]<=layer_shape[1] and proj_type =='std') or (layer_shape[0]>=layer_shape[1] and proj_type =='reverse_std'):
                    st['left_proj'] = True
                    st['previous_projector']  = low_rank_projector(rank=st["rank"], proj_type='left')
                    lowdim_shape = (st['rank'], layer_shape[1])
                    if st['rank'] > layer_shape[0] :
                        raise ValueError("For left projection, rank cannot be greater than the number of rows in the weight matrix")

                st['m'] = torch.zeros(lowdim_shape, device=p.device, dtype=p.dtype)
                st['v'] = torch.zeros(lowdim_shape, device=p.device, dtype=p.dtype)

                proj_method = group.get('proj_method', proj_method)
                st['use_svd'], st['use_svd_lowrank'], st['use_poweriteration'] = False, False, False # Setting an optimizer state to a string value is not standard practice and is generally not recommended
                if proj_method == 'svd' or proj_method  == 'svd_lowrank' : st['use_svd'] = True
                if proj_method  == 'svd_lowrank': st['use_svd_lowrank'] = True
                if proj_method == 'power_iteration': st['use_poweriteration'] = True

                st['error_feedback'] = group.get('error_feedback', error_feedback)

        self.completed_steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        self._update_lr_wd()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group['enable_lowrank']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.ldadamw_step(p)
            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.adamw_step(p)

        self.completed_steps += 1

        return loss



    @torch.no_grad()
    def adamw_step(self, p):
        completed_steps = self.completed_steps

        st = self.state[p]
        grad = p.grad

        #AdamW hyperparameters
        lr = st['lr']
        wd = st['weight_decay']
        beta1 = st['beta1']
        beta2 = st['beta2']
        eps = st['eps']

        #Adaptive optimization step
        st['m'].mul_(beta1)
        st['m'].add_(grad, alpha=(1 - beta1))

        st['v'].mul_(beta2)
        st['v'].addcmul_(grad, grad, value=(1 - beta2))

        ### MODEL UPDATE
        descent_direction = st['v'].div(1 - beta2**(completed_steps+1))
        descent_direction.sqrt_()
        descent_direction.add_(eps)

        descent_direction.reciprocal_()
        descent_direction.mul_(st['m'])
        descent_direction.div_((1 - beta1**(completed_steps+1)))

        p.mul_(1 - lr * wd) #decoupled weight decay
        p.add_(descent_direction, alpha=-lr)

        del descent_direction



    @torch.no_grad()
    def ldadamw_step(self, p):
        completed_steps = self.completed_steps

        st = self.state[p]
        grad = p.grad

        #AdamW hyperparameters
        lr = st['lr']
        wd = st['weight_decay']
        beta1 = st['beta1']
        beta2 = st['beta2']
        eps = st['eps']

        #LDAdamW hyperparameters
        rho = st['rho']
        rank = st['rank']
        left_proj = st['left_proj']
        right_proj = st['right_proj']
        use_svd = st['use_svd']
        use_svd_lowrank = st['use_svd_lowrank']
        use_poweriteration = st['use_poweriteration']

        ### LEARNING SUBSPACE ADAPTATION
        if left_proj : projector = low_rank_projector(rank=rank, proj_type='left')
        elif right_proj : projector = low_rank_projector(rank=rank, proj_type='right')

        previous_projector = st['previous_projector']

        if completed_steps==0 :
            projector.get_orthogonal_matrix_svd(grad, svd_lowrank=use_svd_lowrank) #init power iteration process with SVD
            previous_projector.ortho_matrix = torch.zeros_like(projector.ortho_matrix)
        else :
            b = previous_projector.project_back(st['m'])
            b.div_(1-beta1**completed_steps)
            b.mul_(rho)
            b.add_(grad, alpha=(1 - rho))
            if use_svd :
                projector.get_orthogonal_matrix_svd(b, svd_lowrank=use_svd_lowrank)
            elif use_poweriteration :
                projector.power_iteration(b, init=previous_projector.ortho_matrix)

        lowdim_grad = projector.project(grad)

        ### ERROR BUFFER LOADING - gradient compression
        if st['error_feedback'] :
            lowrank_grad = projector.project_back(lowdim_grad)
            grad.sub_(lowrank_grad) #store error in grad tensor
            del lowrank_grad

        ### OPTIMIZER STATES PROJECTION-AWARE UPDATE - gradient first-order statistic
        if left_proj :
            mat_change_of_subspace = projector.ortho_matrix.t() @ previous_projector.ortho_matrix
            lowdim_updated_momentum = mat_change_of_subspace @ st['m']

        elif right_proj :
            mat_change_of_subspace = previous_projector.ortho_matrix @ projector.ortho_matrix.t()
            lowdim_updated_momentum = st['m'] @ mat_change_of_subspace

        ### GENERALIZED ERROR BUFFER LOADING - optimizer states compression
        if st['error_feedback']:
            lowrank_previous_momentum = previous_projector.project_back(st['m'])
            grad.add_(lowrank_previous_momentum, alpha=(beta1 / (1 - beta1))) #store error in grad tensor
            del lowrank_previous_momentum

            grad.sub_(projector.project_back(lowdim_updated_momentum), alpha=(beta1 / (1-beta1)))  #store error in grad tensor

        del previous_projector

        ### OPTIMIZER STATES PROJECTION-AWARE UPDATE - gradient second-order statistic
        if completed_steps > 0:
            #Optimizer states projection-aware update - gradient second-order statistic
            bias1_correction = 1 - beta1**completed_steps
            bias2_correction = 1 - beta2**completed_steps

            mat_change_of_subspace.mul_(mat_change_of_subspace)

            st['v'].mul_(1/bias2_correction)
            st['v'].addcmul_(st['m'], st['m'], value=-1/(bias1_correction**2))

            if left_proj :
                st['v'] = torch.matmul(mat_change_of_subspace, st['v'])
            elif right_proj :
                st['v'] = torch.matmul(st['v'], mat_change_of_subspace)
            del mat_change_of_subspace

            st['v'].addcmul_(lowdim_updated_momentum, lowdim_updated_momentum, value=1/(bias1_correction**2))
            st['v'].mul_(bias2_correction)
            st['v'].abs_()

            st['m'].copy_(lowdim_updated_momentum)
            del lowdim_updated_momentum

        ### OPTIMIZER STATES ADAM-TYPE UPDATE
        st['m'].mul_(beta1)
        st['m'].add_(lowdim_grad, alpha=(1 - beta1))

        st['v'].mul_(beta2)
        st['v'].addcmul_(lowdim_grad, lowdim_grad, value=(1 - beta2))

        ### MODEL UPDATE
        lowdim_descent_direction = st['v'].div(1 - beta2**(completed_steps+1))
        lowdim_descent_direction.sqrt_()
        lowdim_descent_direction.add_(eps)

        lowdim_descent_direction.reciprocal_()
        lowdim_descent_direction.mul_(st['m'])
        lowdim_descent_direction.div_((1 - beta1**(completed_steps+1)))

        descent_direction = projector.project_back(lowdim_descent_direction)
        del lowdim_descent_direction

        st['previous_projector'] = projector

        p.mul_(1 - lr * wd) #decoupled weight decay
        p.add_(descent_direction, alpha=-lr)

        del descent_direction



    def _update_lr_wd(self):
        # copy the learning rate group to parameter state because the lr scheduler updates the one in the group
        for group in self.param_groups:
            lr = group.get('lr', self.lr) # if the param groups do not have learning rate, then use the external one
            wd = group.get('weight_decay', self.weight_decay)  # if the param groups do not have weight decay, then use the external one
            for p in group['params']:
                self.state[p]['lr'] = lr
                self.state[p]['weight_decay'] = wd



    ### GRADIENT ACCUMULATION AND ERROR BUFFER LOADING
    def zero_grad(self):
        for group in self.param_groups:
            error_feedback = group.get('error_feedback', self.error_feedback)
            if not(group['enable_lowrank']) or not(error_feedback):
                for p in group['params']:
                    p.grad = None