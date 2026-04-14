import torch


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