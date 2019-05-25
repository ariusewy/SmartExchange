from math import ceil
from joblib import Parallel, delayed

import numpy as np
import scipy.io as sio
import torch


def factors(n):
    """Factor a positive integer `n` into the product of primes in ascending order."""
    i = 2
    fct = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            fct.append(i)
    if n > 1:
        fct.append(n)
    return fct

preset = {
    (3072, 3): 192,
    (8364, 3): 204
}

def isprime(n):
    """check if integer n is a prime"""
    # make sure n is a positive integer
    n = abs(int(n))
    # 0 and 1 are not primes
    if n < 2:
        return False
    # 2 is the only even prime number
    if n == 2:
        return True
    # all other even numbers are not primes
    if not n & 1:
        return False
    # range starts with 3 and only needs to go up the squareroot of n
    # for all odd numbers
    for x in range(3, int(n**0.5)+1, 2):
        if n % x == 0:
            return False
    return True


def nearestpow2(v):
    """Find the nearest power of 2 of `v`.

    :v: numpy.Ndarray, supposed to to all positive.
    :returns: numpy.Ndarray, nearest powers of 2 of `v`.

    """
    assert v.size > 0 and np.all(v > 0)
    nextpow2 = np.ceil(np.log2(v))
    lerr = v - 2**(nextpow2-1)
    rerr = 2**nextpow2 - v
    lbetter = (lerr <= rerr).astype(np.float32)
    nearest = (nextpow2 - 1) * lbetter + nextpow2 * (1 - lbetter)
    return nearest


def factor_short(n):
    """Find a proper """
    if n <= 12 or isprime(n):
        return n
    fct = factors(n)
    if fct[0] > 2:
        return fct[0]
    if fct[1] >= 6:
        return fct[1]
    return fct[0] * fct[1]


def factor_long(m, n0):
    m0 = preset.get((m,n0), False)
    if not m0:
        if m <= 32 * n0 or isprime(m):
            return m
        fct = factors(m)
        for i in range(len(fct)):
            m0 = np.prod(fct[i:])
            if m0 <= 32 * n0:
                return m0
    return m0


def core_decompose(A, **opts):
    """Core function for the SmartExchange decomposition.

    Find a decomposition of matrix `A`:
        A \approx Ce * B, such that
    elements in `Ce` are powers of 2 or zero and `B` is a small matrix.

    :A: numpy.Ndarray, supposed to be a tall matrix.
    :**opts: options for the decomposition:
        :decompose_iternum: integer, maximum number of iterations.
        :decompose_threshold: float, threshold used to promote sparsity.
        :decompose_decay: float, decay rate for the threshold when sparsity is too high.
        :decompose_scale: boolean, whether scaling is performed before quantization.
        :decompose_tol: float, stopping tolerance for the algorithm.
        :decompose_rcond: float, for computation stability in least square;
                          refer to `numpy.linalg.lstsq`.
        :threshold_row: boolean, whether threshold a row as a whole for row sparsity.
    :returns: Ce, B, Out

    """
    if len(A.shape) != 2:
        raise ValueError('The input should be a 2-D matrix')
    m, n = A.shape
    if m < n:
        raise ValueError('The input should be a tall matrix')
    A = np.transpose(A)
    m, n = A.shape

    init_method = opts.get('init_method', 'trivial')
    if init_method.lower() is 'ksvd':
        raise NotImplementedError("KSVD initialization not implemented yet.")
    decompose_iternum = opts.get('decompose_iternum', 50)
    decompose_threshold = opts.get('decompose_threshold', 2e-1)
    decompose_decay = opts.get('decompose_decay', 0.1)
    decompose_scale = opts.get('decompose_scale', True)
    decompose_tol = opts.get('decompose_tol', 1e-6)
    decompose_rcond = opts.get('decompose_rcond', 1e-10)
    threshold_row = opts.get('threshold_row', False)

    # if init_method is 'trivial':
    B = np.eye(m)
    Ce = np.copy(A)

    # Initialize the output
    Out = dict()
    Out['B_init'] = np.transpose(B)
    Out['Ce_init'] = np.transpose(Ce)
    Out['B_hist'] = np.zeros((decompose_iternum, m, m))
    Out['Ce_hist'] = np.zeros((decompose_iternum, n, m))
    Out['err_hist'] = np.zeros(decompose_iternum)
    Out['nnz_hist'] = np.zeros(decompose_iternum)
    if decompose_scale:
        Out['dhist'] = np.zeros((m, decompose_iternum))

    for i in range(decompose_iternum-1):
        # quantization
        Ce_sign = np.sign(Ce)
        Ce_abs = np.abs(Ce)
        nz_idx = Ce_abs > 0
        # find scaling matrix D for minimum quantization error
        if decompose_scale is True:
            d = np.ones((m,1))
            for j in range(m):
                c = Ce_abs[j,:]
                cnz = c[c>0]
                if cnz.size == 0:
                    opts['decompose_threshold'] *= decompose_decay
                    return core_decompose(np.transpose(A), **opts)
                cnz_log = np.log2(cnz)
                cnz_round = np.round(cnz_log)
                d[j] = 2 ** np.mean(cnz_round-cnz_log)
            Ce_abs = d * Ce_abs
            Out['dhist'][:,i] = d.reshape(-1)
        Ce_abs[nz_idx] = 2 ** nearestpow2(Ce_abs[nz_idx])
        if np.max(Ce_abs) > 32:
            Ce_abs = Ce_abs / np.max(Ce_abs) * 32.0
        Ce_quant = np.reshape(Ce_abs, Ce.shape) * Ce_sign

        # quit condition
        if i == 0:
            Ce_quant_prev = np.copy(Ce_quant)
        else:
            diff = np.linalg.norm(Ce_quant - Ce_quant_prev, ord='fro')
            if diff <= decompose_tol:
                break
            Ce_quant_prev = np.copy(Ce_quant)

        # least square to update B
        B = np.transpose(np.linalg.lstsq(Ce_quant.T, A.T, rcond=decompose_rcond)[0])

        # update history
        Out['Ce_hist'][i,:,:] = Ce_quant.T
        Out['B_hist'][i,:,:] = B.T
        Out['nnz_hist'][i] = np.count_nonzero(Ce_quant)
        Out['err_hist'][i] = np.linalg.norm(A - np.matmul(B, Ce_quant), 'fro')

        # least square to update C
        Ce = np.linalg.lstsq(B, A, rcond=decompose_rcond)[0]

        # promote sparsity in C
        if threshold_row:
            # NOTE: the `row`s correspond to the columns in Ce here because of
            # the transpose
            Ce[:, np.sum(np.abs(Ce), axis=0) < decompose_threshold * 5] = 0.0
        # elif decompose_threshold_type == 'element':
        Ce[np.abs(Ce) < decompose_threshold] = 0.0

    # quantization
    Ce_sign = np.sign(Ce)
    Ce_abs = np.abs(Ce)
    nz_idx = Ce_abs > 0
    # find scaling matrix D for minimum quantization error
    if decompose_scale is True:
        d = np.ones((m,1))
        for j in range(m):
            c = Ce_abs[j,:]
            cnz = c[c>0]
            if cnz.size == 0:
                opts['decompose_threshold'] *= decompose_decay
                return core_decompose(np.transpose(A), **opts)
            cnz_log = np.log2(cnz)
            cnz_round = np.round(cnz_log)
            d[j] = 2 ** np.mean(cnz_round-cnz_log)
        Ce_abs = Ce_abs * d
        Out['dhist'][:,i] = d.reshape(-1)
    Ce_abs[nz_idx] = 2 ** nearestpow2(Ce_abs[nz_idx])
    if np.max(Ce_abs) > 32:
        Ce_abs = Ce_abs / np.max(Ce_abs) * 32.0
    Ce = np.reshape(Ce_abs, Ce.shape) * Ce_sign

    # least square to update B
    B = np.transpose(np.linalg.lstsq(Ce.T, A.T, rcond=1e-10)[0])

    # update history
    Out['Ce_hist'][-1,:,:] = Ce.T
    Out['B_hist'][-1,:,:] = B.T
    Out['nnz_hist'][-1] = np.count_nonzero(Ce)
    Out['err_hist'][-1] = np.linalg.norm(A - np.matmul(B, Ce), 'fro')

    return np.transpose(Ce), np.transpose(B), Out


def vector_decompose(col, **opts):
    """Perform the SmartExchange decomposition for a (long) vector.
    This can be used for fully connected layers. Refer to our paper.

    `vector_decompose` first reshapes the vector into a matrix of shape (m,3).
    Then it calls `matrix_decompose` to perform the SmartExchange decomposition.
    Finally it reshapes the reconstructed matrix into a vector, and returns it
    along with the lists of `Ce` and `B` matrices.

    NOTE: the selection for the `newsize` is for more balanced partition in
    `matrix_decompose`.

    :col: numpy.Ndarray, the input vector.
    :**opts: options for SmartExchange, will be eventually passed to `core_decompose`.

    """
    if opts.get('verbose', False):
        print(col.shape)
    assert len(col.shape) == 1
    size = col.size
    if size == 512:
        newsize = 513
    if size == 2048:
        newsize = 2052
    elif size == 4096:
        newsize = 4104
    elif size == 512 * 7 * 7: # vgg fc #1
        newsize = 25092
    else:
        newsize = int(ceil(size/3.0) * 3)
    newcol = np.zeros(newsize, dtype=col.dtype)
    newcol[:size] = col
    mat = newcol.reshape(newsize//3, 3)
    if opts.get('verbose', False):
        print(mat.shape)
    matrecon, Ces, Bs = matrix_decompose(mat, **opts)
    colrecon = matrecon.reshape(-1)[:size]
    return colrecon, Ces, Bs


def matrix_decompose(A, **opts):
    """Perform SmartExchange decomposition for a matrix.

    The input matrix `A` can have arbitrary dimensions but we expect them to be
    tall. If not, we will transpose it, and also transpose the reconstruction
    back before returning.

    The matrix `A` will be partitioned into smaller matrices who have much more
    rows than columns.

    :A: numpy.Ndarray, the input matrix to be decomposed.
    :**opts: options for SmartExchange, will be eventually passed to `core_decompose`.
    :returns:
        :Arecon: numpy.Ndarray, the reconstruction of matrix `A`.
        :Ces: [numpy.Ndarray], the list of `Ce` matrices.
        :Bs: [numpy.Ndarray], the list of `B` matrices.

    """
    assert len(A.shape) == 2
    m, n = A.shape

    # transpose weight if weight is a fat and short matrix.
    # `transpose_flag` will be used to transpose the reconstructed matrix
    #   back before returning the results.
    if m <= n:
        A = np.transpose(A)
        m, n = A.shape
        transpose_flag = 1
    else:
        transpose_flag = 0

    # decide how to partition the matrix for decompositions
    n0 = factor_short(n)
    m0 = factor_long(m,n0)

    return_decomps = opts.get('return_decomps', True)

    Arecon = np.zeros(A.shape)
    Ces = [] if return_decomps else None
    Bs = [] if return_decomps else None

    for j in range(n // n0):
        for i in range(m // m0):
            upper = i * m0
            lower = (i+1) * m0
            left = j * n0
            right = (j+1) * n0
            Ce, B, _ = core_decompose(A[upper:lower,left:right], **opts)
            Arecon[upper:lower,left:right] = np.matmul(Ce, B)
            if return_decomps:
                Ces.append(Ce)
                Bs.append(B)

    if transpose_flag:
        Arecon = np.transpose(Arecon)
    return Arecon, Ces, Bs


def parfun_vector_decompose(i, vec, **opts):
    """Handle used for parallelization to accelerate the SmartDecomposition."""
    decomps = dict(type='vector', shape=vec.size)
    vecrecon, Ces, Bs = vector_decompose(vec, **opts)
    decomps['Ces'] = Ces
    decomps['Bs'] = Bs
    return i, vecrecon, decomps


def parfun_matrix_decompose(i, Wp, **opts):
    """Handle used for parallelization to accelerate the SmartDecomposition."""
    decomps = dict(type='matrix', shape=Wp.shape)
    Wprecon, Ces, Bs = matrix_decompose(Wp, **opts)
    decomps['Ces'] = Ces
    decomps['Bs'] = Bs
    return i, Wprecon, decomps


def smart_decompose(W, **opts):
    """Black-box function that takes a weight (CONV or FC) as input, returns its
    reconstruction after SmartExchange decomposition and the list of decomposed
    matrices.

    :W: numpy.Ndarray, the input weight, either CONV or FC.
    :**opts: options for SmartExchange, will be eventually passed to `core_decompose`.
    :returns:
        :Wrecon: numpy.Ndarray, the reconstruction after SmartExchange decomposition.
        :decomps: dict, contains the decomposition information and decomposed matrices.

    """
    threshold_row = opts.pop('threshold_row', False)
    num_workers = opts.pop('num_workers', 8)
    decomps = dict()
    if len(W.shape) == 2:
        decomps['type'] = 'fc'
        decomps['shape'] = tuple(W.shape)
        dout, din = W.shape
        Wrecon = np.zeros_like(W)
        opts['threshold_row'] = False # do not do row thresholding for fc layers
        results = Parallel(n_jobs=num_workers)(delayed(parfun_vector_decompose)(i, W[i,:], **opts) for i in range(dout))
        for i, rowrecon, row_decomps in results:
            decomps['r%d'%(i+1)] = row_decomps
            Wrecon[i,] = rowrecon
    else:
        decomps['type'] = 'conv'
        decomps['shape'] = tuple(W.shape)
        cout, cin, kh, kw = W.shape
        opts['threshold_row'] = threshold_row and (kh == 3)
        if kh != 1 and kh != 3:
            return W, decomps
        Wrecon = np.zeros_like(W)
        if kh == 1:
            W = np.reshape(W, (cout,-1))
            results = Parallel(n_jobs=num_workers)(
                delayed(parfun_vector_decompose)(i, W[i, :], **opts) for i in range(cout))
        else:
            W = np.reshape(W, (cout, cin*kh, kw))
            results = Parallel(n_jobs=num_workers)(
                delayed(parfun_matrix_decompose)(i, W[i,:,:], **opts) for i in range(cout))
        for c, Wprecon, Wp_decomps in results:
            Wrecon[c,:,:,:] = np.reshape(Wprecon, (cin, kh, kw))
            decomps['k%d'%(c+1)] = Wp_decomps
    return Wrecon, decomps


def smart_net(net, **opts):
    """Black-box function to layerwisely perform SmartExchange decomposition for
    a deep neural network.

    :net: pytorch module, the network to be decomposed.
    :**opts: options for SmartExchange, will be eventually passed to `core_decompose`.
    :returns: dict, contains the decomposition information and decomposed matrices.

    """
    i = 0
    decomps = dict()
    for param in net.parameters():
        if len(param.shape) >= 2:
            i = i + 1
            print('decompose layer {}...'.format(i), end=' ')
            if param.is_cuda:
                w = param.detach().cpu().numpy()
            else:
                w = param.detach().numpy()
            wrecon, layer_decomps = smart_decompose(w, **opts)
            wrecon_tensor = torch.FloatTensor(wrecon)
            if param.is_cuda:
                wrecon_tensor = wrecon_tensor.cuda()
            param.data = wrecon_tensor
            print('done')
            decomps['l%d'%i] = layer_decomps
    return decomps


def smart_state_dict(state, **opts):
    """Black-box function to layerwisely perform SmartExchange decomposition for
    a deep neural network in the state dict form.

    NOTE: modify a state_dict of a pytorch module will not directly modify the
    weights in the network.

    :state: state_dict of a pytorch module, the network to be decomposed.
    :**opts: options for SmartExchange, will be eventually passed to `core_decompose`.
    :returns: dict, contains the decomposition information and decomposed matrices.

    """
    i = 0
    decomps = dict()
    for k,v in state.items():
        if ('weight' in k) and (len(v.shape) >= 2):
        # if ('weight' in k) and ('classifier' in k) and (len(v.shape) >= 2):
        # if len(param.shape) >= 2:
            i = i + 1
            print('decompose layer {}...'.format(i), end=' ')
            if v.data.is_cuda:
                w = v.data.detach().cpu().numpy()
            else:
                w = v.data.detach().numpy()
            wrecon, layer_decomps = smart_decompose(w, **opts)
            wrecon_tensor = torch.FloatTensor(wrecon)
            if v.data.is_cuda:
                wrecon_tensor = wrecon_tensor.cuda()
            v.data = wrecon_tensor
            print('done')
            decomps['l%d'%i] = layer_decomps
    return decomps


if __name__ == "__main__":
    mat = sio.loadmat('./pytorch-vgg-cifar10/vgg19_cifar10.mat')
    w1 = mat['w1'].astype(np.float32)
    opts = dict(decompose_iternum=50,
                decompose_threshold=1e-2,
                decompose_decay=0.1,
                decompose_scale=True,
                decompose_tol=1e-6,
                decompose_rcond=1e-10,
                save_Ce=True)
    wrecon1, Out1 = smart_decompose(w1, **opts)
    print(np.count_nonzero(Out1['Ce']))
