import torch
from scipy.optimize import brentq 
from matplotlib import pyplot as plt
plt.style.use("ggplot")

# Function to compute kernels
def rbf(x, y, lengthscale, var=1.):
    if x.ndim == 1:
        x = x[..., None]
    if y.ndim == 1:
        y = y[..., None]
    return var * torch.exp(-(x - y.T)**2 / (2.*lengthscale**2))

def get_matrices(test_x, lscale, obs_noise, K_T, train_x, train_y, var=1.):
    if type(test_x) is float:
        test_x = torch.tensor([test_x])
    if test_x.ndim == 0:
        test_x = test_x[..., None]
    
    X_tilde = test_x - train_x
    if X_tilde.shape < torch.Size([2]):
        X_tilde = X_tilde[..., None]

    # Repeat for each task
    X_tilde = X_tilde.repeat(train_y.shape[1], 1)
    
    K_XX = rbf(train_x, train_x, lscale, var)
    K_xx = torch.kron(K_T, K_XX)

    k_X = rbf(test_x, train_x, lscale, var)
    k_x = torch.kron(K_T, k_X)

    noise_matrix = torch.kron(torch.diag(obs_noise), torch.eye(train_y.shape[0]))
    inverse = torch.inverse(K_xx + noise_matrix)
    
    alpha = torch.matmul(inverse, torch.flatten(train_y.T))
    alpha = alpha.reshape((alpha.shape[0], 1))
    
    return X_tilde, k_x, alpha

def df_dx(test_x, lscale, obs_noise, K_T, train_x, train_y, var=1.):
    X_tilde, k_x, alpha = get_matrices(test_x, lscale, obs_noise, K_T, train_x, train_y, var)

    # Note this is for 1D input so lscale is 1D
    Lambda_X_tilde_T = -(1./lscale)**2 * X_tilde.T

    return torch.matmul(Lambda_X_tilde_T, k_x.T * alpha)

def df_dx_brent(test_x, lscale, obs_noise, K_T, train_x, train_y, task, var=1.):
    return df_dx(test_x, lscale, obs_noise, K_T, train_x, train_y, var)[:, task]

def df2_dx2(test_x, lscale, obs_noise, K_T, train_x, train_y, var=1.):
    X_tilde, k_x, alpha = get_matrices(test_x, lscale, obs_noise, K_T, train_x, train_y, var)

    term1 = -(1./lscale**2) * k_x
    term2 = 1./lscale**4 * X_tilde.T * X_tilde.T * k_x
    return torch.matmul((term1 + term2), alpha)

def find_zeros(f, x, rbf_lscale, obs_noise, K_T, train_x, train_y, var=1.):
    signs = torch.sign(f)
    dx_zeros = []

    for task in range(train_y.shape[1]):
        task_list = []
        for i in range(signs.shape[0]-1):
            if signs[i, task] + signs[i+1, task] == 0:
                if torch.abs(f[i, task] - f[i+1, task]) > 1e-3:
                    root = brentq(df_dx_brent, float(x[i]), float(x[i+1]), args=(rbf_lscale, obs_noise, K_T, train_x, train_y, task, var))
                    task_list.append(root)
        dx_zeros.append(task_list)
    
    return dx_zeros

def get_mean_dx(test_x, rbf_lscale, obs_noise, K_T, train_x, train_y, var=1.):
    with torch.no_grad():
        mean_dx = torch.zeros((test_x.shape[0], train_y.shape[1]))
        i = 0
        for x in test_x:
            mean_dx[i,:] = df_dx(x, rbf_lscale, obs_noise, K_T, train_x, train_y, var)
            i = i+1
    return mean_dx

def get_mean_dx2(dx_zeros, rbf_lscale, obs_noise, K_T, train_x, train_y, var=1.):
    with torch.no_grad():
        n_roots = sum([len(i) for i in dx_zeros])
        mean_dx2 = []
        task = 0
        for task_list in dx_zeros:
            task_derivatives = []
            if task_list != []:
                for zero in task_list:
                    task_derivatives.append(df2_dx2(zero, rbf_lscale, obs_noise, K_T, train_x, train_y, var)[task])
            mean_dx2.append(task_derivatives)
            task = task+1 
    return mean_dx2

def desirable_max(test_x, rbf_lscale, obs_noise, K_T, train_x, train_y, a=torch.tensor(0.), b=torch.tensor(1.), var=1.):
    mean_dx = get_mean_dx(test_x, rbf_lscale, obs_noise, K_T, train_x, train_y, var)
    dx_zeros = find_zeros(mean_dx, test_x, rbf_lscale, obs_noise, K_T, train_x, train_y, var)
    mean_dx2 = get_mean_dx2(dx_zeros, rbf_lscale, obs_noise, K_T, train_x, train_y, var)
    min_second_derivative = -10.
    
    is_there_max = torch.zeros(len(dx_zeros))
    min_dx2 = []

    for task in range(len(dx_zeros)):
        task_dx2 = torch.tensor(mean_dx2[task])

        max_num = torch.sum(task_dx2 < min_second_derivative)
        is_max = False 
        is_last_max = False

        if max_num > 0.:
            max_indices = torch.where(task_dx2 < min_second_derivative)[0]
            # Find the lowest 2nd derivative for the task
            min_dx2.append([torch.min(task_dx2[max_indices])])
            # Note that first_max can be close to b if it's
            # the only max
            first_max = dx_zeros[task][max_indices[0]]
            if max_indices.shape >= torch.Size([2]):
                last_max = dx_zeros[task][max_indices[-1]]
                is_last_max = True
            is_max = True
                
        if is_max:
            # If the first max is not too close to either a or b
            if (torch.abs(first_max - a) > 1e-2 and torch.abs(first_max - b) > 1e-2):
                # Then there is a desirable max
                is_there_max[task] = 1.
            # Or, if there is another max
            elif is_last_max:
                # that is not too close to b
                if torch.abs(last_max - b) > 1e-2:
                    # Then there is a desirable max 
                    is_there_max[task] = 1.
                    
    return is_there_max, mean_dx, mean_dx2, dx_zeros, min_dx2
