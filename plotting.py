import torch
from matplotlib import pyplot as plt
plt.style.use("ggplot")
import numpy as np

def plot_gp(train_x, train_y, test_x, f_pred, observed_pred, title=None, new_point_idx=None):
    lower, upper = observed_pred.confidence_region()

    if len(train_y.shape) > 1:
        num_tasks = train_y.shape[1]
        test_x = torch.reshape(test_x, (test_x.shape[0], 1))
        for task in range(num_tasks):
            plt.figure(figsize=(5,3))
            plt.plot(train_x[:,0].numpy(), train_y[:, task].numpy(), 'k*', label='Observations')
            if new_point_idx is not None:
                plt.plot(train_x.numpy()[new_point_idx,0], train_y.numpy()[new_point_idx, task], 'r*', 
                         label='New observation')
            plt.plot(test_x[:,0].numpy(), f_pred.mean[:, task].numpy(), color='blue', label='Mean')    
            plt.fill_between(test_x[:,0], lower[:, task].numpy(), upper[:, task].numpy(), 
                             alpha=0.5, color="C0", label="Confidence")
            plt.fill_between(test_x[:,0], (f_pred.mean[:, task] - 2*torch.sqrt(f_pred.variance[:, task])).numpy(), (f_pred.mean[:, task] + 2*torch.sqrt(f_pred.variance[:, task])).numpy(), alpha=0.5, color="C1", label="GP Confidence")
            if title is not None:
                plt.title(title)
            plt.legend()
            plt.show()

    else:
        plt.figure(figsize=(5,3))
        plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Observations')
        if new_point_idx is not None:
            plt.plot(train_x.numpy()[new_point_idx], train_y.numpy()[new_point_idx], 'r*', label='New observation')
        plt.plot(test_x.numpy(), f_pred.mean.numpy(), color='blue', label='Mean')    
        plt.fill_between(test_x, lower.numpy(), upper.numpy(), alpha=0.5, color="C0", label="Confidence")
        if title is not None:
            plt.title(title)
        plt.show()

def plot_gp_sampled(train_x, train_y, num_orig_train_pts, test_x, f_pred, observed_pred, title=None):

    lower, upper = observed_pred.confidence_region()

    if len(train_y.shape) > 1:
        num_tasks = train_y.shape[1]
        # it expects test_x to be torch.linspace so reshaping to 2D shape
        if len(test_x.shape) < 2:
            test_x = torch.reshape(test_x, (test_x.shape[0], 1))
        for task in range(num_tasks):
            plt.figure(figsize=(5,3))
            plt.plot(train_x[:num_orig_train_pts,0].numpy(), train_y[:num_orig_train_pts, task].numpy(), 'k*', label='Observations')
            
            # If only one pt has been sampled so far, plot it in different colour as current
            num_sampled = train_x.shape[0] - num_orig_train_pts
            if num_sampled == 1:
                plt.plot(train_x[-1,0].numpy(), train_y[-1,task].numpy(), color='magenta', marker='*', label='Current')
            # Otherwise, go through the list of sampled points and plot last one as current
            else:
                for pt in range(num_sampled):
                    if pt == 0:
                        plt.plot(train_x[num_orig_train_pts+pt,0].numpy(), train_y[num_orig_train_pts+pt, task].numpy(), 'r*', label='Sampled')
                    if pt == num_sampled-1:
                        plt.plot(train_x[num_orig_train_pts+pt,0].numpy(), train_y[num_orig_train_pts+pt, task].numpy(), color='magenta', marker='*', label='Current')
                    else:
                        plt.plot(train_x[num_orig_train_pts+pt,0].numpy(), train_y[num_orig_train_pts+pt, task].numpy(),'r*')

            plt.plot(test_x[:,0].numpy(), f_pred.mean[:, task].numpy(), color='blue', label='Mean')    
            plt.fill_between(test_x[:,0], lower[:, task].numpy(), upper[:, task].numpy(), 
                             alpha=0.5, color="C0", label="Confidence")
            plt.fill_between(test_x[:,0], (f_pred.mean[:, task] - 2*torch.sqrt(f_pred.variance[:, task])).numpy(), (f_pred.mean[:, task] + 2*torch.sqrt(f_pred.variance[:, task])).numpy(), alpha=0.5, color="C1", label="GP Confidence")
            if title is not None:
                plt.title(title[task])
            plt.legend()
            plt.show()

def plot_data_sampled(train_x, train_y, num_orig_train_pts, title=None):

    if len(train_y.shape) > 1:
        num_tasks = train_y.shape[1]
        # it expects test_x to be torch.linspace so reshaping to 2D shape
        for task in range(num_tasks):
            plt.figure(figsize=(5,3))
            plt.plot(train_x[:num_orig_train_pts,0].numpy(), train_y[:num_orig_train_pts, task].numpy(), 'k*', label='Observations')
            
            # If only one pt has been sampled so far, plot it in different colour as current
            num_sampled = train_x.shape[0] - num_orig_train_pts
            if num_sampled == 1:
                plt.plot(train_x[-1,0].numpy(), train_y[-1,task].numpy(), color='magenta', marker='*', label='Current')
            # Otherwise, go through the list of sampled points and plot last one as current
            else:
                for pt in range(num_sampled):
                    if pt == 0:
                        plt.plot(train_x[num_orig_train_pts+pt,0].numpy(), train_y[num_orig_train_pts+pt, task].numpy(), 'r*', label='Sampled')
                    if pt == num_sampled-1:
                        plt.plot(train_x[num_orig_train_pts+pt,0].numpy(), train_y[num_orig_train_pts+pt, task].numpy(), color='magenta', marker='*', label='Current')
                    else:
                        plt.plot(train_x[num_orig_train_pts+pt,0].numpy(), train_y[num_orig_train_pts+pt, task].numpy(),'r*')
            if title is not None:
                plt.title(title[task])
            plt.legend()
            plt.show()

def plot_data(train_x, train_y, xlabel, ylabel, label=None, title=None):
    plt.figure(figsize=(10,6))
    if label is not None:
        plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label=label)
        plt.legend()
    else:
        plt.plot(train_x.numpy(), train_y.numpy(), 'k*')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_loss(loss, xlabel, ylabel, label=None, title=None):
    plt.figure(figsize=(10,6))
    if label is None:
        plt.plot(loss)
    else:
        plt.plot(loss, label=label)
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_acq(x, acq_f, x_max, xlabel, ylabel, title=None):
    plt.figure(figsize=(5,3))
    plt.plot(x.numpy(), acq_f.numpy())
    plt.axvline(x_max.item(), linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.show()  
