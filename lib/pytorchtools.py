<<<<<<< HEAD
import torch
import numpy as np
import matplotlib.pyplot as plt 



def check_gpu(gpu_id):

    torch.cuda.set_device(gpu_id)

    device_count = torch.cuda.device_count() 
    current_device = torch.cuda.current_device() 
    device_name = torch.cuda.get_device_name(current_device)
    device_capability = torch.cuda.get_device_capability(current_device) 
    device_properties = torch.cuda.get_device_properties(current_device)
    is_available = torch.cuda.is_available()

    print('device_count: {device_count}'.format(device_count=device_count))
    print('current_device: {current_device}'.format(current_device=current_device))
    print('device_name: {device_name}'.format(device_name=device_name))
    print('device_capability: {device_capability}'.format(device_capability=device_capability))
    print('device_properties: {device_properties}'.format(device_properties=device_properties))
    print('is_available: {is_available}'.format(is_available=is_available))



def plot_pred(pred_df, target_names, model_file, meta_dir, poses_rng, **kwargs):

    plt.rcParams.update({'font.size': 18+2})
    n = len(target_names)
    fig, axes = plt.subplots(ncols=n, figsize=(7*n, 7), dpi = 500)

    # fig.suptitle(model_file.replace(os.environ['DATAPATH'],'') + '\n' + 
    #              os.path.dirname(meta_file.replace(os.environ['DATAPATH'],'')))
    
    # fig.suptitle(model_file + '\n' + 
    #              meta_dir)

    fig.subplots_adjust(wspace=0.3)
    n_smooth = int(pred_df.shape[0]/20)    

    for i, ax in enumerate(axes): 

        sort_df = pred_df.sort_values(by=[f"target_{i+1}"])
        ax.scatter(sort_df[f"target_{i+1}"], sort_df[f"pred_{i+1}"], s=1, c = 'k', cmap="inferno") ##c=sort_df[f"target_{i+1}"], 
        ax.plot(sort_df[f"target_{i+1}"].rolling(n_smooth).mean(), sort_df[f"pred_{i+1}"].rolling(n_smooth).mean(), c="red")
        # ax.set(xlabel=f"target {target_names[i]}", ylabel=f"predicted {target_names[i]}")
        ax.set(xlabel=r"Target $Y$ (mm)" if i == 0 else r"Target $\theta_{Roll}$ (deg)", ylabel=r"Predicted $Y$ (mm)" if i == 0 else r"Predicted $\theta_{Roll}$ (deg)")
        ind = int(target_names[i][-1])-1
        ax.set_xlim(poses_rng[0][ind], poses_rng[1][ind])
        ax.set_ylim(poses_rng[0][ind], poses_rng[1][ind])
        ax.text(0.05,0.9, 'MAE='+str(sort_df[f"error_{i+1}"].mean())[0:4], transform=ax.transAxes)    
        ax.grid(True)

    return fig



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func


    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}' + '\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        
=======
import torch
import numpy as np
import matplotlib.pyplot as plt 



def check_gpu(gpu_id):

    torch.cuda.set_device(gpu_id)

    device_count = torch.cuda.device_count() 
    current_device = torch.cuda.current_device() 
    device_name = torch.cuda.get_device_name(current_device)
    device_capability = torch.cuda.get_device_capability(current_device) 
    device_properties = torch.cuda.get_device_properties(current_device)
    is_available = torch.cuda.is_available()

    print('device_count: {device_count}'.format(device_count=device_count))
    print('current_device: {current_device}'.format(current_device=current_device))
    print('device_name: {device_name}'.format(device_name=device_name))
    print('device_capability: {device_capability}'.format(device_capability=device_capability))
    print('device_properties: {device_properties}'.format(device_properties=device_properties))
    print('is_available: {is_available}'.format(is_available=is_available))



def plot_pred(pred_df, target_names, model_file, meta_dir, poses_rng, **kwargs):

    plt.rcParams.update({'font.size': 18+2})
    n = len(target_names)
    fig, axes = plt.subplots(ncols=n, figsize=(7*n, 7), dpi = 500)

    # fig.suptitle(model_file.replace(os.environ['DATAPATH'],'') + '\n' + 
    #              os.path.dirname(meta_file.replace(os.environ['DATAPATH'],'')))
    
    # fig.suptitle(model_file + '\n' + 
    #              meta_dir)

    fig.subplots_adjust(wspace=0.3)
    n_smooth = int(pred_df.shape[0]/20)    

    for i, ax in enumerate(axes): 

        sort_df = pred_df.sort_values(by=[f"target_{i+1}"])
        ax.scatter(sort_df[f"target_{i+1}"], sort_df[f"pred_{i+1}"], s=1, c = 'k', cmap="inferno") ##c=sort_df[f"target_{i+1}"], 
        ax.plot(sort_df[f"target_{i+1}"].rolling(n_smooth).mean(), sort_df[f"pred_{i+1}"].rolling(n_smooth).mean(), c="red")
        # ax.set(xlabel=f"target {target_names[i]}", ylabel=f"predicted {target_names[i]}")
        ax.set(xlabel=r"Target $Y$ (mm)" if i == 0 else r"Target $\theta_{Roll}$ (deg)", ylabel=r"Predicted $Y$ (mm)" if i == 0 else r"Predicted $\theta_{Roll}$ (deg)")
        ind = int(target_names[i][-1])-1
        ax.set_xlim(poses_rng[0][ind], poses_rng[1][ind])
        ax.set_ylim(poses_rng[0][ind], poses_rng[1][ind])
        ax.text(0.05,0.9, 'MAE='+str(sort_df[f"error_{i+1}"].mean())[0:4], transform=ax.transAxes)    
        ax.grid(True)

    return fig



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func


    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}' + '\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        
>>>>>>> d0f4c51ddca5ada159842ebc02158d73a574e86f
        