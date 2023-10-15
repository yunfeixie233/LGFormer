import torch
import numpy as np
import h5py

def to_h5(**kwargs):
    """
    Save input tensors or numpy arrays to an H5 file.
    
    Usage:
    >>> a = torch.tensor([1,2,3])
    >>> b = np.array([4,5,6])
    >>> save_tensors_to_h5(a=a, b=b, filename="output.h5")
    
    Arguments:
    **kwargs : Tensors or numpy arrays to save.
    filename : Name                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             of the H5 file to save to.
    
    Returns:
    None
    """
    filename = kwargs.pop('filename', 'default_output.h5')
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as f:
        for key, value in kwargs.items():
            # If it's a tensor on GPU, detach and move it to CPU.
            if torch.is_tensor(value) and value.device.type == 'cuda':
                value = value.detach().cpu().numpy()
            # If it's a tensor (not on GPU), just detach and convert.
            elif torch.is_tensor(value):
                value = value.detach().numpy()
            # Else, it's assumed to be a numpy array.
            f.create_dataset(key, data=value)

