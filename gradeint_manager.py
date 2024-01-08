import torch
import numpy as np
import torchvision.models as models


from collections import OrderedDict


class GradientManager:
    '''
    Class to capture the gradients of the layers specified in the layers list
    Just pass the model and the layers you want to capture the gradients of
    After each backward pass, the gradients will be stored in the grad attribute
    '''
    def __init__(self, layers_or_model, detach=False, max_len = 20):
        if isinstance(layers_or_model, list):
            self.layers = layers_or_model 
        else:
            all_layers_to_track = self.get_layer_names(layers_or_model)
            self.layers = all_layers_to_track

        self.grad = {}
        self.detach = detach
        self.handles = []
        self.hooks_enabled = False  # To keep track of the state of hooks
        # for layer_name in self.layers:
        #     self.grad[layer_name] = None

        self.list_of_grads = []
        self.max_len = max_len
        self.n_samples = 0

        



    def register_hooks(self, model):
        if not self.hooks_enabled:
            for layer_name in self.layers:
                layer = dict(model.named_modules())[layer_name]
                handle = layer.weight.register_hook(self.capture_grad(layer_name + ".weight"))
                self.handles.append(handle)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    # print('Yes')
                    handle = layer.bias.register_hook(self.capture_grad(layer_name + ".bias"))
                    self.handles.append(handle)
            self.hooks_enabled = True
        
        self.running_sum = OrderedDict()
        self.running_sum_squared = OrderedDict()
        self.grad_mean = OrderedDict()
        self.grad_std = OrderedDict()
        for key, value in model.state_dict().items():
            self.running_sum[key] = torch.zeros_like(value).detach().cpu()
            self.running_sum_squared[key] = torch.zeros_like(value).detach().cpu()
            self.grad_mean[key] = torch.zeros_like(value).detach().cpu()
            self.grad_std[key] = torch.zeros_like(value).detach().cpu()

            self.running_sum[key].requires_grad = False
            self.running_sum_squared[key] = False
            self.grad_mean[key] = False
            self.grad_std[key] = False

    def capture_grad(self, layer_name):
        def hook(grad):
            if self.hooks_enabled:
                if self.detach:
                    captured_grad = grad.detach().cpu().numpy()
                else:
                    captured_grad = grad
                self.grad.update({layer_name: captured_grad})
        return hook

    def enable_hooks(self):
        self.hooks_enabled = True

    def disable_hooks(self):
        self.hooks_enabled = False

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.hooks_enabled = False

    def get_flattened_gradients(self):
        flattened_grads = []
        for layer in self.layers:
            layer_grad = self.grad[layer]
            if isinstance(layer_grad, torch.Tensor):
                layer_grad = layer_grad.cpu()
                layer_flattened_grad = torch.flatten(layer_grad)
            else:
                layer_flattened_grad = layer_grad.flatten()
            flattened_grads.append(layer_flattened_grad)
        if isinstance(flattened_grads[0], torch.Tensor):
            flattened_grads = torch.cat(flattened_grads)
        else:
            flattened_grads = np.concatenate(flattened_grads)
        return flattened_grads
    
    def get_dict_gradients(self):

        dict_grads = {}
        for layer in self.layers:
            layer_grad = self.grad[layer]
            if isinstance(layer_grad, torch.Tensor):
                layer_grad = layer_grad.cpu()
                layer_flattened_grad = torch.flatten(layer_grad)
            else:
                layer_flattened_grad = layer_grad.flatten()
            # flattened_grads.append(layer_flattened_grad)
            dict_grads[layer] =  layer_grad
    
        return dict_grads
        

    def get_layer_names(self, model):
        layer_names = []
        for name, module in model.named_modules():
            if hasattr(module, "weight") or hasattr(module, "bias"):
            # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # if True:
                layer_names.append(name)
        return layer_names
    
    def store_gradients(self):
        # g = self.get_flattened_gradients()
        g = self.get_dict_gradients()
        if len(self.list_of_grads) == self.max_len:
            self.list_of_grads.pop(0)
        self.list_of_grads.append(g)
    
    def update_stats(self):
        self.n_samples += 1
        for grad_name, grad_value in self.grad.items():
            self.running_sum[grad_name] = grad_value.detach().cpu()
            self.running_sum_squared[grad_name] = (grad_value ** 2).detach().cpu()

    def calculate_mean_std(self):
        # print('Calculating mean and std')
        print(f'There are {len(self.running_sum_squared.keys())} keys!')
        for key in self.running_sum.keys():
            self.grad_mean[key] = (self.running_sum[key] / self.n_samples).detach().cpu()
            self.grad_std[key]  = (self.running_sum_squared[key] / self.n_samples - self.grad_mean[key] ** 2).detach().cpu()
        # print('deleting running sum and sum_squared')
        del self.running_sum, self.running_sum_squared
        self.running_sum, self.running_sum_squared = None, None

