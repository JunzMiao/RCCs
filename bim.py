# -*- coding: UTF-8 -*-

import torch
import numpy as np
from torch import nn

def criterion_adv(loss_name, outputs, labels, target_labels):
    
    if loss_name=="ce":
        loss = nn.CrossEntropyLoss()
        if target_labels:
            criterion = -loss(outputs, target_labels)
        else:
            criterion = loss(outputs, labels)

    return criterion

class BIM:
    """
    Description
    -----------
        [1] Kurakin A, Goodfellow I, Bengio S. Adversarial examples in the physical world[J]. arXiv preprint arXiv:1607.02533, 2016. (https://arxiv.org/abs/1607.02533)

    Example
    -------
        >>> bim_cls = BIM(model, "cuda", np.inf, 4/255, 1/255, 20, "ce")
        >>> adv_images = bim_cls(images, labels)
    """

    def __init__(self, model, device="cuda", norm=np.inf, eps=4/255, step_size=1/255, steps=20, loss="ce"):
        """
        Parameters
        ----------
            model : torch.nn.Module
            
            device : torch.device, default="cuda"
            
            norm : float, default=np.inf, options={1, 2, np.inf}
            
            eps : float, default=4/255
            
            step_size : float, default=1/255
            
            steps : int, default=20
            
            loss : str, default="ce"
        """

        self.model = model
        self.device = device
        self.norm = norm
        self.eps = eps
        self.step_size = step_size
        self.steps = steps
        self.loss = loss
    
    def __call__(self, images=None, labels=None, target_labels=None):
        """
        Parameters
        ----------
            images : torch.Tensor, shape : torch.Size([batch_size, num_channel, height, width]), default=None
            
            labels : torch.Tensor, shape : torch.Size([batch_size]), default=None
            
            target_labels : torch.Tensor, shape : torch.Size([batch_size]), default=None
            
        Return
        ------
            adv_images : torch.Tensor, shape : torch.Size([batch_size, num_channel, height, width])
        """
        
        num_images = images.shape[0]
        images, labels = images.to(self.device), labels.to(self.device)

        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        adv_images = images

        for _ in range(self.steps):
            adv_images = adv_images.clone().detach().requires_grad_(True).to(self.device)
            outputs, *_ = self.model(adv_images)

            loss = criterion_adv(self.loss, outputs, labels, target_labels)

            delta_adv = torch.autograd.grad(loss, [adv_images])[0]

            if self.norm == np.inf:
                delta_adv = delta_adv.sign()
            else:
                norm_val = torch.norm(delta_adv.reshape((num_images, -1)), self.norm, 1)
                delta_adv = delta_adv / norm_val.reshape((num_images, 1, 1, 1))
            
            adv_images = adv_images + self.step_size * delta_adv
            delta_adv = adv_images - images

            if self.norm == np.inf:
                delta_adv = torch.clamp_(delta_adv, -self.eps, self.eps)
            else:
                norm_val = torch.norm(delta_adv.reshape(num_images, -1), self.norm, 1)
                mask = norm_val <= self.eps
                scaling = self.eps / norm_val
                scaling[mask] = 1
                delta_adv = delta_adv * scaling.reshape(num_images, 1, 1, 1)
            adv_images = images + delta_adv
            
            adv_images = torch.clamp_(adv_images, 0, 1)
        
        return adv_images
