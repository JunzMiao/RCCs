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

class FGSM:
    """
    Description
    -----------
        [1] Goodfellow I J, Shlens J, Szegedy C. Explaining and harnessing adversarial examples[J]. arXiv preprint arXiv:1412.6572, 2014. (https://arxiv.org/abs/1412.6572)

    Example
    -------
        >>> fgsm_cls = FGSM(model, "cuda", np.inf, 4/255, "ce")
        >>> adv_images = fgsm_cls(images, labels)
    """

    def __init__(self, model, device="cuda", norm=np.inf, eps=4/255, loss="ce"):
        """
        Parameters
        ----------
            model : torch.nn.Module
            
            device : torch.device, default="cuda"
            
            norm : float, default=np.inf, options={1, 2, np.inf}
            
            eps : float, default=4/255
            
            loss : str, default="ce"
        """

        self.model = model
        self.device = device
        self.norm = norm
        self.eps = eps
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
        
        adv_images = images.clone().detach().requires_grad_(True).to(self.device)
        outputs, *_ = self.model(adv_images)

        loss = criterion_adv(self.loss, outputs, labels, target_labels)

        delta_adv = torch.autograd.grad(loss, [adv_images])[0]

        if self.norm == np.inf:
            delta_adv = delta_adv.sign()
        else:
            norm_val = torch.norm(delta_adv.reshape((num_images, -1)), self.norm, 1)
            delta_adv = delta_adv / norm_val.reshape((num_images, 1, 1, 1))
        
        adv_images = adv_images + self.eps * delta_adv

        adv_images = torch.clamp_(adv_images, 0, 1)
        
        return adv_images
