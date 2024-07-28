

"""
Abstract base class for all dynamics models.
"""

from __future__ import annotations

import abc
import os
import torch
from torch import nn as nn
from typing import Any, Literal

from .base_layers import CNN, MLP
from .model_base_cfg import BaseModelCfg
from .resnet import PerceptNet, ResNet, ResNetFPN
from .s4_rnn import S4RNN


class Model(nn.Module, abc.ABC):
    """Base abstract class for all dynamics models.

    All classes derived from `Model` must implement the following methods:

        - ``forward``: computes the model output.
        - ``loss``: computes a loss tensor that can be used for backpropagation.
        - ``eval_score``: computes a non-reduced tensor that gives an evaluation score
          for the model on the input data (e.g., squared error per element).
        - ``save``: saves the model to a given path.
        - ``load``: loads the model from a given path.

    Subclasses may also want to overrides :meth:`sample` and :meth:`reset`.

    Args:
        device (str or torch.device): device to use for the model. Note that the
            model is not actually sent to the device. Subclasses must take care
            of this.
    """

    _MODEL_FNAME = "model.pth"

    def __init__(
        self,
        cfg: BaseModelCfg,
        device,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

    @property
    def number_of_parameters(self) -> int:
        """Returns the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @property
    def layer_parameters(self) -> dict[str, int]:
        """
        Count the number of parameters in each layer.

        Returns:
            dict: A dictionary containing layer names as keys and the corresponding number of parameters as values.
        """
        params_count = {}
        for name, param in self.named_parameters():
            layer_name = name.split(".")[0]
            if layer_name not in params_count:
                params_count[layer_name] = 0
            params_count[layer_name] += param.numel()
        return params_count

    def forward(self, x: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, ...]:
        """Computes the output of the dynamics model.

        Args:
            x (tensor): the input to the model.

        Returns:
            (tuple of tensors): all tensors predicted by the model (e.g., .mean and logvar).
        """
        pass

    @abc.abstractmethod
    def loss(
        self,
        model_out: torch.Tensor | tuple[torch.Tensor, ...],
        target: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        mode: str = "train",
        suffix: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Computes a loss that can be used to update the model using backpropagation.

        Args:
            odel_out (tensor): the outputs to the model.
            target (tensor, optional): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.

        Returns:
            (tuple of tensor and optional dict): the loss tensor and, optionally,
                any additional metadata computed by the model,
                 as a dictionary from strings to objects with metadata computed by
                 the model (e.g., reconstruction, entropy) that will be used for logging.
        """
        raise NotImplementedError

    def evaluate(
        self,
        model_in: torch.Tensor | tuple[torch.Tensor, ...],
        target: None | torch.Tensor | tuple[torch.Tensor, ...] = None,
        eval_in: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        mode: Literal["eval", "test"] = "eval",
        suffix: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Computes an evaluation score for the model over the given input/target.

        This method should compute a non-reduced score for the model, intended mostly for
        logging/debugging purposes (so, it should not keep gradient information).
        For example, the following could be a valid
        implementation of ``eval_score``:

        .. code-block:: python

           with torch.no_grad():
               return torch.functional.mse_loss(model(model_in), target, reduction="none")


        Args:
            model_in: the inputs to the model.
            target: the expected output for the given inputs, if it
                cannot be computed from ``model_in``.
            eval_in: the inputs to the model for evaluation.

        Returns:
            (tuple of tensor and optional dict): a non-reduced tensor score, and a dictionary
                from strings to objects with metadata computed by the model
                (e.g., reconstructions, entropy, etc.) that will be used for logging.
        """
        self.eval()
        with torch.no_grad():
            model_out = self.forward(model_in)
            loss, meta = self.loss(model_out, target, mode, suffix)
            meta = self.eval_metrics(model_out, target, eval_in, meta, mode, suffix)
        return loss.item(), meta

    @abc.abstractmethod
    def eval_metrics(
        self,
        model_out: torch.Tensor | tuple[torch.Tensor, ...],
        target: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        eval_in: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        meta: dict[str, Any] | None = None,
        mode: str = "train",
        suffix: str = "",
    ) -> dict[str, Any]:
        """Computes evaluation metrices for the model over the given input/target."""
        raise NotImplementedError

    def update(
        self,
        model_in: torch.Tensor | tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
        target: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        eval_in: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Updates the model using backpropagation with given input and target tensors.

        Provides a basic update function, following the steps below:

        .. code-block:: python

           optimizer.zero_grad()
           loss = self.loss(model_in, target)
           loss.backward()
           optimizer.step()

        Args:
            model_in (tensor or batch of transitions): the inputs to the model.
            optimizer (torch.optimizer): the optimizer to use for the model.
            target (tensor or sequence of tensors): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.

        Returns:
             (float): the numeric value of the computed loss.
             (dict): any additional metadata dictionary computed by :meth:`loss`.
        """
        self.train()
        optimizer.zero_grad()
        model_out = self.forward(model_in)
        loss, meta = self.loss(model_out, target)
        meta = self.eval_metrics(model_out, target, eval_in, meta)
        loss.backward()
        if self.cfg.max_grad_norm:
            nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)
        optimizer.step()
        return loss.item(), meta

    def reset(self, obs: torch.Tensor, rng: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Initializes the model to start a new simulated trajectory.

        This method can be used to initialize data that should be kept constant during
        a simulated trajectory starting at the given observation (for example model
        indices when using a bootstrapped ensemble with TSinf propagation). It should
        also return any state produced by the model that the :meth:`sample()` method
        will require to continue the simulation (e.g., predicted observation,
        latent state, last action, beliefs, propagation indices, etc.).

        Args:
            obs (tensor): the observation from which the trajectory will be
                started.
            rng (`torch.Generator`, optional): an optional random number generator
                to use.

        Returns:
            (dict(str, tensor)): the model state necessary to continue the simulation.
        """
        raise NotImplementedError("ModelEnv requires that model has a reset() method defined.")

    def sample(
        self,
        act: torch.Tensor,
        model_state: dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: torch.Generator | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, torch.Tensor] | None,
    ]:
        """Samples a simulated transition from the dynamics model.

        This method will be used by :class:`ModelEnv` to simulate a transition of the form.
            o_t+1, r_t+1, d_t+1, st = sample(at, s_t), where

            - a_t: action taken at time t.
            - s_t: model state at time t (as returned by :meth:`reset()` or :meth:`sample()`.
            - r_t: reward at time t.
            - d_t: terminal indicator at time t.

        If the model doesn't simulate rewards and/or terminal indicators, it can return
        ``None`` for those.

        Args:
            act (tensor): the action at.
            model_state (tensor): the model state st.
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (`torch.Generator`, optional): an optional random number generator
                to use.

        Returns:
            (tuple): predicted observation, rewards, terminal indicator and model
                state dictionary. Everything but the observation is optional, and can
                be returned with value ``None``.
        """
        raise NotImplementedError("ModelEnv requires that model has a sample() method defined.")

    def __len__(self):
        return 1

    def get_model_path(self, save_dir: str, suffix: str = "") -> str:
        if self._MODEL_FNAME.endswith(".pth"):
            model_base_name, model_ext = os.path.splitext(self._MODEL_FNAME)
        else:
            model_base_name = self._MODEL_FNAME
            model_ext = ".pth"
        return os.path.join(save_dir, model_base_name + suffix + model_ext)

    def save(self, path: str):
        """Saves the model to the given path."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Loads the model from the given path."""
        self.load_state_dict(torch.load(path))

    def _construct_layer(self, layer_cfg) -> nn.Module:
        """Constructs a layer from the given configuration."""
        if isinstance(layer_cfg, BaseModelCfg.MLPConfig):
            layer = MLP(layer_cfg)
        elif isinstance(layer_cfg, BaseModelCfg.CNNConfig):
            layer = CNN(layer_cfg)
        elif isinstance(layer_cfg, BaseModelCfg.S4RNNConfig):
            layer = S4RNN(**layer_cfg.to_dict())
        elif isinstance(layer_cfg, BaseModelCfg.ResNetConfig):
            if layer_cfg.multi_scale_features:
                layer = ResNetFPN(layer_cfg)
            else:
                layer = ResNet(layer_cfg)
        elif isinstance(layer_cfg, BaseModelCfg.PerceptNetCfg):
            layer = PerceptNet(**layer_cfg.to_dict())
        else:
            layer_args = layer_cfg.to_dict()
            layer_type = layer_args.pop("type")
            layer = getattr(nn, layer_type)(**layer_args)

            # change initialization for GRU and LSTM
            if layer_type == "GRU" or layer_type == "LSTM":
                # set forget gate iz and hz bias to 1
                for names in layer._all_weights:
                    for name in filter(lambda n: "bias" in n, names):
                        bias = getattr(layer, name)
                        n = bias.size(0)
                        start, end = n // 3, n // 3 * 2
                        bias.data[start:end].fill_(1.0)
        return layer
