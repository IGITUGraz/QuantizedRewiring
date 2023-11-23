"""Spiking neuron models"""

import math
from abc import ABC
from abc import abstractmethod
from typing import Tuple, Type

import torch
import torch.nn.functional
from torch import Tensor

from layers.autograd_functions import SpikeFunction


class NeuronModel(ABC):
    @abstractmethod
    def __call__(
            self, x: torch.Tensor, states: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        pass

    @staticmethod
    @abstractmethod
    def initial_states(
            batch_size: int,
            hidden_size: int,
            dtype: Type[torch.dtype],
            device: Type[torch.device],
    ) -> Tuple[torch.Tensor, ...]:
        pass


class IafPscDelta(NeuronModel):
    def __init__(self, thr: float = 1.0, perfect_reset: bool = False, refractory_timesteps: int = 1,
                 tau_mem: float = 20.0, spike_function: Type[torch.autograd.Function] = SpikeFunction,
                 dampening_factor: float = 1.0, ) -> None:
        super().__init__()
        self.thr = thr
        self.perfect_reset = perfect_reset
        self.refractory_timesteps = refractory_timesteps

        self.decay_mem = math.exp(-1.0 / tau_mem)

        self.spike_function = lambda x: spike_function.apply(x, dampening_factor)

    def __call__(self, x_t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        z_t, v_t, r_t = states

        is_refractory = torch.greater(r_t, 0.0)

        # Integrate membrane.
        v_t, v_scaled = self.integrate(
            x_t, z_t, v_t, self.thr, self.perfect_reset, self.decay_mem
        )

        # Spike generation.
        z_t = self.spike_function(v_scaled)
        z_t = torch.where(is_refractory, torch.zeros_like(z_t), z_t)

        # Update refractory period.
        r_t = torch.where(
            is_refractory,
            (r_t - 1.0).clamp(0.0, self.refractory_timesteps),
            self.refractory_timesteps * z_t,
        )

        return z_t, (z_t, v_t, r_t)

    @staticmethod
    @torch.jit.script
    def integrate(x_t: torch.Tensor, z_t: torch.Tensor, v_t: torch.Tensor, thr: float, perfect_reset: bool,
                  decay_mem: float, ) -> Tuple[torch.Tensor, torch.Tensor]:
        if perfect_reset:
            v_t = decay_mem * v_t * (1.0 - z_t) + (1.0 - decay_mem) * x_t
        else:
            v_t = decay_mem * v_t + (1.0 - decay_mem) * x_t - z_t * thr

        v_scaled = (v_t - thr) / thr

        return v_t, v_scaled

    @staticmethod
    def initial_states(batch_size: int, hidden_size: int, dtype: torch.dtype, device: torch.device) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
        )


class AdaIafPscDelta(NeuronModel):
    def __init__(self, thr: float = 1.0, perfect_reset: bool = False, refractory_timesteps: int = 1,
                 tau_mem: float = 20.0, tau_adaptation: float = 20.0, adaptation_constant: Tensor = 1.0,
                 spike_function: Type[torch.autograd.Function] = SpikeFunction,
                 dampening_factor: float = 1.0, ) -> None:
        super().__init__()
        self.thr = thr
        self.perfect_reset = perfect_reset
        self.refractory_timesteps = refractory_timesteps
        self.adaptation_constant = adaptation_constant

        self.decay_mem = math.exp(-1.0 / tau_mem)
        self.decay_ada = math.exp(-1.0 / tau_adaptation)

        self.spike_function = lambda x: spike_function.apply(x, dampening_factor)

    def __call__(self, x_t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        z_t, v_t, r_t, a_t = states

        is_refractory = torch.greater(r_t, 0.0)

        # Update threshold
        thr, a_t = self.update_threshold(
            a_t, z_t, self.thr, self.decay_ada, self.adaptation_constant
        )

        # Integrate membrane.
        v_t, v_scaled = self.integrate(
            x_t, z_t, v_t, thr, self.perfect_reset, self.decay_mem
        )

        # Spike generation.
        z_t = self.spike_function(v_scaled)
        z_t = torch.where(is_refractory, torch.zeros_like(z_t), z_t)

        # Update refractory period.
        r_t = torch.where(
            is_refractory,
            (r_t - 1.0).clamp(0.0, self.refractory_timesteps),
            self.refractory_timesteps * z_t,
        )

        return z_t, (z_t, v_t, r_t, a_t)

    @staticmethod
    @torch.jit.script
    def update_threshold(a_t: torch.Tensor, z_t: torch.Tensor, thr: float, decay_ada: float,
                         adaptation_constant: Tensor, ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_t = decay_ada * a_t + (1.0 - decay_ada) * z_t
        return thr + adaptation_constant * a_t, a_t

    @staticmethod
    @torch.jit.script
    def integrate(x_t: torch.Tensor, z_t: torch.Tensor, v_t: torch.Tensor, thr: torch.Tensor, perfect_reset: bool,
                  decay_mem: float, ) -> Tuple[torch.Tensor, torch.Tensor]:
        if perfect_reset:
            v_t = decay_mem * v_t * (1.0 - z_t) + (1.0 - decay_mem) * x_t
        else:
            v_t = decay_mem * v_t + (1.0 - decay_mem) * x_t - z_t * thr

        v_scaled = (v_t - thr) / thr

        return v_t, v_scaled

    @staticmethod
    def initial_states(batch_size: int, hidden_size: int, dtype: torch.dtype, device: torch.device) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
        )


class AdaIafPscDeltaModule(torch.nn.Module):
    def __init__(
            self,
            n_regular_neurons: int,
            n_adaptive_neurons: int,
            thr: float = 1.0,
            perfect_reset: bool = False,
            refractory_timesteps: int = 1,
            tau_mem: float = 20.0,
            tau_adaptation: float = 20.0,
            adaptation_constant: float = 1.0,
            spike_function: Type[torch.autograd.Function] = SpikeFunction,
            dampening_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.thr = thr
        self.perfect_reset = perfect_reset
        self.refractory_timesteps = refractory_timesteps
        self.register_buffer('adaptation_constant', torch.concat([
            torch.full((n_adaptive_neurons,), adaptation_constant),
            torch.full((n_regular_neurons,), 0.0),
        ]))

        self.decay_mem = math.exp(-1.0 / tau_mem)
        self.decay_ada = math.exp(-1.0 / tau_adaptation)

        self.spike_function = lambda x: spike_function.apply(x, dampening_factor)

    def forward(
            self,
            x_t: torch.Tensor,
            states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        z_t, v_t, r_t, a_t = states

        is_refractory = torch.greater(r_t, 0.0)

        # Update threshold
        thr, a_t = self.update_threshold(
            a_t, z_t, self.thr, self.decay_ada, self.adaptation_constant
        )

        # Integrate membrane.
        v_t, v_scaled = self.integrate(
            x_t, z_t, v_t, thr, self.perfect_reset, self.decay_mem
        )

        # Spike generation.
        z_t = self.spike_function(v_scaled)
        z_t = torch.where(is_refractory, torch.zeros_like(z_t), z_t)

        # Update refractory period.
        r_t = torch.where(
            is_refractory,
            (r_t - 1.0).clamp(0.0, self.refractory_timesteps),
            self.refractory_timesteps * z_t,
        )

        return z_t, (z_t, v_t, r_t, a_t)

    @staticmethod
    @torch.jit.script
    def update_threshold(
            a_t: torch.Tensor,
            z_t: torch.Tensor,
            thr: float,
            decay_ada: float,
            adaptation_constant: Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_t = decay_ada * a_t + (1.0 - decay_ada) * z_t
        return thr + adaptation_constant * a_t, a_t

    @staticmethod
    @torch.jit.script
    def integrate(
            x_t: torch.Tensor,
            z_t: torch.Tensor,
            v_t: torch.Tensor,
            thr: torch.Tensor,
            perfect_reset: bool,
            decay_mem: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if perfect_reset:
            v_t = decay_mem * v_t * (1.0 - z_t) + (1.0 - decay_mem) * x_t
        else:
            v_t = decay_mem * v_t + (1.0 - decay_mem) * x_t - z_t * thr

        v_scaled = (v_t - thr) / thr

        return v_t, v_scaled

    @staticmethod
    def initial_states(
            batch_size: int, hidden_size: int, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
        )


class AdaIafPscDeltaModuleDelay(torch.nn.Module):
    def __init__(
            self,
            n_regular_neurons: int,
            n_adaptive_neurons: int,
            thr: float = 1.0,
            perfect_reset: bool = False,
            refractory_timesteps: int = 1,
            tau_mem: float = 20.0,
            tau_adaptation: float = 20.0,
            adaptation_constant: float = 1.0,
            spike_function: Type[torch.autograd.Function] = SpikeFunction,
            dampening_factor: float = 1.0,
            max_delay: int = 1,
    ) -> None:
        super().__init__()
        self.thr = thr
        self.perfect_reset = perfect_reset
        self.refractory_timesteps = refractory_timesteps
        self.register_buffer('adaptation_constant', torch.concat([
            torch.full((n_adaptive_neurons,), adaptation_constant),
            torch.full((n_regular_neurons,), 0.0),
        ]))

        self.decay_mem = math.exp(-1.0 / tau_mem)
        self.decay_ada = math.exp(-1.0 / tau_adaptation)
        self.max_delay = max_delay

        self.spike_function = lambda x: spike_function.apply(x, dampening_factor)

    def forward(
            self,
            input_recurrent_out: torch.Tensor,
            states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        z_t, v_t, r_t, a_t, state_i_future_buffer = states

        is_refractory = torch.greater(r_t, 0.0)

        # Update threshold
        thr, a_t = self.update_threshold(a_t, z_t, self.thr, self.decay_ada, self.adaptation_constant)

        # Integrate membrane.
        i_future_buffer = input_recurrent_out + state_i_future_buffer
        i_t = i_future_buffer[:, :, 0]
        v_t, v_scaled = self.integrate(i_t, z_t, v_t, thr, self.perfect_reset, self.decay_mem)
        i_future_buffer = torch.roll(i_future_buffer, -1, dims=2)
        i_future_buffer[:, :, -1] = 0

        # Spike generation.
        z_t = self.spike_function(v_scaled)
        z_t = torch.where(is_refractory, torch.zeros_like(z_t), z_t)

        # Update refractory period.
        r_t = torch.where(
            is_refractory,
            (r_t - 1.0).clamp(0.0, self.refractory_timesteps),
            self.refractory_timesteps * z_t,
        )

        return z_t, (z_t, v_t, r_t, a_t, i_future_buffer)

    @staticmethod
    @torch.jit.script
    def update_threshold(
            a_t: torch.Tensor,
            z_t: torch.Tensor,
            thr: float,
            decay_ada: float,
            adaptation_constant: Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_t = decay_ada * a_t + (1.0 - decay_ada) * z_t
        return thr + adaptation_constant * a_t, a_t

    @staticmethod
    @torch.jit.script
    def integrate(
            x_t: torch.Tensor,
            z_t: torch.Tensor,
            v_t: torch.Tensor,
            thr: torch.Tensor,
            perfect_reset: bool,
            decay_mem: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if perfect_reset:
            v_t = decay_mem * v_t * (1.0 - z_t) + (1.0 - decay_mem) * x_t
        else:
            v_t = decay_mem * v_t + (1.0 - decay_mem) * x_t - z_t * thr

        v_scaled = (v_t - thr) / thr

        return v_t, v_scaled

    @staticmethod
    def initial_states(
            batch_size: int, hidden_size: int, max_delay: int, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, hidden_size, max_delay, dtype=dtype, device=device)
        )
