"""SNN models"""

import math
from typing import Tuple, Optional, Dict

import torch
import torch.nn.functional
from torch import nn

from layers.neuron_models import NeuronModel
from layers.quantized_linear import QLinearDelay as QLinear
from layers.quantized_linear_multi import QLinearMulti
from layers.utility_functions import exp_convolve
from layers.utils import NoOpContextManager
from layers.quantized_linear_multi_delay import QLinearMultiDelay as QLMD


class QRecurrentSNN(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            n_regular_neurons: int,
            n_adaptive_neurons: int,
            output_size: int,
            autapses: bool,
            tau_trace: float,
            dynamics: NeuronModel,
            rewiring_config: Dict,
            average_output_ms=None,
            multi_baseweight=False,
            return_state_sequence=False,
            quantized_readout=False,
            no_grad_snn=False,
    ) -> None:
        super(QRecurrentSNN, self).__init__()
        self.input_size = input_size
        self.n_regular_neurons = n_regular_neurons
        self.n_adaptive_neurons = n_adaptive_neurons
        self.hidden_size = n_regular_neurons + n_adaptive_neurons
        self.output_size = output_size
        self.autapses = autapses
        self.tau_trace = tau_trace
        self.dynamics = dynamics
        self.average_output_ms = average_output_ms
        self.rewiring_config = rewiring_config
        self.return_state_sequence = return_state_sequence
        self.no_grad_snn = no_grad_snn

        self.decay_trace = math.exp(-1.0 / tau_trace)
        assert rewiring_config['num_baseweights'] == 2
        self.input_recurrent_layer = QLinearMulti(input_size + self.hidden_size, self.hidden_size,
                                                  rewiring_config=rewiring_config, bias=False)
        if quantized_readout:
            self.output_layer = QLinear(self.hidden_size, output_size, rewiring_config=rewiring_config)
        else:
            self.output_layer = nn.Linear(self.hidden_size, output_size)

        self.reset_parameters()

    def forward(
            self, x: torch.Tensor, states: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        batch_size, sequence_length, _ = x.size()

        if states is None:
            states = self.dynamics.initial_states(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)

        trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        if self.no_grad_snn:
            context_mgr = torch.no_grad()
        else:
            context_mgr = NoOpContextManager()

        spikes_sequence, trace_sequence = [], []
        with context_mgr:
            if self.return_state_sequence:
                state_sequence = []
            for t in range(sequence_length):
                input_state = torch.concat([x.select(1, t), states[0]], dim=1)
                i_t = self.input_recurrent_layer(input_state)
                z_t, states = self.dynamics(i_t, states)

                # trace shape: [batch_size, num_neurons]
                trace = exp_convolve(z_t, trace, self.decay_trace)

                spikes_sequence.append(z_t)
                trace_sequence.append(trace)
                if self.return_state_sequence:
                    state_sequence.append(states)
            spikes_sequence = torch.stack(spikes_sequence, dim=1)
            trace_sequence = torch.stack(trace_sequence, dim=1)

        if self.average_output_ms is not None:
            output = self.output_layer(trace_sequence[:, -self.average_output_ms:, :]).mean(dim=1)
        else:
            output = self.output_layer(trace)

        # trace sequence shape: [batch_size, time, num_neurons]
        return (
            output,
            spikes_sequence,
            trace_sequence,
            states if not self.return_state_sequence else state_sequence,
        )

    def reset_parameters(self) -> None:
        if self.input_recurrent_layer.bias is not None:
            self.input_recurrent_layer.bias.data.fill_(0.0)
        self.output_layer.bias.data.fill_(0.0)

        if not self.autapses:
            self.input_recurrent_layer.C_shadow.data[:, self.input_size:, 0].fill_diagonal_(0.0)
            self.input_recurrent_layer.C_shadow.data[:, self.input_size:, 1].fill_diagonal_(0.0)

            def autapses_hook(gradient):
                gradient = gradient.clone()
                gradient[:, self.input_size:, 0].fill_diagonal_(0.0)
                gradient[:, self.input_size:, 1].fill_diagonal_(0.0)
                self.input_recurrent_layer.C_shadow.grad = gradient
                return gradient

            self.input_recurrent_layer.C.register_hook(autapses_hook)


class QRecurrentSNNDelay(QRecurrentSNN):
    def __init__(
            self,
            input_size: int,
            n_regular_neurons: int,
            n_adaptive_neurons: int,
            output_size: int,
            autapses: bool,
            tau_trace: float,
            dynamics: NeuronModel,
            rewiring_config: Dict,
            average_output_ms=None,
            multi_baseweight=False,
            return_state_sequence=False
    ) -> None:
        super(QRecurrentSNNDelay, self).__init__(
            input_size=input_size,
            n_regular_neurons=n_regular_neurons,
            n_adaptive_neurons=n_adaptive_neurons,
            output_size=output_size,
            autapses=autapses,
            tau_trace=tau_trace,
            dynamics=dynamics,
            average_output_ms=average_output_ms,
            rewiring_config=rewiring_config,
            return_state_sequence=return_state_sequence,
            multi_baseweight=multi_baseweight,
        )

        self.decay_trace = math.exp(-1.0 / tau_trace)
        assert rewiring_config['num_baseweights'] == 2
        self.input_recurrent_layer = QLMD(input_size + self.hidden_size, self.hidden_size,
                                          rewiring_config=rewiring_config, bias=False)
        self.output_layer = nn.Linear(self.hidden_size, output_size)
        self.reset_parameters()

    def forward(
            self, x: torch.Tensor, states: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        batch_size, sequence_length, _ = x.size()

        if states is None:
            states = self.dynamics.initial_states(batch_size, self.hidden_size, self.rewiring_config['max_delay'],
                                                  dtype=x.dtype, device=x.device)

        trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)

        spikes_sequence, trace_sequence = [], []
        if self.return_state_sequence:
            state_sequence = []
        for t in range(sequence_length):
            input_state = torch.concat([x.select(1, t), states[0]], dim=1)
            i_future_buffer = self.input_recurrent_layer(input_state)
            z_t, states = self.dynamics(i_future_buffer, states)

            # trace shape: [batch_size, num_neurons]
            trace = exp_convolve(z_t, trace, self.decay_trace)

            spikes_sequence.append(z_t)
            trace_sequence.append(trace)
            if self.return_state_sequence:
                state_sequence.append(states)
        spikes_sequence = torch.stack(spikes_sequence, dim=1)
        trace_sequence = torch.stack(trace_sequence, dim=1)
        if self.average_output_ms is not None:
            output = self.output_layer(trace_sequence[:, -self.average_output_ms:, :]).mean(dim=1)
        else:
            output = self.output_layer(trace)

        # trace sequence shape: [batch_size, time, num_neurons]
        return (
            output,
            spikes_sequence,
            trace_sequence,
            states if not self.return_state_sequence else state_sequence,
        )

    def reset_parameters(self) -> None:
        if self.input_recurrent_layer.bias is not None:
            self.input_recurrent_layer.bias.data.fill_(0.0)
        self.output_layer.bias.data.fill_(0.0)

        if not self.autapses:
            self.input_recurrent_layer.C_shadow.data[:, self.input_size:, 0].fill_diagonal_(0.0)
            self.input_recurrent_layer.C_shadow.data[:, self.input_size:, 1].fill_diagonal_(0.0)

            def autapses_hook(gradient):
                gradient = gradient.clone()
                gradient[:, self.input_size:, 0].fill_diagonal_(0.0)
                gradient[:, self.input_size:, 1].fill_diagonal_(0.0)
                self.input_recurrent_layer.C_shadow.grad = gradient
                return gradient

            self.input_recurrent_layer.C.register_hook(autapses_hook)
