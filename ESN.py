import torch
import torch.nn as nn
from utils.formula import spectral_radius as _spectral_radius
from matrix import Matrix
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
logger = logging.getLogger(__name__)
from readout import Ridge 
class ESN(nn.Module):
    def __init__(self, W: Matrix, readout, spectral_radius, f, washout, training_noise=0, input_bias=None):
        super(ESN, self).__init__()
        self.W = W
        self.spectral_radius = spectral_radius
        self.f = f
        self.washout = washout
        self.readout = readout
        self.training_noise = training_noise
        self.input_bias = input_bias
        self.W_back = W.W_back
        self.hidden_nodes = len(self.W.W_res)
        spec_rad = _spectral_radius(self.W.W_res)
        if self.spectral_radius is not None and spec_rad != 0:
            self.W.W_res *= self.spectral_radius / spec_rad
    def forward(self, u, y=None, kq=False):
        len_timeseries = u.size()[0]
        X = torch.zeros(len_timeseries, self.hidden_nodes)
        x = torch.zeros(self.hidden_nodes)


        for t in range(len_timeseries):
            w_in_signal = self.input_bias if self.input_bias is not None else u[t]
            state_input = self.W.W_in * w_in_signal + self.W.W_res.mv(x)
            if y is not None and self.W_back is not None and t > 0:
                state_input = state_input + self.W_back * y[t - 1]
                if self.training_noise > 0:
                    noise = torch.FloatTensor(self.hidden_nodes).uniform_(
                        -self.training_noise, self.training_noise
                    )
                    state_input = state_input + noise
            x = self.f(state_input)
            X[t] = x

        self.X = X
        X = X[self.washout:]
        y = y[self.washout:] if y is not None else y

        if not kq:
            y_pred = self.readout(X, y)
            return y_pred
        else:
            # When kq=True, we just need to store X, don't compute readout
            return None
    
    def closed_loop_test(self, u_test, prediction_horizon, teacher_forcing_len=1000):
        """
        Closed-loop free-run evaluation (Jaeger's approach).
        Assumes readout is already trained and reservoir state is primed (self.X[-1]).

        Splits u_test into consecutive windows of (teacher_forcing_len + prediction_horizon).
        For each window:
          1. Teacher-force for teacher_forcing_len steps
          2. Free-run for prediction_horizon steps
          3. Record only the final (84th) prediction

        Returns (predictions, ground_truths) tensors of length num_trials.
        """
        window_len = teacher_forcing_len + prediction_horizon
        num_trials = len(u_test) // window_len
        predictions = torch.zeros(num_trials)
        ground_truths = torch.zeros(num_trials)

        x = self.X[-1].clone()

        for trial in range(num_trials):
            start = trial * window_len

            # Teacher-force for teacher_forcing_len steps
            for t in range(teacher_forcing_len):
                w_in_signal = self.input_bias if self.input_bias is not None else u_test[start + t]
                state_input = self.W.W_in * w_in_signal + self.W.W_res.mv(x)
                if self.W_back is not None:
                    state_input = state_input + self.W_back * u_test[start + t]
                x = self.f(state_input)

            # Free-run for prediction_horizon steps, record only the last prediction
            x_free = x.clone()
            for step in range(prediction_horizon - 1):
                pred = self.readout.model.predict(x_free.detach().numpy().reshape(1, -1))[0]
                u_pred = torch.tensor(pred, dtype=torch.float32).clamp(-1, 1)
                w_in_signal = self.input_bias if self.input_bias is not None else u_pred
                state_input = self.W.W_in * w_in_signal + self.W.W_res.mv(x_free)
                if self.W_back is not None:
                    state_input = state_input + self.W_back * u_pred
                x_free = self.f(state_input)

            final_pred = self.readout.model.predict(x_free.detach().numpy().reshape(1, -1))[0]
            predictions[trial] = final_pred
            ground_truths[trial] = u_test[start + window_len - 1]

            # Advance true state through the free-run portion for next trial
            for t in range(prediction_horizon):
                w_in_signal = self.input_bias if self.input_bias is not None else u_test[start + teacher_forcing_len + t]
                state_input = self.W.W_in * w_in_signal + self.W.W_res.mv(x)
                if self.W_back is not None:
                    state_input = state_input + self.W_back * u_test[start + teacher_forcing_len + t]
                x = self.f(state_input)

        return predictions, ground_truths

    def memory_capacity(self, washout, u_train, u_test, plot=False):

        output_nodes = int(1.4*self.hidden_nodes)
        washout_len = washout.shape[0]
        train_len = u_train.shape[0]

        self(torch.cat((washout, u_train, u_test), 0), kq=True)
        self.X_train = self.X[washout_len:washout_len+train_len]

        self.w_outs = [0]*output_nodes


        for k in range(1, output_nodes+1):
            self.w_outs[k-1] = Ridge()
            self.w_outs[k-1](self.X_train[k:, :], u_train[:-k])
            

        self.X_test = self.X[washout_len+train_len:]

        ys = torch.FloatTensor([rr(self.X_test) for rr in self.w_outs])
      

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(u_test)
            for k in range(output_nodes):
                plt.plot(ys[k][k+1:])
            plt.show()

        mc = 0
        self.mcs = []
        for k in range(output_nodes):
            u_tk = u_test[:-(k+1)]
            numerator = (np.cov(u_tk, ys[k][k+1:], bias=True)[0][1])**2
            denominator = torch.var(u_tk)*torch.var(ys[k][k+1:])
            _mc = numerator/denominator
            mc += _mc
            self.mcs.append(_mc)
        return float(mc)

    def plot_node_states(
        self,
        node_indices=None,
        num_nodes=10,
        start=0,
        end=None,
        show=True,
        save_path=None,
    ):
        """
        Plot reservoir node activations over time.

        Args:
            node_indices (Iterable[int], optional): Specific node indices to plot.
            num_nodes (int): Max number of nodes to plot if node_indices is None.
            start (int): Starting timestep (inclusive).
            end (int, optional): Ending timestep (exclusive). Defaults to full length.
            show (bool): Whether to display the figure interactively.
            save_path (str, optional): Path to save the figure (PNG, PDF, etc.).
        """
        if not hasattr(self, "X"):
            raise RuntimeError(
                "Reservoir states not found. Run a forward pass before plotting."
            )

        states = self.X.detach().cpu().numpy()
        total_steps, total_nodes = states.shape

        start = max(0, int(start))
        end = total_steps if end is None else min(total_steps, int(end))
        if start >= end:
            raise ValueError("Invalid time window: 'start' must be < 'end'.")

        if node_indices is None:
            selected_nodes = list(range(min(num_nodes, total_nodes)))
        else:
            selected_nodes = [int(idx) for idx in node_indices if 0 <= idx < total_nodes]
            if not selected_nodes:
                raise ValueError("Provided node_indices are outside valid range.")
            if num_nodes is not None:
                selected_nodes = selected_nodes[:num_nodes]

        timesteps = np.arange(start, end)

        plt.figure(figsize=(12, 6))
        for idx in selected_nodes:
            plt.plot(timesteps, states[start:end, idx], label=f"Node {idx}")

        plt.xlabel("Timestep")
        plt.ylabel("Node value")
        plt.title(f"Reservoir node activations ({len(selected_nodes)} nodes)")
        plt.legend(loc="upper right", ncol=2, fontsize=8)
        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, "state_plot.png"), dpi=300)

        if show:
            plt.show()
        else:
            plt.close()
