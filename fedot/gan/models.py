import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    def __init__(self, h_inputs, h, z_dim, n, rw_len, temp, state='structure'):
        """
            H_inputs: input dimension
            H:        hidden dimension
            z_dim:    latent dimension
            N:        number of nodes (needed for the up and down projection)
            rw_len:   number of LSTM cells
            temp:     temperature for the gumbel softmax
        """
        super(Generator, self).__init__()
        self.intermediate = nn.Linear(z_dim, h).dtype(torch.float64)
        torch.nn.init.xavier_uniform_(self.intermediate.weight)
        torch.nn.init.zeros_(self.intermediate.bias)
        self.intermediate_lines = nn.Linear(z_dim, h).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.intermediate_lines.weight)
        torch.nn.init.zeros_(self.intermediate_lines.bias)

        self.c_up = nn.Linear(h, h).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.c_up.weight)
        torch.nn.init.zeros_(self.c_up.bias)

        self.h_up = nn.Linear(h, h).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.h_up.weight)
        torch.nn.init.zeros_(self.h_up.bias)

        self.c_up_lines = nn.Linear(h, h).dtype(torch.float64)
        torch.nn.init.xavier_uniform_(self.c_up_lines.weight)
        torch.nn.init.zeros_(self.c_up_lines.bias)

        self.h_up_lines = nn.Linear(h, h).dtype(torch.float64)
        torch.nn.init.xavier_uniform_(self.h_up_lines.weight)
        torch.nn.init.zeros_(self.h_up_lines.bias)

        self.lstm_cell = LSTMCell(h_inputs, h).type(torch.float64)
        self.lstm_cell_lines = LSTMCell(h_inputs, h).type(torch.float64)

        self.W_up = nn.Linear(h, h).type(torch.float64)
        self.W_down = nn.Linear(n, h_inputs, bias=False).type(torch.float64)
        self.W_down.lines = nn.Linear(n, h_inputs, bias=False).type(torch.float64)
        self.W_out_lines = nn.Linear(h, 1).type(torch.float64)

        self.rw_len = rw_len
        self.temp = temp
        self.H = h
        self.latent_dim = z_dim
        self.N = n
        self.H_inputs = h_inputs
        self.freeze_param(state)

    def forward(self, latent, inputs, device='cpu'):
        intermediate = torch.tanh(self.intermediate(latent))
        intermediate_lines = torch.tanh(self.intermediate_lines(latent))
        hc = (torch.tanh(self.h_up(intermediate)), torch.tanh(self.c_up(intermediate)))
        hc_lines = (torch.tanh(self.h_up_lines(intermediate_lines)), torch.tanh(self.c_up_lines(intermediate_lines)))

        out, out_lines = [], []

        for i in range(self.rw_len):
            hh, cc = self.lstm_cell(inputs, hc)
            hc = (hh, cc)
            h_up = self.W_up(hh)
            h_sample = self.gumbel_softmax_sample(h_up, self.temp, device)
            inputs = self.W_down(h_sample)
            out.append(h_sample)

        for j in range(self.rw_len):
            inputs_lines = self.W_down_lines(out[j])
            hh_lines, cc_lines = self.lstm_cell(inputs_lines, hc_lines)
            hc_lines = (hc_lines, cc_lines)
            hh_out = self.W_out_lines(hh_lines)
            out_lines.append(hh_out)

        return torch.stack(out, dim=1), torch.stack(out_lines, dim=1)

    def sample_latenet(self, num_samples, device='cpu'):
        return torch.randn((num_samples, self.latent_dim)).type(torch.float64).to(device)

    def sample(self, num_samples, device):
        noise = self.sample_latenet(num_samples, device)
        input_zeros = self.init_hidden(num_samples).contiguous().type(torch.float64).to(device)
        generated_data, generated_weights = self(noise, input_zeros, device)

        return generated_data, generated_weights

    def sample_discrete(self, num_samples, device):
        with torch.no_grad():
            proba, proba_weights = self.sample(num_samples, device)

        return np.argmax(proba.cpu().numpy(), axis=2), proba_weights.cpu().numpy()

    def sample_gumbel(self, logits, eps=1e-20):#
        U = torch.rand(logits.shape, dtype=torch.float64)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, device, hard=True):
        gumbel = self.sample_gumbel(logits).type(torch.float64).to(device)
        y = logits + gumbel
        y = F.softmax(y / temperature, dim=1)

        if hard:
            y_hard = torch.max(y, 1, keepdim=True)[0].eq(y).type(torch.float64).to(device)
            y = (y_hard - y).detach() + y

        return y

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new(batch_size, self.H_inputs).zero_().type(torch.float64)

    def freeze_params(self, state):
        self.intermediate_lines.weight.requires_grad_(False if state == 'structure' else True)
        self.intermediate_lines.bias.requires_grad_(False if state == 'structure' else True)
        self.h_up_lines.weight.requires_grad_(False if state == 'structure' else True)
        self.h_up_lines.bias.requires_grad_(False if state == 'structure' else True)
        self.c_up_lines.weight.requires_grad_(False if state == 'structure' else True)
        self.c_up_lines.bias.requires_grad_(False if state == 'structure' else True)
        self.lstmcell_lines.cell.weight.requires_grad_(False if state == 'structure' else True)
        self.lstmcell_lines.cell.bias.requires_grad_(False if state == 'structure' else True)
        self.W_down_lines.weight.requires_grad_(False if state == 'structure' else True)
        self.W_out_lines.weight.requires_grad_(False if state == 'structure' else True)
        self.W_out_lines.bias.requires_grad_(False if state == 'structure' else True)

        self.intermediate.weight.requires_grad_(True if state == 'structure' else False)
        self.intermediate.bias.requires_grad_(True if state == 'structure' else False)
        self.h_up.weight.requires_grad_(True if state == 'structure' else False)
        self.h_up.bias.requires_grad_(True if state == 'structure' else False)
        self.c_up.weight.requires_grad_(True if state == 'structure' else False)
        self.c_up.bias.requires_grad_(True if state == 'structure' else False)
        self.lstmcell.cell.weight.requires_grad_(True if state == 'structure' else False)
        self.lstmcell.cell.bias.requires_grad_(True if state == 'structure' else False)
        self.W_down.weight.requires_grad_(True if state == 'structure' else False)
        self.W_up.weight.requires_grad_(True if state == 'structure' else False)
        self.W_up.bias.requires_grad_(True if state == 'structure' else False)


class Discriminator(nn.Module):
    def __init__(self, h_inputs, h, n, rw_len):
        """
            H_inputs: input dimension
            H:        hidden dimension
            N:        number of nodes (needed for the up and down projection)
            rw_len:   number of LSTM cells
        """
        super(Discriminator, self).__init__()
        self.W_down = nn.Linear(n, h_inputs, bias=False).type(torch.float64)
        torch.nn.init.xavier_uniform_()

        self.lstm_cell = LSTMCell(h_inputs + 1, h).type(torch.float64)
        self.lin_out = nn.Linear(h, 1, bias=True).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.lin_out.bias)
        torch.nn.init.zeros_(self.lin_out.bias)

        self.H = h
        self.N = n
        self.rw_len = rw_len
        self.H_inputs = h_inputs

    def forward(self, x):
        x_rw = x[:, :, :self.N]
        x_weights = x[:, :, -1:]
        x_rw = x_rw.view(-1, self.N)
        xa = self.W_down(x_rw)
        xa = xa.view(-1, self.rw_len, self.H_inputs)
        xc = torch.cat((xa, x_weights), dim=2)
        hc = self.init_hidden(xc.size(0))

        for i in range(self.rw_len):
            hc = self.lstm_cell(xc[:, i, :], hc)

        out = hc[0]
        pred = self.lin_out(out)

        return pred

    def init_inputs(self, num_samples):
        weight = next(self.parameters()).data
        return weight.new(num_samples, self.H_inputs).zero_().type(torch.float64)

    def init_hidden(self, num_samples):
        weight = next(self.parameters()).data
        return (weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64),
                weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64)
        )


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=True)
        torch.nn.init.xavier_uniform_(self.cell.weight)
        torch.nn.init.zeros_(self.cell.bias)

    def forward(self, x, hidden):
        hx, cx = hidden
        gates = torch.cat((x, hx), dim=1)
        gates = self.cell(gates)

        in_gate, cell_gate, forget_gate, out_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(torch.add(forget_gate, 1.))
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = torch.mul(cx, forget_gate) + torch.mul(in_gate, cell_gate)
        hy = torch.mul(out_gate, torch.tanh(cy))

        return hy, cy
