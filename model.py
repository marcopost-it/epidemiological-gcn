import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from abc import abstractmethod
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, ARMAConv

class EGCN_Base(BaseModel):
    def __init__(self,
                 n_nodes,
                 gcn_input_size,
                 gcn_hidden_size,
                 gcn_output_size,
                 lstm_hidden_size,
                 lstm_bias,
                 lstm_num_layers,
                 dropout,
                 N_per_nodes,
                 i0):
        super(EGCN_Base, self).__init__()

        self.n_nodes = n_nodes

        self._make_gcn(gcn_input_size, gcn_hidden_size, gcn_output_size)

        self._make_lstm(gcn_output_size, lstm_hidden_size,lstm_bias, lstm_num_layers)

        self.dropout = dropout
        self.lstm_num_layers = lstm_num_layers

        self.N_per_nodes = nn.parameter.Parameter(torch.tensor(N_per_nodes, dtype=torch.float32), requires_grad=False)
        self.i0 = nn.parameter.Parameter(torch.tensor(i0, dtype=torch.float32),requires_grad=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _make_gcn(self, input_size, hidden_size, output_size):
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, output_size)

    def _make_lstm(self, input_size, hidden_size, bias, num_layers):
        self.lstm = nn.LSTM(input_size, hidden_size, bias=bias, num_layers=num_layers)
        self.l2b = nn.Linear(hidden_size, 1)

    @abstractmethod
    def _forward_init(self):
        raise NotImplementedError

    def _forward_gcn(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr[:,0]

        x = self.conv1(x, edge_index, edge_weights)
        x = F.relu(x)
        x = F.dropout(x, training=self.dropout)

        x = self.conv2(x, edge_index)#, edge_weights)
        x = F.relu(x)
        x = F.dropout(x, training=self.dropout)

        x = self.conv3(x, edge_index)#, edge_weights)
        x = F.relu(x)
        x = F.dropout(x, training=self.dropout)

        return x

    def _forward_lstm(self, data):
        b_inter, (self.h_t, self.c_t) = self.lstm(data[None, ...], (self.h_t, self.c_t))
        b = torch.relu(self.l2b(b_inter)).squeeze()
        return b

    @abstractmethod
    def _forward_update_state(self, hidden, prev_h, betas, t):
        raise NotImplementedError

    @abstractmethod
    def _forward_output(self, hidden):
        """It may be a good idea to update this in subclasses, and necessarily
        must do so is the order of the SIR-like compartments does NOT begin with
        I, R"""
        raise NotImplementedError

    def _forward_cleanup(self):
        del self.h_t
        del self.c_t

    def forward(self, X):
        #time_steps = len(X) #.len()
        time_steps = X.slices['x'].shape[0]-1

        # lstm initialization
        self.h_t = torch.randn(self.lstm_num_layers, self.n_nodes, self.lstm.hidden_size,device = self.device)
        self.c_t = torch.randn(self.lstm_num_layers, self.n_nodes, self.lstm.hidden_size,device = self.device)

        comportamental_states = self._forward_init()

        prev_h = comportamental_states.clone()  # init previous state
        hiddens = []
        outputs = []
        betas_list = []
        for t in range(time_steps):
            graph_t = X.get(t)

            node_embeddings = self._forward_gcn(graph_t)

            betas = self._forward_lstm(node_embeddings)
            comportamental_states = self._forward_update_state(comportamental_states, prev_h, betas, t)

            prev_h = comportamental_states.clone()
            outputs.append(self._forward_output(prev_h))
            hiddens.append(prev_h)
            betas_list.append(betas)

        # End of loop cleanup
        self._forward_cleanup()

        return torch.stack(hiddens), torch.stack(outputs),torch.stack(betas_list)


class EGCN_SIR_Regioni(EGCN_Base):
    def __init__(self,
                 n_nodes,
                 gcn_input_size,
                 gcn_hidden_size,
                 gcn_output_size,
                 lstm_hidden_size,
                 lstm_bias,
                 lstm_num_layers,
                 update_gammas,
                 dropout,
                 N_per_nodes,
                 i0,
                 gammas):
        super().__init__(n_nodes,
                         gcn_input_size,
                         gcn_hidden_size,
                         gcn_output_size,
                         lstm_hidden_size,
                         lstm_bias,
                         lstm_num_layers,
                         dropout,
                         N_per_nodes,
                         i0
                         )

        self.gammas = nn.parameter.Parameter(torch.tensor(gammas, dtype=torch.float32),requires_grad=update_gammas)

        self.INFECTED = 0
        self.RECOVERED = 1
        self.SUSCEPTIBLE = 2

    def _forward_init(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        comportamental_states = torch.zeros(
           self.n_nodes, 3
        ).to(device=device)

        comportamental_states[:, self.INFECTED] = self.i0
        comportamental_states[:, self.SUSCEPTIBLE] = 1 - self.i0
        return comportamental_states

    def _forward_update_state(self, hidden, prev_h, betas, t):
        INFECTED = self.INFECTED
        RECOVERED = self.RECOVERED
        SUSCEPTIBLE = self.SUSCEPTIBLE

        drdt = F.relu(self.gammas.clone()) * prev_h[:, INFECTED].clone()
        hidden[:, INFECTED] = prev_h[:,INFECTED].clone() + prev_h[:,INFECTED].clone() * betas.clone() * prev_h[:,SUSCEPTIBLE].clone() - drdt
        hidden[:, RECOVERED] = prev_h[:,RECOVERED].clone() + drdt
        hidden[:, SUSCEPTIBLE] = 1.0 - hidden[:, INFECTED] - hidden[:, RECOVERED]

        return hidden

    def _forward_output(self, hidden):
        return (hidden[:, 0:2])[..., None]


class EGCN_SIRD_Regioni(EGCN_Base):
    def __init__(self,
                 n_nodes,
                 gcn_input_size,
                 gcn_hidden_size,
                 gcn_output_size,
                 lstm_hidden_size,
                 lstm_bias,
                 lstm_num_layers,
                 update_gammas,
                 update_mus,
                 dropout,
                 N_per_nodes,
                 i0,
                 gammas,
                 mus):
        super().__init__(n_nodes,
                         gcn_input_size,
                         gcn_hidden_size,
                         gcn_output_size,
                         lstm_hidden_size,
                         lstm_bias,
                         lstm_num_layers,
                         dropout,
                         N_per_nodes,
                         i0
                         )

        gammas = np.array([gammas for i in range(72)])
        mus = np.array([mus for i in range(72)])

        self.gammas = nn.parameter.Parameter(torch.tensor(gammas, dtype=torch.float32),requires_grad=update_gammas)
        self.mus = nn.parameter.Parameter(torch.tensor(mus, dtype=torch.float32), requires_grad=update_mus)

        self.INFECTED = 0
        self.RECOVERED = 1
        self.DECEASED = 2
        self.SUSCEPTIBLE = 3


    def _forward_init(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        comportamental_states = torch.zeros(
           self.n_nodes, 4
        ).to(device=device)

        comportamental_states[:, self.INFECTED] = self.i0
        comportamental_states[:, self.SUSCEPTIBLE] = 1 - self.i0
        return comportamental_states

    def _forward_update_state(self, hidden, prev_h, betas, t):
        # update the hidden state SIR model @formatter:off

        INFECTED = self.INFECTED
        RECOVERED = self.RECOVERED
        SUSCEPTIBLE = self.SUSCEPTIBLE
        DECEASED = self.DECEASED


        dr_dt = self.gammas[t].clone() * prev_h[:, INFECTED].clone()
        dd_dt = self.mus[t].clone() * prev_h[:, INFECTED].clone()

        hidden[:, INFECTED] = prev_h[:,INFECTED].clone() + prev_h[:,SUSCEPTIBLE].clone() * prev_h[:,INFECTED].clone() * betas.clone() - dr_dt - dd_dt
        hidden[:, RECOVERED] = prev_h[:,RECOVERED].clone() + dr_dt
        hidden[:, DECEASED] = prev_h[:, DECEASED].clone() + dd_dt
        hidden[:, SUSCEPTIBLE] = 1.0 - hidden[:, INFECTED] - hidden[:, RECOVERED] - hidden[:, DECEASED]

        return hidden

    def _forward_output(self, hidden):
        return (hidden[:, 0:3])[..., None]

class EGCN_Model_Province(TDEGCN_Base):
    def __init__(self,
                 n_nodes,
                 gcn_input_size,
                 gcn_hidden_size,
                 gcn_output_size,
                 lstm_hidden_size,
                 lstm_bias,
                 lstm_num_layers,
                 update_gammas,
                 dropout,
                 N_per_nodes,
                 i0,
                 gammas):
        super().__init__(n_nodes,
                         gcn_input_size,
                         gcn_hidden_size,
                         gcn_output_size,
                         lstm_hidden_size,
                         lstm_bias,
                         lstm_num_layers,
                         update_gammas,
                         dropout,
                         N_per_nodes,
                         i0,
                         gammas
                         )

    def _forward_output(self, hidden):
        return (hidden[:, self.INFECTED] + hidden[:, self.RECOVERED])[..., None]  # add dimension w/ None

