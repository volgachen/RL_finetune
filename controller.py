import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        # TODO
        self.num_choices = args.num_choices
        self.num_layers = args.num_layers
        self.lstm_size = args.lstm_size
        # self.lstm_num_layers = args.lstm_num_layers # TODO: Multilayer LSTM
        self.temperature = args.temperature
        self.tanh_constant = args.controller_tanh_constant

        self.encoder = nn.Embedding(self.num_choices, self.lstm_size)

        self.lstm = nn.LSTMCell(self.lstm_size, self.lstm_size)
        self.w_soft = nn.Linear(self.lstm_size, self.num_choices, bias=False)
        self.reset_param()

    def reset_param(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, prev_c=None, prev_h=None, use_bias=False):
        if prev_c is None:
            # TODO: multi-layer LSTM
            #prev_c = [torch.zeros(1, self.lstm_size).cuda() for _ in range(self.lstm_num_layers)]
            #prev_h = [torch.zeros(1, self.lstm_size).cuda() for _ in range(self.lstm_num_layers)]
            prev_c = torch.zeros(1, self.lstm_size).cuda()
            prev_h = torch.zeros(1, self.lstm_size).cuda()

        inputs = self.encoder(torch.zeros(1).long().cuda())

        op_seq = []

        entropy = []
        log_prob = []

        for layer_id in range(self.num_layers):
            embed = inputs
            next_h, next_c = self.lstm(embed, (prev_h, prev_c))
            prev_c, prev_h = next_c, next_h
            logits = self.w_soft(next_h)
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                logits = self.tanh_constant * torch.tanh(logits)
            prob = F.softmax(logits, dim=-1)
            op_id = torch.multinomial(prob, 1).long().view(1)
            op_seq.append(op_id)
            curr_log_prob = F.cross_entropy(logits, op_id)
            log_prob.append(curr_log_prob)
            curr_ent = -torch.mean(torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()
            entropy.append(curr_ent)
            inputs = self.encoder(op_id+1)

        op_seq = torch.tensor(op_seq)
        entropy = sum(entropy)
        log_prob = sum(log_prob)

        return op_seq, log_prob, entropy

