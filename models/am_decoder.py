import torch
import torch.nn as nn
import math

class AMDecoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.w_q = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.w_k = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.norm_factor = 1. / math.sqrt(emb_dim)
        self.tanh_clipping = 10.
        self.reset_paramters()

    def reset_paramters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, node_context, agent_context, mask, decode_type):
        """
        Paramters
        ---------
        node_context: torch.tensor [batch_size x num_nodes x emb_dim]
            context of nodes obtained from the node-encoder
        agent_context: torch.tensor [batch_size x num_agents x emb_dim]
            context of agents obtained from the agent-encoder
        mask: torch.tensor [batch_size x num_nodes]
            mask that removes infeasible nodes (0: infeasible, 1: feasible)

        Returns
        -------
        probs: torch.tensor [batch_size x num_nodes]
            probabilities of visiting nodes 
        next_node_id: torch.tensor [batch_size]
            id of a node visited in the next
        """
        query  = torch.matmul(agent_context, self.w_q) # [batch_size x 1 x emb_dim]
        key    = torch.matmul(node_context, self.w_k)  # [batch_size x num_nodes x emb_dim]
        logits = self.norm_factor * torch.matmul(query, key.transpose(-1, -2)).squeeze(1)  # [batch_size x 1 x num_nodes] -> [batch_size x num_nodes]
        logits = torch.tanh(logits) * self.tanh_clipping
        # masking
        logits[mask < 1] = -math.inf
        # get probs and determine nex node
        probs = torch.softmax(logits, dim=-1)
        if probs.isnan().any():
            batch_size = node_context.size(0)
            for i in range(batch_size):
                if probs[i].isnan().any():
                    print(f"batch: {i}, prob: {probs[i]}, h_node: {logits[i]}, mask: {mask[i]}")
            assert False
        # print(probs); import time; time.sleep(10)
        next_node_id = self.select_node(probs, mask, decode_type)
        return probs, next_node_id
    
    def select_node(self, probs, mask, decode_type):
        """
        Paramters
        ---------
        probs: torch.tensor [batch_size x num_nodes]
        mask: torch.tensor [batch_size x num_nodes]
        decode_type: str
            decoding type {sampling, greedy}

        Returns
        --------
        selected: torch.tensor [batch_size]
            id of a node visited in the next
        """
        assert (probs == probs).all(), "Probs should not contain any nans"

        if decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            # check if sampling went OK, can go wrong due to bug on GPU:
            #   points with zero probability are sampled occasionally
            # see https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while (mask.gather(1, selected.unsqueeze(-1)) == 0).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        elif decode_type == "greedy":
            _, selected = probs.max(1)
            assert not (mask.gather(1, selected.unsqueeze(-1)) == 0).data.any(), "Decode greedy: infeasible action has maximum probability"
        else:
            assert False, f"decode type:{self.decode_type} is not supported."
        return selected