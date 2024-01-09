import torch
import torch.nn as nn
import math

class AMEncoder(nn.Module):
    def __init__(self, 
                 loc_dim: int,
                 depot_dim: int,
                 vehicle_dim: int,
                 emb_dim: int,
                 num_heads: int,
                 num_mha_layers: int,
                 dropout: float = 0.0):
        super().__init__()
        self.loc_dim = loc_dim
        self.depot_dim = depot_dim 
        self.emb_dim = emb_dim
        self.dim_feedforward = 2 * emb_dim
        self.num_mha_layers = num_mha_layers

        # initial embedding
        self.init_linear_loc     = nn.Linear(loc_dim, emb_dim)
        self.init_linear_depot   = nn.Linear(depot_dim, emb_dim)
        self.init_linear_vehicle = nn.Linear(vehicle_dim, emb_dim)
        # Transformer Encoder
        # for nodes (locations + depots)
        node_mha_layer = nn.TransformerEncoderLayer(d_model=emb_dim, 
                                                    nhead=num_heads,
                                                    dim_feedforward=self.dim_feedforward,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.node_mha = nn.TransformerEncoder(node_mha_layer, num_layers=num_mha_layers)
        # for vehicles
        vehicle_mha_layer = nn.TransformerEncoderLayer(d_model=emb_dim, 
                                                       nhead=num_heads,
                                                       dim_feedforward=self.dim_feedforward,
                                                       dropout=dropout,
                                                       batch_first=True)
        self.vehicle_mha = nn.TransformerEncoder(vehicle_mha_layer, num_layers=num_mha_layers)

        # params initialization
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self,
                loc_feats: torch.Tensor,
                depot_feats: torch.Tensor,
                vehicle_feats: torch.Tensor):
        """
        Paramters
        ---------

        Returns
        -------

        """
        # initial embeddings
        loc_emb   = self.init_linear_loc(loc_feats)     # [batch_size x num_locs x emb_dim]
        depot_emb = self.init_linear_depot(depot_feats) # [batch_size x num_depots x emb_dim]
        node_emb  = torch.cat((loc_emb, depot_emb), 1)  # [batch_size x num_nodes x emb_dim]
        vehicle_emb = self.init_linear_vehicle(vehicle_feats) # [batch_size x num_vehicles x emb_dim]
        # transformer encoding
        node_emb = self.node_mha(node_emb) # [batch_size x num_nodes x emb_dim]
        vehicle_emb = self.vehicle_mha(vehicle_emb) # [batch_size x num_vehicles x emb_dim]
        return node_emb, vehicle_emb

# class SkipConnection(nn.Module):
#     def __init__(self, module):
#         super().__init__()
#         self.module = module

#     def forward(self, input):
#         return input + self.module(input)

# class MultiHeadAttention(nn.Module):
#     def __init__(self,
#                  n_heads,
#                  input_dim,
#                  embed_dim):
#         super().__init__()
#         self.n_heads = n_heads
#         self.input_dim = input_dim
#         self.embed_dim = embed_dim
#         self.head_dim = embed_dim // n_heads
#         self.norm_factor = 1 / math.sqrt(self.head_dim)
        
#         self.w_q = nn.Parameter(torch.Tensor(n_heads, input_dim, self.head_dim))
#         self.w_k = nn.Parameter(torch.Tensor(n_heads, input_dim, self.head_dim))
#         self.w_v = nn.Parameter(torch.Tensor(n_heads, input_dim, self.head_dim))
#         self.w_o = nn.Parameter(torch.Tensor(n_heads, self.head_dim, embed_dim))
        
#         self.reset_parameters()

#     def reset_parameters(self):
#         for param in self.parameters():
#             stdv = 1. / math.sqrt(param.size(-1))
#             param.data.uniform_(-stdv, stdv)

#     def forward(self, embedded_inputs):
#         """
#         Transformer's self-attention-based aggregation
        
#         Parameters
#         -----------
#         embedded_inputs: torch.FloatTensor [batch_size x num_nodes(or num_agents) x emb_dim]
#             embeddings of nodes / agents
#         Returns
#         -------
#         out: torch.FloatTensor [batch_size x num_nodes(or num_agents) x context_dim] 
#             context of nodes / agents
#         """
#         batch_size, seq_length, input_dim = embedded_inputs.size()

#         # compute query, key, value
#         hflat = embedded_inputs.contiguous().view(-1, input_dim) # [(batch_size*seq_length) x input_size]
#         shp = (self.n_heads, batch_size, seq_length, self.head_dim) # split embeddings into num. of heads
#         # [n_heads x (batch_size*seq_length) x head_dim] -> [n_heads x batch_size x seq_length x head_dim]
#         q = torch.matmul(hflat, self.w_q).view(shp)
#         k = torch.matmul(hflat, self.w_k).view(shp)
#         v = torch.matmul(hflat, self.w_v).view(shp)

#         # compute attention coefficients
#         # [H x B x L x D] x [H x B x D x L] -> [H x B x L x L]
#         compatibility = self.norm_factor * torch.matmul(q, k.transpose(2, 3)) # dim. is the same as attn's one
#         attn = torch.softmax(compatibility, dim=-1) # [num_head x batch_size x seq_length x seq_length(attention coef.)]

#         # attention-based neighbor aggregation
#         # [H x B x L x L] x [H x B x L x H] -> [H x B x L x H]
#         heads = torch.matmul(attn, v) # [num_head x batch_size x seq_length x head_dim]

#         # aggregate heads
#         out = torch.mm(
#             heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.head_dim), # concat: [batch_size x seq_length x embed_dim]
#             self.w_o.view(-1, self.embed_dim) # [embed_dim x embed_dim]
#         ).view(batch_size, seq_length, self.embed_dim)

#         return out

# class Normalization(nn.Module):
#     def __init__(self, embed_dim, normalization='batch'):
#         super().__init__()

#         normalizer_class = {
#             'batch': nn.BatchNorm1d,
#             'instance': nn.InstanceNorm1d
#         }.get(normalization, None)

#         self.normalizer = normalizer_class(embed_dim, affine=True)

#         # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
#         # self.init_parameters()

#     def init_parameters(self):
#         for name, param in self.named_parameters():
#             stdv = 1. / math.sqrt(param.size(-1))
#             param.data.uniform_(-stdv, stdv)

#     def forward(self, input):
#         if isinstance(self.normalizer, nn.BatchNorm1d):
#             return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
#         elif isinstance(self.normalizer, nn.InstanceNorm1d):
#             return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
#         else:
#             assert self.normalizer is None, "Unknown normalizer type"
#             return input

# class MultiHeadAttentionLayer(nn.Sequential):
#     def __init__(self,
#                  n_heads,
#                  embed_dim,
#                  feed_forward_hidden=512,
#                  normalization='batch'):
#         super().__init__(
#             SkipConnection(
#                 MultiHeadAttention(
#                     n_heads,
#                     input_dim=embed_dim,
#                     embed_dim=embed_dim
#                 )
#             ),
#             Normalization(embed_dim, normalization),
#             SkipConnection(
#                 nn.Sequential(
#                     nn.Linear(embed_dim, feed_forward_hidden),
#                     nn.ReLU(),
#                     nn.Linear(feed_forward_hidden, embed_dim)
#                 ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
#             ),
#             Normalization(embed_dim, normalization)
#         )

# class AMEncoder(nn.Module):
#     def __init__(self,
#                  node_dim,
#                  emb_dim,
#                  n_heads,
#                  n_layers,
#                  normalization='batch',
#                  feed_forward_hidden=512):
#         super().__init__()

#         # To map input to embedding space
#         self.init_embed = nn.Linear(node_dim, emb_dim)
        
#         self.layers = nn.Sequential(*(
#             MultiHeadAttentionLayer(n_heads, emb_dim, feed_forward_hidden, normalization)
#             for _ in range(n_layers)
#         ))

#     def forward(self, x):
#         # Batch multiply to get initial embeddings of nodes
#         h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)
#         h = self.layers(h)
#         return h