# Re-implemented Sparse Structure Learning (SSL) module in dgl
# Original code from: https://github.com/qkrdmsghk/TextSSL/blob/master/ssl_graphmodels/pyg_models/layers_docs.py#L102, https://github.com/qkrdmsghk/TextSSL/blob/master/ssl_graphmodels/utils/SSL_GCN.py

import dgl
from sympy import galois_group
import torch
from torch.nn import Parameter

from typing import Union, Optional

import dgl

class StructureLearning(torch.nn.Module):
    def __init__(self, 
                 input_dim:int, 
                 sparse:Union['sparse','hard'], 
                 emb_layer:int=1, 
                 activation: Union['relu','sigmoid','leaky_relu']='leaky_relu',
                 threshold:float=0.5, 
                 temperature:float=0.5):
        super(StructureLearning, self).__init__()
        self.att = Parameter(torch.Tensor(1, input_dim*2))
        torch.nn.init.xavier_uniform_(self.att)

        self.emb_layer = emb_layer
        if emb_layer > 0:
            self.gnns = torch.nn.ModuleList()
            for _ in range(emb_layer):
                self.gnns.append(dgl.nn.GraphConv(input_dim, input_dim))

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()

        self.sparse = sparse
        self.threshold = threshold
        self.temperature = temperature

    def edge_attention(self, edges):
        src, tgt = edges.src['learned_node_feat'], edges.dst['learned_node_feat']
        edge_weights = (torch.cat([src, tgt], 1) * self.att).sum(-1) 
        edge_weights = self.activation(edge_weights)
        return {'learned_edge_weight': edge_weights}

    def forward(self,
                g: dgl.graph,
                x: torch.Tensor,
                layer):
        """
        g: DGLGraph
        x: Node features (e.g., cell embeddings)
        pos_enc: Positional encodings (e.g., spatial coordinates
        layer: Current layer index (for edge mask updates)
        """

        # Edge weight: 1 (intra-subgraph edges) or 0 (inter-subgraph edges) or 1 (inter-subgraph edges rescued in this layer)
        # Edge mask: -1 (intra-subgraph edges) or 0 (inter-subgraph edges) or layer+1 (inter-subgraph edges rescued in this layer)
        with g.local_scope():
            print("the number of nodes/edges before adding self-loops:")
            print(g.num_nodes(), g.num_edges())
            # Add self-loops
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g, edge_feat_names=['mask'], fill_data=-1)
            g = dgl.add_self_loop(g, edge_feat_names=['weight'], fill_data=0)
            print(g.num_nodes(), g.num_edges())
            # GGN on intra-subgraph edges only
            if self.emb_layer > 0:
                for i in range(self.emb_layer):
                    # Message passing within intra-subgraph edges + learned edges only
                    edge_mask = g.edata['mask'] != 0
                    edge_ids = edge_mask.nonzero(as_tuple=True)[0]
                    g_intra = dgl.edge_subgraph(g, edge_ids, store_ids=True)
                    x = self.gnns[i](g_intra, x)
                    x = self.activation(x)
            g.ndata['learned_node_feat'] = x

            # Make edge weights learnable in dgl
            g.apply_edges(self.edge_attention)

            # Sparse structure learning (if sparse, apply softmax and if hard, apply gumbel softmax)
            g.edata['score_softmax'] = dgl.nn.functional.edge_softmax(g, g.edata['learned_edge_weight'])  # grouped by col (dst)
            g.apply_edges(lambda edges: {'learned_edge_weight': edges.data['learned_edge_weight']+ edges.data['weight']})

            edges_soft = dgl.nn.functional.edge_softmax(g, g.edata['learned_edge_weight'])

            # Sampling edges with Gumbel-softmax
            probs = edges_soft

            dist = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(self.temperature, probs=probs)
            edges_soft = dist.rsample()
            edges_hard = (edges_soft > self.threshold).float()
            y = (edges_hard - edges_soft).detach() + edges_soft  # Straight-through
            print('------------')

            # TODO: confine edge addition to neighboring subgraphs only
            # Update inter-subgraph edges
            inter_mask = (g.edata['weight'] == 0)
            g.edata['weight'][inter_mask] = y[inter_mask]
            print(f"the number of {layer}th-step rescued edges: {g.edata['weight'].sum()} / {g.num_edges()}")
            # Update edge mask
            edge_mask = g.edata['mask']
            g.edata['mask'][(edge_mask == 0) & (y > self.threshold)] = layer + 1
            print(f"the number of {layer}th-step rescued edges: {(g.edata['mask']!=0).sum()} / {g.num_edges()}")
 
            updated_weight = g.edata['weight'].clone()
            updated_mask = g.edata['mask'].clone()

        g.edata['weight'] = updated_weight
        g.edata['mask'] = updated_mask
        print(f"the number of cumulative rescued edges: {(g.edata['mask']!=0).sum()} / {g.num_edges()}")
        return g
    
class GRAPHLayer(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: Union['relu','sigmoid','leaky_relu'] = 'leaky_relu',
                 dropout: float = 0,
                 bias: bool = True,
                 num_layers: int = 1,
                 threshold: float = 0.5,
                 temperature: float = 0.5,
                 ):
        super(GRAPHLayer, self).__init__()
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()

        # Graph node embedding layer w/o message passing
        self.node_embedding = torch.nn.ModuleList([torch.nn.Dropout(p=dropout), torch.nn.Linear(input_dim, output_dim), self.activation, torch.nn.Dropout(p=dropout)])

        # Initialize the GCN layers and structure learning layers
        self.convs = torch.nn.ModuleList()
        self.sls = torch.nn.ModuleList()
        self.num_layers = num_layers
        for _ in range(self.num_layers):
            self.convs.append(dgl.nn.GraphConv(output_dim, output_dim))
            self.sls.append(StructureLearning(output_dim, sparse='hard', emb_layer=1, threshold=threshold, temperature=temperature))
        self.threshold = threshold
        self.temperature = temperature
        self.weight = Parameter(torch.Tensor(output_dim, output_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        # TODO
        return
        
    def forward(self, 
                g: dgl.graph, 
                h: torch.Tensor,
                ):

        for layer in self.node_embedding:
            h = layer(h)

        # Apply GCN/structure learning layers
        g = dgl.add_self_loop(g)

        self.kl_terms = []  # Store KL divergence terms for each forward pass
        for i in range(self.num_layers):
            h = self.convs[i](g, h)
            h = self.activation(h)
            g = self.sls[i](g, h, layer=i)
            # Regularization
            kl = self.compute_kl_divergence(g)
            self.kl_terms.append(kl)

        return h
    
    def compute_kl_divergence(self, g, use_edge_norm=True):
        """
        g: DGLGraph with edge fields:
            - 'weight': sampling result (e.g., y_soft)
            - 'mask': edge_mask (-1: intra/self-loop)
        """
        # 1. intra-edge mask
        intra_mask = g.edata['mask'] == -1

        # 2. p (reference distribution)
        if use_edge_norm:
            raw_edge_intra_weight = torch.ones_like(g.edata['weight'][intra_mask])
            raw_edge_intra_weight = torch.nn.functional.softmax(raw_edge_intra_weight, dim=0)  # uniform or soft
        else:
            raw_edge_intra_weight = torch.ones_like(g.edata['weight'][intra_mask]) / intra_mask.sum()

        # 3. q (sampled soft edge weight)
        intra_soft_edge = g.edata['weight'][intra_mask]

        # 4. KL(p‖q)
        log_p = torch.log(raw_edge_intra_weight + 1e-12)
        log_q = torch.log(intra_soft_edge + 1e-12)
        kl = torch.mean(raw_edge_intra_weight * (log_p - log_q))

        return kl
    
class READOUTLayer(torch.nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 aggregate: Union['mean','max','add']= 'max', 
                 activation: Union['relu', 'sigmoid', 'leaky_relu'] = 'leaky_relu',
                 dropout: float = 0,
                 ):
        super(READOUTLayer, self).__init__()
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()

        self.node_embedding = torch.nn.ModuleList([torch.nn.Dropout(p=dropout), torch.nn.Linear(input_dim, output_dim), self.activation, torch.nn.Dropout(p=dropout)])

        if aggregate == 'mean':
            self.pool = dgl.nn.AvgPooling()
        elif aggregate == 'max':
            self.pool = dgl.nn.MaxPooling()
        elif aggregate == 'add':
            self.pool = dgl.nn.SumPooling()

        self.mlp = torch.nn.ModuleList([torch.nn.Linear(input_dim, output_dim),
                                        torch.nn.BatchNorm1d(output_dim),
                                        self.activation,
                                        torch.nn.Dropout(p=dropout),])
        
    def forward(self, g, h):
        # Embedding layer
        for layer in self.node_embedding:
            h = layer(h)

        # Pooling on graph
        h = self.pool(g, h)

        # MLP
        for layer in self.mlp:
            h = layer(h)
        return h