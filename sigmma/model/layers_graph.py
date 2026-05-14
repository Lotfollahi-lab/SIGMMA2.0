# Re-implemented Sparse Structure Learning (SSL) module in dgl
# Original code from: https://github.com/qkrdmsghk/TextSSL/blob/master/ssl_graphmodels/pyg_models/layers_docs.py#L102, https://github.com/qkrdmsghk/TextSSL/blob/master/ssl_graphmodels/utils/SSL_GCN.py

import dgl
from sympy import galois_group
import torch
from torch.nn import Parameter

from typing import Union, Literal, Optional

import dgl


# =====================================================================
# Legacy implementations (per-graph Python loop, retained for unit-test
# equivalence checks against the vectorized versions below).
# =====================================================================

class StructureLearningLegacy(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 sparse: Literal['sparse', 'hard'],
                 emb_layer: int = 1,
                 activation: Literal['relu', 'sigmoid', 'leaky_relu'] = 'leaky_relu',
                 threshold: float = 0.5,
                 temperature: float = 0.5,
                 candidate_top_k: int = 8,
                 candidate_sim_threshold: float = 0.4):
        super().__init__()
        self.att = Parameter(torch.Tensor(1, input_dim * 2))

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
        self.candidate_top_k = candidate_top_k
        self.candidate_sim_threshold = candidate_sim_threshold

        torch.nn.init.xavier_uniform_(self.att)

    def edge_attention(self, edges):
        src, tgt = edges.src['learned_node_feat'], edges.dst['learned_node_feat']
        edge_weights = (torch.cat([src, tgt], 1) * self.att).sum(-1)
        edge_weights = self.activation(edge_weights)
        return {'learned_edge_weight': edge_weights}

    def _propose_candidates(self, gi, h, layer_id):
        n = gi.num_nodes()
        if n < 2 or self.candidate_top_k <= 0:
            empty = torch.empty(0, dtype=torch.long, device=h.device)
            return empty, empty
        BLOCK = 2 * (layer_id + 1)
        x_pos = gi.ndata['x_pos']
        y_pos = gi.ndata['y_pos']
        bx = torch.div(x_pos, BLOCK, rounding_mode='floor')
        by = torch.div(y_pos, BLOCK, rounding_mode='floor')
        M = int(max(bx.max().item(), by.max().item())) + 1
        block_key = bx * M + by
        h_norm = torch.nn.functional.normalize(h, dim=-1)
        sim = h_norm @ h_norm.t()
        same_block = block_key.unsqueeze(0) == block_key.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=h.device)
        sim = sim.masked_fill(~(same_block & ~eye), float('-inf'))
        k = min(self.candidate_top_k, n - 1)
        topk_vals, topk_idx = sim.topk(k, dim=-1)
        valid = torch.isfinite(topk_vals) & (topk_vals > self.candidate_sim_threshold)
        rows = torch.arange(n, device=h.device).unsqueeze(1).expand(-1, k)
        src = rows[valid]
        dst = topk_idx[valid]
        src_sym = torch.cat([src, dst])
        dst_sym = torch.cat([dst, src])
        if src_sym.numel() == 0:
            return src_sym, dst_sym
        existing = gi.has_edges_between(src_sym, dst_sym)
        keep = ~existing
        return src_sym[keep], dst_sym[keep]

    def forward(self, g, x, layer_id):
        with g.local_scope():
            edge_feat_specs = {"mask": -1, "weight": 0}
            g_unbatched = []
            node_offset = 0
            for gi in dgl.unbatch(g):
                n_nodes = gi.num_nodes()
                gi = dgl.remove_self_loop(gi)
                for feat, fill_val in edge_feat_specs.items():
                    gi = dgl.add_self_loop(
                        gi, edge_feat_names=[feat], fill_data=fill_val
                    )
                h_local = x[node_offset: node_offset + n_nodes]
                new_src, new_dst = self._propose_candidates(gi, h_local, layer_id)
                if new_src.numel() > 0:
                    n_new = new_src.numel()
                    gi.add_edges(new_src, new_dst, data={
                        'mask': torch.zeros(n_new, dtype=gi.edata['mask'].dtype, device=x.device),
                        'weight': torch.zeros(n_new, dtype=gi.edata['weight'].dtype, device=x.device),
                    })
                g_unbatched.append(gi)
                node_offset += n_nodes
            g = dgl.batch(g_unbatched)

            if self.emb_layer > 0:
                for i in range(self.emb_layer):
                    edge_mask = g.edata['mask'] != 0
                    g_intra = edge_subgraph_per_graph(g, edge_mask, store_ids=True)
                    x = self.gnns[i](g_intra, x)
                    x = self.activation(x)
            g.ndata['learned_node_feat'] = x

            g.apply_edges(self.edge_attention)
            g.edata['score_softmax'] = dgl.nn.functional.edge_softmax(g, g.edata['learned_edge_weight'])
            g.apply_edges(lambda edges: {'learned_edge_weight': edges.data['learned_edge_weight'] + edges.data['weight']})
            edges_soft = dgl.nn.functional.edge_softmax(g, g.edata['learned_edge_weight'])
            probs = edges_soft
            dist = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(self.temperature, probs=probs)
            edges_soft = dist.rsample()
            edges_hard = (edges_soft > self.threshold).float()
            y = (edges_hard - edges_soft).detach() + edges_soft

            inter_mask = (g.edata['weight'] == 0)
            u, v = g.edges()
            x_pos = g.ndata['x_pos']
            y_pos = g.ndata['y_pos']
            BLOCK = 2 * (layer_id + 1)
            bx = torch.div(x_pos, BLOCK, rounding_mode='floor')
            by = torch.div(y_pos, BLOCK, rounding_mode='floor')
            bx_u, bx_v = bx[u], bx[v]
            by_u, by_v = by[u], by[v]
            same_block = (bx_u == bx_v) & (by_u == by_v)
            non_neighbor_edge_mask = same_block
            g.edata['weight'][inter_mask & non_neighbor_edge_mask] = y[inter_mask & non_neighbor_edge_mask]
            edge_mask = g.edata['mask']
            g.edata['mask'][(edge_mask == 0) & (y > self.threshold) & non_neighbor_edge_mask] = layer_id + 1
            updated_weight = g.edata['weight'].clone()
            updated_mask = g.edata['mask'].clone()
        g.edata['weight'] = updated_weight
        g.edata['mask'] = updated_mask
        return g


class GRAPHLayerLegacy(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: Literal['relu', 'sigmoid', 'leaky_relu'] = 'leaky_relu',
                 dropout: float = 0,
                 bias: bool = True,
                 num_gnn_layers: int = 1,
                 threshold: float = 0.5,
                 temperature: float = 0.5,
                 candidate_top_k: int = 8,
                 candidate_sim_threshold: float = 0.4):
        super().__init__()
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()
        self.node_embedding = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim * 2),
            torch.nn.LayerNorm(input_dim * 2),
            self.activation,
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(input_dim * 2, output_dim),
            torch.nn.LayerNorm(output_dim),
            self.activation,
            torch.nn.Dropout(p=dropout),
        ])
        self.convs = torch.nn.ModuleList()
        self.num_gnn_layers = num_gnn_layers
        for _ in range(self.num_gnn_layers):
            self.convs.append(dgl.nn.SAGEConv(output_dim, output_dim, 'lstm'))
        self.sls = StructureLearningLegacy(
            output_dim, sparse='hard', emb_layer=1,
            threshold=threshold, temperature=temperature,
            candidate_top_k=candidate_top_k,
            candidate_sim_threshold=candidate_sim_threshold,
        )
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, g, h, layer_id):
        for layer in self.node_embedding:
            h = layer(h)
        g_unbatched = [dgl.add_self_loop(graph) for graph in dgl.unbatch(g)]
        g = dgl.batch(g_unbatched)
        self.kl_terms = []
        for i in range(self.num_gnn_layers):
            h = self.convs[i](g, h)
            h = self.activation(h)
        g = self.sls(g, h, layer_id=layer_id)
        kl = self.compute_kl_divergence(g)
        self.kl_terms.append(kl)
        return g, h

    def compute_kl_divergence(self, g, use_edge_norm=True):
        intra_mask = g.edata['mask'] == -1
        if use_edge_norm:
            raw_edge_intra_weight = torch.ones_like(g.edata['weight'][intra_mask])
            raw_edge_intra_weight = torch.nn.functional.softmax(raw_edge_intra_weight, dim=0)
        else:
            raw_edge_intra_weight = torch.ones_like(g.edata['weight'][intra_mask]) / intra_mask.sum()
        intra_soft_edge = g.edata['weight'][intra_mask]
        log_p = torch.log(raw_edge_intra_weight + 1e-12)
        log_q = torch.log(intra_soft_edge + 1e-12)
        kl = torch.mean(raw_edge_intra_weight * (log_p - log_q))
        return kl


# =====================================================================
# Vectorized implementations (default; replace the per-graph Python
# loop with batched torch + DGL ops). Produce a new batched DGLGraph
# constructed in graph-major edge order matching the Legacy layout
# (orig non-self / self-loop set 1 / self-loop set 2 / candidates) so
# downstream RelaxedBernoulli sampling order matches Legacy edge-for-
# edge.
# =====================================================================

class StructureLearning(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 sparse: Literal['sparse', 'hard'],
                 emb_layer: int = 1,
                 activation: Literal['relu', 'sigmoid', 'leaky_relu'] = 'leaky_relu',
                 threshold: float = 0.5,
                 temperature: float = 0.5,
                 candidate_top_k: int = 8,
                 candidate_sim_threshold: float = 0.4):
        super().__init__()
        self.att = Parameter(torch.Tensor(1, input_dim * 2))

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
        self.candidate_top_k = candidate_top_k
        self.candidate_sim_threshold = candidate_sim_threshold

        torch.nn.init.xavier_uniform_(self.att)

    def edge_attention(self, edges):
        src, tgt = edges.src['learned_node_feat'], edges.dst['learned_node_feat']
        edge_weights = (torch.cat([src, tgt], 1) * self.att).sum(-1)
        edge_weights = self.activation(edge_weights)
        return {'learned_edge_weight': edge_weights}

    def _propose_candidates_batched(self, g, h, layer_id, existing_src, existing_dst):
        """Vectorized candidate edge proposal over a batched DGL graph.

        Functionally equivalent to running the Legacy ``_propose_candidates``
        independently on every unbatched sub-graph, but executed as a single
        masked NxN similarity / top-K over the whole batched node set.

        Dedupe is done against the explicit ``existing_src`` / ``existing_dst``
        pair (typically the original non-self edges) via hashed integer IDs,
        instead of ``dgl.has_edges_between`` on an ad-hoc graph that would
        otherwise lose batch-info and corrupt the same-graph mask.
        """
        n_total = g.num_nodes()
        if n_total < 2 or self.candidate_top_k <= 0:
            empty = torch.empty(0, dtype=torch.long, device=h.device)
            return empty, empty

        device = h.device
        batch_num_nodes = g.batch_num_nodes()
        B = batch_num_nodes.numel()
        graph_id_per_node = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=batch_num_nodes.dtype),
            batch_num_nodes,
        )

        BLOCK = 2 * (layer_id + 1)
        x_pos = g.ndata['x_pos']
        y_pos = g.ndata['y_pos']
        bx = torch.div(x_pos, BLOCK, rounding_mode='floor')
        by = torch.div(y_pos, BLOCK, rounding_mode='floor')

        h_norm = torch.nn.functional.normalize(h, dim=-1)
        sim = h_norm @ h_norm.t()

        same_graph = graph_id_per_node.unsqueeze(0) == graph_id_per_node.unsqueeze(1)
        same_bx = bx.unsqueeze(0) == bx.unsqueeze(1)
        same_by = by.unsqueeze(0) == by.unsqueeze(1)
        eye = torch.eye(n_total, dtype=torch.bool, device=device)
        valid_mask = same_graph & same_bx & same_by & ~eye
        sim = sim.masked_fill(~valid_mask, float('-inf'))

        # k is bounded by min(candidate_top_k, n_total - 1) for matrix sanity;
        # per-row validity (#valid neighbours) is enforced by the finite +
        # threshold check below, which matches the Legacy per-graph behaviour
        # (Legacy used k = min(top_k, n_i - 1) per sub-graph; here -inf entries
        # are filtered out so rows in small sub-graphs naturally produce fewer
        # candidates).
        k = min(self.candidate_top_k, n_total - 1)
        topk_vals, topk_idx = sim.topk(k, dim=-1)
        valid = torch.isfinite(topk_vals) & (topk_vals > self.candidate_sim_threshold)

        rows = torch.arange(n_total, device=device).unsqueeze(1).expand(-1, k)
        src = rows[valid]
        dst = topk_idx[valid]

        src_sym = torch.cat([src, dst])
        dst_sym = torch.cat([dst, src])

        if src_sym.numel() == 0:
            return src_sym, dst_sym

        # Dedupe via hashed (src, dst) IDs. Candidates already exclude self-pairs
        # via the eye-mask, so existing self-loops would never collide with new
        # candidate edges; deduping against original non-self edges suffices to
        # match Legacy semantics.
        new_ids = src_sym.to(torch.int64) * n_total + dst_sym.to(torch.int64)
        existing_ids = existing_src.to(torch.int64) * n_total + existing_dst.to(torch.int64)
        existing_hit = torch.isin(new_ids, existing_ids)
        keep = ~existing_hit
        return src_sym[keep], dst_sym[keep]

    def _rebuild_batched_graph(self, g, h, layer_id):
        """Build a new batched graph whose edge tensor matches the per-graph
        Legacy layout exactly: for each graph i, edges appear as
            [orig_non_self_i, self_loop_mask=-1_i (n_i edges),
             self_loop_mask=0_i (n_i edges), candidates_i]
        so that downstream apply_edges / edge_softmax / RelaxedBernoulli sees
        edges in identical positions to the Legacy implementation.
        """
        device = h.device
        batch_num_nodes = g.batch_num_nodes()
        B = batch_num_nodes.numel()
        n_total = g.num_nodes()

        # ---- Category 0: original non-self edges.
        # NOTE: dgl.add_self_loop on a batched graph appends self-loops at the
        # END of the global edge tensor rather than interleaving them per-graph,
        # which makes g.batch_num_edges() (even after manual restore in
        # GRAPHLayer.forward) disagree with the edge tensor's per-graph slice
        # boundaries. We side-step that by deriving each edge's owning graph
        # from its src node's graph id (edges never cross sub-graphs), which is
        # always reliable.
        src_orig, dst_orig = g.edges()
        mask_orig = g.edata['mask']
        weight_orig = g.edata['weight']
        gid_per_node = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=batch_num_nodes.dtype),
            batch_num_nodes,
        )
        edge_graph_id_orig = gid_per_node[src_orig]
        not_self = src_orig != dst_orig
        src_c0 = src_orig[not_self]
        dst_c0 = dst_orig[not_self]
        mask_c0 = mask_orig[not_self]
        weight_c0 = weight_orig[not_self]
        gid_c0 = edge_graph_id_orig[not_self]

        # ---- Category 1: self-loops with mask=-1, weight=0
        # ---- Category 2: self-loops with mask=0, weight=0
        # (Legacy code calls dgl.add_self_loop twice with different edge_feat
        # fill values; this results in 2N self-loops per node count, half of
        # them carrying mask=-1 and half mask=0. Preserved verbatim.)
        arange_nodes = torch.arange(n_total, device=device, dtype=src_c0.dtype)
        # gid_per_node already defined above for Category 0.

        src_c1 = arange_nodes
        dst_c1 = arange_nodes
        mask_c1 = torch.full((n_total,), -1, dtype=mask_c0.dtype, device=device)
        weight_c1 = torch.zeros(n_total, dtype=weight_c0.dtype, device=device)
        gid_c1 = gid_per_node

        src_c2 = arange_nodes
        dst_c2 = arange_nodes
        mask_c2 = torch.zeros(n_total, dtype=mask_c0.dtype, device=device)
        weight_c2 = torch.zeros(n_total, dtype=weight_c0.dtype, device=device)
        gid_c2 = gid_per_node

        # ---- Category 3: candidate edges.
        # Propose using the ORIGINAL batched g (which carries valid batch info
        # for the same-graph mask) and dedupe against the original non-self
        # edges via hashed-ID isin. Self-loops never collide with candidates
        # (candidate proposal eye-masks self-pairs), so we don't need to feed
        # self-loops into the dedupe set.
        cand_src, cand_dst = self._propose_candidates_batched(
            g, h, layer_id, existing_src=src_c0, existing_dst=dst_c0,
        )
        n_cand = cand_src.numel()
        if n_cand > 0:
            mask_c3 = torch.zeros(n_cand, dtype=mask_c0.dtype, device=device)
            weight_c3 = torch.zeros(n_cand, dtype=weight_c0.dtype, device=device)
            gid_c3 = gid_per_node[cand_src]
        else:
            cand_src = cand_src.to(src_c0.dtype)
            cand_dst = cand_dst.to(dst_c0.dtype)
            mask_c3 = torch.empty(0, dtype=mask_c0.dtype, device=device)
            weight_c3 = torch.empty(0, dtype=weight_c0.dtype, device=device)
            gid_c3 = torch.empty(0, dtype=gid_c0.dtype, device=device)

        # ---- Concatenate categories then stable-sort by (graph_id, category).
        all_src = torch.cat([src_c0, src_c1, src_c2, cand_src])
        all_dst = torch.cat([dst_c0, dst_c1, dst_c2, cand_dst])
        all_mask = torch.cat([mask_c0, mask_c1, mask_c2, mask_c3])
        all_weight = torch.cat([weight_c0, weight_c1, weight_c2, weight_c3])
        all_gid = torch.cat([gid_c0, gid_c1, gid_c2, gid_c3])
        all_cat = torch.cat([
            torch.zeros_like(gid_c0),
            torch.ones_like(gid_c1),
            torch.full_like(gid_c2, 2),
            torch.full_like(gid_c3, 3),
        ])

        # Stable sort by (graph_id, category): max category < 4 so this key is
        # unique-per-position; argsort with stable=True preserves within-key
        # original order.
        sort_key = all_gid.to(torch.int64) * 4 + all_cat.to(torch.int64)
        sort_idx = torch.argsort(sort_key, stable=True)
        all_src = all_src[sort_idx]
        all_dst = all_dst[sort_idx]
        all_mask = all_mask[sort_idx]
        all_weight = all_weight[sort_idx]
        all_gid_sorted = all_gid[sort_idx]

        new_batch_num_edges = torch.bincount(all_gid_sorted, minlength=B).to(batch_num_nodes.dtype)

        g_new = dgl.graph((all_src, all_dst), num_nodes=n_total, idtype=g.idtype, device=device)
        for k_, v_ in g.ndata.items():
            g_new.ndata[k_] = v_
        g_new.edata['mask'] = all_mask
        g_new.edata['weight'] = all_weight
        g_new.set_batch_num_nodes(batch_num_nodes)
        g_new.set_batch_num_edges(new_batch_num_edges)
        return g_new

    def forward(self, g, x, layer_id):
        with g.local_scope():
            # Build the post-self-loop / post-candidate batched graph in one shot
            # (no per-graph Python loop).
            g = self._rebuild_batched_graph(g, x, layer_id)

            if self.emb_layer > 0:
                for i in range(self.emb_layer):
                    edge_mask = g.edata['mask'] != 0
                    # Use the per-graph helper to match Legacy edge ordering
                    # exactly. (dgl.edge_subgraph on batched + relabel_nodes
                    # behaves subtly differently w.r.t. edge ordering than the
                    # unbatch + per-graph + rebatch path, which would otherwise
                    # break bit-identical equivalence with Legacy.)
                    g_intra = edge_subgraph_per_graph(g, edge_mask, store_ids=True)
                    x = self.gnns[i](g_intra, x)
                    x = self.activation(x)
            g.ndata['learned_node_feat'] = x

            g.apply_edges(self.edge_attention)
            g.edata['score_softmax'] = dgl.nn.functional.edge_softmax(g, g.edata['learned_edge_weight'])
            g.apply_edges(lambda edges: {'learned_edge_weight': edges.data['learned_edge_weight'] + edges.data['weight']})
            edges_soft = dgl.nn.functional.edge_softmax(g, g.edata['learned_edge_weight'])
            probs = edges_soft
            dist = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(self.temperature, probs=probs)
            edges_soft = dist.rsample()
            edges_hard = (edges_soft > self.threshold).float()
            y = (edges_hard - edges_soft).detach() + edges_soft

            inter_mask = (g.edata['weight'] == 0)
            u, v = g.edges()
            x_pos = g.ndata['x_pos']
            y_pos = g.ndata['y_pos']
            BLOCK = 2 * (layer_id + 1)
            bx = torch.div(x_pos, BLOCK, rounding_mode='floor')
            by = torch.div(y_pos, BLOCK, rounding_mode='floor')
            bx_u, bx_v = bx[u], bx[v]
            by_u, by_v = by[u], by[v]
            same_block = (bx_u == bx_v) & (by_u == by_v)
            non_neighbor_edge_mask = same_block

            g.edata['weight'][inter_mask & non_neighbor_edge_mask] = y[inter_mask & non_neighbor_edge_mask]
            edge_mask = g.edata['mask']
            g.edata['mask'][(edge_mask == 0) & (y > self.threshold) & non_neighbor_edge_mask] = layer_id + 1

            updated_weight = g.edata['weight'].clone()
            updated_mask = g.edata['mask'].clone()
        g.edata['weight'] = updated_weight
        g.edata['mask'] = updated_mask
        return g


class GRAPHLayer(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: Literal['relu', 'sigmoid', 'leaky_relu'] = 'leaky_relu',
                 dropout: float = 0,
                 bias: bool = True,
                 num_gnn_layers: int = 1,
                 threshold: float = 0.5,
                 temperature: float = 0.5,
                 candidate_top_k: int = 8,
                 candidate_sim_threshold: float = 0.4):
        super().__init__()
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()

        self.node_embedding = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim * 2),
            torch.nn.LayerNorm(input_dim * 2),
            self.activation,
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(input_dim * 2, output_dim),
            torch.nn.LayerNorm(output_dim),
            self.activation,
            torch.nn.Dropout(p=dropout),
        ])

        self.convs = torch.nn.ModuleList()
        self.num_gnn_layers = num_gnn_layers
        for _ in range(self.num_gnn_layers):
            self.convs.append(dgl.nn.SAGEConv(output_dim, output_dim, 'lstm'))
        self.sls = StructureLearning(
            output_dim, sparse='hard', emb_layer=1,
            threshold=threshold, temperature=temperature,
            candidate_top_k=candidate_top_k,
            candidate_sim_threshold=candidate_sim_threshold,
        )
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, g, h, layer_id):
        for layer in self.node_embedding:
            h = layer(h)

        # Keep per-graph add_self_loop here (matches Legacy edge ordering for
        # the SAGEConv that follows). SAGEConv with the 'lstm' aggregator
        # consumes neighbours in their edge-tensor order, so a global-tail
        # batched add_self_loop would shift each node's self-loop relative to
        # its incoming neighbours and silently change the LSTM output. The
        # hot-spot vectorization lives inside StructureLearning below.
        g = dgl.batch([dgl.add_self_loop(gi) for gi in dgl.unbatch(g)])

        self.kl_terms = []
        for i in range(self.num_gnn_layers):
            h = self.convs[i](g, h)
            h = self.activation(h)
        g = self.sls(g, h, layer_id=layer_id)
        kl = self.compute_kl_divergence(g)
        self.kl_terms.append(kl)
        return g, h

    def compute_kl_divergence(self, g, use_edge_norm=True):
        intra_mask = g.edata['mask'] == -1
        if use_edge_norm:
            raw_edge_intra_weight = torch.ones_like(g.edata['weight'][intra_mask])
            raw_edge_intra_weight = torch.nn.functional.softmax(raw_edge_intra_weight, dim=0)
        else:
            raw_edge_intra_weight = torch.ones_like(g.edata['weight'][intra_mask]) / intra_mask.sum()
        intra_soft_edge = g.edata['weight'][intra_mask]
        log_p = torch.log(raw_edge_intra_weight + 1e-12)
        log_q = torch.log(intra_soft_edge + 1e-12)
        kl = torch.mean(raw_edge_intra_weight * (log_p - log_q))
        return kl


class BaseGRAPHLayer(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: Literal['relu', 'sigmoid', 'leaky_relu'] = 'leaky_relu',
                 dropout: float = 0,
                 num_gnn_layers: int = 1):
        super().__init__()
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()
        self.node_embedding = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim * 2),
            torch.nn.LayerNorm(input_dim * 2),
            self.activation,
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(input_dim * 2, output_dim),
            torch.nn.LayerNorm(output_dim),
            self.activation,
            torch.nn.Dropout(p=dropout),
        ])
        self.convs = torch.nn.ModuleList()
        self.num_gnn_layers = num_gnn_layers
        for _ in range(self.num_gnn_layers):
            self.convs.append(dgl.nn.SAGEConv(output_dim, output_dim, 'lstm'))

    def forward(self, g, h):
        for layer in self.node_embedding:
            h = layer(h)
        # See GRAPHLayer.forward: per-graph add_self_loop to keep SAGEConv
        # 'lstm' aggregation order identical to the original implementation.
        g = dgl.batch([dgl.add_self_loop(gi) for gi in dgl.unbatch(g)])
        for i in range(self.num_gnn_layers):
            h = self.convs[i](g, h)
            h = self.activation(h)
        return h


class READOUTLayer(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 aggregate: Literal['mean', 'max', 'add', 'attn_pool'] = 'max',
                 activation: Literal['relu', 'sigmoid', 'leaky_relu'] = 'leaky_relu',
                 dropout: float = 0):
        super().__init__()
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()
        self.node_embedding = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim * 2),
            torch.nn.LayerNorm(input_dim * 2),
            self.activation,
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(input_dim * 2, output_dim),
            torch.nn.LayerNorm(output_dim),
            self.activation,
            torch.nn.Dropout(p=dropout),
        ])
        if aggregate == 'mean':
            self.pool = dgl.nn.AvgPooling()
        elif aggregate == 'max':
            self.pool = dgl.nn.MaxPooling()
        elif aggregate == 'add':
            self.pool = dgl.nn.SumPooling()
        elif aggregate == 'attn_pool':
            gate_nn = torch.nn.Sequential(
                torch.nn.Linear(output_dim, output_dim),
                torch.nn.LayerNorm(output_dim),
                self.activation,
                torch.nn.Linear(output_dim, 1),
            )
            self.pool = dgl.nn.GlobalAttentionPooling(gate_nn)
        self.mlp = torch.nn.ModuleList([
            torch.nn.Linear(output_dim, output_dim),
            torch.nn.LayerNorm(output_dim),
            torch.nn.Dropout(p=dropout),
        ])

    def forward(self, g, h):
        for layer in self.node_embedding:
            h = layer(h)
        h = self.pool(g, h)
        for layer in self.mlp:
            h = layer(h)
        return h


def edge_subgraph_per_graph(bg, edge_mask_bool, store_ids=True):
    """Legacy helper used by StructureLearningLegacy. The vectorized
    StructureLearning uses dgl.edge_subgraph(..., relabel_nodes=False) directly
    on the batched graph instead.
    """
    assert edge_mask_bool.dtype == torch.bool and edge_mask_bool.numel() == bg.num_edges()
    graphs = dgl.unbatch(bg)
    e_sizes = bg.batch_num_edges().tolist()
    masks = torch.split(edge_mask_bool, e_sizes)
    subs = [dgl.edge_subgraph(g, m, store_ids=store_ids) for g, m in zip(graphs, masks)]
    return dgl.batch(subs)
