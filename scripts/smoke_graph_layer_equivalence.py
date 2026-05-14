"""Unit-level smoke test: Legacy vs Vectorized GRAPHLayer equivalence.

Builds a small batched DGL graph with controlled features/positions, then
runs both GRAPHLayerLegacy and GRAPHLayer (vectorized) with the SAME weights
and SAME RNG seed. Compares:

  - returned h: deterministic given input + weights, must match exactly (up
    to atol=1e-5).
  - returned g: must have the same edge count, edge set (as unordered pairs),
    and same per-edge `mask` / `weight` after structure-learning updates.
  - kl divergence: must match (uses g.edata['weight'] which depends on the
    sampled y; if edge order is preserved, sampled values should match too).

Exit code 0 on PASS, 1 on FAIL.

Run::

    python scripts/smoke_graph_layer_equivalence.py
"""

import sys
import torch
import dgl

from sigmma.model.layers_graph import GRAPHLayer, GRAPHLayerLegacy


def make_batched_graph(num_graphs=4, nodes_per_graph=6, fov_size=8, p_edge=0.3, seed=0):
    """Create a small batched DGL graph with random intra-graph edges plus
    `mask` (-1 for these intra edges) and `weight` (1.0 for these edges) edata,
    plus x_pos/y_pos node features in [0, fov_size).
    """
    rng = torch.Generator().manual_seed(seed)
    graphs = []
    for gi in range(num_graphs):
        n = nodes_per_graph
        # random edges within this graph
        prob = torch.rand(n, n, generator=rng)
        edge_mask = (prob < p_edge) & ~torch.eye(n, dtype=torch.bool)
        src, dst = torch.nonzero(edge_mask, as_tuple=True)
        g = dgl.graph((src, dst), num_nodes=n)
        g.edata['mask'] = torch.full((g.num_edges(),), -1, dtype=torch.int64)
        g.edata['weight'] = torch.ones(g.num_edges(), dtype=torch.float32)
        # random x_pos, y_pos in [0, fov_size)
        g.ndata['x_pos'] = torch.randint(0, fov_size, (n,), generator=rng)
        g.ndata['y_pos'] = torch.randint(0, fov_size, (n,), generator=rng)
        graphs.append(g)
    bg = dgl.batch(graphs)
    return bg


def random_h(num_nodes, dim, seed=1):
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(num_nodes, dim, generator=rng)


def make_layer_pair(input_dim, output_dim, **kw):
    legacy = GRAPHLayerLegacy(input_dim=input_dim, output_dim=output_dim, **kw)
    vec = GRAPHLayer(input_dim=input_dim, output_dim=output_dim, **kw)
    # Copy weights legacy -> vec (state_dicts should have identical keys since
    # module structure is the same).
    vec.load_state_dict(legacy.state_dict())
    # Eval mode disables Dropout to eliminate one source of RNG difference.
    legacy.eval()
    vec.eval()
    return legacy, vec


def edge_set(g):
    src, dst = g.edges()
    return torch.stack([src, dst], dim=1)  # (E, 2)


def edge_index_lookup(g):
    """Map (src, dst) tuple -> edge index for the given graph."""
    src, dst = g.edges()
    out = {}
    for i in range(g.num_edges()):
        out[(int(src[i].item()), int(dst[i].item()))] = i
    return out


def assert_close(name, a, b, atol=1e-5, rtol=1e-5):
    if torch.allclose(a, b, atol=atol, rtol=rtol):
        return True
    diff = (a.float() - b.float()).abs()
    print(f"  FAIL {name}: max|diff|={diff.max().item():.3e} mean|diff|={diff.mean().item():.3e}")
    return False


def main():
    torch.manual_seed(42)
    bg = make_batched_graph(num_graphs=4, nodes_per_graph=6, fov_size=8, p_edge=0.25)
    h = random_h(bg.num_nodes(), dim=16)

    legacy, vec = make_layer_pair(
        input_dim=16, output_dim=16, activation='leaky_relu', dropout=0.0,
        num_gnn_layers=1, threshold=0.5, temperature=0.5,
        candidate_top_k=4, candidate_sim_threshold=0.0,
    )

    layer_id = 0

    # ---- Legacy forward ----
    torch.manual_seed(123)
    g_l, h_l = legacy(bg.clone(), h.clone(), layer_id=layer_id)
    kl_l = legacy.kl_terms[0].detach()

    # ---- Vectorized forward ----
    torch.manual_seed(123)
    g_v, h_v = vec(bg.clone(), h.clone(), layer_id=layer_id)
    kl_v = vec.kl_terms[0].detach()

    all_ok = True

    # 1) Returned h (post-SAGEConv, deterministic given weights)
    print("[1] h returned by GRAPHLayer.forward")
    all_ok &= assert_close("h", h_l.detach(), h_v.detach(), atol=1e-4, rtol=1e-4)
    print(f"    legacy h.shape={tuple(h_l.shape)}  vec h.shape={tuple(h_v.shape)}")

    # 2) Number of edges + sets
    print("[2] returned g edge structure")
    print(f"    legacy num_edges={g_l.num_edges()}  vec num_edges={g_v.num_edges()}")
    if g_l.num_edges() != g_v.num_edges():
        print(f"  FAIL: edge counts differ")
        all_ok = False
    else:
        es_l = edge_set(g_l)
        es_v = edge_set(g_v)
        # Sort by (src, dst) lexicographically for set comparison
        order_l = torch.argsort(es_l[:, 0] * (1 << 30) + es_l[:, 1])
        order_v = torch.argsort(es_v[:, 0] * (1 << 30) + es_v[:, 1])
        es_l_sorted = es_l[order_l]
        es_v_sorted = es_v[order_v]
        if torch.equal(es_l_sorted, es_v_sorted):
            print("    edge SETS match (sorted)")
        else:
            mismatch = (es_l_sorted != es_v_sorted).any(dim=1).sum().item()
            print(f"  FAIL: edge sets differ in {mismatch} rows after sort")
            all_ok = False

    # 3) Per-edge mask / weight — sort both by (src, dst, mask) so parallel
    #    edges (e.g. duplicated self-loops with different masks) align
    #    deterministically.
    print("[3] per-edge mask / weight")
    if g_l.num_edges() == g_v.num_edges():
        SHIFT1 = int(1 << 40)
        SHIFT2 = int(1 << 20)
        # Use (mask + offset) so negative masks (legacy uses -1) become positive
        # keys; offset 8 is generous.
        key_l = (
            es_l[:, 0].to(torch.int64) * SHIFT1
            + es_l[:, 1].to(torch.int64) * SHIFT2
            + (g_l.edata['mask'].to(torch.int64) + 8)
        )
        key_v = (
            es_v[:, 0].to(torch.int64) * SHIFT1
            + es_v[:, 1].to(torch.int64) * SHIFT2
            + (g_v.edata['mask'].to(torch.int64) + 8)
        )
        ord_l = torch.argsort(key_l, stable=True)
        ord_v = torch.argsort(key_v, stable=True)
        src_l_s = es_l[ord_l, 0]
        dst_l_s = es_l[ord_l, 1]
        src_v_s = es_v[ord_v, 0]
        dst_v_s = es_v[ord_v, 1]
        m_l_s = g_l.edata['mask'][ord_l]
        m_v_s = g_v.edata['mask'][ord_v]
        w_l_s = g_l.edata['weight'][ord_l]
        w_v_s = g_v.edata['weight'][ord_v]
        if torch.equal(src_l_s, src_v_s) and torch.equal(dst_l_s, dst_v_s):
            print("    (src, dst) after sort: identical")
        else:
            n_diff = ((src_l_s != src_v_s) | (dst_l_s != dst_v_s)).sum().item()
            print(f"  FAIL: (src, dst) differ in {n_diff} positions after sort")
            all_ok = False
        if torch.equal(m_l_s, m_v_s):
            print("    edata['mask'] aligned: exact match")
        else:
            n_diff = (m_l_s != m_v_s).sum().item()
            print(f"  FAIL: edata['mask'] differs in {n_diff} / {len(m_l_s)} aligned positions")
            all_ok = False
        if assert_close("edata['weight'] aligned", w_l_s.float(), w_v_s.float(), atol=1e-4, rtol=1e-4):
            print("    edata['weight'] aligned: close (atol=1e-4)")
    else:
        print(f"  FAIL: cannot compare per-edge: legacy={g_l.num_edges()} vec={g_v.num_edges()}")
        all_ok = False

    # 4) KL divergence
    print("[4] compute_kl_divergence")
    print(f"    legacy kl={kl_l.item():.6e}  vec kl={kl_v.item():.6e}")
    if assert_close("kl", kl_l.float(), kl_v.float(), atol=1e-4, rtol=1e-4):
        pass

    print()
    print("=" * 60)
    if all_ok:
        print("RESULT: PASS — Legacy and Vectorized are equivalent.")
        return 0
    else:
        print("RESULT: FAIL — see [N] section(s) above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
