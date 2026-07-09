import dgl 
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata

def dgl_to_anndata(g: dgl.DGLGraph, prefix, X=None):
    N = g.num_nodes()

    if X is None:
        X = np.zeros((N, 1))  # dummy

    # edges
    src, dst = g.edges()
    src = src.numpy()
    dst = dst.numpy()

    # adjacency matrix
    adj = sp.coo_matrix(
        (np.ones(len(src)), (src, dst)),
        shape=(N, N)
    ).tocsr()

    adata = anndata.AnnData(X=X)
    adata.obsp[f"{prefix}_connectivities"] = adj

    # edge table
    edge_dict = {
        "src": src,
        "dst": dst,
    }

    for k, v in g.edata.items():
        edge_dict[k] = v.cpu().numpy()

    adata.uns["edges"] = pd.DataFrame(edge_dict)
    
    return adata


def add_graph_edge_to_adata(adata,
                            batched_graphs,
                            mask_key="mask",
                            obsp_key="sigmma_connectivities"):
    adata = adata.copy()
    N = adata.n_obs

    rows, cols, data = [], [], []

    for batch in batched_graphs:
        graphs = dgl.unbatch(batch)
        for g in graphs:
            
            src, dst = g.edges(order="eid")

            mask = g.edata[mask_key]
            keep = (mask != 0)

            src_k = src[keep]
            dst_k = dst[keep]
            if src_k.numel() == 0:
                continue

            cell_id = g.ndata["cell_id"]
            gi = cell_id[src_k].detach().cpu().numpy()
            gj = cell_id[dst_k].detach().cpu().numpy()

            rows.append(gi)
            cols.append(gj)
            data.append(np.ones(len(gi), dtype=np.float32))

    if len(rows) == 0:
        A = sp.csr_matrix((N, N), dtype=np.float32)
    else:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        assert rows.min() >= 0 and cols.min() >= 0
        assert rows.max() < N and cols.max() < N

        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        A.data[:] = 1.0
        A = A.maximum(A.T)

    adata.obsp[obsp_key] = A
    return adata