from __future__ import annotations

from typing import Any, Dict, List

import torch

from binding_affinity.types import BatchGraph


def _collate_graph(samples: List[Dict[str, Any]], *, key: str, default_is_ligand: bool | None = None) -> BatchGraph:
    z_list: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []
    edge_list: List[torch.Tensor] = []
    edge_attr_list: List[torch.Tensor] = []
    batch_list: List[torch.Tensor] = []
    is_ligand_list: List[torch.Tensor] = []
    fa_t_list: List[torch.Tensor] = []
    fa_R_list: List[torch.Tensor] = []
    atom_name_list: List[torch.Tensor] = []
    ligand_feat_list: List[torch.Tensor] = []

    has_atom_name = any("atom_name" in (s.get(key) or {}) for s in samples)
    has_ligand_feat = any("ligand_feat" in (s.get(key) or {}) for s in samples)
    ligand_feat_dim = None
    if has_ligand_feat:
        for s in samples:
            g = s[key]
            if "ligand_feat" in g:
                ligand_feat_dim = int(g["ligand_feat"].shape[1])
                break
        if ligand_feat_dim is None:
            has_ligand_feat = False

    node_offset = 0
    for graph_idx, s in enumerate(samples):
        g: Dict[str, Any] = s[key]
        z = g["z"].long()
        pos = g["pos"].float()
        edge_index = g["edge_index"].long()
        edge_attr = g.get("edge_attr", None)
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.shape[1],), dtype=torch.long)
        else:
            edge_attr = edge_attr.long().view(-1)
            if edge_attr.shape[0] != edge_index.shape[1]:
                raise ValueError(f"{key}.edge_attr must match number of edges: {edge_attr.shape[0]} vs {edge_index.shape[1]}")

        if edge_index.numel() > 0:
            edge_index = edge_index + node_offset
        node_offset += int(z.shape[0])

        z_list.append(z)
        pos_list.append(pos)
        edge_list.append(edge_index)
        edge_attr_list.append(edge_attr)
        batch_list.append(torch.full((z.shape[0],), int(graph_idx), dtype=torch.long))

        fa_t_list.append(g["fa_t"].float())
        fa_R_list.append(g["fa_R"].float())
        if has_atom_name:
            if "atom_name" in g:
                atom_name = g["atom_name"].long().view(-1)
                if atom_name.shape[0] != z.shape[0]:
                    raise ValueError(f"{key}.atom_name must match number of nodes: {atom_name.shape[0]} vs {z.shape[0]}")
            else:
                atom_name = torch.zeros((z.shape[0],), dtype=torch.long)
            atom_name_list.append(atom_name)
        if has_ligand_feat:
            assert ligand_feat_dim is not None
            if "ligand_feat" in g:
                ligand_feat = g["ligand_feat"].float()
                if ligand_feat.ndim != 2 or ligand_feat.shape[0] != z.shape[0] or ligand_feat.shape[1] != ligand_feat_dim:
                    raise ValueError(
                        f"{key}.ligand_feat must be (N,{ligand_feat_dim}): got {tuple(ligand_feat.shape)} for N={int(z.shape[0])}"
                    )
            else:
                ligand_feat = torch.zeros((z.shape[0], ligand_feat_dim), dtype=torch.float32)
            ligand_feat_list.append(ligand_feat)
        if "is_ligand" in g:
            is_ligand_list.append(g["is_ligand"].bool())
        elif default_is_ligand is not None:
            is_ligand_list.append(torch.full((z.shape[0],), bool(default_is_ligand), dtype=torch.bool))

    z_cat = torch.cat(z_list, dim=0) if z_list else torch.zeros((0,), dtype=torch.long)
    pos_cat = torch.cat(pos_list, dim=0) if pos_list else torch.zeros((0, 3), dtype=torch.float32)
    edge_cat = torch.cat(edge_list, dim=1) if edge_list else torch.zeros((2, 0), dtype=torch.long)
    edge_attr_cat = torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.zeros((0,), dtype=torch.long)
    batch_cat = torch.cat(batch_list, dim=0) if batch_list else torch.zeros((0,), dtype=torch.long)
    fa_t = torch.stack(fa_t_list, dim=0) if fa_t_list else torch.zeros((0, 3), dtype=torch.float32)
    fa_R = torch.stack(fa_R_list, dim=0) if fa_R_list else torch.zeros((0, 4, 3, 3), dtype=torch.float32)

    is_ligand = None
    if is_ligand_list:
        is_ligand = torch.cat(is_ligand_list, dim=0)

    atom_name = None
    if atom_name_list:
        atom_name = torch.cat(atom_name_list, dim=0)

    ligand_feat = None
    if ligand_feat_list:
        ligand_feat = torch.cat(ligand_feat_list, dim=0)

    return BatchGraph(
        z=z_cat,
        pos=pos_cat,
        edge_index=edge_cat,
        edge_attr=edge_attr_cat,
        batch=batch_cat,
        fa_t=fa_t,
        fa_R=fa_R,
        is_ligand=is_ligand,
        atom_name=atom_name,
        ligand_feat=ligand_feat,
    )


def collate_lba_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    complex_g = _collate_graph(samples, key="complex")
    protein_g = _collate_graph(samples, key="protein", default_is_ligand=False)
    ligand_g = _collate_graph(samples, key="ligand", default_is_ligand=True)

    # IMPORTANT: to preserve ΔE additivity, protein/ligand must share the same
    # frame-averaging (fa_t/fa_R) reference as the complex graph. Otherwise,
    # E_c(with cross edges removed) != E_p + E_l even for the same model.
    protein_g = BatchGraph(
        z=protein_g.z,
        pos=protein_g.pos,
        edge_index=protein_g.edge_index,
        batch=protein_g.batch,
        fa_t=complex_g.fa_t,
        fa_R=complex_g.fa_R,
        edge_attr=protein_g.edge_attr,
        is_ligand=protein_g.is_ligand,
        atom_name=protein_g.atom_name,
        ligand_feat=protein_g.ligand_feat,
    )
    ligand_g = BatchGraph(
        z=ligand_g.z,
        pos=ligand_g.pos,
        edge_index=ligand_g.edge_index,
        batch=ligand_g.batch,
        fa_t=complex_g.fa_t,
        fa_R=complex_g.fa_R,
        edge_attr=ligand_g.edge_attr,
        is_ligand=ligand_g.is_ligand,
        atom_name=ligand_g.atom_name,
        ligand_feat=ligand_g.ligand_feat,
    )

    batch: Dict[str, Any] = {
        "complex": complex_g,
        "protein": protein_g,
        "ligand": ligand_g,
        "y": torch.tensor([float(s["y"]) for s in samples], dtype=torch.float32),
        "id": [str(s["id"]) for s in samples],
    }
    return batch
