from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple, Union, Any

import numpy as np
import pandas as pd


def format_protein_string_to_list(protein_group: Any) -> List[str]:
    """
    Robustly parse a protein field into a list of protein accessions/IDs.

    Accepts:
      - list/tuple/set of proteins
      - a string with separators like ';' or ',' (common in spectral libraries)
      - a single string accession

    Notes
    -----
    We intentionally do NOT split on '|' because many accessions contain pipe-delimited
    fields (e.g., 'sp|P12345|PROT_HUMAN').
    """
    if protein_group is None:
        return []
    if isinstance(protein_group, (list, tuple, set)):
        return [str(x).strip() for x in protein_group if str(x).strip()]
    s = str(protein_group).strip()
    if not s or s.lower() == "nan":
        return []
    # normalize common separators
    s = s.replace(";;", ";").replace(",,", ",")
    if ";" in s:
        parts = [p.strip() for p in s.split(";")]
    elif "," in s:
        parts = [p.strip() for p in s.split(",")]
    elif "\t" in s:
        parts = [p.strip() for p in s.split("\t")]
    elif " " in s:
        # only split on spaces if it looks like a list rather than a descriptive name
        parts = [p.strip() for p in s.split(" ") if p.strip()]
    else:
        parts = [s]
    return [p for p in parts if p]


def identify_high_confidence_proteins(peptide_df: pd.DataFrame, protein_column: str = "protein") -> Set[Union[str, Tuple[str, ...]]]:
    """
    Run the IDPicker-style protein parsimony algorithm.

    Parameters
    ----------
    peptide_df:
      DataFrame with at least columns:
        - 'peptide'
        - protein_column (default 'protein'), possibly containing protein groups
    protein_column:
      name of the column containing proteins or protein groups.

    Returns
    -------
    leading_proteins:
      Set of protein nodes selected by parsimony. Nodes may be strings (single protein)
      or tuples (indistinguishable protein groups after collapsing).
    """
    edges = initialize__format_peptide_protein_connections(peptide_df, protein_column=protein_column)
    edges = collapse__group_identically_connected_peptides_and_proteins(edges)
    edges["cluster"] = separate__identify_and_label_independent_clusters(edges)
    return reduce__identify_minimum_number_of_most_connected_proteins(edges)


def initialize__format_peptide_protein_connections(peptide_df: pd.DataFrame, protein_column: str = "protein") -> pd.DataFrame:
    peptide_protein_connections: List[Tuple[str, str]] = []
    if len(peptide_df) == 0:
        return pd.DataFrame([], columns=["peptide", "protein"])
    for i in range(len(peptide_df)):
        peptide = str(peptide_df["peptide"].iloc[i])
        protein_group = peptide_df[protein_column].iloc[i]
        for protein in format_protein_string_to_list(protein_group):
            peptide_protein_connections.append((peptide, protein))
    return pd.DataFrame(peptide_protein_connections, columns=["peptide", "protein"])


def collapse__group_identically_connected_peptides_and_proteins(peptide_protein_connections_df: pd.DataFrame) -> pd.DataFrame:
    if len(peptide_protein_connections_df) == 0:
        return peptide_protein_connections_df.copy()
    df = peptide_protein_connections_df.copy()
    df["peptideGroup"] = group_nodes_by_identical_edges(df, is_peptide_nodes=True)
    df["proteinGroup"] = group_nodes_by_identical_edges(df, is_peptide_nodes=False)
    df = df[["peptideGroup", "proteinGroup"]].drop_duplicates(["peptideGroup", "proteinGroup"]).reset_index(drop=True)
    df.columns = ["peptide", "protein"]
    return df


def group_nodes_by_identical_edges(df: pd.DataFrame, is_peptide_nodes: bool):
    main_node = "peptide" if is_peptide_nodes else "protein"
    connected_node = "protein" if is_peptide_nodes else "peptide"
    temp = df.copy()
    # For each main node, capture its full set of connections
    temp["allConnections"] = df.groupby(main_node)[connected_node].transform(
        lambda x: [tuple(x.tolist())] * len(x)
    )
    # Nodes with identical connection sets get grouped together
    return temp.groupby("allConnections")[main_node].transform(
        lambda x: [tuple(sorted(set(map(str, x.tolist()))))] * len(x)
    )


def separate__identify_and_label_independent_clusters(peptide_protein_connections_df: pd.DataFrame) -> np.ndarray:
    if len(peptide_protein_connections_df) == 0:
        return np.array([], dtype=int)
    cluster_column = np.array([-1] * len(peptide_protein_connections_df.index))
    clusters = extract_clusters_from_dataframe(peptide_protein_connections_df)
    for cluster_num, cluster_idx in enumerate(clusters):
        cluster_column[cluster_idx] = cluster_num
    return cluster_column


def extract_clusters_from_dataframe(df: pd.DataFrame):
    df = df.copy()
    clusters = []
    while len(df.index) > 0:
        peptide_set, protein_set = initialize_peptide_protein_sets_of_next_cluster(df)
        peptide_set, _ = identify_next_cluster_in_dataframe_recursively(df, peptide_set, protein_set)
        cluster_df = df[df["peptide"].isin(peptide_set)]
        clusters.append(cluster_df.index)
        df = df[~df.index.isin(cluster_df.index)]
    return clusters


def initialize_peptide_protein_sets_of_next_cluster(df: pd.DataFrame):
    peptide_set = set([df.iloc[0]["peptide"]])
    protein_set = set([df.iloc[0]["protein"]])
    return peptide_set, protein_set


def identify_next_cluster_in_dataframe_recursively(df: pd.DataFrame, old_peptide_set: Set, old_protein_set: Set):
    new_peptide_set, new_protein_set = identify_all_matching_peptide_proteins_in_cluster_from_old_set(
        df, old_peptide_set, old_protein_set
    )
    if new_peptide_set == old_peptide_set and new_protein_set == old_protein_set:
        return old_peptide_set, new_peptide_set
    return identify_next_cluster_in_dataframe_recursively(df, new_peptide_set, new_protein_set)


def identify_all_matching_peptide_proteins_in_cluster_from_old_set(df: pd.DataFrame, peptide_set: Set, protein_set: Set):
    sub = df[df["peptide"].isin(peptide_set) | df["protein"].isin(protein_set)]
    return set(sub["peptide"]), set(sub["protein"])


def reduce__identify_minimum_number_of_most_connected_proteins(peptide_protein_connections_df: pd.DataFrame):
    if len(peptide_protein_connections_df) == 0:
        return set()
    leading_proteins = set()
    for _, cluster_df in peptide_protein_connections_df.groupby("cluster"):
        cluster_df = cluster_df.copy()
        cluster_df["originalProteinCount"] = label_proteins_by_original_protein_count_for_breaking_ties(cluster_df)
        sorted_cluster_df, accepted = initialize_protein_identification_recursion_parameters(cluster_df)
        leading_proteins.update(identify_acceptable_proteins_recursively(sorted_cluster_df, accepted))
    return leading_proteins


def initialize_protein_identification_recursion_parameters(cluster_df: pd.DataFrame):
    sorted_cluster_df = sort_dataframe_by_descending_protein_count(cluster_df)
    initial_accepted = set([sorted_cluster_df.iloc[0]["protein"]])
    return sorted_cluster_df, initial_accepted


def sort_dataframe_by_descending_protein_count(df: pd.DataFrame):
    return (
        df.assign(currentProteinCount=df.groupby("protein")["protein"].transform("count"))
        .sort_values(by=["currentProteinCount", "originalProteinCount", "protein"], ascending=[False, False, True])
        .drop(["currentProteinCount"], axis=1)
    )


def label_proteins_by_original_protein_count_for_breaking_ties(df: pd.DataFrame):
    return df.groupby("protein")["protein"].transform("count")


def identify_acceptable_proteins_recursively(sorted_cluster_df: pd.DataFrame, accepted_protein_set: Set):
    accepted_df = sorted_cluster_df[sorted_cluster_df["protein"].isin(accepted_protein_set)]
    unclaimed_df = sorted_cluster_df[~sorted_cluster_df["peptide"].isin(accepted_df["peptide"])]
    if len(unclaimed_df.index) == 0:
        return set(accepted_df["protein"])
    unclaimed_df = sort_dataframe_by_descending_protein_count(unclaimed_df)
    next_protein = unclaimed_df.iloc[0]["protein"]
    accepted_protein_set.add(next_protein)
    return identify_acceptable_proteins_recursively(sorted_cluster_df, accepted_protein_set)


def build_protein_table_from_peptides(
    peptide_df: pd.DataFrame,
    score_col: str = "finalScore",
    q_col: str = "q",
    protein_col: str = "protein",
    peptide_col: str = "peptide",
    intensity_cols: Sequence[str] = ("ionCount", "ionCountSumPSM"),
) -> pd.DataFrame:
    """
    Create a protein table using IDPicker parsimony on a peptide table.

    The input peptide_df is typically the peptide-best table (1 row per peptide) and
    ideally already filtered to high-confidence targets (e.g., q<=0.01, isDecoy==0).

    Output columns:
      - protein: protein accession or ';'-joined indistinguishable group
      - nPeptides: # unique peptides assigned to this leading protein node
      - peptides: ';'-joined list of peptides
      - bestPeptideScore: max peptide score among its peptides
      - minPeptideQ: min peptide q among its peptides (if q_col exists)
      - ionCountSum: sum of peptide intensities for the peptides assigned to the protein (if available)
      - ionCountSumPSM: sum of per-peptide summed intensities across all PSMs (if available)
    """
    if peptide_df is None or len(peptide_df) == 0:
        return pd.DataFrame(
            columns=["protein", "nPeptides", "peptides", "bestPeptideScore", "minPeptideQ", "ionCountSum", "ionCountSumPSM"]
        )

    # Run parsimony
    leading = identify_high_confidence_proteins(peptide_df, protein_column=protein_col)

    # Build peptide->protein edges (non-collapsed) for reporting
    rows = []
    for _, r in peptide_df.iterrows():
        pep = str(r[peptide_col])
        prot_group = r[protein_col]
        score = float(r.get(score_col, np.nan))
        qv = float(r.get(q_col, np.nan)) if q_col in peptide_df.columns else np.nan
        for prot in format_protein_string_to_list(prot_group):
            rows.append((pep, prot, score, qv))
    edge = pd.DataFrame(rows, columns=["peptide", "protein_raw", "peptideScore", "peptideQ"])

    # Pull per-peptide intensity columns (peptide_df should already be 1 row / peptide)
    intensity_cols = tuple([c for c in intensity_cols if c in peptide_df.columns])
    pep_intensity = {}
    if intensity_cols:
        tmp = peptide_df[[peptide_col] + list(intensity_cols)].drop_duplicates(peptide_col)
        tmp = tmp.set_index(peptide_col)
        for c in intensity_cols:
            pep_intensity[c] = tmp[c].to_dict()

    # Map leading protein nodes to their constituent raw proteins
    def node_to_string(node):
        if isinstance(node, tuple):
            return ";".join(map(str, node))
        return str(node)

    leading_nodes = [node_to_string(p) for p in leading]
    leading_set = set(leading_nodes)

    # For each leading node, assign peptides that are connected to ANY raw protein in that node.
    out_rows = []
    for node in leading:
        node_str = node_to_string(node)
        raw_prots = list(node) if isinstance(node, tuple) else [str(node)]
        sub = edge[edge["protein_raw"].isin(set(map(str, raw_prots)))]
        peps = sorted(set(sub["peptide"].tolist()))
        if not peps:
            # This can happen rarely due to collapsing; skip empty nodes.
            continue
        sub2 = sub[sub["peptide"].isin(peps)]
        best_score = float(np.nanmax(sub2["peptideScore"].to_numpy())) if len(sub2) else np.nan
        min_q = float(np.nanmin(sub2["peptideQ"].to_numpy())) if len(sub2) else np.nan

        # Sum intensities across the unique peptides assigned to this protein node
        ion_sum = np.nan
        ion_sum_psm = np.nan
        if intensity_cols:
            if "ionCount" in intensity_cols:
                ion_sum = float(
                    np.nansum([float(pep_intensity["ionCount"].get(p, 0.0) or 0.0) for p in peps])
                )
            if "ionCountSumPSM" in intensity_cols:
                ion_sum_psm = float(
                    np.nansum([float(pep_intensity["ionCountSumPSM"].get(p, 0.0) or 0.0) for p in peps])
                )

        out_rows.append(
            dict(
                protein=node_str,
                nPeptides=len(peps),
                peptides=";".join(peps),
                bestPeptideScore=best_score,
                minPeptideQ=min_q,
                ionCountSum=ion_sum,
                ionCountSumPSM=ion_sum_psm,
            )
        )

    prot_df = pd.DataFrame(out_rows)
    if len(prot_df) == 0:
        return pd.DataFrame(
            columns=["protein", "nPeptides", "peptides", "bestPeptideScore", "minPeptideQ", "ionCountSum", "ionCountSumPSM"]
        )
    prot_df = prot_df.sort_values(["nPeptides", "bestPeptideScore"], ascending=[False, False]).reset_index(drop=True)
    return prot_df
