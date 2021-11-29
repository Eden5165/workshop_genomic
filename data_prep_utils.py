import os
import pandas as pd

# Data preperation

SAMPLE_ID_FIELD = "SampleID"

def get_df(table_name, sep='\t', lineterminator='\n'):
    """
    param table_name: in ["beat_drug", "beat_rnaseq", "drug_mut_cor", "drug_mut_cor_lables", "tcga_mut", "tcga_rna"]
    param sep: Columns seperator characters. Optional, default='\t'
    param lineterminator: Rows seperator characters. Optional, default='\n'
    return: df for the requested table
    """
    table_fp = os.path.join(os.getcwd(), "medical_genomics_2021_data", table_name)
    return pd.read_csv(table_fp, sep=sep, lineterminator=lineterminator)


def filter_beat_by_tcga(beat_rna, tcga_rna):
    """
    param beat_rna tcga_rna: original data frames such that rows are genes
    return beat_rna_filtered: beat_rna df only with genes that appear in tcga_rna
    """
    beat_rna_gene = beat_rna.reset_index().rename(columns={"index": "gene"})
    tcga_rna_gene = tcga_rna.reset_index().rename(columns={"index": "gene"})
    beat_rna_filtered = beat_rna_gene[beat_rna_gene["gene"].isin(tcga_rna_gene["gene"])]
    beat_rna_filtered.set_index("gene", inplace=True)
    return beat_rna_filtered


def transpose_df(df):
    """
    param df: df in wich columns are samples
    return: trsanposed df such that first column named SampleID, rows are samples
    """
    return df.transpose().reset_index().rename(columns={"index": SAMPLE_ID_FIELD})

def get_data_reorgenized(genes_df, other_data_df):
    """
    param genes_df: genes df such that rows are samples
    param other_data_df: drugs or mutations df such that rows are sample
    return: other_data_df reoregenized such that row i represents the sample i that is
    represented by row i in genes_df
    """
    merged = pd.merge(genes_df, other_data_df, on=SAMPLE_ID_FIELD)
    first_other_field = len(genes_df.columns)
    last_other_field = len(merged.columns)
    data_reorg = merged.iloc[:, [0] + list(range(first_other_field, last_other_field))]
    return data_reorg

