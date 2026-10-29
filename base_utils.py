import pandas as pd
import numpy as np
import json
import pickle

def get_new_cols(incl_cols, ref_cols, sort=True):
    new_cols = [c for c in incl_cols if c in ref_cols]
    return sorted(new_cols) if sort else new_cols

def check_filename(fn, extension):
    return f'{fn}.{extension}' if (len(fn) <= len(f'.{extension}')) or (fn[-len(f'.{extension}'):] != f'.{extension}') else fn

def drop_useless_cols(df, drop_constant=False):
    new_df = df.dropna(axis=1, how='all')
    if drop_constant:
        const = [col for col in new_df.columns if new_df[col].fillna('na').nunique() == 1]
        new_df = new_df.drop(columns=const)
    return new_df

def load_data(kind, filename, data_path, clean=False, **kwargs):
    filename = check_filename(filename, {'list':'txt', 'json':'json', 'pd':'pkl', 'csv':'csv', 'raw txt':'txt', 'tsv':'tsv', 'pkl':'pkl'}[kind])
    if kind in ['list', 'json']:
        with open(data_path+filename, 'r') as fin:
            return json.load(fin) if kind == 'json' else [line.strip() for line in fin.readlines()]
    elif kind == 'pd':
        return pd.read_pickle(data_path+filename)
    elif kind in ['csv', 'raw txt', 'tsv']:
        loaded = pd.read_csv(data_path+filename, sep=',' if kind == 'csv' else '\t', na_values=['[Not Available]','[NOT AVAILABLE]','[Not Applicable]','Not Available','NOT AVAILABLE','nan','none','[Unknown]','Not reported','Not Reported','not reported','[Not Evaluated]','[Discrepancy]',"'--"], **kwargs)
        return drop_useless_cols(loaded) if clean else loaded
    elif kind == 'pkl':
        with open(data_path+filename, 'rb') as fin:
            return pickle.load(fin)

def write_file(kind, filename, data, data_path):
    filename = check_filename(filename, {'list':'txt', 'json':'json', 'pd':'pkl', 'csv':'csv', 'pkl': 'pkl'}[kind])
    if kind == 'list':
        with open(data_path+filename, 'w') as fout: fout.writelines('\n'.join(data))
    elif kind == 'json':
        with open(data_path+filename, 'w') as fout: json.dump(data, fout)
    elif kind == 'pd': 
        data.to_pickle(data_path+filename)
    elif kind == 'csv': 
        data.to_csv(data_path+filename)
    elif kind == 'pkl':
        with open(data_path+filename, 'wb') as fout: pickle.dump(data, fout)

def get_hgnc26():
    hgnc = load_data('json', 'HGNC_dicts.json', data_path='/gpfs/data/dgamsiz/jpatrico/HGNC2026/')
    for key in hgnc: # standardize gene names to uppercase
        hgnc[key] = {str(k).strip().upper(): str(v).strip().upper() for k, v in hgnc[key].items()}
    return hgnc

def limit_df(df, included):
    return df[df.index.isin(included)].dropna(axis=1, how='all').copy()

def set_idx(df, idx_col='sample_id', also_drop=None):
    also_drop = [] if also_drop is None else also_drop
    assert df[idx_col].is_unique, f"Duplicate index values in {idx_col}"
    df.index = df[idx_col].tolist()
    return df.drop(columns = [idx_col] + also_drop)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False