import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import importlib
from pandas.api.types import is_numeric_dtype
import importlib
from decimal import Decimal, ROUND_HALF_UP

import base_utils; importlib.reload(base_utils)
from base_utils import load_data, write_file, set_idx

def get_data(filename, dp, lower=False, clean=True, **kwargs):
    kind = {'csv':'csv', 'txt':'raw txt', 'tsv':'tsv'}[filename.split('.')[1]] if '.' in filename else 'raw txt'
    df = load_data(kind, filename, dp, clean=clean, comment='#', **kwargs)
    if lower: df.columns = [col.lower() for col in df.columns]
    return df

def does_it_add_value(df, base_col, supp_col):
    missing_in_base = df.loc[df[base_col].isna()].index.tolist()
    included_in_supp = df.loc[df[supp_col].notna()].index.tolist()
    return not set(missing_in_base).isdisjoint(included_in_supp)

def column_agreement(df, col1, col2):
    no_nulls = df.loc[(df[col1].notna())&(df[col2].notna())]
    return bool((no_nulls[col1] == no_nulls[col2]).all())

def fill_from_backup_cols(df, target, backups, drop_backups=True, override_agreement=False):
    column_pairs = list(combinations([target] + backups, 2))
    if not override_agreement:
        for pair in column_pairs:
            assert column_agreement(df, pair[0], pair[1]), f"Column Disagreement between {pair[0]} and {pair[1]}"
    work = df[[target] + backups].copy()
    filled_target = work.bfill(axis=1).iloc[:, 0] # Take first non-null across [target] + backups (row-wise)
    df[target] = filled_target
    return df.drop(columns=backups).copy() if drop_backups else df.copy()

MAF_KEEP = {
    'Missense_Mutation', 'Nonsense_Mutation', 'Nonstop_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins',
    'Translation_Start_Site', 'Splice_Site',
}
ANNOVAR_EXONIC_KEEP = {
    'nonsynonymous snv', 'nonsynonymous', 'stopgain snv', 'immediate-stopgain', 'stoploss snv', 'frameshift substitution', 'frameshift insertion',
    'nonframeshift substitution', 'nonframeshift insertion',
}
ANNOVAR_REGION_KEEP = {'exonic', 'splicing', 'exonic;splicing'}

# root = '../data'
root = 'data'

class TCGA: # window_positives must be False, otherwise there's an issue with the ttr follow-up for censored
    def __init__(self, ignore, hgnc, cutoff, window=1825, req_NT=False, window_positives=False, print_excl_numbers=False, harmonize_mut=True, keep_censored=False, min_censor_days=30):
        """ keep_censored: retain recurrence-free patients whose follow-up stops before `window` instead of dropping them. They get label_any = label_early = <NA> (there is no determinate 
        answer for them) but a usable fu_time/event_obs pair, so a time-to-event loss can train on them. Default False reproduces the previous cohort exactly. min_censor_days: censored 
        patients with less follow-up than this are still dropped """
        self.name = 'tcga'
        self.paths = dict(fh=f'{root}/firehose/', pc=f'{root}/pancan/', xena=f'{root}/xena/', combined_xena=f'{root}/combined_xena/', gdc=f'{root}/gdc/')
        self.ignore, self.hgnc, self.cutoff, self.window, self.req_NT = ignore, hgnc, cutoff, window, req_NT
        self.window_positives, self.print_excl_numbers, self.harmonize = window_positives, print_excl_numbers, harmonize_mut
        self.keep_censored, self.min_censor_days = keep_censored, min_censor_days
        self.exclude, self.id_tracker = set(), {}
        self.explicit_cases = defaultdict(set)
        self.get_clinical()
        self.get_genomic()
        self.finalize_data()

    def get_clinical(self):
        self.xena_clin = self.load_and_prep_xena('_1', src='xena')
        self.combined_xena = self.load_and_prep_xena('_7', src='combined_xena')
        self.pc_clin = self.load_and_prep_cbio('_2', src='pc')
        self.fh_clin = self.load_and_prep_cbio('_3', src='fh')
        self.cdr = self.load_and_prep_gdc('_4', key='cdr')
        self.cfu = self.load_and_prep_gdc('_5', key='followup')
        self.gdc_clin = self.load_and_prep_gdc('_6', key='luad_gdc_phenotypes')
        self.clin = pd.concat([self.xena_clin, self.pc_clin, self.fh_clin, self.cdr, self.cfu, self.gdc_clin, self.combined_xena], axis=1, join='outer').dropna(axis=1, how='all')
        self.clin = self.clin.loc[:, ~self.clin.columns.str.startswith(('_PANCAN', '_INTEGRATION', 'submitter_id', 'entity_submitter_id', 'sample_type', 'SAMPLE_TYPE'))].copy()
        self.clin = self.clin[sorted(self.clin.columns, key=str.lower)]
        print(f"N primary LUAD patients from all sources combined (pre-criteria): {len(self.clin)}")
        self.clin = self.clin.drop(index=self.exclude)
        if self.print_excl_numbers: # this gets excluded outside of remove_subset() so it has to be printed independently
            print(f"Exclusion: Lost to follow-up/redacted (n={len(self.exclude)}) --> {len(self.clin)} remaining")
        self.compile_clinical_data()
        self.extract_timeline_data()
        self.clin = self.clin.join(self.status_dates, how='left')
        self.clin = self.clin.join(self.tx_dates, how='left')
        # Adjust DFS DAYS (which is from initial treatment, while the others like days_to_NTE, are from diagnosis); only for pancan because that's what the timeline is from
        self.clin['DFS_DAYS_adj_2'] = self.clin['DFS_DAYS_2'] + self.clin['earliest_tx']
        self.clin = self.remove_subset(self.clin.index[self.clin['smoking_status'].isna()].tolist(), 'Missing smoking status')
        self.clin = self.remove_subset(self.clin.index[self.clin['AGE_3'].isna()].tolist(), 'Missing age')
        self.label_cases()
        self.clin = self.remove_subset(self.clin.index[self.clin['gender_5'].isna()].tolist(), 'Missing sex') # this is actually sex as far as I can determine
        self.clin = self.remove_subset(self.clin.index[self.clin['pathologic_stage_5'].isna()].tolist(), 'Missing stage')
        self.clin = self.remove_subset(self.clin.index[self.clin['pathologic_stage_5']=='Stage IV'].tolist(), 'Stage IV')
        self.clin = self.remove_subset(self.clin.index[self.clin['pathologic_stage_5']=='Stage IIIB'].tolist(), 'Stage IIIB')
        self.clin = self.remove_subset(self.clin.index[(self.clin['system_version_5']!='7th')&(self.clin['pathologic_T_5']=='T4')].tolist(), 'Pre-7th edition with T4')
        self.clin['pathologic_stage_5'] = self.clin['pathologic_stage_5'].str.replace('Stage ', '')
        if self.req_NT:
            self.clin = self.remove_subset(self.clin.index[self.clin['pathologic_N_5'].isna()].tolist(), 'Missing pathologic N')
            self.clin = self.remove_subset(self.clin.index[self.clin['pathologic_N_5']=='NX'].tolist(), 'Pathologic NX')
            self.clin = self.remove_subset(self.clin.index[self.clin['pathologic_T_5'].isna()].tolist(), 'Missing pathologic T')
    
    def extract_timeline_data(self):
        def extract(tline, pts, basecol, flags, suffix, earliest=True):
            dates = tline[tline[basecol].isin(flags.values())].groupby(['SAMPLE_ID', basecol])['START_DATE']
            dates = dates.min() if earliest else dates.max()
            dates = dates.unstack(basecol).reindex(index=pts, columns=list(flags.values())).rename(columns={v: f'{k}_{suffix}' for k, v in flags.items()})
            dates.columns.name = None
            return dates.astype('Int64')
        # data_timeline_status; _S indicates date is from STATUS
        stline = get_data('data_timeline_status', dp=self.paths['pc'], lower=False)
        stline['SAMPLE_ID'] = stline['PATIENT_ID'] + '-01'
        patients = pd.Index(stline['SAMPLE_ID'].unique(), name='SAMPLE_ID')
        flags1 = dict(death='DECEASED', last_follow_up='Last Follow Up')
        dates1 = extract(stline, patients, 'STATUS', flags1, 'S', earliest=False)
        flags2 = dict(dist_metas='Distant Metastasis', local_rec='Locoregional Recurrence', local_dist='Locoregional Recurrence|Distant Metastasis', dist_newprim='Distant Metastasis|New Primary Tumor')
        dates2 = extract(stline, patients, 'STATUS', flags2, 'S')
        self.status_dates = dates1.join(dates2)
        self.status_dates['last_status_S'] = stline.groupby('SAMPLE_ID')['START_DATE'].max().reindex(patients).astype('Int64')
        # Get tumor status at last follow-up
        temp = set_idx(stline.loc[stline['STATUS']=='Last Follow Up', ['SAMPLE_ID', 'TUMOR_STATUS']], idx_col='SAMPLE_ID').reindex(patients).rename(columns={'TUMOR_STATUS': 'final_tumor_status_S'})
        self.status_dates = self.status_dates.join(temp)
        # data_timeline_treatment; _TS indicates date is from anatomic treatment site, _RI indicates date is from regimen indication
        ttline = get_data('data_timeline_treatment', dp=self.paths['pc'], lower=False)
        ttline['SAMPLE_ID'] = ttline['PATIENT_ID'] + '-01'
        tx_patients = pd.Index(ttline['SAMPLE_ID'].unique(), name='SAMPLE_ID')
        flags3 = dict(dist_rec='Distant Recurrence', local_rec='Local Recurrence')
        dates3 = extract(ttline, tx_patients, 'ANATOMIC_TREATMENT_SITE', flags3, 'TS')
        flags4 = dict(rec='Recurrence')
        dates4 = extract(ttline, tx_patients, 'REGIMEN_INDICATION', flags4, 'RI')
        self.tx_dates = dates3.join(dates4)
        # Get earliest treatment date because some metrics like DFS are from treatment, not from diagnosis
        start_tx = ttline.groupby('SAMPLE_ID')['START_DATE'].min().reindex(tx_patients).astype('Int64')
        self.tx_dates.insert(0, 'earliest_tx', start_tx)
        stop_tx = ttline.groupby('SAMPLE_ID')['STOP_DATE'].max().reindex(tx_patients).astype('Int64')
        self.tx_dates.insert(1, 'latest_tx', stop_tx)
    
    def label_cases(self):
        self.id_tracker['Labeling'] = defaultdict(list)
        keep = self.clin.apply(self.filter_cases, axis=1)
        self.clin = self.remove_subset(self.clin.index[~keep].tolist(), 'No usable recurrence information')
        assert len(self.id_tracker['Labeling']['Explicit not otherwise caught']) == 0, f"There are {len(self.id_tracker['Labeling']['Explicit not otherwise caught'])} explicit patients not otherwise flagged as recurrent"
        self.id_tracker['Window'] = defaultdict(list)
        self.ttr, self.censor, self.ttr_src = {}, {}, {}
        df2 = self.clin[self.clin.apply(self.restrict_window, axis=1)].copy()
        unusable = self.clin.index.difference(df2.index).tolist()
        self.clin = self.remove_subset(unusable, f'No usable {self.window}-day label')
        assert len(df2) == len(self.clin)
        positives = [i for k, v in self.id_tracker['Window'].items() if k.startswith('1: pass') for i in v]
        # Determinate = the window could answer the question: a recurrence was seen ('1: ...'), or follow-up reached the window ('0: pass'). Patients retained by keep_censored 
        # are recurrence-free but under-followed, so they get <NA>, NOT 0 — scoring them as negatives would relabel every one of them
        censored = [i for k, v in self.id_tracker['Window'].items() if k.startswith('0: fail') for i in v]
        censored = [i for i in censored if i in self.clin.index]
        lab = pd.Series(pd.NA, index=self.clin.index, dtype='Int64')
        lab.loc[self.clin.index.isin(positives)] = 1
        lab.loc[~self.clin.index.isin(positives) & ~self.clin.index.isin(censored)] = 0
        self.clin['label_any'] = lab
        self.clin['ttr'] = pd.Series(self.ttr, dtype='Float64').reindex(self.clin.index).astype('Int64')
        # Time-to-event view, defined for everyone retained: time = days to recurrence if observed, else days of recurrence-free follow-up; 
        # event_obs = 1 if a recurrence was observed at that time, 0 if the patient was still recurrence-free then
        cens_time = pd.Series(self.censor, dtype='Float64').reindex(self.clin.index)
        self.clin['event_obs'] = self.clin.index.isin(positives).astype(int)
        self.clin['fu_time'] = self.clin['ttr'].astype('Float64').where(self.clin['event_obs'] == 1, cens_time).astype('Int64')
        if self.print_excl_numbers and self.keep_censored:
            print(f"Retained {len(censored)} censored patients (recurrence-free, follow-up < {self.window}d): label_any is <NA>, fu_time/event_obs usable for a time-to-event loss")
    
    def filter_cases(self, row):
        rec_status = row[['dist_metas_S', 'local_rec_S', 'local_dist_S', 'dist_newprim_S', 'dist_rec_TS', 'local_rec_TS', 'rec_RI']].notna().any()
        rec_nte = row[['new_neoplasm_event_type_5', 'nte_type_4']].isin({'Distant Metastasis', 'Locoregional Recurrence', 'Locoregional Recurrence|Distant Metastasis', 'Distant Metastasis|New Primary Tumor'}).any()
        dfs_1 = (row.get('DFI_7') == 1) or (row.get('DFS_STATUS_3') == '1:Recurred/Progressed')
        explicit = row.name in self.explicit_cases
        met_proc = row.get('additional_surgery_metastatic_procedure_1') == 'YES'
        dfs_0 = (row.get('DFI_7') == 0) or (row.get('DFS_STATUS_3') == '0:DiseaseFree')
        tumor_free = (row.get('person_neoplasm_cancer_status_1') == 'TUMOR FREE') or (row.get('final_tumor_status_S') == 'Tumor Free')
        
        def track(reason, keep):
            self.id_tracker['Labeling'][reason].append(row.name); return keep
        
        if (rec_status or rec_nte or dfs_1 or explicit or met_proc) and dfs_0: self.id_tracker['Labeling']['Conflicting sources'].append(row.name)
        if rec_status: return track('Recurrence status', True) # 1
        if rec_nte: return track('Recurrence NTE', True) # 1
        if met_proc: return track('Metastatic procedure', True) # 1
        if dfs_1: return track('Has DFI/DFS 1', True) # 1
        if explicit: return track('Explicit not otherwise caught', True) # 1
        if dfs_0: return track('Has DFI/DFS 0', True) # 0
        if tumor_free: return track('Tumor free', True) # 0
        return track('No recurrence indication', True) # 0 switched this from False, it will still be windowed
    
    def restrict_window(self, row):
        follow_ups = [['death_S', 'last_follow_up_S', 'last_status_S', 'latest_tx'], ['last_follow_up', 'days_to_death'], ['last_contact_days_to_4']]
        dfs_days_to = [['DFI.time_7'], ['DFS_DAYS_adj_2'], ['DFS_DAYS_3']]
        status_days_to = [['dist_metas_S', 'local_rec_S', 'local_dist_S', 'dist_newprim_S'], ['dist_rec_TS', 'local_rec_TS', 'rec_RI']]
        nte_days_to = [['days_to_nte']]
        met_proc = [['days_to_additional_surgery_metastatic_procedure_1']]
        
        def track(reason, keep):
            self.id_tracker['Window'][reason].append(row.name); return keep
        
        def flatten(tiers):
            return [c for tier in tiers for c in tier]
        
        def check_neg(tiers, tag):
            vals = pd.to_numeric(row[flatten(tiers)], errors='coerce')
            if vals.isna().all(): return track(f'0: missing ({tag})', False) # no follow-up time at all: unusable either way
            valid = vals[vals >= 0]
            if len(valid): self.censor[row.name] = valid.max() # longest recurrence-free follow-up on record
            if (vals >= self.window).any(): return track(f'0: pass ({tag})', True)
            # Recurrence-free but under-followed: no determinate label, but 'recurrence-free through censor time' is real information
            keep = bool(self.keep_censored) and len(valid) > 0 and valid.max() >= self.min_censor_days
            return track(f'0: fail ({tag})', keep)
        
        def check_pos(tiers, tag):
            flat = flatten(tiers)
            vals = pd.to_numeric(row[flat], errors='coerce')
            valid = vals[vals >= 0]
            for tier in tiers:
                tier_valid = valid[valid.index.isin(tier)]
                if len(tier_valid):
                    m = tier_valid.min()
                    self.ttr[row.name] = m
                    # all columns in the winning tier that hit the min, so ties are visible rather than resolved arbitrarily
                    self.ttr_src[row.name] = '|'.join(tier_valid.index[tier_valid == m])
                    break
            if vals.isna().all(): return track(f'1: missing ({tag})', False)
            if (vals.dropna() <= 0).all(): return track(f'1: invalid ({tag})', False)
            if not self.window_positives: return track(f'1: pass ({tag})', True)
            if valid.between(0, self.window).any(): return track(f'1: pass ({tag})', True)
            return track(f'1: fail ({tag})', True) # recurred after window -> disease-free through it -> label 0 (hence keep=True)
        
        if row.name in self.id_tracker['Labeling']['Has DFI/DFS 0']: return check_neg(follow_ups + dfs_days_to, 'dfi')
        if row.name in self.id_tracker['Labeling']['Tumor free']: return check_neg(follow_ups + dfs_days_to, 'tumor free')
        if row.name in self.id_tracker['Labeling']['No recurrence indication']: return check_neg(follow_ups + dfs_days_to, 'no ind')
        if row.name in self.id_tracker['Labeling']['Recurrence status']: return check_pos(status_days_to, 'status') # no need for dfs because all statuses have a date
        if row.name in self.id_tracker['Labeling']['Recurrence NTE']: return check_pos(nte_days_to + dfs_days_to, 'nte')
        if row.name in self.id_tracker['Labeling']['Metastatic procedure']: return check_pos(met_proc + dfs_days_to, 'dfi')
        if row.name in self.id_tracker['Labeling']['Has DFI/DFS 1']: return check_pos(dfs_days_to, 'dfi')
        raise ValueError(f"UNCLASSIFIED ROW IN WINDOW STEP: {row.name}")
    
    def get_genomic(self):
        self.expr = self.process_expression_data('data_mrna_seq_v2_rsem.txt', self.paths['pc'], gene_col='Hugo_Symbol')
        self.mut = self.process_mutation_data('data_mutations.txt', self.paths['pc'], gene_col='Hugo_Symbol', id_col='Tumor_Sample_Barcode', chr_str='Chromosome', varclass_str='Variant_Classification')
        self.expr, self.mut = [df[df.index.str.endswith('-01')].copy() for df in [self.expr, self.mut]] # limit to primary samples only
        print(f"Expression shape: {self.expr.shape}, Mutation shape: {self.mut.shape}")
    
    def update_id_tracker(self, key, ids):
        self.id_tracker[key] = list(ids)

    def bulk_update_id_tracker(self, mapping):
        self.id_tracker.update({k: list(v) for k, v in mapping.items()})
    
    def remove_subset(self, ids, key):
        self.update_id_tracker(key, ids)
        subsetted = self.clin[~self.clin.index.isin(ids)].copy()
        if self.print_excl_numbers: print(f"Exclusion: {key} (n={len(ids)}) --> {len(subsetted)} remaining")
        return subsetted
    
    def compile_clinical_data(self):
        self.clin = fill_from_backup_cols(self.clin, 'AGE_3', ['age_at_initial_pathologic_diagnosis_1', 'AGE_2', 'age_at_initial_pathologic_diagnosis_4', 'age_at_initial_pathologic_diagnosis_5', 'age_at_index_6', 'age_at_initial_pathologic_diagnosis_7'])
        for c in [*[f'gender_{i}' for i in [1,4,5,6,7]], 'SEX_2', 'SEX_3', 'RACE_2', 'race_6', 'ETHNICITY_2', 'ethnicity_6', 'NTE_AFTER_INITIAL_TREATMENT_2', 'RADIATION_THERAPY_2', 'PERSON_NEOPLASM_CANCER_STATUS_2']:
            self.clin[c] = self.clin[c].str.upper()
        self.clin['AJCC_PATHOLOGIC_TUMOR_STAGE_2'] = self.clin['AJCC_PATHOLOGIC_TUMOR_STAGE_2'].str.replace('STAGE', 'Stage')
        for c in ['PATH_T_STAGE_2', 'PATH_M_STAGE_2']:
            self.clin[c] = self.clin[c].str.replace('B', 'b').str.replace('A', 'a')
        self.clin['AJCC_STAGING_EDITION_2'] = self.clin['AJCC_STAGING_EDITION_2'].str.lower()
        groupings1 = {
            'gender_5': ['gender_4', 'SEX_3', 'gender_1', 'gender_6', 'gender_7', 'SEX_2'],
            'race_4': ['RACE_3', 'race_5', 'race_6', 'RACE_2'],
            'ethnicity_5': ['ETHNICITY_3', 'ethnicity_6', 'ETHNICITY_2'],
            'pathologic_stage_5': ['AJCC_PATHOLOGIC_TUMOR_STAGE_3', 'ajcc_pathologic_tumor_stage_4', 'pathologic_stage_1', 'AJCC_PATHOLOGIC_TUMOR_STAGE_2', 'ajcc_pathologic_stage_6', 'pathologic_stage_7'],
            'pathologic_T_5': ['AJCC_TUMOR_PATHOLOGIC_PT_3', 'pathologic_T_1', 'ajcc_pathologic_t_6', 'pathologic_T_7', 'PATH_T_STAGE_2'],
            'pathologic_M_5': ['AJCC_METASTASIS_PATHOLOGIC_PM_3', 'pathologic_M_1', 'ajcc_pathologic_m_6', 'pathologic_M_7', 'PATH_M_STAGE_2'],
            'pathologic_N_5': ['AJCC_NODES_PATHOLOGIC_PN_3', 'pathologic_N_1', 'ajcc_pathologic_n_6', 'pathologic_N_7', 'PATH_N_STAGE_2'],
            'number_pack_years_smoked_5': ['SMOKING_PACK_YEARS_3', 'number_pack_years_smoked_1', 'number_pack_years_smoked_7', 'pack_years_smoked_6'],
            'year_of_tobacco_smoking_onset_1': ['SMOKING_YEAR_STARTED_3', 'year_of_tobacco_smoking_onset_5', 'year_of_tobacco_smoking_onset_7'],
            'stopped_smoking_year_1': ['SMOKING_YEAR_STOPPED_3', 'stopped_smoking_year_5', 'stopped_smoking_year_7'],
            'TOBACCO_SMOKING_HISTORY_INDICATOR_3': ['tobacco_smoking_history_1', 'tobacco_smoking_history_7'],
            'system_version_5': ['AJCC_STAGING_EDITION_3', 'system_version_1', 'system_version_7', 'ajcc_staging_system_edition_6', 'AJCC_STAGING_EDITION_2'],
        }
        for target, backups in groupings1.items(): self.clin = fill_from_backup_cols(self.clin, target, backups)
        self.clin = fill_from_backup_cols(self.clin, 'tobacco_smoking_history_5', ['tobacco_smoking_history_indicator_1', 'tobacco_smoking_history_indicator_7'], override_agreement=True) # 2 have duration reformed unknown, NaN for those in the other
        coerce_num = ['days_to_death_1', 'death_days_to_4', 'days_to_death_5', 'days_to_death_6', 'days_to_death_7', 'days_to_last_followup_1', 'DAYS_LAST_FOLLOWUP_2', 'days_to_last_followup_5', 'days_to_last_follow_up_6', 'days_to_last_followup_7', 'days_to_nte_after_initial_treatment_1', 'nte_dx_days_to_4', 'days_to_nte_after_initial_treatment_5', 'days_to_nte_after_initial_treatment_7', 'days_to_birth_1', 'DAYS_TO_BIRTH_2', 'days_to_birth_5', 'days_to_birth_6', 'days_to_birth_7', 'DSS.time_1', 'DSS.time_4', 'DSS.time_7', 'DFS_MONTHS_2', 'DFS_MONTHS_3']
        self.clin[coerce_num] = self.clin[coerce_num].apply(pd.to_numeric, errors='coerce')
        self.clin['DFS_DAYS_2'], self.clin['DFS_DAYS_3'] = [(self.clin[f'DFS_MONTHS_{i}'] * 30.4167).round() for i in [2, 3]]
        # days_to_death_6 matches the others for all but one patient TCGA-55-A4DF-01 (for which all the others are uniform 440 vs 614)
        self.clin['days_to_death'] = self.clin[['days_to_death_1', 'death_days_to_4', 'days_to_death_5', 'days_to_death_6', 'days_to_death_7']].max(axis=1)
        self.clin['last_follow_up'] = self.clin[['days_to_last_followup_1', 'DAYS_LAST_FOLLOWUP_2', 'days_to_last_followup_5', 'days_to_last_follow_up_6', 'days_to_last_followup_7']].max(axis=1)
        groupings2 = {
            'days_to_birth_5': ['days_to_birth_1', 'days_to_birth_6', 'days_to_birth_7', 'DAYS_TO_BIRTH_2'],
            'DSS_7': ['DSS_1', 'DSS_4'],
            'DSS.time_7': ['DSS.time_1', 'DSS.time_4'],
            'PFI_7': ['PFI_1', 'PFI_4'],
            'PFI.time_7': ['PFI.time_1', 'PFI.time_4'],
            'HISTORY_NEOADJUVANT_TRTYN_2': ['HISTORY_NEOADJUVANT_TRTYN_3'],
            'nte_after_initial_treatment_5': ['nte_after_initial_treatment_1'],
            'new_neoplasm_event_type_5': ['new_neoplasm_event_type_1'],
            'icd_10_5': ['ICD_10_3', 'icd_10_7', 'icd_10_code_6', 'icd_10_1', 'ICD_10_2'],
            'icd_o_3_histology_5': ['ICD_O_3_HISTOLOGY_3', 'icd_o_3_histology_7', 'icd_o_3_histology_1', 'ICD_O_3_HISTOLOGY_2'],
            'icd_o_3_site_5': ['ICD_O_3_SITE_3', 'icd_o_3_site_7', 'icd_o_3_site_1', 'ICD_O_3_SITE_2'],
            'days_to_nte_after_initial_treatment_1': ['days_to_nte_after_initial_treatment_7']
        }
        for target, backups in groupings2.items(): self.clin = fill_from_backup_cols(self.clin, target, backups)
        """ So I looked at all cases where the days_to_nte columns weren't identical, in the context of each counts associated nte_type; before taking the 'min' across the days_to_nte, 
        I first resolved instances where the difference is due to one type being new primary """
        # mask = self.clin[['days_to_nte_after_initial_treatment_1', 'nte_dx_days_to_4', 'days_to_nte_after_initial_treatment_5']].nunique(axis=1) > 1
        # self.clin[mask][['nte_dx_days_to_4', 'days_to_nte_after_initial_treatment_1', 'days_to_nte_after_initial_treatment_5', 'nte_type_4', 'new_neoplasm_event_type_5', 'new_neoplasm_event_type_7']]
        # TCGA-49-AARO-01 is LR for all three but 4 is 1144.0 while 1/5 is 3521.0; same issue for TCGA-50-5049-01, TCGA-50-5946-01, TCGA-86-7701-01, TCGA-MP-A4TK-01
        self.clin['days_to_nte'] = self.clin[['days_to_nte_after_initial_treatment_1', 'nte_dx_days_to_4', 'days_to_nte_after_initial_treatment_5']].min(axis=1)
        self.clin.loc['TCGA-49-AARR-01', 'days_to_nte'] = 4812.0 # 1099.0 corresponds to New Primary Tumor
        self.clin.loc['TCGA-73-4659-01', 'days_to_nte'] = 316.0 # 35.0 doesn't correspond to a nte_type while the others do
        self.clin.loc['TCGA-97-7547-01', 'days_to_nte'] = 1147.0 # 775.0 doesn't correspond to a nte_type while the others do
        self.clin = self.clin.drop(columns=['days_to_death_1', 'death_days_to_4', 'days_to_death_5', 'days_to_death_6', 'days_to_death_7', 'days_to_last_followup_1', 'DAYS_LAST_FOLLOWUP_2', 'days_to_last_followup_5', 'days_to_last_follow_up_6', 'days_to_last_followup_7', 'days_to_nte_after_initial_treatment_1', 'nte_dx_days_to_4', 'days_to_nte_after_initial_treatment_5'])
        groupings3 = {
            'DFI.time_7': ['DFI.time_1', 'DFI.time_4'],
            'DFI_7': ['DFI_1', 'DFI_4'],
            'new_neoplasm_event_type_5': ['new_neoplasm_event_type_7'],
            'treatment_outcome_first_course_4': ['TREATMENT_OUTCOME_FIRST_COURSE_3'],
            'location_in_lung_parenchyma_5': ['LOCATION_LUNG_PARENCHYMA_3', 'location_in_lung_parenchyma_1', 'location_in_lung_parenchyma_7'],
            'SAMPLE_INITIAL_WEIGHT_3': ['initial_weight_1', 'initial_weight_7'],
            'residual_tumor_5': ['RESIDUAL_TUMOR_3', 'residual_tumor_1', 'residual_tumor_7'],
            'radiation_therapy_5': ['RADIATION_THERAPY_2', 'radiation_therapy_1', 'radiation_therapy_7'],
            'LONGEST_DIMENSION_3': ['longest_dimension_7', 'longest_dimension_1'],
            'KRAS_MUTATION_3': ['kras_mutation_found_5'],
            'KRAS_MUTATION_IDENTIFIED_TYPE_3': ['kras_mutation_result_1', 'kras_mutation_result_5', 'kras_mutation_result_7'],
            'days_to_birth_5': ['birth_days_to_4'],
            'dlco_predictive_percent_5': ['CARBON_MONOXIDE_DIFFUSION_DLCO_3'],
            'person_neoplasm_cancer_status_1': ['person_neoplasm_cancer_status_7', 'tumor_status_4'], # matches the status timeline ones so this is the one we'll use (ignoring person_neoplasm_cancer_status_5, TUMOR_STATUS_3, and PERSON_NEOPLASM_CANCER_STATUS_2, which all match eachother)
            'year_of_initial_pathologic_diagnosis_5': ['initial_pathologic_dx_year_4', 'INITIAL_PATHOLOGIC_DX_YEAR_3', 'year_of_initial_pathologic_diagnosis_1', 'year_of_initial_pathologic_diagnosis_7', 'year_of_diagnosis_6']
        }
        for target, backups in groupings3.items(): self.clin = fill_from_backup_cols(self.clin, target, backups)
        groupings_ov = {
            'DFS_STATUS_3': ['DFS_STATUS_2'],
            'nte_after_initial_treatment_5': ['nte_after_initial_treatment_7', 'NTE_AFTER_INITIAL_TREATMENT_2', 'NTE_AFTER_INITIAL_TREATMENT_3'], # only _3 has some disagreement, the rest all align
            'treatment_outcome_first_course_4': ['primary_therapy_outcome_success_5', 'primary_therapy_outcome_success_7', 'primary_therapy_outcome_success_1'], # disagreement for one patient (TCGA-L9-A50W-01) between _1 and _5/_4/_7
            'targeted_molecular_therapy_5': ['TARGETED_MOLECULAR_THERAPY_3', 'targeted_molecular_therapy_1', 'targeted_molecular_therapy_7'], # opting for 3/5 here (which agree) over 1/7 (which agree) as 3/5 are two different sources while 1/7 are both xena
        }
        for target, backups in groupings_ov.items(): self.clin = fill_from_backup_cols(self.clin, target, backups, override_agreement=True)
        for c in ['ABSOLUTE_Ploidy', 'ABSOLUTE_Purity', 'days_to_additional_surgery_locoregional_procedure', 'days_to_additional_surgery_metastatic_procedure', 'additional_surgery_locoregional_procedure', 'additional_surgery_metastatic_procedure', 'bcr_patient_barcode', 'bcr_sample_barcode', 'BRAF', 'CBL', 'CTNNB1', 'EGFR', 'ERBB2', 'ERBB4', 'Expression_Subtype', 'other_dx', 'Pathology', 'PTPN11', 'RET_translocation', 'STK11', 'followup_treatment_success', 'followup_case_report_form_submission_reason', 'KRAS', 'MAP2K1', 'MET', 'patient_id', 'PIK3CA', 'AKT1', 'karnofsky_performance_score', 'Pathology_Updated']:
            self.clin = fill_from_backup_cols(self.clin, f'{c}_1', [f'{c}_7'])
        for c in ['additional_pharmaceutical_therapy', 'additional_radiation_therapy', 'anatomic_neoplasm_subdivision', 'dlco_predictive_percent', 'egfr_mutation_result', 'history_of_neoadjuvant_treatment', 'post_bronchodilator_fev1_fvc_percent', 'post_bronchodilator_fev1_percent', 'pre_bronchodilator_fev1_fvc_percent', 'pre_bronchodilator_fev1_percent', 'tissue_source_site']:
            self.clin = fill_from_backup_cols(self.clin, f'{c}_5', [f'{c}_1', f'{c}_7'])
        for pfx in ['post', 'pre']:
            self.clin = fill_from_backup_cols(self.clin, f'FEV1_FVC_RATIO_{pfx.upper()}BRONCHOLIATOR_3', [f'{pfx}_bronchodilator_fev1_fvc_percent_5'])
            self.clin = fill_from_backup_cols(self.clin, f'FEV1_PERCENT_REF_{pfx.upper()}BRONCHOLIATOR_3', [f'{pfx}_bronchodilator_fev1_percent_5'])
        adds = [c for c in ['age_at_diagnosis_6', 'age_at_earliest_diagnosis_6', 'age_at_earliest_diagnosis_in_years_6', 'days_to_birth_5'] if does_it_add_value(self.clin, 'AGE_3', c)]
        assert not adds, f'Supplemental age col(s) [{adds}] add value for AGE_3, utilize accordingly'
        adds = [c for c in ['cigarettes_per_day_6', 'number_pack_years_smoked_5', 'stopped_smoking_year_1', 'year_of_tobacco_smoking_onset_1', 'years_smoked_6', 'TOBACCO_SMOKING_HISTORY_INDICATOR_3'] if does_it_add_value(self.clin, 'tobacco_smoking_history_5', c)]
        assert not adds, f'Supplemental smoking col(s) [{adds}] add value for tobacco_smoking_history_5, utilize accordingly'
        SMOKING_MAP = {'current reformed smoker for < or = 15 years': 'ref_sub15', 'current reformed smoker for > 15 years': 'ref_sup15', 'current reformed smoker, duration not specified': 'ref_unk', 'current smoker': 'current', 'lifelong non-smoker': 'never'}
        s = self.clin['tobacco_smoking_history_5'].str.strip().str.lower()
        mapped = s.map(SMOKING_MAP)
        assert not set(s[s.notna() & mapped.isna()]), f'Unmapped smoking status: {set(s[s.notna() & mapped.isna()])}'
        self.clin['smoking_status'] = mapped
        self.clin = self.clin.drop(columns=['bcr_patient_barcode_1', 'diagnosis_5', 'days_to_diagnosis_6', 'disease_code_5', 'patient_id_1', 'patient_id_5', 'tumor_tissue_site_5', 'person_neoplasm_cancer_status_5', 'TUMOR_STATUS_3', 'PERSON_NEOPLASM_CANCER_STATUS_2', 'tobacco_smoking_history_5', 'cigarettes_per_day_6', 'stopped_smoking_year_1', 'year_of_tobacco_smoking_onset_1', 'years_smoked_6', 'TOBACCO_SMOKING_HISTORY_INDICATOR_3', 'DFS_MONTHS_2', 'DFS_MONTHS_3', 'age_at_diagnosis_6', 'age_at_earliest_diagnosis_6', 'age_at_earliest_diagnosis_in_years_6', 'days_to_birth_5', 'DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS_2', 'DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS_3'])

    def update_explicit(self, df, col, id_col=None):
        ids = list(df.index.to_series().str[:-2] + '01') if id_col is None else list(df[id_col].str[:-2] + '01')
        for i, idx in enumerate(df.index):
            self.explicit_cases[ids[i]].update([df.loc[idx, col]])
    
    def load_and_prep_xena(self, suffix='', src='xena'):
        df1 = get_data('xena_phenotypes', dp=self.paths[src], lower=False) # phenotypes
        df1 = df1.loc[:, ~df1.columns.isin(self.ignore['xena_phenotypes'])].copy()
        df1 = df1.loc[:, ~df1.columns.str.startswith('_GENOMIC_ID')].copy()
        if src == 'combined_xena':
            df1 = df1.loc[df1['_primary_disease']=='lung adenocarcinoma'].copy()
        df1 = set_idx(df1.dropna(subset=['sampleID']), idx_col='sampleID')
        explicit = df1[(df1['sample_type']=='Recurrent Tumor')|(df1.index.str.endswith('-02'))]
        self.update_explicit(explicit, 'days_to_initial_pathologic_diagnosis')
        df1 = df1.loc[(df1['sample_type']=='Primary Tumor')|(df1.index.str.endswith('-01'))].copy()
        df2 = get_data('xena_endpoints', dp=self.paths[src], lower=False) # curated endpoints preferred by TCGA
        df2 = df2.loc[:, ~df2.columns.isin(self.ignore['xena_endpoints'])].copy()
        df2 = set_idx(df2.dropna(subset=['sample']), idx_col='sample')
        df = df1.join(df2, how='outer')
        lost_to_follow_up = df.loc[df['lost_follow_up']=='YES'].index.tolist()
        self.update_id_tracker('lost follow up', lost_to_follow_up)
        self.exclude.update(lost_to_follow_up)
        df = df.drop(columns=['sample_type', 'lost_follow_up', '_primary_disease', 'days_to_initial_pathologic_diagnosis'])
        df.columns = [f"{col.replace('new_tumor_event', 'nte').replace('NEW_TUMOR_EVENT', 'NTE')}{suffix}" for col in df.columns]
        return df

    def load_and_prep_cbio(self, suffix='', src='fh'):
        def get_primary(df):
            if 'SAMPLE_TYPE' in df.columns:
                explicit = df[(df['SAMPLE_TYPE']=='Recurrence')|(df.index.str.endswith('-02'))]
                self.update_explicit(explicit, 'DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS')
                return df.loc[(df['SAMPLE_TYPE']=='Primary')|(df.index.str.endswith('-01'))].drop(columns=['SAMPLE_TYPE']).copy()
            else:
                explicit = df[df.index.str.endswith('-02')]
                self.update_explicit(explicit, 'DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS')
                return df.loc[df.index.str.endswith('-01')].copy()
        patient = get_data('data_clinical_patient', dp=self.paths[src], lower=False)
        sample = get_data('data_clinical_sample', dp=self.paths[src], lower=False)
        patient = patient.loc[:, ~patient.columns.isin(self.ignore['data_clinical_patient'])].copy()
        sample = sample.loc[:, ~sample.columns.isin(self.ignore['data_clinical_sample'])].copy()
        df = patient.merge(sample, on='PATIENT_ID')
        df = set_idx(df.dropna(subset=['SAMPLE_ID']), idx_col='SAMPLE_ID', also_drop=['PATIENT_ID'])
        df = get_primary(df)
        df.columns = [f"{col.replace('new_tumor_event', 'nte').replace('NEW_TUMOR_EVENT', 'NTE')}{suffix}" for col in df.columns]
        return df
    
    def load_and_prep_gdc(self, suffix='', key='cdr'):
        if key in ['cdr', 'followup']: # these 2 are by patient, not sample (static to the patient), so we can tack on -01
            if key == 'cdr':
                df = load_data('csv', 'tcga_cdr', data_path=self.paths['gdc'], clean=True, index_col=0)
                df = df.loc[df['type']=='LUAD'].dropna(axis=1, how='all')
            else:
                df = load_data('tsv', 'clinical_PANCAN_patient_with_followup', self.paths['gdc'], clean=True, encoding='latin1', low_memory=False)
                df = df.loc[df['acronym']=='LUAD'].dropna(axis=1, how='all')
            df = df.loc[:, ~df.columns.isin(self.ignore[key])].copy()
            df['sample_id'] = df['bcr_patient_barcode'] + '-01'
            df = set_idx(df, idx_col='sample_id', also_drop=['bcr_patient_barcode', 'type' if key == 'cdr' else 'acronym'])
        elif key == 'luad_gdc_phenotypes':
            df = load_data('tsv', key, data_path=self.paths['gdc'])
            df = df.loc[:, ~df.columns.isin(self.ignore[key])].copy()
            df = df.drop_duplicates()
            # All sample_ids have an additional letter at the end (A, B, C, etc.) that indicate vial
            df['sample_id'] = df['sample'].str[:-1] 
            df = df.dropna(subset=['sample_id'])
            explicit = df[(df['sample_type.samples']=='Recurrent Tumor')|(df['tumor_descriptor.samples']=='Recurrence')|(df['sample_id'].str.endswith('-02'))]
            self.update_explicit(explicit, 'days_to_diagnosis.diagnoses', id_col='sample_id')
            df = df[(df['sample_type.samples']=='Primary Tumor')|(df['tumor_descriptor.samples']=='Primary')].copy()
            df = set_idx(df.drop(columns = ['sample', 'sample_type.samples', 'tumor_descriptor.samples']).drop_duplicates(), idx_col='sample_id')
            df.columns = [col.split('.')[0] for col in df.columns]
        df.columns = [f"{col.replace('new_tumor_event', 'nte').replace('NEW_TUMOR_EVENT', 'NTE')}{suffix}" for col in df.columns]
        return df
    
    def process_expression_data(self, filename, dp, gene_col): 
        expr_df = get_data(filename, dp=dp)
        expr_df = self.update_gene_symbols(expr_df, gene_col)
        expr_df = expr_df.groupby('approved_symbol').mean() if 'zscores' in filename else expr_df.groupby('approved_symbol').sum()
        expr_df.index = list(expr_df.index) # removes "approved symbol" label
        expr_df = expr_df.T
        expr_df = expr_df.loc[:, (expr_df > 0).any(axis=0)] # drop all-zero columns
        return expr_df.add_suffix('_expr')

    def process_mutation_data(self, filename, dp, gene_col, id_col, chr_str='Chromosome', varclass_str='Variant_Classification'):
        df = get_data(filename, dp=dp, low_memory=False)
        sequenced = pd.Index(sorted(df[id_col].dropna().unique()), name=id_col)
        self.variant_classes = df[varclass_str].value_counts()
        if self.harmonize:
            df = df[df[varclass_str].isin(MAF_KEEP) & ~df[chr_str].isin(['X', 'Y'])].copy()
        else:  # pre-revision filter, retained to reproduce the before-harmonization comparison
            df = df[(~df[chr_str].isin(['X','Y'])) & (~df[varclass_str].isin(['Silent', 'RNA']))].copy()
        df = self.update_gene_symbols(df, gene_col)
        self.raw_mut = df.copy() # make sure to update this later
        mut_df = pd.crosstab(df[id_col], df['approved_symbol']).astype(float)
        mut_df = mut_df.reindex(index=sequenced, fill_value=0.0)
        mut_df.columns = [f"{col}_mut" for col in mut_df.columns]
        return mut_df
    
    def update_gene_symbols(self, df, gene_col, drop_unmapped=True):
        df = df.copy()
        def _mapper_dict(key):
            d = self.hgnc.get(key)
            return {str(k).strip(): v for k, v in d.items()} # normalize keys to string and strip spaces
        def _norm_series(s):
            return s.astype('string').str.strip()
        approved_map = _mapper_dict('Symbol')
        df['approved_symbol'] = _norm_series(df[gene_col]).str.upper().map(approved_map)
        df['approved_symbol'] = df['approved_symbol'].infer_objects(copy=False)
        if drop_unmapped: df = df.dropna(subset=['approved_symbol'])
        return df.drop(columns=[gene_col])

    def get_groupings(self):
        clin_expr, clin_mut = self.expr.index.intersection(self.clin.index), self.mut.index.intersection(self.clin.index) # clin + expr; clin + mut
        all3 = clin_expr.intersection(clin_mut) # clin + expr + mut
        expr_no_mut, mut_no_expr, clin_no_omic = clin_expr.difference(all3), clin_mut.difference(all3), self.clin.index.difference(clin_expr.union(clin_mut)) # clin + expr (no mut); clin + mut (no expr); clin only (no genomic)
        return dict(clin_expr=clin_expr, clin_mut=clin_mut, all3=all3, expr_no_mut=expr_no_mut, mut_no_expr=mut_no_expr, clin_no_omic=clin_no_omic)
    
    def finalize_data(self):
        g = self.get_groupings()
        print(f"All have clinical (for label):   Expr + Mut = {len(g['all3'])}   Expr only = {len(g['expr_no_mut'])}   Mut only = {len(g['mut_no_expr'])}   No genomic = {len(g['clin_no_omic'])}")
        self.cohort = sorted(self.get_groupings()['all3'].tolist())
        self.ttr_src = pd.Series(self.ttr_src, dtype='string').reindex(self.cohort).dropna()
        def slice_and_clean(name):
            df = getattr(self, name)
            df = df.loc[self.cohort].copy()
            df = df.loc[:, df.nunique() > 1].copy()
            return df
        self.clin, self.expr, self.mut = (slice_and_clean(name) for name in ['clin', 'expr', 'mut'])
        self.raw_mut = self.raw_mut[self.raw_mut['Tumor_Sample_Barcode'].isin(self.cohort)].copy()
        y1, y0 = [self.clin[self.clin['label_any']==i].index.tolist() for i in [1, 0]]
        n_cens = int(self.clin['label_any'].isna().sum())
        print(f"{len(self.clin)} patients ({len(y1)} relapsed, {len(y0)} disease-free" + (f", {n_cens} censored" if n_cens else "") + ")")
        # label_early stays <NA> wherever label_any is: a censored patient has no determinate early-recurrence answer either
        early = pd.Series(pd.NA, index=self.clin.index, dtype='Int64')
        determinate = self.clin['label_any'].notna()
        early.loc[determinate] = (self.clin.index[determinate].isin(y1) & (self.clin.loc[determinate, 'ttr'] < self.cutoff)).astype(int)
        self.clin['label_early'] = early
        y1, y0 = [self.clin[self.clin['label_early']==i].index.tolist() for i in [1, 0]]
        print(f"{len(self.clin)} patients ({len(y1)} early, {len(y0)} late or never" + (f", {n_cens} censored" if n_cens else "") + ")\n")

    def print_id_tracker(self, d, path=(), final_cohort_only=False):
        if isinstance(d, dict):
            for k, v in d.items(): self.print_id_tracker(v, path + (k,))
        elif isinstance(d, list) or isinstance(d, set):
            if final_cohort_only: print(" -> ".join(path), ":", len([i for i in d if i in self.clin.index]))
            else: print(" -> ".join(path), ":", len(d))

class TRACERx:
    def __init__(self, hgnc, cutoff, window=1825, req_NT=False, window_positives=False, print_excl_numbers=False, harmonize_mut=True):
        self.name = 'tracerx'
        self.path = f'{root}/tracerx/'
        self.hgnc, self.cutoff, self.window, self.req_NT = hgnc, cutoff, window, req_NT
        self.window_positives, self.print_excl_numbers, self.harmonize = window_positives, print_excl_numbers, harmonize_mut
        self.id_tracker = {}
        self.get_clinical() # https://zenodo.org/records/10932811
        self.get_expression()
        self.get_mutation()
        self.finalize_data()
    
    def update_id_tracker(self, key, ids):
        self.id_tracker[key] = list(ids)
    
    def print_id_tracker(self, d, path=(), final_cohort_only=False):
        if isinstance(d, dict):
            for k, v in d.items(): self.print_id_tracker(v, path + (k,))
        elif isinstance(d, list) or isinstance(d, set):
            if final_cohort_only: print(" -> ".join(path), ":", len([i for i in d if i in self.clin.index]))
            else: print(" -> ".join(path), ":", len(d))

    def remove_subset(self, ids, key):
        self.update_id_tracker(key, ids)
        subsetted = self.clin[~self.clin.index.isin(ids)].copy()
        if self.print_excl_numbers: print(f"Exclusion: {key} (n={len(ids)}) --> {len(subsetted)} remaining")
        return subsetted

    def get_clinical(self):
        def quick_format(key):
            df = load_data('csv', key, data_path=self.path, clean=False)
            return df.copy()
        chist, all_tm, all_pt = [quick_format(fn) for fn in ['clinicohistopathological_data', 'TRACERx421_all_tumour_df', 'TRACERx421_all_patient_df']]
        self.chist = chist.loc[chist['Is.primary.tumour.Tx421']=='primary', ['tumour_id_muttable_cruk', 'sample_name_cruk', 'Histology_per_tumour_id_muttable', 'smoking_status_merged', 'pathologyTNM', 'LUAD_pred_subtype_with.IMA_per_tumour']].drop_duplicates()
        self.all_tm = all_tm[all_tm['histology_3']=='LUAD'].copy()
        self.all_pt = all_pt[all_pt['histology_multi_full_genomically.confirmed'].isin(['LUAD', 'LUADx2', 'LUADx3'])].copy()
        temp = self.all_tm.merge(self.all_pt, on='tumour_id_muttable_cruk', how='outer')
        self.clin = self.chist.merge(temp, on='tumour_id_muttable_cruk', how='inner') # because we need primary and LUAD only
        self.primary_luad_tumor_samples = self.clin[['sample_name_cruk', 'tumour_id_muttable_cruk']].drop_duplicates().set_index('sample_name_cruk')
        # Make sure that the columns of self.clin do not vary based on sample_name_cruk (vary by patient, not by sample region)
        n_tumors = self.clin['tumour_id_muttable_cruk'].nunique()
        self.clin = self.clin.drop(columns=['sample_name_cruk']).drop_duplicates().copy()
        assert n_tumors == len(self.clin), "Patient data in self.clin varies by tumor region, wait until after expression QC"
        self.clin = set_idx(self.clin, idx_col='tumour_id_muttable_cruk')
        self.clin = self.clin.drop(columns=['Histology_per_tumour_id_muttable_x', 'Histology_per_tumour_id_muttable_y', 'histology_3', 'histology_lesion1', 'histology_lesion1_merged', 'lesion1_sampled', 'lesion2_sampled'])
        self.clin = self.clin[sorted(self.clin.columns, key=str.lower)]
        self.clin[['age_x', 'years_smoking_x']] = self.clin[['age_x', 'years_smoking_x']].astype('float64')
        for c in ['is.family.lung_x', 'is.family.lung_y']:
            assert self.clin[c].isnull().sum() == 0, f"Null values in {c}, casting to bool will erroneously map NaN to True"
            self.clin[c] = self.clin[c].astype('boolean')
        groupings = {
            **{f'{c}_x': [f'{c}_y'] for c in ['LUAD_pred_subtype_with.IMA_per_tumour', 'age', 'ethnicity', 'cruk_id', 'cigs_perday', 'years_smoking', 'is.family.lung']}, # 'is.family.lung' # only x, y exist
            **{c: [f'{c}_x', f'{c}_y'] for c in ['pathologyTNM', 'smoking_status_merged']}, # base, x, and y exist
            'sex': ['clinical_sex'],
            'pack_years': ['packyears']
        }
        for target, backups in groupings.items(): self.clin = fill_from_backup_cols(self.clin, target, backups)
        self.clin = self.clin.rename(columns={f'{c}_x': c for c in ['LUAD_pred_subtype_with.IMA_per_tumour', 'age', 'ethnicity', 'cruk_id', 'cigs_perday', 'years_smoking', 'is.family.lung']})
        s = self.clin['smoking_status_merged'].str.strip().str.lower()
        mapped = s.map({'ex-smoker': 'ref', 'smoker': 'current', 'never smoked': 'never'})
        assert not set(s[s.notna() & mapped.isna()]), f'Unmapped smoking status: {set(s[s.notna() & mapped.isna()])}'
        self.clin['smoking_status'] = mapped
        print(f"N primary LUAD patients from all sources combined (pre-criteria): {len(self.clin)}")
        self.clin = self.remove_subset(self.clin.index[self.clin['smoking_status'].isna()].tolist(), 'Missing smoking status')
        self.clin = self.remove_subset(self.clin.index[self.clin['age'].isna()].tolist(), 'Missing age')
        self.label_cases()
        self.clin = self.remove_subset(self.clin.index[self.clin['sex'].isna()].tolist(), 'Missing sex')
        self.clin = self.remove_subset(self.clin.index[self.clin['pathologyTNM'].isna()].tolist(), 'Missing stage')
        self.clin = self.remove_subset(self.clin.index[self.clin['pathologyTNM']=='IV'].tolist(), 'Stage IV')
        self.clin = self.remove_subset(self.clin.index[self.clin['pathologyTNM']=='IIIB'].tolist(), 'Stage IIIB')
        if self.req_NT:
            self.clin = self.remove_subset(self.clin.index[self.clin['pN_stage_per_patient'].isna()].tolist(), 'Missing pathologic N')
            assert set(self.clin['pN_stage_per_patient'].unique().tolist()) == {0, 1, 2}, "Unknown value in pN_stage_per_patient" # making sure no NX, like there were in TCGA
            self.clin = self.remove_subset(self.clin.index[self.clin['pT_stage_per_patient'].isna()].tolist(), 'Missing pathologic T')
        self.primary_luad_tumor_samples = self.primary_luad_tumor_samples[self.primary_luad_tumor_samples['tumour_id_muttable_cruk'].isin(self.clin.index)].copy()
        if self.print_excl_numbers:
            print("TRACERx421 cohort is all 7th edition so there is no need to exclude non-7th edition T4 patients") # https://www.nature.com/articles/s41586-023-05783-5
        self.clin = self.clin.drop(columns=['smoking_status_merged'])

    def label_cases(self):
        self.id_tracker['Labeling'] = defaultdict(list)
        keep = self.clin.apply(self.filter_cases, axis=1)
        self.clin = self.remove_subset(self.clin.index[~keep].tolist(), 'No usable recurrence information')
        self.id_tracker['Window'] = defaultdict(list)
        self.ttr = {}
        df2 = self.clin[self.clin.apply(self.restrict_window, axis=1)].copy()
        unusable = self.clin.index.difference(df2.index).tolist()
        self.clin = self.remove_subset(unusable, f'No usable {self.window}-day label')
        assert len(df2) == len(self.clin)
        positives = [i for k, v in self.id_tracker['Window'].items() if k.startswith('1: pass') for i in v]
        self.clin['label_any'] = self.clin.index.isin(positives).astype(int)
        self.clin['ttr'] = pd.Series(self.ttr, dtype='Float64').reindex(self.clin.index).astype('Int64')
    
    def filter_cases(self, row):
        """ Disease-free survival (DFS) - measured from the time of study registration to date of first lung recurrence or death from any cause. Patients who do not have these events are 
        censored at the date last known to be alive (including patients who developed a new primary tumour that has been shown biologically to not be linked to the initial primary lung 
        tumour). Overall survival - measured from the time of study registration to date of death from any cause. (https://www.nature.com/articles/s41586-023-05783-5#Sec52 reporting summary) """
        rec = row.get('first_dfs_any_event_rec.or.new.primary') == 'recurrence'
        no_rec = row.get('cens_dfs_any_event') == 0 # this way we exclude new primaries
        non_rec_death = row.get('first_event_during_followup') == 'Death'
        def track(reason, keep):
            self.id_tracker['Labeling'][reason].append(row.name); return keep
        if rec and no_rec:
            self.id_tracker['Labeling']['Conflicting sources'].append(row.name)
        if rec: return track('Recurrence', True) # 1
        if no_rec: return track('No recurrence', True) # 0
        if non_rec_death: return track('Non recurrence death', True) # 0
        return track('Non recurrence event', False) # second primary, etc (which is why no_rec == 1)
    
    def restrict_window(self, row):
        def track(reason, keep):
            self.id_tracker['Window'][reason].append(row.name); return keep

        def flatten(tiers):
            return [c for tier in tiers for c in tier]

        def check_neg(tiers, tag):
            vals = pd.to_numeric(row[flatten(tiers)], errors='coerce')
            if vals.isna().all(): return track(f'0: missing ({tag})', False)
            return track(f'0: pass ({tag})', True) if (vals >= self.window).any() else track(f'0: fail ({tag})', False)

        def check_pos(tiers, tag):
            flat = flatten(tiers)
            vals = pd.to_numeric(row[flat], errors='coerce')
            valid = vals[vals >= 0]
            for tier in tiers:
                tier_valid = valid[valid.index.isin(tier)]
                if len(tier_valid):
                    self.ttr[row.name] = tier_valid.min()
                    break
            if vals.isna().all(): return track(f'1: missing ({tag})', False)
            if (vals.dropna() <= 0).all(): return track(f'1: invalid ({tag})', False)
            if not self.window_positives: return track(f'1: pass ({tag})', True)
            if valid.between(0, self.window).any(): return track(f'1: pass ({tag})', True)
            return track(f'1: fail ({tag})', True) # recurred after window -> disease-free through it -> label 0 (hence keep=True)
        
        if row.name in self.id_tracker['Labeling']['No recurrence']: return check_neg([['dfs_time'], ['os_time']], 'no rec')
        if row.name in self.id_tracker['Labeling']['Non recurrence death']: return check_neg([['os_time'], ['dfs_time']], 'non rec death')
        if row.name in self.id_tracker['Labeling']['Recurrence']: return check_pos([['Recurrence_time_use'], ['dfs_time']], 'rec') # recurrence_time_use should be populated for all
        raise ValueError(f"UNCLASSIFIED ROW IN WINDOW STEP: {row.name}")

    def get_expression(self):
        expr = load_data('csv', 'rsem_counts_mat', data_path=self.path)
        expr = self._update_gene_symbols(expr, 'gene_id')
        expr = expr.groupby('approved_symbol').sum()
        expr.index = list(expr.index) # removes "approved symbol" label
        expr = expr.T
        # expr regions are pre-filtered for QC (as described in the paper - confirmed with Carlos)
        self.samples_passing_expr_QC = expr.index.tolist()
        pts_passing_QC = self.primary_luad_tumor_samples.loc[self.primary_luad_tumor_samples.index.isin(self.samples_passing_expr_QC), 'tumour_id_muttable_cruk'].unique()
        self.clin = self.remove_subset(self.clin.index[~self.clin.index.isin(pts_passing_QC)].tolist(), 'No expression data passing QC')
        self.primary_luad_tumor_samples = self.primary_luad_tumor_samples[self.primary_luad_tumor_samples['tumour_id_muttable_cruk'].isin(self.clin.index)].copy()
        # now limit regions in expr to those from primary tumors
        expr = pd.concat([expr, self.primary_luad_tumor_samples], axis=1, join='inner')
        self.expr = expr.groupby('tumour_id_muttable_cruk').median() # median across regions (Carlos's suggestion)
        self.expr.index = list(self.expr.index)
        self.expr = self.expr.loc[:, (self.expr > 0).any(axis=0)].add_suffix('_expr') # drop all-zero columns
    
    def get_mutation(self):
        # ANNOVAR variables (https://annovar.openbioinformatics.org/en/latest/articles/wANNOVAR/)
        mut = load_data('csv', 'TRACERx421_mutation_table', data_path=self.path, clean=False)
        sequenced = pd.Index(sorted(mut['tumour_id'].dropna().unique()), name='tumour_id')   # before any filtering (NEW)
        func = mut['func'].astype('string').str.lower().str.strip()          # variant region (exonic, intronic, UTR, etc)
        efunc = mut['exonic.func'].astype('string').str.lower().str.strip()  # exonic consequence (nonsynonymous, stopgain, etc)
        self.variant_classes = func.where(~func.isin(['exonic']), efunc).value_counts()
        is_xy = mut['chr'].astype('string').isin(['chrX','chrY'])
        if self.harmonize:
            is_splice = func.isin(['splicing', 'exonic;splicing'])   # splicing rows have no exonic.func, keep on region alone
            keep = func.isin(ANNOVAR_REGION_KEEP) & (efunc.isin(ANNOVAR_EXONIC_KEEP) | is_splice)
            mut = mut[keep & ~is_xy].copy()
        else:  # pre-revision filter, retained to reproduce the before-harmonization comparison
            is_silent = efunc.isin(['synonymous snv', 'synonymous'])
            is_rna = func.str.startswith('ncrna_', na=False)
            mut = mut[~is_silent & ~is_rna & ~is_xy].copy()
        mut = self._update_gene_symbols(mut, 'Hugo_Symbol')
        mut = mut[mut['tumour_id'].isin(self.clin.index)]
        self.raw_mut = mut.copy()
        self.mut = pd.crosstab(mut['tumour_id'], mut['approved_symbol']).astype(float)
        self.mut = self.mut.reindex(index=sequenced.intersection(self.clin.index), fill_value=0.0) # NEW
        self.mut.columns = [f"{col}_mut" for col in self.mut.columns]

    def _update_gene_symbols(self, df, gene_col, drop_unmapped=True):
        df = df.copy()
        def _mapper_dict(key):
            d = self.hgnc.get(key)
            return {str(k).strip(): v for k, v in d.items()} # normalize keys to string and strip spaces
        def _norm_series(s):
            return s.astype('string').str.strip()
        approved_map = _mapper_dict('Symbol')
        df['approved_symbol'] = _norm_series(df[gene_col]).str.upper().map(approved_map)        
        df['approved_symbol'] = df['approved_symbol'].infer_objects(copy=False)
        if drop_unmapped:
            df = df.dropna(subset=['approved_symbol'])
        return df.drop(columns=[gene_col])
    
    def get_groupings(self):
        """ Expr and mut need clinical otherwise there's no label. """
        clin_expr, clin_mut = self.expr.index.intersection(self.clin.index), self.mut.index.intersection(self.clin.index) # clin + expr; clin + mut
        all3 = clin_expr.intersection(clin_mut) # clin + expr + mut
        expr_no_mut, mut_no_expr, clin_no_omic = clin_expr.difference(all3), clin_mut.difference(all3), self.clin.index.difference(clin_expr.union(clin_mut)) # clin + expr (no mut); clin + mut (no expr); clin only (no genomic)
        return dict(clin_expr=clin_expr, clin_mut=clin_mut, all3=all3, expr_no_mut=expr_no_mut, mut_no_expr=mut_no_expr, clin_no_omic=clin_no_omic)
    
    def finalize_data(self):
        g = self.get_groupings()
        print(f"All have clinical (for label):   Expr + Mut = {len(g['all3'])}   Expr only = {len(g['expr_no_mut'])}   Mut only = {len(g['mut_no_expr'])}   No genomic = {len(g['clin_no_omic'])}")
        self.cohort = sorted(self.get_groupings()['all3'].tolist())
        def slice_and_clean(name):
            df = getattr(self, name)
            df = df.loc[self.cohort].copy()
            if name != 'clin': 
                assert not df.isna().values.any(), f"NaNs detected in {name}"
                df = df.loc[:, df.nunique() > 1].copy()
            return df
        self.clin, self.expr, self.mut = (slice_and_clean(name) for name in ['clin', 'expr', 'mut'])
        self.raw_mut = self.raw_mut[self.raw_mut['tumour_id'].isin(self.cohort)].copy()
        y1, y0 = [self.clin[self.clin['label_any']==i].index.tolist() for i in [1, 0]]
        print(f"{len(self.clin)} patients ({len(y1)} relapsed, {len(y0)} disease-free)") 
        self.clin['label_early'] = (self.clin.index.isin(y1) & (self.clin['ttr'] < self.cutoff)).astype(int)
        y1, y0 = [self.clin[self.clin['label_early']==i].index.tolist() for i in [1, 0]]
        print(f"{len(self.clin)} patients ({len(y1)} early, {len(y0)} late or never)")

def create_smoking_cols(df, words):
    flat_list = [item for sublist in words.values() for item in sublist]
    assert (set(df['smoking_status'].unique()) == set(flat_list)) and (df['smoking_status'].isna().sum() == 0)
    for col, sublist in words.items():
        df[col] = df['smoking_status'].isin(sublist).astype(int)
    return df

def encode_clin(df, maps, use_NT=False):
    clin = df.copy()
    for col, map_ in maps.items():
        if col.startswith('path_') and not use_NT: continue
        unmapped = set(clin[col].dropna().unique()) - set(map_)
        assert not unmapped, f"UNMAPPED VALUES IN {col}: {sorted(unmapped)}"
        clin[col] = clin[col].map(map_)
    return clin

# TracerX expression: Upper-quartile normalize each sample/tumor (row) to 1000 using 75th percentile of nonzero values (TCGA data arrives this way)
def uqx1000(tracerx_expr):
    X = tracerx_expr.to_numpy(dtype=np.float32)
    out = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        v = X[i]
        nz = v[v > 0]
        s = np.percentile(nz, 75) if nz.size else 1.0
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        out[i] = (v / s) * 1000.0
    return pd.DataFrame(out, index=tracerx_expr.index, columns=tracerx_expr.columns)

def transform_expr(tcga_expr, tracerx_expr):
    if not tcga_expr.columns.equals(tracerx_expr.columns):
        raise ValueError('Expression columns do not match (names/order)')
    edfs = [tcga_expr.copy(), uqx1000(tracerx_expr)]
    outs = [np.log2(edf.to_numpy(dtype=np.float32) + 1.0) for edf in edfs]
    return [pd.DataFrame(o, index=edf.index, columns=edf.columns) for o, edf in zip(outs, edfs)]

def transform_mut(kind, tcga_mut, tracerx_mut):
    if not tcga_mut.columns.equals(tracerx_mut.columns):
        raise ValueError('Mutation columns do not match (names/order)')
    mdfs = [tcga_mut.copy(), tracerx_mut.copy()]
    if kind is None:
        return mdfs
    if kind == 'binarize':
        return [(mdf > 0).astype(np.float32) for mdf in mdfs]
    if kind == 'log1p':
        outs = [np.log1p(mdf.to_numpy(dtype=np.float32)) for mdf in mdfs]
        return [pd.DataFrame(o, index=mdf.index, columns=mdf.columns) for o, mdf in zip(outs, mdfs)]
    if kind == 'cap2':
        outs = [np.clip(mdf.to_numpy(dtype=np.float32), 0, 2) / 2.0 for mdf in mdfs]
        return [pd.DataFrame(o, index=mdf.index, columns=mdf.columns) for o, mdf in zip(outs, mdfs)]
    raise ValueError(f'Invalid transformation type: {kind}')

# 1. FLAGS (https://pmc.ncbi.nlm.nih.gov/articles/PMC4267152/ Table S1) — artifact-prone, length/repeat-driven
def get_FLAGS(hgnc): # 
    given_flags = get_data('FLAGS.txt', f'{root}/')['FLAGS'].dropna().tolist()
    given_flags = set(g.upper() for g in given_flags)
    flags = set()
    for sym in given_flags:
        apprvd = hgnc['Symbol'].get(sym)
        if apprvd is not None: flags.add(apprvd)
    return flags

# 2. Shift-adjusted concordance
def binarize(mut, suffix='_mut'):
    b = (mut > 0).astype(int)
    b.columns = [c[:-len(suffix)] if c.endswith(suffix) else c for c in b.columns]
    return b

def concordance_filter(tcga, trcr, min_count=10, ratio_bounds=(0.5, 2.0), nbins=4, adjust='none'):
    """ min_count = floor for 'assessable' (reporting only), either cohort satisfies it; ratio_bounds = (lo, hi) acceptable range for the adjusted frequency ratio; nbins = quantile
    bins for adjust='stratified'; Concordance is judged on the frequency RATIO only (absolute differences scale with prevalence, so a gene mutated in 2% of patients can never 
    exceed a 0.02 difference and passes automatically, while only the most frequently mutated genes can fail. The ratio is scale-free and catches the artifact cases it is meant to 
    catch (FLG 0.13, MUC17 0.32). Use adjust='none' (after harmonization, cohort-wide frequency offset is near 1.0)"""
    # Presence/absence per patient per gene, suffix stripped so the index is bare symbols
    a, b = binarize(tcga.mut), binarize(trcr.mut)
    n_a, n_b = len(a), len(b)                # cohort sizes (denominators)
    k_a, k_b = a.sum(axis=0), b.sum(axis=0)  # mutated-patient counts per gene
    out = pd.DataFrame({'n_tcga': k_a, 'n_trcr': k_b, 'freq_tcga': k_a / n_a, 'freq_trcr': k_b / n_b})
    # abs_diff is reported (and used for the rare-gene exemption below) but is not a rejection criterion
    out['abs_diff'] = (out['freq_trcr'] - out['freq_tcga']).abs()
    out['ratio'] = np.where(out['freq_tcga'] > 0, out['freq_trcr'] / out['freq_tcga'], np.nan)
    out['assessable'] = (out['n_tcga'] >= min_count) | (out['n_trcr'] >= min_count)
    out['usable'] = (out['n_tcga'] > 0) & (out['n_trcr'] > 0)
    ass = out.index[out['assessable']]
    if adjust == 'stratified':
        # TRACERx multi-region sampling inflates rare genes more than common ones, so the correction is estimated within quartiles of TCGA mutation frequency
        out.loc[ass, 'bin'] = pd.qcut(out.loc[ass, 'freq_tcga'], nbins, duplicates='drop')
        shifts = out.loc[ass].groupby('bin', observed=True)['ratio'].median()
        out.loc[ass, 'adj_ratio'] = out.loc[ass, 'ratio'] / out.loc[ass, 'bin'].astype(object).map(shifts).astype(float)
    elif adjust == 'global':
        shifts = out.loc[ass, 'ratio'].median()
        out.loc[ass, 'adj_ratio'] = out.loc[ass, 'ratio'] / shifts
    elif adjust == 'none':
        shifts = 1.0
        out.loc[ass, 'adj_ratio'] = out.loc[ass, 'ratio']
    else: raise ValueError(f"unknown adjust: {adjust}")
    # Concordant = frequencies agree within a factor of two. The abs_diff <= 0.01 clause exempts genes whose frequencies are so close that the ratio is unstable
    lo, hi = ratio_bounds
    out['retain'] = out['usable'] & (out['adj_ratio'].between(lo, hi) | (out['abs_diff'] <= 0.01))
    out.attrs['adjust'], out.attrs['shifts'] = adjust, shifts
    out.attrs['min_count'] = min_count
    return out.sort_values('n_tcga', ascending=False)

def format_for_modeling(tcga, trcr, mut_transform=None, use_NT=False, conc_min_count=0):
    # TCGA: clin
    tcga_map = {'AGE_3': 'age', 'gender_5': 'sex', 'pathologic_stage_5': 'stage', 'pathologic_N_5': 'path_N', 'pathologic_T_5': 'path_T'}
    tcga.clin = tcga.clin.rename(columns=tcga_map)
    tcga.clin_for_table1 = tcga.clin.copy()
    # Time-to-event view, kept aside before clin is subset to model features (TRACERx has no equivalent and none is needed - it's the external test set and is only scored on determinate labels
    tcga.surv = tcga.clin[['fu_time', 'event_obs']].copy() if 'fu_time' in tcga.clin.columns else None
    tcga.clin = create_smoking_cols(tcga.clin, dict(ref_smoker=['ref_sup15', 'ref_sub15', 'ref_unk'], current_smoker=['current'], never_smoker=['never']))
    tcga.clin = tcga.clin[['age', 'sex', 'stage', 'path_N', 'path_T', 'ref_smoker', 'current_smoker', 'never_smoker', 'label_early']] # ttr
    tcga_enc = dict(
        stage={'I': 1, 'IA': 1, 'IB': 1, 'II': 2, 'IIA': 2, 'IIB': 2, 'IIIA': 3}, sex={'FEMALE': 1, 'MALE': 0},
        path_N={'N0': 0, 'N1': 1, 'N2': 2}, 
        path_T={'T1': 1, 'T1a': 1, 'T1b': 1, 'T2': 2, 'T2a': 2, 'T2b': 2, 'T3': 3, 'T4': 4} # no 3a/b or 4a/b
    )
    tcga.clin = encode_clin(tcga.clin, tcga_enc, use_NT)
    if not use_NT: 
        tcga.clin = tcga.clin.drop(columns=['path_N', 'path_T'])
    # label_early is <NA> for censored patients (kept for a time-to-event loss, never scored); features must still be complete
    feat_cols = [c for c in tcga.clin.columns if c != 'label_early']
    bad = tcga.clin[feat_cols].columns[tcga.clin[feat_cols].isna().any()].tolist()
    assert not bad, f'NaNs in TCGA clinical features: {bad}'
    assert tcga.clin['label_early'].notna().any(), 'no determinate labels in TCGA'
    tcga.clin = tcga.clin.reindex(columns=sorted(tcga.clin.columns))
    # TRACERx : clin
    trcr_map = {'pathologyTNM': 'stage', 'pN_stage_per_patient': 'path_N', 'pT_stage_per_patient': 'path_T'}
    trcr.clin = trcr.clin.rename(columns=trcr_map)
    trcr.clin_for_table1 = trcr.clin.copy()
    trcr.clin = create_smoking_cols(trcr.clin, dict(ref_smoker=['ref'], current_smoker=['current'], never_smoker=['never']))
    trcr.clin = trcr.clin[['age', 'sex', 'stage', 'path_N', 'path_T', 'ref_smoker', 'current_smoker', 'never_smoker', 'label_early']] # ttr
    trcr_enc = dict(
        stage={'IA': 1, 'IB': 1, 'IIA': 2, 'IIB': 2, 'IIIA': 3}, sex={'Female': 1, 'Male': 0},
        path_T={'1a': 1, '1b': 1, '2a': 2, '2b': 2, '3': 3, '4': 4} # path_N already pure floats {1,2,3}
    )
    trcr.clin = encode_clin(trcr.clin, trcr_enc, use_NT)
    if not use_NT: 
        trcr.clin = trcr.clin.drop(columns=['path_N', 'path_T'])
    assert not trcr.clin.isna().values.any()
    trcr.clin = trcr.clin.reindex(columns=sorted(trcr.clin.columns))
    assert tcga.clin.columns.tolist() == trcr.clin.columns.tolist(), f"CLIN COLUMN MISMATCH: tcga = {tcga.clin.columns.tolist()}, trcr = {trcr.clin.columns.tolist()}"
    # Both: expr
    overlapping_expr = sorted(list(set(tcga.expr.columns.tolist()) & set(trcr.expr.columns.tolist())))
    tcga.expr, trcr.expr = [df.reindex(columns=overlapping_expr).copy() for df in [tcga.expr, trcr.expr]]
    tcga.expr, trcr.expr = transform_expr(tcga.expr, trcr.expr)
    # Both: mut
    overlapping_mut = sorted(list(set(tcga.mut.columns.tolist()) & set(trcr.mut.columns.tolist())))
    tcga.mut, trcr.mut = [df.reindex(columns=overlapping_mut).copy() for df in [tcga.mut, trcr.mut]]
    conc = concordance_filter(tcga, trcr, min_count=conc_min_count, adjust='none')
    FLAGS = get_FLAGS(tcga.hgnc)
    pool = sorted([g for g in conc.index[conc['retain']] if g not in FLAGS])
    pool_cols = [f'{g}_mut' for g in pool]
    missing = set(pool_cols) - set(tcga.mut.columns)
    assert not missing, f'Pool columns absent from tcga.mut: {sorted(missing)[:10]}'
    print(f"Overlapping genes: {len(overlapping_mut)}; concordant: {conc['retain'].sum()}; after FLAGS removal: {len(pool_cols)}")
    print(f"(reporting only) mutated in >={conc_min_count} patients in either cohort: {conc['assessable'].sum()}")
    tcga.mut, trcr.mut = [df.reindex(columns=pool_cols).copy() for df in [tcga.mut, trcr.mut]]
    tcga.mut, trcr.mut = transform_mut(mut_transform, tcga.mut, trcr.mut)
    return tcga, trcr, {'conc': conc, 'FLAGS': FLAGS, 'overlapping_mut': overlapping_mut, 'pool': pool, 'pool_cols': pool_cols}

class LUNG:
    def __init__(self, tcga, trcr, dp=''):
        self.tcga, self.trcr, self.dp = tcga, trcr, dp
        self.prechecks()
    
    def prechecks(self):
        for mod in ['clin', 'expr', 'mut']: assert getattr(self.tcga, mod).columns.tolist() == getattr(self.trcr, mod).columns.tolist(), f"Column Mismatch: {mod}"
        # Columns are already sorted
        self.cols = {'clin': [c for c in self.tcga.clin.columns.tolist() if c not in ['label_early']], 'expr': self.tcga.expr.columns.tolist(), 'mut': self.tcga.mut.columns.tolist()}
        self.tcga_all, self.trcr_all = [ds.clin.join(ds.expr, how='inner').join(ds.mut, how='inner') for ds in [self.tcga, self.trcr]]
        assert self.tcga_all.columns.tolist() == self.trcr_all.columns.tolist() # confirm post-join uniformity
        for df in [self.tcga_all, self.trcr_all]:
            # label_early is legitimately <NA> for censored patients, so it is judged on its non-null values; every other column is judged with dropna=False
            const = [c for c in df.columns if df[c].nunique(dropna=(c == 'label_early')) <= 1]
            assert not const, f'Constant column(s): {const}'
        self.tcga_x, self.trcr_x = [df.drop(columns=['label_early']).copy() for df in [self.tcga_all, self.trcr_all]]
        self.tcga_y, self.trcr_y = [df['label_early'].copy() for df in [self.tcga_all, self.trcr_all]]
        surv = getattr(self.tcga, 'surv', None)
        self.tcga_surv = None if surv is None else surv.reindex(self.tcga_x.index)
        n_cens = int(self.tcga_y.isna().sum())
        if n_cens:
            assert self.tcga_surv is not None, 'censored patients present but no survival frame; rebuild TCGA with keep_censored=True'
            assert self.tcga_surv.loc[self.tcga_y.isna(), 'fu_time'].notna().all(), 'censored patient without fu_time'
            print(f"  {n_cens} censored patients (label <NA>, usable for a time-to-event loss), {int(self.tcga_y.notna().sum())} with determinate {'early-recurrence'} labels")
        print(f"Clin cols: {len(self.cols['clin'])}  Expr cols: {len(self.cols['expr'])}  Mut cols: {len(self.cols['mut'])}")

class GenePrior:
    def __init__(self, hgnc):
        self.hgnc = hgnc
        self.get_luad_genes()
    
    def get_luad_genes(self):
        symbols = ['ABCB1', 'ABCG2', 'ABL2', 'ALDH1A1', 'ALDOA', 'ALK', 'AQP2', 'ARID1A', 'ARID1B', 'ARID2', 'ASPM', 'ATM', 'ATP1B1', 'BCL2', 'BIRC6', 'BRAF', 'CAT', 'CAV2', 'CCDC174', 'CCDC6', 'CCND3', 'CCNE1', 'CD109', 'CD274', 'CD44', 'CD74', 'CDK4', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'CDPF1', 'CHRNA3', 'CHRNA5', 'CLPTM1L', 'CLTC', 'COL1A1', 'CTLA4', 'CTNNB1', 'CTNND2', 'CYP1A1', 'CYP2D6', 'CYP2E1', 'DCTN1', 'DEPP1', 'DLGAP5', 'EGFR', 'EML4', 'ENTPD2', 'ERBB2', 'ERBB3', 'EXO1', 'EZR', 'FAM83A', 'FGFR1', 'FOXO3', 'FOXRED2', 'FRRS1', 'GATA2', 'GATA4', 'GATA5', 'GCC2', 'GNPNAT1', 'GOPC', 'GPX3', 'GSTM1', 'GSTP1', 'GSTT1', 'HES1', 'HEY1', 'HEY2', 'HIC1', 'HIF1A', 'HIP1', 'HOXA9', 'HOXD13', 'HRAS', 'HYKK', 'IGF1R', 'JAK1', 'JAK2', 'KDELR2', 'KDR', 'KEAP1', 'KIF5B', 'KLC1', 'KRAS', 'LDHA', 'LMO7', 'LRIG3', 'MAP2K1', 'MAP2K5', 'MDM2', 'MECOM', 'MGAM', 'MMP2', 'MMP9', 'MYC', 'NANOG', 'NBPF1', 'NF1', 'NFE2L2', 'NKX2-1', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4', 'NRAS', 'NRG1', 'NTRK1', 'NTRK2', 'NTRK3', 'PCP4', 'PDGFA', 'PDGFB', 'PDGFRA', 'PDGFRB', 'PEX3', 'PHACTR1', 'PIK3CA', 'POU4F2', 'POU5F1', 'PROM1', 'PTEN', 'PTPN11', 'PTPN3', 'RAD52', 'RASSF1', 'RB1', 'RET', 'RHOV', 'RIT1', 'ROS1', 'S100P', 'SDC4', 'SETD2', 'SFRP1', 'SLC34A2', 'SLC3A2', 'SLC4A5', 'SLTM', 'SMARCA4', 'SOX17', 'SOX2', 'SPOCK2', 'SQSTM1', 'STK11', 'STRN', 'STX2', 'TCN1', 'TERT', 'TFG', 'TIMP2', 'TOP1', 'TP53', 'TPM3', 'TPR', 'TPTE', 'TRIM33', 'TRRAP', 'TYMS', 'U2AF1', 'VEGFA', 'WIF1', 'ZEB1', 'ZKSCAN1']
        df = pd.DataFrame({'symbol': sorted(symbols)})
        # Already mapped to their approved symbols but for the sake of consistency...
        df = self.update_gene_symbols(df, 'symbol')
        self.luad_genes = set(df['approved_symbol'])
        
    def update_gene_symbols(self, df, gene_col, drop_unmapped=True):
        df = df.copy()
        def _mapper_dict(key):
            d = self.hgnc.get(key)
            return {str(k).strip(): v for k, v in d.items()} # normalize keys to string and strip spaces
        def _norm_series(s):
            return s.astype('string').str.strip()
        approved_map = _mapper_dict('Symbol')
        df['approved_symbol'] = _norm_series(df[gene_col]).str.upper().map(approved_map)
        df['approved_symbol'] = df['approved_symbol'].infer_objects(copy=False)
        if drop_unmapped: df = df.dropna(subset=['approved_symbol'])
        return df.drop(columns=[gene_col])

""" Table1 columns are All / Early / Late-or-never, defined on the determinate labels only: censored patients (label_early <NA>) train through the survival loss but are never scored, so 
putting them in 'Late/Never' would relabel them as negatives. Timing rows are restricted to the subgroup they describe:
  * time to recurrence exists only for patients who recurred, so the Late/Never cell covers the late recurrences alone and the never-recurred patients contribute nothing to it
  * follow-up time answers 'how long were the event-free patients actually watched', so it is reported for the never-recurred patients alone 
follow-up is the longest time on record, taken row-wise across these columns — the same rule check_neg uses to decide whether a patient reached the window, and the same rule that builds 
TCGA's fu_time (censor = valid.max() across every follow-up tier). TRACERx has no fu_time/event_obs of its own (scored only on determinate labels, no training loss) so its time-at-risk 
comes from the cohort's disease-free and overall survival times, which coincide for an event-free patient and for a non-recurrence death, and differ only where one censors earlier """

BASE_COLS = ['sex', 'stage', 'path_N', 'path_T', 'smoking_status', 'age']
EXTRA_COLS = {'tcga': ['race_4', 'ethnicity_5'], 'trcr': ['ethnicity']}
FU_COL = {'tcga': ['fu_time'], 'trcr': ['dfs_time', 'os_time']}
TTR_COL = 'ttr'
LABELS = {'sex': 'Sex', 'stage': 'Pathologic stage', 'path_N': 'Pathologic N', 'path_T': 'Pathologic T', 'smoking_status': 'Smoking status', 'age': 'Age (years)',  'race_4': 'Race', 'ethnicity_5': 'Ethnicity', 'ethnicity': 'Ethnicity'}

class Table1:
    def __init__(self, tcga, tracerx, decpts=2, base_cols=None, extra_cols=None, fu_col=None, labels=None, clean_up=True):
        self.tcga = tcga
        self.trcr = tracerx
        self.decpts = decpts
        self.clean_up = clean_up
        self.base_cols = list(BASE_COLS if base_cols is None else base_cols)
        self.extra_cols = dict(EXTRA_COLS if extra_cols is None else extra_cols)
        self.fu_col = dict(FU_COL if fu_col is None else fu_col)
        self.labels = dict(LABELS, **(labels or {}))
        self.n_censored = {} # per dataset; a single attribute would be overwritten by get_tables
        self.skipped = {}
        
    def clean(self, df, key):
        tcga_enc = dict(
            stage={'I': 'IA', 'IA': 'IA', 'IB': 'IB', 'II': 'IIA', 'IIA': 'IIA', 'IIB': 'IIB', 'IIIA': 'IIIA'},
            path_T={'T1': 'T1', 'T1a': 'T1', 'T1b': 'T1', 'T2': 'T2', 'T2a': 'T2', 'T2b': 'T2', 'T3': 'T3', 'T4': 'T4'}, # no 3a/b or 4a/b
            smoking_status={'ref_sup15': 'former', 'ref_sub15': 'former', 'ref_unk': 'former', 'current': 'current', 'never': 'never'}
        )
        trcr_enc = dict(
            path_T={'1a': 'T1', '1b': 'T1', '2a': 'T2', '2b': 'T2', '3': 'T3', '4': 'T4'},
            smoking_status={'ref': 'former', 'current': 'current', 'never': 'never'}
        )
        enc = {'tcga': tcga_enc, 'trcr': trcr_enc}[key]
        return encode_clin(df, enc, True)
  
    def get_idxs(self, df, dataset):
        """ All / Early / Late-or-never, plus the recurrence mask the timing rows need. 'Recurred' is ttr.notna(): ttr is populated only on the positive branch of restrict_window, in 
        both cohorts; where event_obs exists it must agree (the assertion catches a labelling change that breaks that) """
        det = df['label_early'].notna()
        self.n_censored[dataset] = int((~det).sum())
        recurred = df[TTR_COL].notna() if TTR_COL in df.columns else pd.Series(False, index=df.index)
        if 'event_obs' in df.columns:
            assert recurred.eq(df['event_obs'].astype(bool)).all(), 'ttr and event_obs disagree on who recurred'
        self.recurred = recurred
        return df.index[det], df.index[det & (df['label_early'] == 1)], df.index[det & (df['label_early'] == 0)]
 
    def is_continuous(self, s, max_cats=10): # numeric with >max_cats distinct non-nan values -> continuous
        if not is_numeric_dtype(s):
            return False
        return s.dropna().nunique() > max_cats
  
    def _half_up(self, x, decpts=None): # round half away from zero, not numpy/Python's half-to-even
        decpts = self.decpts if decpts is None else decpts
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        exp = Decimal(1).scaleb(-decpts)
        return float(Decimal(repr(float(x))).quantize(exp, rounding=ROUND_HALF_UP))
 
    def median(self, idxs, s, as_str=True):
        v = pd.to_numeric(s.loc[idxs], errors='coerce').dropna() if len(idxs) else pd.Series(dtype=float)
        med = float(np.median(v)) if len(v) else np.nan      # np.nanmedian warns on an all-nan slice; this does not
        med_out = self._half_up(med)
        if not as_str:
            return med_out
        return '—' if np.isnan(med_out) else f"{med_out:.{self.decpts}f}"
 
    def iqr(self, x, as_str=True):
        x = pd.to_numeric(x, errors='coerce').dropna().to_numpy()
        if x.size == 0:
            return '—' if as_str else np.nan
        q1, q3 = (self._half_up(v) for v in np.percentile(x, [25, 75]))
        return f"{q1:.{self.decpts}f}-{q3:.{self.decpts}f}" if as_str else (q1, q3)
 
    def count(self, idxs, lvl, sc, add_pct=True):
        ct = int((sc.loc[idxs] == lvl).sum())
        if not add_pct:
            return ct
        if len(idxs) == 0: # an empty group would divide by zero
            return f"{ct} (—)"
        pct = int((Decimal(ct * 100) / Decimal(len(idxs))).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        return f"{ct} ({pct})"
  
    def _cont_rows(self, label, s, idx_all, idx_early, idx_late, with_n=False):
        rows = []
        if with_n: # restricted blocks state their own denominator
            rows.append((f"{label} — n", len(idx_all), len(idx_early), len(idx_late)))
        rows.append((f"{label} — median", self.median(idx_all, s), self.median(idx_early, s), self.median(idx_late, s)))
        rows.append((f"{label} — IQR", self.iqr(s.loc[idx_all]), self.iqr(s.loc[idx_early]), self.iqr(s.loc[idx_late])))
        return rows
 
    def _cat_rows(self, label, s, idx_all, idx_early, idx_late):
        sc = s.astype('object').where(s.notna(), 'Missing')
        levels = sorted(sc.unique(), key=lambda x: str(x))
        if 'Missing' in levels:
            levels = [l for l in levels if l != 'Missing'] + ['Missing']
        return [(f"{label} = {lvl}", self.count(idx_all, lvl, sc), self.count(idx_early, lvl, sc), self.count(idx_late, lvl, sc)) for lvl in levels]
 
    def _timing_rows(self, df, dataset, idx_early, idx_late):
        """ Time to recurrence among those who recurred, and follow-up among those who did not. Late/Never is split by the recurrence mask: the late recurrences supply time to recurrence, 
        the never-recurred supply follow-up (writing both on the full column would average two different quantities) """
        rows, rec = [], self.recurred
        late_rec = idx_late[rec.loc[idx_late]] # recurred, but after the window
        never = idx_late[~rec.loc[idx_late]]
        if TTR_COL in df.columns:
            rows += self._cont_rows('Time to recurrence (days), recurred only', df[TTR_COL], idx_early.union(late_rec), idx_early, late_rec, with_n=True)
        fu_cols = self.fu_col.get(dataset) or []
        fu_cols = [fu_cols] if isinstance(fu_cols, str) else list(fu_cols)
        have = [c for c in fu_cols if c in df.columns]
        self.skipped.setdefault(dataset, []).extend([c for c in fu_cols if c not in df.columns])
        if have:
            fu = df[have].apply(pd.to_numeric, errors='coerce').max(axis=1) # longest time on record, as check_neg does
            empty = df.index[[]] # follow-up is reported for the never-recurred alone
            rows += self._cont_rows('Follow-up (days), never recurred', fu, never, empty, never, with_n=True)
        return rows
  
    def table1(self, dataset='tcga', fn=''):
        assert dataset in ('tcga', 'trcr'), "Dataset must be 'tcga' or 'trcr'"
        df = (self.tcga if dataset == 'tcga' else self.trcr).clin_for_table1.copy()
        if self.clean_up:
            df = self.clean(df, dataset)
        all_idx, early_idx, late_idx = self.get_idxs(df, dataset)
        wanted = self.base_cols + list(self.extra_cols.get(dataset, []))
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            self.skipped.setdefault(dataset, []).extend(missing)
            print(f"[table1:{dataset}] not in clin_for_table1, skipped: {missing}")
        cols = [c for c in wanted if c in df.columns]
        rows = [("N (patients)", len(all_idx), len(early_idx), len(late_idx))]
        for col in cols:
            label, s = self.labels.get(col, col), df[col]
            rows += self._cont_rows(label, s, all_idx, early_idx, late_idx) if self.is_continuous(s) else self._cat_rows(label, s, all_idx, early_idx, late_idx)
        rows += self._timing_rows(df, dataset, early_idx, late_idx)
        if self.n_censored.get(dataset):
            rows.append(("Censored, excluded above (training only)", self.n_censored[dataset], '—', '—'))
        out = pd.DataFrame(rows, columns=['Feature', 'All', 'Early', 'Late/Never'], dtype=object)
        if fn:
            write_file('csv', fn, out, 'results/')
        return out
 
    def get_tables(self, tcga_fn='', trcr_fn=''):
        self.tcga_table1 = self.table1('tcga', tcga_fn)
        self.trcr_table1 = self.table1('trcr', trcr_fn)