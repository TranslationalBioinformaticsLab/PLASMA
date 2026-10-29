import inspect
import os
import pandas as pd
import numpy as np
from lung_data_prep import TCGA, TRACERx, format_for_modeling, get_data
from base_utils import load_data

def variant_class_tables(raw_tcga, raw_trcr):
    """ Consequence distributions in each cohort BEFORE harmonization, returns (tcga_table, trcr_region_table, trcr_exonic_table, noncoding_shares). Note: synonymous/silent, splice-region 
    (near miss), and TRACERx's "unknown" label are excluded by harmonization but are not counted as "non-coding" here, since none of the three is non-coding by genomic definition """
    t = raw_tcga['Variant_Classification'].value_counts(dropna=False)
    t = pd.DataFrame({'count': t, 'proportion': t / t.sum()})
    r_region = raw_trcr['func'].astype('string').str.lower().str.strip().value_counts(dropna=False)
    r_region = pd.DataFrame({'count': r_region, 'proportion': r_region / r_region.sum()})
    r_exonic = raw_trcr['exonic.func'].astype('string').str.lower().str.strip().value_counts(dropna=False)
    r_exonic = pd.DataFrame({'count': r_exonic, 'proportion': r_exonic / r_exonic.sum()})
    # Non-coding share — the headline asymmetry in the response
    TCGA_NONCODING = {"Intron", "3'UTR", "5'UTR", "5'Flank", "3'Flank", "RNA", "IGR"}
    TRCR_NONCODING_PREFIXES = ('intronic', 'utr5', 'utr3', 'intergenic', 'upstream', 'downstream', 'ncrna')
    tcga_nc = t.loc[[i for i in t.index if i in TCGA_NONCODING], 'count'].sum() / t['count'].sum()
    trcr_nc = r_region.loc[[i for i in r_region.index if isinstance(i, str) and i.startswith(TRCR_NONCODING_PREFIXES)], 'count'].sum() / r_region['count'].sum()
    shares = {'tcga_noncoding_pct': 100 * tcga_nc, 'trcr_noncoding_pct': 100 * trcr_nc}
    return t, r_region, r_exonic, shares

""" Each bucket's native label(s) in each cohort. The union of every 'maf' set equals MAF_KEEP, and the union of every 'annovar_exonic' set equals ANNOVAR_EXONIC_KEEP (Translation start site 
excepted, see TRANSLATION_START_NOTE) """
HARMONIZED_BUCKETS = {
    'Missense':               {'maf': {'Missense_Mutation'}, 'annovar_exonic': {'nonsynonymous snv', 'nonsynonymous'}},
    'Nonsense (stopgain)':    {'maf': {'Nonsense_Mutation'}, 'annovar_exonic': {'stopgain snv', 'immediate-stopgain'}},
    'Stop-loss':              {'maf': {'Nonstop_Mutation'}, 'annovar_exonic': {'stoploss snv'}},
    'Frameshift indel':       {'maf': {'Frame_Shift_Del', 'Frame_Shift_Ins'}, 'annovar_exonic': {'frameshift substitution', 'frameshift insertion'}},
    'In-frame indel':         {'maf': {'In_Frame_Del', 'In_Frame_Ins'}, 'annovar_exonic': {'nonframeshift substitution', 'nonframeshift insertion'}},
    'Splice-altering':        {'maf': {'Splice_Site'}, 'annovar_region': {'splicing', 'exonic;splicing'}},
    'Translation start site': {'maf': {'Translation_Start_Site'}, 'annovar_exonic': None},
}
TRANSLATION_START_NOTE = "No distinct TRACERx label: start-codon-loss call is absorbed into nonsynonymous SNV, frameshift, or nonframeshift depending on type. Retained as this is not a dropped or absent call. Not double-counted under the parent categories above."
FRAMESHIFT_INDEL_NOTE = "TRACERx has no separate 'frameshift deletion' label; reported as frameshift substitution. Verified against the full observed exonic.func value set."
INFRAME_INDEL_NOTE = "TRACERx has no separate 'nonframeshift deletion' label; reported as nonframeshift substitution. Verified against the full observed exonic.func value set."
SPLICE_REGION_NOTE = "TRACERx/ANNOVAR's splicing window has no separate near-splice category to map this onto; if a variant falls inside ANNOVAR's own splicing window, it's captured as splice-altering above. Otherwise, it is indistinguishable from an ordinary intronic/exonic call."

def _wide_to_long(tab):
    rows = []
    for label, row in tab.drop(index='Total').iterrows():
        if pd.notna(row['n_tcga']): rows.append((label, 'TCGA', int(row['n_tcga']), round(row['pct_tcga'] / 100, 4)))
        if pd.notna(row['n_trcr']): rows.append((label, 'TRACERx', int(row['n_trcr']), round(row['pct_trcr'] / 100, 4)))
    return pd.DataFrame(rows, columns=['harmonized_category', 'cohort', 'count', 'proportion'])

def harmonized_composition(tcga_raw_mut, trcr_raw_mut, maf_col='Variant_Classification', trcr_func_col='func', trcr_efunc_col='exonic.func', layout='wide'):
    """ Composition of the retained (post-harmonization) calls, in shared bucket labels, for both cohorts side by side. Note: a row with func == 'exonic;splicing' is counted only under 
    Splice-altering, never also under whatever its own exonic.func value happens to be. Those rows have a real exonic.func label, so counting them under both buckets would double-count a 
    handful of rows and the per-cohort totals below would stop reconciling """
    if layout not in ('wide', 'long'):
        raise ValueError(f"layout must be 'wide' or 'long', got {layout!r}")
    maf_counts = tcga_raw_mut[maf_col].value_counts()
    func = trcr_raw_mut[trcr_func_col].astype('string').str.lower().str.strip()
    efunc = trcr_raw_mut[trcr_efunc_col].astype('string').str.lower().str.strip()
    is_splice_row = func.isin({'splicing', 'exonic;splicing'})
    is_pure_exonic = func == 'exonic'
    rows = []
    for bucket, spec in HARMONIZED_BUCKETS.items():
        n_tcga = int(maf_counts.reindex(spec.get('maf', set()), fill_value=0).sum())
        if 'annovar_region' in spec:
            n_trcr = int(is_splice_row.sum())
        elif spec.get('annovar_exonic') is None:
            n_trcr = np.nan # see TRANSLATION_START_NOTE
        else:
            n_trcr = int((is_pure_exonic & efunc.isin(spec['annovar_exonic'])).sum())
        rows.append((bucket, n_tcga, n_trcr))
    tab = pd.DataFrame(rows, columns=['harmonized_category', 'n_tcga', 'n_trcr']).set_index('harmonized_category')
    n_tcga_total, n_trcr_total = len(tcga_raw_mut), len(trcr_raw_mut)
    tab['pct_tcga'] = (100 * tab['n_tcga'] / n_tcga_total).round(2)
    tab['pct_trcr'] = (100 * tab['n_trcr'] / n_trcr_total).round(2)
    got_tcga = int(tab['n_tcga'].sum())
    got_trcr = int(tab.loc[tab['n_trcr'].notna(), 'n_trcr'].sum())
    assert got_tcga == n_tcga_total, f"TCGA bucket counts ({got_tcga}) don't sum to retained rows ({n_tcga_total}) -- a MAF_KEEP category is missing from HARMONIZED_BUCKETS"
    assert got_trcr == n_trcr_total, f"TRACERx bucket counts ({got_trcr}) don't sum to retained rows ({n_trcr_total}) -- an ANNOVAR_EXONIC_KEEP/ANNOVAR_REGION_KEEP category is missing from HARMONIZED_BUCKETS"
    tab.loc['Total'] = [n_tcga_total, n_trcr_total, 100.0, 100.0]
    tab = tab[['n_tcga', 'pct_tcga', 'n_trcr', 'pct_trcr']]
    return _wide_to_long(tab) if layout == 'long' else tab

EXCLUDED_BUCKETS = {
    'Synonymous / silent':   {'maf': {'Silent'}, 'annovar_exonic': {'synonymous snv', 'synonymous'}},
    'Intronic':              {'maf': {'Intron'}, 'annovar_region_prefix': ('intronic',)},
    "UTR (5'/3')":           {'maf': {"5'UTR", "3'UTR"}, 'annovar_region_prefix': ('utr5', 'utr3')},
    'Flanking / intergenic': {'maf': {"5'Flank", "3'Flank", 'IGR'}, 'annovar_region_prefix': ('upstream', 'downstream', 'intergenic')},
    'Non-coding RNA':        {'maf': {'RNA'}, 'annovar_region_prefix': ('ncrna',)},
    'Splice region (near-splice, non-canonical)': {'maf': {'Splice_Region'}, 'annovar_exonic': None},
    'Unknown / ambiguous':   {'maf': None, 'annovar_exonic': {'unknown'}},
}

def consequence_composition_before(raw_tcga, raw_trcr, maf_col='Variant_Classification', trcr_func_col='func', trcr_efunc_col='exonic.func', layout='wide'):
    """ Composition of all variants, before harmonization, in the same harmonized category labels """
    if layout not in ('wide', 'long'):
        raise ValueError(f"layout must be 'wide' or 'long', got {layout!r}")
    maf_counts = raw_tcga[maf_col].value_counts()
    func = raw_trcr[trcr_func_col].astype('string').str.lower().str.strip()
    efunc = raw_trcr[trcr_efunc_col].astype('string').str.lower().str.strip()
    is_splice_row = func.isin({'splicing', 'exonic;splicing'})
    is_pure_exonic = func == 'exonic'
    rows = []
    for bucket, spec in {**HARMONIZED_BUCKETS, **EXCLUDED_BUCKETS}.items():
        n_tcga = int(maf_counts.reindex(spec['maf'], fill_value=0).sum()) if spec.get('maf') else 0
        if 'annovar_region' in spec:
            n_trcr = int(is_splice_row.sum())
        elif 'annovar_region_prefix' in spec:
            n_trcr = int(func.str.startswith(spec['annovar_region_prefix']).sum())
        elif spec.get('annovar_exonic') is None:
            n_trcr = np.nan # see TRANSLATION_START_NOTE
        else:
            n_trcr = int((is_pure_exonic & efunc.isin(spec['annovar_exonic'])).sum())
        rows.append((bucket, n_tcga, n_trcr))
    tab = pd.DataFrame(rows, columns=['harmonized_category', 'n_tcga', 'n_trcr']).set_index('harmonized_category')
    n_tcga_total, n_trcr_total = len(raw_tcga), len(raw_trcr)
    tab['pct_tcga'] = (100 * tab['n_tcga'] / n_tcga_total).round(2)
    tab['pct_trcr'] = (100 * tab['n_trcr'] / n_trcr_total).round(2)
    got_tcga = int(tab['n_tcga'].sum())
    got_trcr = int(tab.loc[tab['n_trcr'].notna(), 'n_trcr'].sum())
    assert got_tcga == n_tcga_total, f"TCGA bucket counts ({got_tcga}) don't sum to all rows ({n_tcga_total}) -- some Variant_Classification label isn't covered by HARMONIZED_BUCKETS or EXCLUDED_BUCKETS"
    assert got_trcr == n_trcr_total, f"TRACERx bucket counts ({got_trcr}) don't sum to all rows ({n_trcr_total}) -- some func/exonic.func label isn't covered by HARMONIZED_BUCKETS or EXCLUDED_BUCKETS"
    tab.loc['Total'] = [n_tcga_total, n_trcr_total, 100.0, 100.0]
    tab = tab[['n_tcga', 'pct_tcga', 'n_trcr', 'pct_trcr']]
    return _wide_to_long(tab) if layout == 'long' else tab

def consequence_crosswalk():
    """ Which native labels in each cohort map onto each harmonized consequence bucket (both retained and excluded) """
    rows = [
        ('Missense', 'Missense_Mutation', 'nonsynonymous SNV', 'Yes', ''),
        ('Nonsense (stopgain)', 'Nonsense_Mutation', 'stopgain SNV', 'Yes', ''),
        ('Stop-loss', 'Nonstop_Mutation', 'stoploss SNV', 'Yes', ''),
        ('Frameshift indel', 'Frame_Shift_Del, Frame_Shift_Ins', 'frameshift substitution / insertion', 'Yes', FRAMESHIFT_INDEL_NOTE),
        ('In-frame indel', 'In_Frame_Del, In_Frame_Ins', 'nonframeshift substitution / insertion', 'Yes', INFRAME_INDEL_NOTE),
        ('Splice-altering', 'Splice_Site', 'splicing, exonic;splicing', 'Yes', ''),
        ('Translation start site', 'Translation_Start_Site', '(none)', 'Yes (TCGA only)', TRANSLATION_START_NOTE),
        ('Synonymous / silent', 'Silent', 'synonymous SNV', 'No', 'Not protein-altering'),
        ('Intronic', 'Intron', 'intronic', 'No', ''),
        ("UTR (5'/3')", "5'UTR, 3'UTR", 'utr5, utr3', 'No', ''),
        ('Flanking / intergenic', "5'Flank, 3'Flank, IGR", 'upstream, downstream, intergenic', 'No', ''),
        ('Non-coding RNA', 'RNA', 'ncrna_exonic, ncrna_intronic, ncrna_splicing, ncrna_utr3/5', 'No', ''),
        ('Splice region (near-splice, non-canonical)', 'Splice_Region', '(none)', 'No', SPLICE_REGION_NOTE),
        ('Unknown / ambiguous', '(none)', 'unknown', 'No', ''),
    ]
    return pd.DataFrame(rows, columns=['harmonized_category', 'tcga_maf_label(s)', 'tracerx_annovar_label(s)', 'retained', 'notes']).set_index('harmonized_category')

def summary_row(min_count):
    return f'All assessable (min_count={min_count})'

def report_view(conc, min_count):
    out = conc.copy()
    out['assessable'] = (out['n_tcga'] >= min_count) | (out['n_trcr'] >= min_count)
    out.attrs = dict(conc.attrs, min_count=min_count)
    return out

def shift_table(conc, nbins=4):
    """ Median TRACERx:TCGA frequency ratio within quartiles of TCGA frequency; provides core evidence that harmonization worked """
    floor = conc.attrs.get('min_count', '?')
    ass = conc[conc['assessable']].copy()
    ass['bin'] = pd.qcut(ass['freq_tcga'], nbins, duplicates='drop')
    tab = ass.groupby('bin', observed=True).agg(
        n_genes=('ratio', 'size'), median_freq_tcga=('freq_tcga', 'median'), median_freq_trcr=('freq_trcr', 'median'), median_ratio=('ratio', 'median'),
        q1_ratio=('ratio', lambda s: s.quantile(0.25)), q3_ratio=('ratio', lambda s: s.quantile(0.75)),
    )
    tab.index = [f'Q{i+1} ({iv.left:.4f}–{iv.right:.4f})' for i, iv in enumerate(tab.index)]
    tab.loc[summary_row(floor)] = [len(ass), ass['freq_tcga'].median(), ass['freq_trcr'].median(), ass['ratio'].median(), ass['ratio'].quantile(0.25), ass['ratio'].quantile(0.75)]
    tab = tab.round(4)
    tab['n_genes'] = tab['n_genes'].astype(int) # the row assignment above floats the column
    print(f'shift_table: {len(ass)} of {len(conc)} genes assessable at min_count={floor}')
    return tab

def paired_shift(conc_before, conc_after, nbins=4):
    """ Before/after median ratio on one gene set with one set of bins (two shift_table outputs aren't strictly comparable, harmonization removes non-coding
    calls so fewer genes clear the reporting floor afterwards and qcut re-derives its edges from each gene set) """
    b, a = conc_before, conc_after
    floor_b, floor_a = b.attrs.get('min_count', '?'), a.attrs.get('min_count', '?')
    if floor_b != floor_a:
        raise ValueError(f'reporting floors differ: before={floor_b}, after={floor_a}')
    shared = b.index.intersection(a.index)
    keep = shared[(b.loc[shared, 'assessable'].to_numpy() & a.loc[shared, 'assessable'].to_numpy())]
    df = pd.DataFrame({'freq_tcga': b.loc[keep, 'freq_tcga'], 'ratio_before': b.loc[keep, 'ratio'], 'ratio_after': a.loc[keep, 'ratio']})
    df['bin'] = pd.qcut(df['freq_tcga'], nbins, duplicates='drop') # bins from the BEFORE state
    tab = df.groupby('bin', observed=True).agg(n_genes=('ratio_before', 'size'), median_freq_tcga=('freq_tcga', 'median'), median_ratio_before=('ratio_before', 'median'), median_ratio_after=('ratio_after', 'median'))
    tab.index = [f'Q{i+1} ({iv.left:.4f}–{iv.right:.4f})' for i, iv in enumerate(tab.index)]
    tab.loc[summary_row(floor_b)] = [len(df), df['freq_tcga'].median(), df['ratio_before'].median(), df['ratio_after'].median()]
    tab = tab.round(4)
    tab['n_genes'] = tab['n_genes'].astype(int)
    print(f'paired_shift: {len(df)} of {len(shared)} shared genes assessable in both states at min_count={floor_b}')
    return tab

def attrition(conc, FLAGS, pool, n_overlapping=None):
    """ Where genes are lost, step by step """
    n_total = len(conc)
    n_usable = int(conc['usable'].sum())
    n_retain = int(conc['retain'].sum())
    n_pool = len(pool)
    d = {
        'n_overlapping_genes': n_overlapping if n_overlapping is not None else n_total,
        'n_assessable_reporting': int(conc['assessable'].sum()),
        'n_usable_both_cohorts': n_usable,
        'n_removed_not_in_both': n_total - n_usable,
        'n_retained_concordant': n_retain,
        'n_removed_by_concordance': n_usable - n_retain,
        'n_removed_by_FLAGS': n_retain - n_pool,
        'n_final_pool': n_pool,
    }
    return d

def discordance_direction(conc, ratio_bounds=(0.5, 2.0)):
    """ How many excluded genes run high in each cohort, and their frequency profile """
    lo, hi = ratio_bounds
    u = conc[conc['usable']]
    fails = u[~u['retain']]
    r = fails['adj_ratio'].where(fails['adj_ratio'].notna(), fails['ratio'])
    n_lo, n_hi = int((r < lo).sum()), int((r > hi).sum())
    n_untested = len(fails) - n_lo - n_hi
    if u['adj_ratio'].notna().all():
        assert n_untested == 0, f'{n_untested} genes unclassified where every ratio was tested'
    return {
        'n_excluded_tcga_high': n_lo,
        'n_excluded_trcr_high': n_hi,
        'n_excluded_ratio_untested': n_untested,
        'median_freq_tcga_among_tcga_high': float(fails.loc[r < lo, 'freq_tcga'].median()),
        'median_freq_tcga_among_trcr_high': float(fails.loc[r > hi, 'freq_tcga'].median()),
    }

def example_loci(conc, genes=('FLG', 'MUC17', 'AHNAK2')):
    """ Table of the named artifact-prone genes for the reviewer response text """
    present = [g for g in genes if g in conc.index]
    cols = ['n_tcga', 'n_trcr', 'freq_tcga', 'freq_trcr', 'ratio', 'adj_ratio', 'abs_diff', 'usable', 'retain']
    out = conc.loc[present, [c for c in cols if c in conc.columns]].copy()
    out['fold_higher_in_tcga'] = (out['freq_tcga'] / out['freq_trcr'].replace(0, pd.NA)).round(1)
    return out.round(4)

def all_hmzn_numbers(conc, FLAGS, pool, n_overlapping=None, raw_tcga=None, raw_trcr=None, conc_report=None):
    """ conc: the pool view, which the attrition and discordance counts describe. conc_report: the reporting view (report_view(conc, 10)), which the quartile table is computed on. 
    Passing it here rather than overwriting tables['shift_by_quartile'] afterwards keeps what is printed and what is saved the same """
    nums, tables = {}, {}
    tables['shift_by_quartile'] = shift_table(conc if conc_report is None else conc_report)
    nums.update(attrition(conc, FLAGS, pool, n_overlapping))
    if conc_report is not None:
        nums['n_assessable_reporting'] = int(conc_report['assessable'].sum())
    nums.update(discordance_direction(conc))
    tables['example_loci'] = example_loci(conc)
    nums['n_FLAGS_genes_total'] = len(FLAGS)
    if raw_tcga is not None and raw_trcr is not None:
        t, rr, re_, shares = variant_class_tables(raw_tcga, raw_trcr)
        tables['tcga_variant_classes'] = t
        tables['trcr_regions'] = rr
        tables['trcr_exonic_func'] = re_
        nums.update(shares)
    print('=== Numbers ===')
    for k, v in nums.items():
        print(f'  {k:32} {v:.4g}' if isinstance(v, float) else f'  {k:32} {v}')
    for name, tab in tables.items():
        print(f'\n=== {name} ===')
        print(tab.to_string())
    return nums, tables

def get_mutation_hmzn_numbers(ignore, hgnc, pool_min_count=0, report_min_count=10, composition_layout='wide', **init_args):
    """ Two floors, deliberately different. pool_min_count=0 gates the feature pool: every usable gene faces the ratio test and prevalence is left to the training fold. report_min_count=10 
    gates the published concordance statistics: below ~10 events the frequency ratio is pinned to integer multiples of n_tcga/n_trcr, so a median over rarer genes measures that lattice 
    rather than the cohorts """
    trcr_ok = set(inspect.signature(TRACERx.__init__).parameters) - {'self'}
    trcr_args = {k: v for k, v in init_args.items() if k in trcr_ok}
    dropped = sorted(set(init_args) - set(trcr_args))
    if dropped: print(f'(TCGA-only arguments not passed to TRACERx: {", ".join(dropped)})')
    def build(harmonize):
        tcga = TCGA(ignore, hgnc, **init_args, harmonize_mut=harmonize)
        trcr = TRACERx(hgnc, **trcr_args, harmonize_mut=harmonize)
        _, _, cd = format_for_modeling(tcga, trcr, mut_transform='binarize', use_NT=init_args.get('req_NT', False), conc_min_count=pool_min_count)
        # From cd['conc'], not recomputed: format_for_modeling has already overwritten tcga.mut with the retained pool, so a fresh concordance_filter() here would measure the filter's own output
        cd['conc_report'] = report_view(cd['conc'], report_min_count)
        return tcga, trcr, cd
    os.makedirs('results/harmonization', exist_ok=True)
    row = summary_row(report_min_count)
    print("------------------- BEFORE HARMONIZATION -------------------")
    *_, cd0 = build(False)
    before_tab = shift_table(cd0['conc_report'])
    print(before_tab)
    print("\n------------------- AFTER HARMONIZATION --------------------")
    tcga, trcr, cd = build(True)
    raw_tcga = get_data('data_mutations.txt', dp=tcga.paths['pc'], low_memory=False)
    raw_trcr = load_data('csv', 'TRACERx421_mutation_table', data_path=trcr.path, clean=False)
    nums, tables = all_hmzn_numbers(cd['conc'], cd['FLAGS'], cd['pool'], n_overlapping=len(cd['overlapping_mut']), conc_report=cd['conc_report'])
    _, _, _, noncoding_shares = variant_class_tables(raw_tcga, raw_trcr)
    nums.update(noncoding_shares)
    tables['shift_by_quartile_before'] = before_tab
    tables['shift_paired'] = paired_shift(cd0['conc_report'], cd['conc_report'])
    tables['consequence_crosswalk'] = consequence_crosswalk()
    tables['variant_composition_before'] = consequence_composition_before(raw_tcga, raw_trcr, layout=composition_layout)
    tables['variant_composition_after'] = harmonized_composition(tcga.raw_mut, trcr.raw_mut, layout=composition_layout)
    nums['n_genes_paired'] = int(tables['shift_paired'].loc[row, 'n_genes'])
    nums['median_ratio_before_paired'] = float(tables['shift_paired'].loc[row, 'median_ratio_before'])
    nums['median_ratio_after_paired'] = float(tables['shift_paired'].loc[row, 'median_ratio_after'])
    # Pool attrition before harmonization, so the retention change is in `nums` rather than only in format_for_modeling's printout; floor-independent, unlike the median ratio
    nums.update({f'before_{k}': v for k, v in attrition(cd0['conc'], cd0['FLAGS'], cd0['pool'], n_overlapping=len(cd0['overlapping_mut'])).items()})
    nums['median_ratio_before'] = float(before_tab.loc[row, 'median_ratio'])
    nums['median_ratio_after'] = float(tables['shift_by_quartile'].loc[row, 'median_ratio'])
    nums['pct_concordant_before'] = 100 * nums['before_n_retained_concordant'] / nums['before_n_overlapping_genes']
    nums['pct_concordant_after'] = 100 * nums['n_retained_concordant'] / nums['n_overlapping_genes']
    print(f"\nmedian ratio (min_count={report_min_count}): {nums['median_ratio_before']:.4f} -> {nums['median_ratio_after']:.4f}")
    print(f"  paired on {nums['n_genes_paired']} genes assessable in both states: {nums['median_ratio_before_paired']:.4f} -> {nums['median_ratio_after_paired']:.4f}")
    print(f"concordant/overlapping: {nums['pct_concordant_before']:.1f}% -> {nums['pct_concordant_after']:.1f}%")
    print(f"non-coding: {nums['trcr_noncoding_pct']:.1f}% TRACERx / {nums['tcga_noncoding_pct']:.1f}% TCGA (main text: 'non-coding calls accounted for X% of TRACERx variants against Y% in TCGA')")
    for name, tab in tables.items():
        tab.to_csv(f'results/harmonization/{name}.csv')
    return nums, tables