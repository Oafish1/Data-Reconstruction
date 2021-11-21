import numpy as np
import pandas as pd


def process_decennial(decennial_data):
    """Perform pre-processing on decennial data"""
    cols = [
        'totalpop_white', 'totalpop_black', 'totalpop_amindian',
        'totalpop_asian', 'totalpop_pacisland', 'totalpop_other',
        # 'totalpop_twomoreraces', 'totalpop_tworaces', 'totalpop_whiteblack',
        # 'totalpop_whiteamindian', 'totalpop_whiteasian', 'totalpop_whitepacisland',
        # 'totalpop_whiteother', 'totalpop_blackamindian', 'totalpop_blackasian',
        # 'totalpop_blackpacislander', 'totalpop_blackother', 'totalpop_amindianasian',
        # 'totalpop_amindianpacislander', 'totalpop_amindianother',
        # 'totalpop_asianpacislander', 'totalpop_asianother',
        # 'totalpop_pacislanderother', 'totalpop_threeraces'
    ]
    total_col = ['totalpop']
    tags_cols = ['county_name']  # 'st_abbreviation'

    decennial_data = decennial_data[tags_cols + total_col + cols]
    decennial_data = decennial_data.astype({col: 'int32' for col in cols + total_col})
    decennial_data[cols] = (decennial_data[cols] / decennial_data[total_col].values[:])
    decennial_data[tags_cols] = decennial_data[tags_cols].applymap(lambda x: x.upper())
    decennial_data = decennial_data[tags_cols + cols]

    return decennial_data, cols, tags_cols


def process_ppp(ppp_data, extended=True):
    """Perform pre-processing on ppp data"""
    cols = [
        'CurrentApprovalAmount',
        'ForgivenessAmount',
        'UTILITIES_PROCEED',  # Added
        'PAYROLL_PROCEED',   # Added
        'MORTGAGE_INTEREST_PROCEED',   # Added
        'RENT_PROCEED',  # Added
        'REFINANCE_EIDL_PROCEED',   # Added
        'HEALTH_CARE_PROCEED',  # Added
        'DEBT_INTEREST_PROCEED',  # Added
        # Cause Errors
        # 'UndisbursedAmount',  # Added
        # 'MORTGAGE_INTEREST_PROCEED',
        # 'HEALTH_CARE_PROCEED',
    ] if extended else [
        'CurrentApprovalAmount',
        'ForgivenessAmount',
        'MORTGAGE_INTEREST_PROCEED',
        'HEALTH_CARE_PROCEED',
    ]
    tags_cols = ['ProjectCountyName']  # 'OriginatingLenderState'

    # Remove missing
    ppp_data = ppp_data[tags_cols + cols]
    ppp_data = ppp_data.dropna()

    ppp_data = ppp_data.astype({col: 'int32' for col in cols})
    normalize = lambda df: (df - df.min()) / (df.max() - df.min())  # noqa
    ppp_data[cols] = ppp_data[cols].apply(lambda df: np.log(1 + df)).apply(normalize)

    return ppp_data, cols, tags_cols


def merge_data(
    *tagged_datasets,
    agg_by_tag=False,
):
    """
    Aligns datasets with respect to a chosen column

    Arguments
    ---------
    tagged_datasets: Array of tuples
        (pd.data, data_column_names, join_column_names)
    """
    assert len(tagged_datasets) == 2, \
        f'Only two datasets are supported at this time, {len(tagged_datasets)} provided'

    mod1, mod1_cols, mod1_tags_cols = tagged_datasets[0]
    mod2, mod2_cols, mod2_tags_cols = tagged_datasets[1]

    mod1 = mod1.groupby(mod1_tags_cols).mean()
    merged = mod1.merge(mod2, left_on=mod1_tags_cols, right_on=mod2_tags_cols, how='inner')

    # Limit to 1/region
    if agg_by_tag:
        merged = merged.groupby(mod2_tags_cols).mean()

    mod1 = merged[mod1_cols]
    mod2 = merged[mod2_cols]
    annotations = pd.DataFrame(merged.index)

    return mod1, mod2, annotations
