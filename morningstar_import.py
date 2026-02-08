import sys
import re
import pandas as pd
import morningstar_data as md
from morningstar_data.direct import InvestmentIdentifier

client = input("Client: ").strip()
inv_list_id = input("Investment list ID: ").strip()
year = input("Enter year (2022/2023/2024/2025): ").strip()
valid_years = {"2022", "2023", "2024", "2025"}
if year not in valid_years:
    print("❌ Invalid year! Please choose 2022, 2023, 2024 or 2025.")
    sys.exit(1)

# Set dates
start_date = f"{year}-01-01"
end_date = f"{year}-12-31"

# Map year to data_point ID
year_to_datapoint = {
    "2022": "7874523",
    "2023": "7874527",
    "2024": "7892432",
    "2025": "8361564"
}
data_point_id = year_to_datapoint[year]
print(f"➡️ Client: {client}")
print(f"➡️ Year: {year}")
print(f"➡️ Start: {start_date}, End: {end_date}")
print(f"➡️ Data Point ID: {data_point_id}")

print("=== STEP 1: Get investment list ===")
list_df = md.direct.user_items.get_investment_list(
    list_id=inv_list_id
)
list_inv = list_df['secid'].tolist()
print(f"Got {len(list_inv)} investments from list.")

print("\n=== STEP 2: Resolve seed ISINs from investments ===")
fund_id_df = md.direct.get_investment_data(
    investments=list_inv,
    data_points='8132077'
)
inputs_raw = fund_id_df['ISIN'].dropna().drop_duplicates().tolist()
inputs = [x.strip().upper() for x in inputs_raw if isinstance(x, str)]
print(f"Seed ISINs to process: {len(inputs)}")

print("\n=== STEP 3: Pull holdings (per investment) ===")
holdings_frames = []
failed_inv = []
for i, inv in enumerate(list_inv, 1):
    try:
        h = md.direct.get_holdings(
            investments=[inv],
            start_date=start_date,
            end_date=end_date
        )
        holdings_frames.append(h)
        # Save each investment's holdings as a pickle checkpoint
        h.to_pickle(f'holdings_{client}_{year}_{inv}.pkl')
        print(f"[{i}/{len(list_inv)}] {inv}: {len(h)} rows (saved to pkl)")
    except Exception as e:
        failed_inv.append(inv)
        print(f"[{i}/{len(list_inv)}] {inv}: ERROR -> {e}")

if failed_inv:
    print(f"\n⚠️ Failed investments ({len(failed_inv)}): {failed_inv}")
print(f"✅ Successful: {len(holdings_frames)}/{len(list_inv)}")

holdings_frames = [df for df in holdings_frames if not df.empty]
holdings = pd.concat(holdings_frames, ignore_index=True) if holdings_frames else pd.DataFrame()
print(f"Holdings raw rows: {len(holdings)}")

# Drop / rename
holdings.drop(
    columns=['masterPortfolioId', 'holdingId', 'bondId', 'secId',
             'weight', 'marketValue', 'sharesChanged'],
    inplace=True,
    errors='ignore'
)
holdings.rename(columns={
    'portfolioDate': 'holding_date',
    'investmentId': 'fund_id',
    'name': 'holding_name',
    'isin': 'holding_isin',
    'cusip': 'holding_cusip',
    'shares': 'holding_shares',
    'ticker': 'holding_ticker',
    'detailHoldingType': 'holding_type'
}, inplace=True)
print(f"Holdings after cleanup: {len(holdings)}")

print("\n=== STEP 4: Build siblings using fund_id lookup (refactored) ===")
siblings_frames = []
failed_seeds = []

for i, isin in enumerate(inputs, 1):
    try:
        seed_lookup = md.direct.lookup.investments(
            investment=InvestmentIdentifier(isin=isin),
            only_surviving=False
        )
    except Exception as e:
        print(f"[{i}/{len(inputs)}] ISIN {isin}: ERROR during seed lookup -> {e}")
        failed_seeds.append(isin)
        siblings_frames.append(pd.DataFrame({
            "SeedISIN": [isin], "Fund Id": [pd.NA], "ISIN": [isin],
            "Name": [pd.NA], "SecId": [pd.NA]
        }))
        continue

    if seed_lookup.empty:
        print(f"[{i}/{len(inputs)}] ISIN {isin}: seed lookup returned EMPTY. Keeping placeholder (length should be 1).")
        failed_seeds.append(isin)
        siblings_frames.append(pd.DataFrame({
            "SeedISIN": [isin], "Fund Id": [isin], "ISIN": [isin],
            "Name": [pd.NA], "SecId": [pd.NA]
        }))
        continue

    seed_fund_id = seed_lookup.iloc[0].get("Fund Id")

    if pd.isna(seed_fund_id):
        print(f"[{i}/{len(inputs)}] ISIN {isin}: No Fund Id found. Keeping placeholder.")
        failed_seeds.append(isin)
        siblings_frames.append(pd.DataFrame({
            "SeedISIN": [isin], "Fund Id": [pd.NA], "ISIN": [isin],
            "Name": [pd.NA], "SecId": [pd.NA]
        }))
        continue

    try:
        all_share_classes = md.direct.lookup.investments(seed_fund_id)
        print(f"[{i}/{len(inputs)}] ISIN {isin}: Found {len(all_share_classes)} share classes using fund_id {seed_fund_id}")
    except Exception as e:
        print(f"[{i}/{len(inputs)}] ISIN {isin}: ERROR during fund_id lookup -> {e}")
        failed_seeds.append(isin)
        siblings_frames.append(pd.DataFrame({
            "SeedISIN": [isin], "Fund Id": [seed_fund_id], "ISIN": [isin],
            "Name": [pd.NA], "SecId": [pd.NA]
        }))
        continue

    if not all_share_classes.empty:
        siblings = (
            all_share_classes.loc[:, ["Name", "SecId", "ISIN", "Fund Id"]]
                .dropna(subset=["ISIN"])
                .copy()
        )
    else:
        siblings = pd.DataFrame(columns=["Name", "SecId", "ISIN", "Fund Id"])

    seed_row = seed_lookup.loc[[0], ["Name", "SecId", "ISIN", "Fund Id"]]
    siblings = pd.concat([siblings, seed_row], ignore_index=True)
    siblings = siblings.drop_duplicates(subset=["ISIN"])

    siblings = (
        siblings.assign(SeedISIN=isin)
            .loc[:, ["SeedISIN", "Fund Id", "ISIN", "Name", "SecId"]]
            .reset_index(drop=True)
    )
    siblings_frames.append(siblings)

    print(f"[{i}/{len(inputs)}] ISIN {isin}: found {len(siblings)} sibling(s). "
          f"{'OK (length should be >= 1)' if len(siblings) >= 1 else 'WARN (expected at least 1)'}")

all_siblings = (
    pd.concat(siblings_frames, ignore_index=True)
    if siblings_frames else pd.DataFrame(columns=["SeedISIN", "Fund Id", "ISIN", "Name", "SecId"])
).drop_duplicates()

all_siblings["SecId"] = all_siblings["SecId"].astype(str).str.strip()
print(f"\nAll siblings rows: {len(all_siblings)}")
print(f"Unique SeedISINs represented: {all_siblings['SeedISIN'].nunique()} / {len(inputs)}")

missing_after = [s for s in inputs if s not in set(all_siblings['SeedISIN'])]
if missing_after:
    print("Seeds not represented in all_siblings (unexpected):", missing_after)

if failed_seeds:
    print("Seeds with failed/empty seed lookup (placeholders inserted):", failed_seeds)

unique_inputs = set(inputs)
unique_sibs = set(all_siblings['SeedISIN'])

print(f"\nCheck: unique inputs = {len(unique_inputs)}, unique SeedISINs in all_siblings = {len(unique_sibs)}")

if unique_inputs == unique_sibs:
    print("✅ All input ISINs are represented in all_siblings (good).")
else:
    missing = unique_inputs - unique_sibs
    extra = unique_sibs - unique_inputs
    if missing:
        print(f"⚠️ Missing ISINs in siblings: {sorted(missing)}")
    if extra:
        print(f"⚠️ Unexpected extra SeedISINs in siblings: {sorted(extra)}")

# ---------------------------------
# 5) Shares Outstanding (LONG)
# ---------------------------------
print("\n=== STEP 5: Download Shares Outstanding (Daily) for 2022/23/24/25 (LONG) ===")
all_siblings_SecId = all_siblings['SecId'].dropna().drop_duplicates().tolist()
print(f"Querying {len(all_siblings_SecId)} SecIds for time series…")

shr_out = md.direct.get_investment_data(
    investments=all_siblings_SecId,
    data_points=data_point_id,
    time_series_format=md.direct.data_type.TimeSeriesFormat.LONG
)

print(f"Time series raw rows: {len(shr_out)}")

shr_out = (
    shr_out
      .dropna(subset=['Shares Outstanding (Daily)'])
      .rename(columns={
          'Shares Outstanding (Daily)': 'shr_out',
          'Name': 'fund_name',
          'Id': 'fund_id',
          'Date': 'shr_out_date',
          'ISIN': 'fund_isin'
      })
      .copy()
)
shr_out['fund_id'] = shr_out['fund_id'].astype(str).str.strip()
print(f"Time series cleaned rows: {len(shr_out)}")

# Map SecId -> SeedISIN
secid_to_seed_mapping = all_siblings.drop_duplicates('SecId')
shares_with_seed_info = shr_out.merge(
    secid_to_seed_mapping, how='left',
    left_on='fund_id', right_on='SecId'
)
shares_with_seed_info['shr_out_date'] = pd.to_datetime(shares_with_seed_info['shr_out_date'])

# Keep only rows where SeedISIN == ISIN (seed share class)
seed_share_classes = all_siblings.loc[all_siblings["SeedISIN"].eq(all_siblings["ISIN"]), ['SecId', 'ISIN']].copy()
shares_filtered_by_seed = shares_with_seed_info.merge(
    seed_share_classes,
    left_on='SeedISIN',
    right_on='ISIN',
    how='inner',
    suffixes=('', '_new')
)
print(f"Time series after seed filter rows: {len(shares_filtered_by_seed)}")

# Aggregate (SeedISIN x Date)
shares_aggregated = (
    shares_filtered_by_seed
      .groupby(['SeedISIN', 'shr_out_date', 'SecId_new'], as_index=False)['shr_out'].sum()
      .sort_values(['SeedISIN', 'shr_out_date'])
)
name_map = shares_filtered_by_seed[['SeedISIN', 'fund_name']].drop_duplicates('SeedISIN')
shares_aggregated = shares_aggregated.merge(name_map, on='SeedISIN', how='left')
shares_final = shares_aggregated.rename(columns={
    'SeedISIN': 'fund_isin',
    'SecId_new': 'fund_id'
})
print(f"Final daily shares rows: {len(shares_final)}")

# ---------------------------------
# 6) Build fund_id -> fund_isin map
# ---------------------------------
fund_isin_df = shares_final[['fund_id', 'fund_isin']].drop_duplicates().reset_index(drop=True)
print(f"Fund map rows: {len(fund_isin_df)}")

# ---------------------------------
# 7) Merge holdings with fund_isin map and filter equities
# ---------------------------------
holdings_merge = holdings.merge(
    fund_isin_df,
    on='fund_id',
    how='left',
    suffixes=('', '_from_shr')
)

holdings_nna = holdings_merge.dropna(subset=['holding_isin'])
holdings_final = holdings_nna[holdings_nna['holding_type'].str.contains('EQUITY', case=False, na=False)]
print(f"Holdings final rows (equities with ISIN): {len(holdings_final)}")

# ---------------------------------
# 8) Export to Excel
# ---------------------------------
out_file = f'morningstar_data_{year}_{client}.xlsx'
with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
    shares_final.to_excel(writer, sheet_name='shr_out', index=False)
    holdings_final.to_excel(writer, sheet_name='holdings', index=False)

print(f"\n=== DONE ===\nExported to {out_file}")
