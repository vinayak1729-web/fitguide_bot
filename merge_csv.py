import pandas as pd
print("🔄 MERGING CSVs...")

# Load both files
vehicle_df = pd.read_csv('data/tbl_mVehicle.csv')
speaker_df = pd.read_csv('data/tbl_xVehicleDetailsSpeakerFitguideNonPart.csv')

# MERGE: Add Make + Model to Speaker file
merged_df = speaker_df.merge(
    vehicle_df[['vehicleUID', 'Make', 'Model', 'Year']], 
    on='vehicleUID', 
    how='left'
)

# SAVE NEW FILE
merged_df.to_csv('data/tbl_Speakers_Complete.csv', index=False)
print(f"✅ NEW FILE: tbl_Speakers_Complete.csv ({len(merged_df)} rows)")
print("📋 NEW COLUMNS: Make, Model, Year ADDED!")