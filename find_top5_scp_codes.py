import pandas as pd
import os
from collections import Counter

PTBXL_PATH = './'  # Adjust if needed
ANNOTATION_FILE = os.path.join(PTBXL_PATH, 'ptbxl_database.csv')

# Load metadata
df = pd.read_csv(ANNOTATION_FILE, index_col='ecg_id')

# Count nonzero SCP code occurrences
scp_counter = Counter()
for scp_dict in df['scp_codes'].apply(eval):
    for k, v in scp_dict.items():
        if v > 0.0:
            scp_counter[k] += 1

# Get top 5 most common nonzero SCP codes
top5 = scp_counter.most_common(5)

print("Top 5 most common nonzero SCP codes in the database:")
for code, count in top5:
    print(f"{code}: {count} records")

target_scp_codes = ['NORM', 'IMI', 'ASMI', 'LVH', 'NDT'] 