import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
data_dir = "/Users/vishnups/Documents/legal/data/UK-Abs/train-data/summary/"
output_dir = "/Users/vishnups/Documents/legal/processed_data/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all text files
files = glob.glob(os.path.join(data_dir, "*.txt"))

# Read files and create dataset
data = []
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract filename as case ID
    case_id = os.path.basename(file).replace('.txt', '')
    
    data.append({
        'case_id': case_id,
        'summary': content
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Split into train, validation, and test sets (80/10/10)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save to CSV
train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'validation.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

print(f"Processed {len(df)} documents")
print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")