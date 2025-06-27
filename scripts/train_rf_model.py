import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import logging
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    handlers=[
        logging.FileHandler("logs/data_processing_rf_training.log"),
        logging.StreamHandler()
    ]
)

# Import data processing
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, '..')
sys.path.insert(0, project_root)
from scripts import data_processing

# Load and prepare the data
logging.info("Loading the data...")
df, skewed_cols = data_processing.load_and_process_data()

logging.info("Preparing the data for feature importance analysis...")
# Remove unwanted columns
drop_cols = ['id', 'title', 'release_date', 'spoken_languages'] + skewed_cols
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Drop rows with NaN values
nan_cols = df.columns[df.isna().any()].tolist()
if nan_cols:
    logging.info(f"NaN values found in: {nan_cols}")
    initial_rows = df.shape[0]
    df = df.dropna()
    dropped_rows = initial_rows - df.shape[0]
    logging.info(f"Initial rows: {initial_rows}")
    logging.info(f"Total rows dropped due to NaN values: {dropped_rows}")
    logging.info(f"Final data size: {df.shape}")
else:
    logging.info("No columns with NaN values.")

# Prepare features and target
feature_cols = [col for col in df.columns if col != 'popularity' and df[col].dtype in [int, float, bool]]
X = df[feature_cols]
y = df['popularity']

# Fit RandomForest for feature importance
rf = RandomForestRegressor(random_state=42)
logging.info(f"Fitting the model: {rf}")
rf.fit(X, y)

# Save feature importances
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
importances.to_csv('data/feature_importances.csv', header=['importance'])
logging.info(f"Saved the Feature Importances to: data/feature_importances.csv")
logging.info(f"\nFeature Importances:\n{importances}")