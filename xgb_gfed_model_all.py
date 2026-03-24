# Imports
import numpy as np
import pandas as pd
import dask.array as da
import dask
from dask import delayed
import xgboost as xgb
from joblib import dump
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from statsmodels.nonparametric.smoothers_lowess import lowess
import shap

# Load numpy training arrays from .npy files
static_band_names = ['roads_total', 'TPI', 'VRM', 'water_bodies']
annual_band_names = ['bare', 'broadleaf', 'cropland', 'fuel_load_withgrass', 'GDP', 'grass', 'grazing_pressure', 'irrigated', 'needleleaf', 'rmf', 'rural_pop', 'shrub', 'total_pop', 'tree', 'urban_area']
monthly_band_names = ['accum_6m_precipitation', 'CAPE', 'DD', 'DTR', 'GPP', 'GPP_accum_6m', 'GPP_accum_12m', 'max_temp', 'ppfd', 'snow', 'soil_water', 'VPD_DT', 'VPD_NT', 'wind_max', 'wind_mean']

all_vars = static_band_names + annual_band_names + monthly_band_names

fire_types = ['all_fires', 'cropland_fires', 'deforestation_fires', 'intensity']
fire_titles = ['fires (all types)', 'cropland fires', 'deforestation fires', 'intensity']

# Generic function to load arrays lazily with Dask and ensure 2D shape
# Dask is a great module that loads big arrays really quickly
@delayed
def load_npy_file(filepath):
    data = np.load(filepath)
    return data

for fire, title in zip(fire_types, fire_titles):
    for region in range(1, 15):
        print(f'----- GFED region {region} {fire}-----')
        shape = (np.load(f'/rds/general/user/aac115/home/fireveg/fire_data/Monthly/arrays_gfed/arrays_{region}/GFED5_all_fires_2003_2016_array.npy')).shape
        # Load training arrays
        loaded_train_arrays = [
            da.from_delayed(load_npy_file(f'/rds/general/user/aac115/home/fireveg/input_data/Static/arrays_gfed/arrays_{region}/{band}_array.npy'), shape=shape, dtype=np.float32)
            for band in static_band_names
        ] + [
            da.from_delayed(load_npy_file(f'/rds/general/user/aac115/home/fireveg/input_data/Annual/arrays_gfed/arrays_{region}/{band}_2003_2016_array.npy'), shape=shape, dtype=np.float32)
            for band in annual_band_names
        ] + [
            da.from_delayed(load_npy_file(f'/rds/general/user/aac115/home/fireveg/input_data/Monthly/arrays_gfed/arrays_{region}/{band}_2003_2016_array.npy'), shape=shape, dtype=np.float32)
            for band in monthly_band_names
        ]
        
        # Concatenate along axis 1 to stack columns into X_train
        X_train = da.concatenate(loaded_train_arrays, axis=1)
        print('X_train shape:', X_train.shape)
        
        # Load and ravel the training fire array
        if fire == 'intensity':
            train_fire_array = load_npy_file(f'/rds/general/user/aac115/home/fireveg/fire_data/Monthly/arrays_gfed/arrays_{region}/intensity_MODIS_GFA_2003_2016_array.npy')
        else:
            train_fire_array = load_npy_file(f'/rds/general/user/aac115/home/fireveg/fire_data/Monthly/arrays_gfed/arrays_{region}/GFED5_{fire}_2003_2016_array.npy')
        y_train = da.from_delayed(train_fire_array, shape=shape, dtype=np.float32).ravel()
        print('y_train shape:', y_train.shape)
        
        # Compute training arrays and ravel
        X_train, y_train = dask.compute(X_train, y_train)
        y_train = y_train.ravel()
        
        print("Pre-training processes complete")
        
        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=all_vars)
        
        params = {
            "objective": "reg:squarederror", # Standard regression model
            "tree_method": "hist", # Standard
            "max_depth": 12, # Slightly increased (default is 6)
            "subsample": 0.8, # Param to edit if necessary
            "colsample_bytree": 0.8, # Ditto
            "eta": 0.1, # Ditto
            "random_state": 42, # Standard random state
            "device": "cuda" # GPU
        }
        
        # Train 100 boosting rounds
        model = xgb.train(params, dtrain, num_boost_round=100)
        
        # Save the trained model
        model_output_path = f'/rds/general/user/aac115/home/fireveg/saved_models_reduced/gfed_region_{region}_{fire}_reduced.json'
        model.save_model(model_output_path)
        print(f"Model saved to {model_output_path}")
        
        # Feature importances (within the model)
        raw_importances = model.get_score(importance_type='weight')
        
        # Convert to array matching feature list order
        importances = np.array([raw_importances.get(feature, 0.0) for feature in all_vars])
        
        # Normalize (only if sum > 0)
        total = importances.sum()
        if total > 0:
            importances = importances / total
        
        # Print nicely
        print("Feature Importances (normalized weight):")
        for feature, importance in zip(all_vars, importances):
            print(f"{feature}: {importance:.4f}")
        
        # SHAP model
        X_train_df = pd.DataFrame(X_train, columns=all_vars)
        
        # Convert trained XGBoost model to SHAP TreeExplainer
        # The higher the value of 'n' the more complete it will be
        # However it takes longer to plot with higher values and all you do is add noise
        sampled_df = X_train_df.sample(n=150000, random_state=42) # Have a play around with it
        explainer = shap.TreeExplainer(model)
    
        # Compute SHAP values for the original (non-DMatrix) input data
        shap_values = explainer(sampled_df)

        # Original SHAP plot (all vars)
        plt.figure()
        shap.summary_plot(shap_values, sampled_df, show=False, max_display=len(sampled_df.columns))
        plt.tight_layout()
        plt.title('SHAP summary plot for global wildfires')
        plt.savefig(f'/rds/general/user/aac115/home/fireveg/plots_reduced/gfed_region_{region}_{fire}_shap_reduced', dpi=300)
        plt.close()
    
        # SHAP value plots for each individual variable
        # A version of PDP
        n_rows, n_cols = 6, 6
        ymin = (shap_values.values.min(axis=0)).min()
        ymax = (shap_values.values.max(axis=0)).max()
        bins_x = 60 # good starting point for ~150k points
        bins_y = 60
        frac_lowess = 0.1 # LOWESS smoothing fraction - lower = smoother
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))
        axes = axes.flatten()
        fig.suptitle(f'SHAP scatter plots for GFED region {region} {title} - reduced')
        
        cmap = plt.get_cmap('viridis')
        
        # We'll store the middle scatter object for the colorbar
        mid_scatter = None
        
        for i, variable in enumerate(all_vars):
            print(f"Plotting {variable}")
            ax = axes[i]
        
            # ----------------------------
            # Get feature and SHAP values
            # ----------------------------
            feat_idx = shap_values.feature_names.index(variable)
            x = shap_values.data[:, feat_idx]
            y = shap_values.values[:, feat_idx]
        
            # Keep only finite values
            mask = np.isfinite(x) & np.isfinite(y)
            x_valid = x[mask]
            y_valid = y[mask]
        
            # ----------------------------
            # 2D histogram -> density per bin
            # ----------------------------
            counts, xedges, yedges = np.histogram2d(
                x_valid,
                y_valid,
                bins=[bins_x, bins_y]
            )
        
            # Map each point to its (x_bin, y_bin)
            x_idx = np.searchsorted(xedges, x_valid, side='right') - 1
            y_idx = np.searchsorted(yedges, y_valid, side='right') - 1
        
            # Clip indices to be within [0, bins-1]
            x_idx = np.clip(x_idx, 0, counts.shape[0] - 1)
            y_idx = np.clip(y_idx, 0, counts.shape[1] - 1)
        
            # Get the count for each point
            point_counts = counts[x_idx, y_idx]
        
            # Avoid log10(0): treat zeros as 1 (or you could mask them)
            point_counts_safe = np.where(point_counts > 0, point_counts, 1)
            log_density = np.log10(point_counts_safe)
        
            # Normalize log density for colormap
            vmin = np.nanmin(log_density)
            vmax = np.nanmax(log_density)
            # Avoid degenerate normalization if all values equal
            if vmax <= vmin:
                vmax = vmin + 1e-6
        
            norm = Normalize(vmin=vmin, vmax=vmax)
        
            # ----------------------------
            # Scatter with log-density colors
            # ----------------------------
            sc = ax.scatter(
                x_valid,
                y_valid,
                c=log_density,
                cmap=cmap,
                norm=norm,
                s=5,
                linewidths=0
            )
            if i == 11:
                mid_scatter = sc
        
            # ----------------------------
            # Axis formatting
            # ----------------------------
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("")
        
            col = i % n_cols
            if col == 0:
                ax.set_ylabel("SHAP value")
                ax.tick_params(axis="y", which="both", labelleft=True)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", which="both", labelleft=False)
        
            # ----------------------------
            # LOWESS smoothing on original data
            # ----------------------------
            # Line of best fit
            smooth = lowess(y_valid, x_valid, frac=frac_lowess)
            ax.plot(smooth[:, 0], smooth[:, 1], linewidth=2, color='red', alpha=0.7)
        
            ax.set_title(variable, fontsize=10)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        
        # Shared colorbar for log density
        #if mid_scatter is not None:
        #    cbar = fig.colorbar(mid_scatter, ax=axes[:len(all_vars)], fraction=0.02, pad=0.01)
        #    cbar.set_label('log10(point density)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(f'/rds/general/user/aac115/home/fireveg/plots_reduced/gfed_region_{region}_{fire}_ind_vars_reduced.png', dpi=600)
        plt.close()
