import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir[:-3], "data/medical_examination.csv")
output_dir = os.path.join(script_dir[:-3], "output")

df = pd.read_csv(data_dir, encoding="utf-8")

# Add 'overweight' column
df["overweight"] = df.apply(lambda row: 1 if row["weight"] / np.square((row["height"] / 100)) > 25 else 0, axis=1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1,
# make the value 0. If the value is more than 1, make the value 1.

# The method below is an alternative, but it is slower as it needs to loop over every row
# df["gluc"] = df["gluc"].apply(lambda gluc: 0 if gluc == 1 else 1)
# df["cholesterol"] = df["cholesterol"].apply(lambda cholesterol: 0 if cholesterol == 1 else 1)

# This method is optimized and is called the vectorized approach
df["gluc"] = (df["gluc"] != 1).astype("uint8")
df["cholesterol"] = (df["cholesterol"] != 1).astype("uint8")


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke',
    # 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(
        df, id_vars=["cardio"], value_vars=["active", "alco", "cholesterol", "gluc", "overweight", "smoke"]
    )

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename
    # one of the columns for the catplot to work correctly.
    df_cat = (
        df_cat.reset_index()
        .groupby(["variable", "cardio", "value"])
        .agg("count")
        .rename(columns={"index": "total"})
        .reset_index()
    )

    # Draw the catplot with 'sns.catplot()'
    sns_catplot = sns.catplot(x="variable", y="total", col="cardio", hue="value", data=df_cat, kind="bar")

    # Get the figure for the output
    fig = sns_catplot.fig

    # Do not modify the next two lines
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fig.savefig(output_dir + "/catplot.png")

    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    # - diastolic pressure is higher than systolic
    # - height is less than the 2.5th percentile
    # - height is more than the 97.5th percentile
    # - weight is less than the 2.5th percentile
    # - weight is more than the 97.5th percentile
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig = plt.figure(figsize=(12, 6))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, vmin=-0.5, vmax=0.5)

    # Do not modify the next two lines
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fig.savefig(output_dir + "/heatmap.png")

    return fig
