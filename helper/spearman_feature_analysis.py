"""Spearman feature-ranking analysis for Final_Encoded_Normalized.csv.

This script ranks all predictors by Spearman correlation against the CGPA3 target,
saves the full ranking to CSV, and exports the top 29 features for reporting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Final_Encoded_Normalized.csv"
OUTPUT_DIR = BASE_DIR / "data" / "spearman"
FULL_RANKING_PATH = OUTPUT_DIR / "spearman_feature_ranking_cgpa3.csv"
TOP_29_PATH = OUTPUT_DIR / "spearman_top29_features_cgpa3.csv"
TOP_29_FIG_PATH = OUTPUT_DIR / "spearman_top29_correlations_cgpa3.png"
TOP_29_PAIRWISE_PATH = OUTPUT_DIR / "spearman_top29_pairwise_cgpa3.csv"
TOP_29_PAIRWISE_FIG_PATH = OUTPUT_DIR / "spearman_top29_pairwise_heatmap_cgpa3.png"
TOP_29_STRONG_PAIRS_PATH = OUTPUT_DIR / "spearman_top29_strong_pairs_cgpa3.csv"
SPSS_TABLE_ALL_PATH = OUTPUT_DIR / "spearman_spss_table_cgpa3_all_features.csv"
SPSS_TABLE_PREDICTOR_PATH = OUTPUT_DIR / "spearman_spss_table_cgpa3_predictors_only.csv"
TOP_29_TABLE_IMG_PATH = OUTPUT_DIR / "spearman_top29_features_cgpa3_table.png"
TOP_29_STRONG_PAIRS_IMG_PATH = OUTPUT_DIR / "spearman_top29_strong_pairs_cgpa3_table.png"
SPSS_TABLE_ALL_IMG_PATH = OUTPUT_DIR / "spearman_spss_table_cgpa3_all_features.png"
SPSS_TABLE_PREDICTOR_IMG_PATH = OUTPUT_DIR / "spearman_spss_table_cgpa3_predictors_only.png"
TARGET_COLUMN = "CGPA3_Class"
TOP_N = 29
PAIRWISE_THRESHOLD = 0.70


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def compute_spearman_ranking(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the dataset.")

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    target_series = numeric_df[target_column]

    records: list[dict[str, object]] = []
    for feature in numeric_df.columns:
        if feature == target_column:
            continue

        feature_series = numeric_df[feature]
        rho, p_value = spearmanr(feature_series, target_series, nan_policy="omit")

        records.append(
            {
                "Feature": feature,
                "Spearman_Rho": np.nan if pd.isna(rho) else float(rho),
                "P_Value": np.nan if pd.isna(p_value) else float(p_value),
            }
        )

    ranking_df = pd.DataFrame(records)
    ranking_df["Abs_Spearman_Rho"] = ranking_df["Spearman_Rho"].abs()
    ranking_df = ranking_df.sort_values(
        by=["Abs_Spearman_Rho", "Spearman_Rho", "Feature"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking_df.insert(0, "Rank", np.arange(1, len(ranking_df) + 1))
    return ranking_df


def save_top_features_plot(top_features: pd.DataFrame, output_path: Path) -> None:
    plot_df = top_features.sort_values("Spearman_Rho", ascending=True)
    palette = ["#1f77b4" if value >= 0 else "#d62728" for value in plot_df["Spearman_Rho"]]

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.35)))
    sns.barplot(
        data=plot_df,
        x="Spearman_Rho",
        y="Feature",
        hue="Feature",
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(
        "Top 29 Features Ranked by Spearman Correlation with CGPA3_Class",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Spearman correlation coefficient (rho)")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_pairwise_top_features(df: pd.DataFrame, top_features: list[str]) -> pd.DataFrame:
    pairwise_corr = df[top_features].corr(method="spearman")

    records: list[dict[str, object]] = []
    for left_index, left_feature in enumerate(top_features):
        for right_feature in top_features[left_index + 1 :]:
            rho = pairwise_corr.loc[left_feature, right_feature]
            records.append(
                {
                    "Feature_1": left_feature,
                    "Feature_2": right_feature,
                    "Spearman_Rho": float(rho),
                    "Abs_Spearman_Rho": abs(float(rho)),
                }
            )

    pairwise_df = pd.DataFrame(records)
    pairwise_df = pairwise_df.sort_values(
        by=["Abs_Spearman_Rho", "Feature_1", "Feature_2"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return pairwise_df, pairwise_corr


def save_pairwise_heatmap(pairwise_corr: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 12))
    mask = np.triu(np.ones_like(pairwise_corr, dtype=bool), k=1)
    sns.heatmap(
        pairwise_corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        cbar_kws={"label": "Spearman Correlation", "shrink": 0.8},
        square=True,
        ax=ax,
    )
    ax.set_title(
        "Spearman Pairwise Correlation Among Top 29 Target-Related Features",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dataframe_table_image(
    table_df: pd.DataFrame,
    title: str,
    output_path: Path,
    max_rows: int | None = None,
) -> None:
    display_df = table_df.head(max_rows).copy() if max_rows else table_df.copy()

    if display_df.empty:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{title}\nNo rows to display", ha="center", va="center", fontsize=12)
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig_height = max(4, min(18, 1.2 + 0.35 * (len(display_df) + 1)))
    fig_width = max(12, min(24, 0.9 * len(display_df.columns) + 10))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E9EEF7")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def classify_strength(abs_rho: float) -> str:
    if abs_rho >= 0.80:
        return "Very strong"
    if abs_rho >= 0.60:
        return "Strong"
    if abs_rho >= 0.40:
        return "Moderate"
    if abs_rho >= 0.20:
        return "Weak"
    return "Very weak"


def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.3f}".lstrip("0")


def build_spss_table(ranking_df: pd.DataFrame, drop_current_cgpa5: bool) -> pd.DataFrame:
    table_df = ranking_df.copy()
    if drop_current_cgpa5:
        table_df = table_df[table_df["Feature"] != "Current_CGPA5"].copy()

    table_df["Direction"] = np.where(
        table_df["Spearman_Rho"] > 0,
        "Positive",
        np.where(table_df["Spearman_Rho"] < 0, "Negative", "No direction"),
    )
    table_df["Strength"] = table_df["Abs_Spearman_Rho"].apply(classify_strength)

    spss_df = pd.DataFrame(
        {
            "Predictor Variable": table_df["Feature"],
            "Spearman's rho": table_df["Spearman_Rho"].map(lambda value: f"{value:.3f}"),
            "Direction": table_df["Direction"],
            "Strength": table_df["Strength"],
            "p-value": table_df["P_Value"].apply(format_p_value),
        }
    )
    return spss_df.reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA_PATH)
    ranking_df = compute_spearman_ranking(df, TARGET_COLUMN)

    ranking_df.to_csv(FULL_RANKING_PATH, index=False)
    top_29_df = ranking_df.head(TOP_N).copy()
    top_29_df.to_csv(TOP_29_PATH, index=False)
    save_dataframe_table_image(
        top_29_df[["Rank", "Feature", "Spearman_Rho", "P_Value"]],
        "Top 29 Features by Spearman Correlation with CGPA3_Class",
        TOP_29_TABLE_IMG_PATH,
    )
    save_top_features_plot(top_29_df, TOP_29_FIG_PATH)

    top_features = top_29_df["Feature"].tolist()
    pairwise_df, pairwise_corr = compute_pairwise_top_features(df, top_features)
    pairwise_df.to_csv(TOP_29_PAIRWISE_PATH, index=False)
    save_pairwise_heatmap(pairwise_corr, TOP_29_PAIRWISE_FIG_PATH)

    strong_pairs_df = pairwise_df[pairwise_df["Abs_Spearman_Rho"] >= PAIRWISE_THRESHOLD].copy()
    strong_pairs_df.to_csv(TOP_29_STRONG_PAIRS_PATH, index=False)
    save_dataframe_table_image(
        strong_pairs_df,
        f"Top-29 Pairwise Strong Correlations (|rho| >= {PAIRWISE_THRESHOLD:.2f})",
        TOP_29_STRONG_PAIRS_IMG_PATH,
        max_rows=40,
    )

    spss_all_df = build_spss_table(ranking_df, drop_current_cgpa5=False)
    spss_predictor_df = build_spss_table(ranking_df, drop_current_cgpa5=True)
    spss_all_df.to_csv(SPSS_TABLE_ALL_PATH, index=False)
    spss_predictor_df.to_csv(SPSS_TABLE_PREDICTOR_PATH, index=False)
    save_dataframe_table_image(
        spss_all_df,
        "Spearman Table (All Features vs CGPA3_Class)",
        SPSS_TABLE_ALL_IMG_PATH,
        max_rows=40,
    )
    save_dataframe_table_image(
        spss_predictor_df,
        "Spearman Table (Predictors Only, excluding Current_CGPA5)",
        SPSS_TABLE_PREDICTOR_IMG_PATH,
        max_rows=40,
    )

    print("Spearman analysis complete.")
    print(f"Dataset: {DATA_PATH}")
    print(f"Full ranking saved to: {FULL_RANKING_PATH}")
    print(f"Top {TOP_N} features saved to: {TOP_29_PATH}")
    print(f"Plot saved to: {TOP_29_FIG_PATH}")
    print(f"Pairwise matrix saved to: {TOP_29_PAIRWISE_PATH}")
    print(f"Pairwise heatmap saved to: {TOP_29_PAIRWISE_FIG_PATH}")
    print(f"Strong pairs saved to: {TOP_29_STRONG_PAIRS_PATH}")
    print(f"Top-{TOP_N} table image saved to: {TOP_29_TABLE_IMG_PATH}")
    print(f"Strong-pairs table image saved to: {TOP_29_STRONG_PAIRS_IMG_PATH}")
    print(f"SPSS-style table (all features) saved to: {SPSS_TABLE_ALL_PATH}")
    print(f"SPSS-style table (without Current_CGPA5) saved to: {SPSS_TABLE_PREDICTOR_PATH}")
    print(f"SPSS-style image (all features) saved to: {SPSS_TABLE_ALL_IMG_PATH}")
    print(f"SPSS-style image (without Current_CGPA5) saved to: {SPSS_TABLE_PREDICTOR_IMG_PATH}")
    print("\nTop 10 features by absolute Spearman rho:")
    print(top_29_df.head(10).to_string(index=False))
    print(f"\nPairs with |Spearman rho| >= {PAIRWISE_THRESHOLD:.2f}: {len(strong_pairs_df)}")
    if not strong_pairs_df.empty:
        print(strong_pairs_df.head(10).to_string(index=False))
    print("\nSPSS-style table preview (without Current_CGPA5):")
    print(spss_predictor_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()