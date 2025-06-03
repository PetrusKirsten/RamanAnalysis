import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_metrics(csv_path="metrics.csv", out_folder="./figures/deconv_metrics"):
    df = pd.read_csv(csv_path)

    # Extrair grupo e concentração
    df["group"] = df["sample"].apply(lambda x: "_".join(x.split("_")[:2]) if "kC" in x or "iC" in x else x.split("_")[0])
    df["conc_mM"] = df["sample"].str.extract(r"(\d+)").astype(int)

    os.makedirs(out_folder, exist_ok=True)

    palette = {
        "St": "dodgerblue",
        "St_kC": "hotpink",
        "St_iC": "mediumseagreen"
    }

    for region in df["region"].unique():
        df_region = df[df["region"] == region]

        for peak_id in sorted(df_region["peak"].unique()):
            df_peak = df_region[df_region["peak"] == peak_id]

            for metric in ["area", "FWHM", "center"]:
                plt.figure(figsize=(7, 5))

                # Agrupa e plota cada grupo com média ± std
                for group, dfg in df_peak.groupby("group"):
                    df_stats = dfg.groupby("conc_mM")[metric].agg(["mean", "std"]).reset_index()
                    plt.errorbar(
                        df_stats["conc_mM"], df_stats["mean"], yerr=df_stats["std"],
                        fmt='-o', label=group, capsize=4, color=palette.get(group, None)
                    )

                plt.title(f"{metric.upper()} – Pico {peak_id} – Região {region}")
                plt.xlabel("CaCl₂ (mM)")
                plt.ylabel(metric.upper())
                plt.xticks([0, 7, 14, 21])
                plt.grid(True)
                plt.legend(title="Grupo")
                plt.tight_layout()

                fname = f"comparativo_pico{peak_id}_{metric}_{region.replace('–', '_')}.png"
                plt.savefig(os.path.join(out_folder, fname), dpi=300)
                plt.close()
                print(f"✅ {fname} salvo.")


if __name__ == '__main__':
    process_metrics(csv_path="./figures/deconv/813_883/metrics.csv")
