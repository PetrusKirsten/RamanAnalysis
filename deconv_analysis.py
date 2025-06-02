import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_metrics(csv_path="metrics.csv", out_folder="./figures/deconv_metrics"):
    df = pd.read_csv(csv_path)

    # Extrai "group" e concentração da amostra
    df["group"] = df["sample"].apply(lambda x: "_".join(x.split("_")[:2]) if "kC" in x or "iC" in x else x.split("_")[0])
    df["conc_mM"] = df["sample"].str.extract(r"(\d+)").astype(int)

    os.makedirs(out_folder, exist_ok=True)

    for region in df["region"].unique():
        df_region = df[df["region"] == region]

        for peak_id in sorted(df_region["peak"].unique()):
            df_peak = df_region[df_region["peak"] == peak_id]

            for metric in ["area", "FWHM", "center"]:
                plt.figure(figsize=(8, 5))
                sns.boxplot(data=df_peak, x="group", y=metric, hue="conc_mM", palette="Set2")
                plt.title(f"{metric.upper()} – Pico {peak_id} – Região {region}")
                plt.ylabel(metric.upper())
                plt.xlabel("Grupo")
                plt.legend(title="CaCl₂ (mM)")
                plt.tight_layout()

                save_path = os.path.join(out_folder, f"{metric}_pico{peak_id}_{region.replace('–', '_')}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"✅ Salvo: {save_path}")


if __name__ == '__main__':
    process_metrics(csv_path="./figures/deconv/813_883/metrics.csv")
