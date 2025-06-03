import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rasp.plot_utils import config_figure, set_font

def process_metrics(csv_path):
    out_folder = os.path.dirname(csv_path)
    os.makedirs(out_folder, exist_ok=True)

    palette = {
        "St"    : "#EEA65B",
        "St_kC" : "#D221A8",
        "St_iC" : "#1495CA"
    }

    df = pd.read_csv(csv_path)

    # Extrair grupo e concentração
    df["group"] = df["sample"].apply(lambda x: "_".join(x.split("_")[:2]) if "kC" in x or "iC" in x else x.split("_")[0])
    df["conc_mM"] = df["sample"].str.extract(r"(\d+)").astype(int)

    for region in df["region"].unique():
        df_region = df[df["region"] == region]

        for peak_id in sorted(df_region["peak"].unique()):
            df_peak = df_region[df_region["peak"] == peak_id]

            for metric in ["area", "FWHM", "center"]:
                title = f"{metric[0].upper() + metric[1:]} – Peak {peak_id} – Region {region}"
                ax = config_figure(fig_title=title, size=(2000, 2000))

                # Agrupa e plota cada grupo com média ± std
                for group, dfg in df_peak.groupby("group"):
                    df_stats = dfg.groupby("conc_mM")[metric].agg(["mean", "std"]).reset_index()
                    ax.errorbar(
                        df_stats["conc_mM"], df_stats["mean"], yerr=df_stats["std"],
                        marker='o', markersize=8,
                        color=palette.get(group, None), alpha=0.95,
                        mec='w', mew=1.25,
                        ls='-', lw=0.75,
                        capsize=4, 
                        label=group
                    )

                ax.set_xlabel("CaCl$_2$ (mM)")
                ax.set_ylabel(metric[0].upper() + metric[1:])
                ax.set_xticks([0, 7, 14, 21])
                # ax.grid(True)
                # plt.legend(title="Group")
                plt.tight_layout()

                fname = f"comparative_peak{peak_id}_{metric}_{region.replace('–', '_')}.png"
                plt.savefig(os.path.join(out_folder, fname), dpi=300)
                plt.close()
                print(f"✅ {fname} generated.")


if __name__ == '__main__':
    set_font("D:/Documents/GitHub/Raman-Analysis-Software/data/fonts/Helvetica-Light.ttf")

    for folder in ['385_640', '813_883', '885_965', '980_1180']:
        process_metrics(csv_path = f"./figures/deconv/{folder}/metrics.csv")
