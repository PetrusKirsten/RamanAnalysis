import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def set_grid(
        title, image_paths, save_path,
        crop_list: list, cols=4, rows=3, row_labels=["St", "St kCar", "St iCar"],
        facecolor='#09141E', titlecolor='white'
    ):
    
    # === Fonte ===
    font_path = ("D:/Documents/GitHub/Raman-Analysis-Software/data/fonts/Helvetica-Light.ttf")
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()
    plt.rcParams.update({
        'font.family': font_name,
        'figure.facecolor': facecolor,
        'axes.facecolor': facecolor,
        'savefig.dpi': 300,
    })

    # === Recorte ===
    crop_x, crop_y, crop_w, crop_h = crop_list[0], crop_list[1], crop_list[2], crop_list[3]
    col_labels = [f"CL {i * 7}" for i in range(cols)]
    cropped_images = []
    for path in image_paths:
        img = Image.open(path)
        crop_box = (
            crop_x,
            crop_y,
            min(crop_x + crop_w, img.width),
            min(crop_y + crop_h, img.height)
        )
        cropped = img.crop(crop_box)
        cropped_images.append(cropped)

    img_w, img_h = cropped_images[0].size
    final_img = Image.new("RGB", (img_w * cols, img_h * rows))
    for idx, img in enumerate(cropped_images):
        row = idx // cols
        col = idx % cols
        final_img.paste(img, (col * img_w, row * img_h))

    # === Plotagem com título e labels ===
    fig, ax = plt.subplots(figsize=(cols * 3, rows * 3), constrained_layout=True)
    ax.imshow(final_img)
    ax.axis("off")
    ax.set_xlim(-100, final_img.width)
    ax.set_ylim(final_img.height, -100)

    plt.suptitle(title, fontsize=16, color=titlecolor, fontproperties=prop)

    for c in range(cols):
        ax.text((c + 0.5) * img_w, -30, col_labels[c], ha='center', va='bottom',
                fontsize=12, color='w', fontproperties=prop)
    for r in range(rows):
        ax.text(-30, (r + 0.5) * img_h, row_labels[r], ha='right', va='center',
                fontsize=12, color='w', fontproperties=prop, rotation=90)

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()

# === Topograhpy ===
name = 'topography-outliersAndShading_correction.png'

st = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  CLs/run_2025-05-29_15-17-41"
kc = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  kC CLs/run_2025-05-29_15-18-58"
ic = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  iC CLs/run_2025-05-29_15-20-14"

paths_topo = [
        f"{st}/Map_St_CL_0_Region_1/{name}",
        f"{st}/Map_St_CL_7_Region_2/{name}",
        f"{st}/Map_St_CL_14_Region_2/{name}",
        f"{st}/Map_St_CL_21_Region_2/{name}",

        f"{kc}/Map_St_kC_CL_0_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_7_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_14_Region_2/{name}",
        f"{kc}/Map_St_kC_CL_21_Region_1/{name}",

        f"{ic}/Map_St_iC_CL_0_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_7_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_14_Region_2/{name}",
        f"{ic}/Map_St_iC_CL_21_Region_1/{name}",
]

set_grid(
    title=f"Total spectrum sum | Topography map", image_paths=paths_topo, save_path=f"./figures/topography_grid.png",
    crop_list=[60, 260, 
               1720-60, 1880-260]
)

# === 478 cm-1 ===
name = 'band_478-outliersAndShading_correction.png'

st = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  CLs/run_2025-05-29_15-13-57"
kc = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  kC CLs/run_2025-05-29_14-58-27"
ic = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  iC CLs/run_2025-05-29_15-00-08"

paths_topo = [
        f"{st}/Map_St_CL_0_Region_1/{name}",
        f"{st}/Map_St_CL_7_Region_2/{name}",
        f"{st}/Map_St_CL_14_Region_2/{name}",
        f"{st}/Map_St_CL_21_Region_2/{name}",

        f"{kc}/Map_St_kC_CL_0_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_7_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_14_Region_2/{name}",
        f"{kc}/Map_St_kC_CL_21_Region_1/{name}",

        f"{ic}/Map_St_iC_CL_0_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_7_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_14_Region_2/{name}",
        f"{ic}/Map_St_iC_CL_21_Region_1/{name}",
]

grid_title = f"Band map at 478" + " cm$^{-1}$"
set_grid(
    title=grid_title, image_paths=paths_topo, save_path=f"./figures/band_478_grid.png",
    crop_list=[60, 260, 
               1720-60, 1880-260]
)

# === 851 cm-1 ===
name = 'band_851-outliersAndShading_correction.png'

st = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  CLs/run_2025-05-29_15-13-57"
kc = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  kC CLs/run_2025-05-29_14-58-27"
ic = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  iC CLs/run_2025-05-29_15-00-08"

paths_topo = [
        f"{st}/Map_St_CL_0_Region_1/{name}",
        f"{st}/Map_St_CL_7_Region_2/{name}",
        f"{st}/Map_St_CL_14_Region_2/{name}",
        f"{st}/Map_St_CL_21_Region_2/{name}",

        f"{kc}/Map_St_kC_CL_0_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_7_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_14_Region_2/{name}",
        f"{kc}/Map_St_kC_CL_21_Region_1/{name}",

        f"{ic}/Map_St_iC_CL_0_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_7_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_14_Region_2/{name}",
        f"{ic}/Map_St_iC_CL_21_Region_1/{name}",
]

grid_title = f"Band map at 851" + " cm$^{-1}$"
set_grid(
    title=grid_title, image_paths=paths_topo, save_path=f"./figures/band_851_grid.png",
    crop_list=[60, 260, 
               1720-60, 1880-260]
)

# === 939 cm-1 ===
name = 'band_939-outliersAndShading_correction.png'

st = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  CLs/run_2025-05-29_15-13-57"
kc = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  kC CLs/run_2025-05-29_14-58-27"
ic = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  iC CLs/run_2025-05-29_15-00-08"

paths_topo = [
        f"{st}/Map_St_CL_0_Region_1/{name}",
        f"{st}/Map_St_CL_7_Region_2/{name}",
        f"{st}/Map_St_CL_14_Region_2/{name}",
        f"{st}/Map_St_CL_21_Region_2/{name}",

        f"{kc}/Map_St_kC_CL_0_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_7_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_14_Region_2/{name}",
        f"{kc}/Map_St_kC_CL_21_Region_1/{name}",

        f"{ic}/Map_St_iC_CL_0_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_7_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_14_Region_2/{name}",
        f"{ic}/Map_St_iC_CL_21_Region_1/{name}",
]

grid_title = f"Band map at 939" + " cm$^{-1}$"
set_grid(
    title=grid_title, image_paths=paths_topo, save_path=f"./figures/band_939_grid.png",
    crop_list=[60, 260, 
               1720-60, 1880-260]
)

# === Multibands ===
name = 'multibands_global_continuous.png'

st = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  CLs/run_2025-05-29_15-13-57"
kc = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  kC CLs/run_2025-05-29_14-58-27"
ic = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  iC CLs/run_2025-05-29_15-00-08"

paths_topo = [
        f"{st}/Map_St_CL_0_Region_1/{name}",
        f"{st}/Map_St_CL_7_Region_2/{name}",
        f"{st}/Map_St_CL_14_Region_2/{name}",
        f"{st}/Map_St_CL_21_Region_2/{name}",

        f"{kc}/Map_St_kC_CL_0_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_7_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_14_Region_2/{name}",
        f"{kc}/Map_St_kC_CL_21_Region_1/{name}",

        f"{ic}/Map_St_iC_CL_0_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_7_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_14_Region_2/{name}",
        f"{ic}/Map_St_iC_CL_21_Region_1/{name}",
]

grid_title = "Multibands map | Red: 851 cm$^{-1}$ and Blue: 478 cm$^{-1}$"
set_grid(
    title=grid_title, image_paths=paths_topo, save_path=f"./figures/multibands_grid.png",
    crop_list=[60, 140, 
               1940-60, 1960-140]
)

# === Multibands composition ===
name = 'composition_multibands_global_continuous.png'

st = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  CLs/run_2025-05-29_15-13-57"
kc = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  kC CLs/run_2025-05-29_14-58-27"
ic = f"D:/Documents/GitHub/Raman-Analysis-Software/figures/maps-St  iC CLs/run_2025-05-29_15-00-08"

paths_topo = [
        f"{st}/Map_St_CL_0_Region_1/{name}",
        f"{st}/Map_St_CL_7_Region_2/{name}",
        f"{st}/Map_St_CL_14_Region_2/{name}",
        f"{st}/Map_St_CL_21_Region_2/{name}",

        f"{kc}/Map_St_kC_CL_0_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_7_Region_1/{name}",
        f"{kc}/Map_St_kC_CL_14_Region_2/{name}",
        f"{kc}/Map_St_kC_CL_21_Region_1/{name}",

        f"{ic}/Map_St_iC_CL_0_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_7_Region_1/{name}",
        f"{ic}/Map_St_iC_CL_14_Region_2/{name}",
        f"{ic}/Map_St_iC_CL_21_Region_1/{name}",
]

grid_title = "Pixel Distribution by Raman Band Contribution (Red = 478 cm$^{-1}$ and Blue = 851 cm$^{-1}$)"
set_grid(
    title=grid_title, image_paths=paths_topo, save_path=f"./figures/composition_multibands_grid.png",
    crop_list=[60, 180, 
               1940-60, 1920-180],
    facecolor='#FFFFFF', titlecolor='#383838'
)

# region
# st = f"./figures/maps/St CLs/bands_280to1780_no-bg_nearest"
# kc = f"./figures/maps/St kC CLs/bands_280to1780_no-bg_nearest"
# ic = f"./figures/maps/St iC CLs/bands_280to1780_no-bg_nearest"

# # === 478 cm-1 band - raw ===
# band = '478'
# name = f'band_{band}_raw_global'

# paths_band = [
#         f"{st}/{band}/St_CL_0_Region_{1}_{name}.png",
#         f"{st}/{band}/St_CL_7_Region_{2}_{name}.png",
#         f"{st}/{band}/St_CL_14_Region_{1}_{name}.png",
#         f"{st}/{band}/St_CL_21_Region_{2}_{name}.png",

#         f"{kc}/{band}/St_kC_CL_0_Region_{1}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_{1}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_{2}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_{1}_{name}.png",

#         f"{ic}/{band}/St_iC_CL_0_Region_{1}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_{1}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_{2}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_{1}_{name}.png",
# ]

# set_grid(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{name}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )

# # === 862 cm-1 band - raw ===
# band = '862'
# name = f'band_{band}_raw_global'

# paths_band = [
#         f"{st}/{band}/St_CL_0_Region_{1}_{name}.png",
#         f"{st}/{band}/St_CL_7_Region_{2}_{name}.png",
#         f"{st}/{band}/St_CL_14_Region_{1}_{name}.png",
#         f"{st}/{band}/St_CL_21_Region_{2}_{name}.png",

#         f"{kc}/{band}/St_kC_CL_0_Region_{1}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_{1}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_{2}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_{1}_{name}.png",

#         f"{ic}/{band}/St_iC_CL_0_Region_{1}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_{1}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_{2}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_{1}_{name}.png",
# ]

# set_grid(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{name}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )

# # === 939 cm-1 band - raw ===
# band = '939'
# name = f'band_{band}_raw_global'

# paths_band = [
#         f"{st}/{band}/St_CL_0_Region_{1}_{name}.png",
#         f"{st}/{band}/St_CL_7_Region_{2}_{name}.png",
#         f"{st}/{band}/St_CL_14_Region_{1}_{name}.png",
#         f"{st}/{band}/St_CL_21_Region_{2}_{name}.png",

#         f"{kc}/{band}/St_kC_CL_0_Region_{1}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_{1}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_{2}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_{1}_{name}.png",

#         f"{ic}/{band}/St_iC_CL_0_Region_{1}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_{1}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_{2}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_{1}_{name}.png",
# ]

# set_grid(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{name}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )

# # === 1080 cm-1 band - raw ===
# band = '1080'
# name = f'band_{band}_raw_global'

# paths_band = [
#         f"{st}/{band}/St_CL_0_Region_{1}_{name}.png",
#         f"{st}/{band}/St_CL_7_Region_{2}_{name}.png",
#         f"{st}/{band}/St_CL_14_Region_{1}_{name}.png",
#         f"{st}/{band}/St_CL_21_Region_{2}_{name}.png",

#         f"{kc}/{band}/St_kC_CL_0_Region_{1}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_{1}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_{2}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_{1}_{name}.png",

#         f"{ic}/{band}/St_iC_CL_0_Region_{1}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_{1}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_{2}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_{1}_{name}.png",
# ]

# set_grid(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{name}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )


# # === 1650 cm-1 band - raw ===
# band = '1650'
# name = f'band_{band}_raw_global'

# paths_band = [
#         f"{st}/{band}/St_CL_0_Region_{1}_{name}.png",
#         f"{st}/{band}/St_CL_7_Region_{2}_{name}.png",
#         f"{st}/{band}/St_CL_14_Region_{1}_{name}.png",
#         f"{st}/{band}/St_CL_21_Region_{2}_{name}.png",

#         f"{kc}/{band}/St_kC_CL_0_Region_{1}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_{1}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_{2}_{name}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_{1}_{name}.png",

#         f"{ic}/{band}/St_iC_CL_0_Region_{1}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_{1}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_{2}_{name}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_{1}_{name}.png",
# ]

# set_grid(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{name}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )

# # === multiband - raw ===
# st = f"./figures/maps/St CLs/multi_280to1780_no-bg_nearest"
# kc = f"./figures/maps/St kC CLs/multi_280to1780_no-bg_nearest"
# ic = f"./figures/maps/St iC CLs/multi_280to1780_no-bg_nearest"

# name = f'spectrum_raw'

# paths_band = [
#         f"{st}/St_CL_0_Region_{1}_{name}.png",
#         f"{st}/St_CL_7_Region_{2}_{name}.png",
#         f"{st}/St_CL_14_Region_{1}_{name}.png",
#         f"{st}/St_CL_21_Region_{2}_{name}.png",

#         f"{kc}/St_kC_CL_0_Region_{2}_{name}.png",
#         f"{kc}/St_kC_CL_7_Region_{1}_{name}.png",
#         f"{kc}/St_kC_CL_14_Region_{2}_{name}.png",
#         f"{kc}/St_kC_CL_21_Region_{1}_{name}.png",

#         f"{ic}/St_iC_CL_0_Region_{2}_{name}.png",
#         f"{ic}/St_iC_CL_7_Region_{1}_{name}.png",
#         f"{ic}/St_iC_CL_14_Region_{2}_{name}.png",
#         f"{ic}/St_iC_CL_21_Region_{3}_{name}.png",
# ]

# set_grid(
#     title="Multiband maps | R: 862 cm$^{-1}$; G: 939 cm$^{-1}$; B: 1650 cm$^{-1}$;", 
#     image_paths=paths_band, save_path=f"{name}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )
# endregion
