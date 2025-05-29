
import ramanspy as rp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from preprocessing import preprocess
from loaders       import load_spectrum


def normalize(y):
    return (y - y.min()) / (y.max() - y.min() + 1e-12)

def interactive_preprocessing_tuner(spectrum):

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0.25, bottom=0.55)

    # Inicial
    crop_min, crop_max = 300, 1800

    axcolor = 'lightgoldenrodyellow'
    ax_k    = plt.axes([0.25, 0.45, 0.65, 0.03], facecolor=axcolor)
    ax_t    = plt.axes([0.25, 0.40, 0.65, 0.03], facecolor=axcolor)
    ax_w    = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor=axcolor)
    ax_p    = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
    ax_cmin = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_cmax = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)

    s_k    = Slider(ax_k,    'Kernel Size', 1,    15,   valinit=5,        valstep=1)
    s_t    = Slider(ax_t,    'Threshold',   1,    100,  valinit=15,       valstep=1)
    s_w    = Slider(ax_w,    'Smooth Win',  3,    31,   valinit=11,       valstep=2)
    s_p    = Slider(ax_p,    'Polyorder',   1,    5,    valinit=3,        valstep=1)
    s_cmin = Slider(ax_cmin, 'Crop Min',    30,   400,  valinit=crop_min, valstep=5)
    s_cmax = Slider(ax_cmax, 'Crop Max',    1000, 1800, valinit=crop_max, valstep=5)

    # Plot inicial vazio
    l_raw, = ax.plot([], [], color='slategray',  lw=1.25, alpha=0.6, label='Raw')
    l_prc, = ax.plot([], [], color='dodgerblue', lw=1.25, alpha=0.6, label='Processed')

    ax.set_xlabel("Raman Shift")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title("Interactive Preprocessing Tuner")
    ax.legend()

    def update(val):
        crop_range = (int(s_cmin.val), int(s_cmax.val))

        x = spectrum.spectral_axis
        y = spectrum.spectral_data
        mask = (x >= crop_range[0]) & (x <= crop_range[1])
        x_crop = x[mask]
        y_crop = y[mask]

        l_raw.set_xdata(x_crop)
        l_raw.set_ydata(normalize(y_crop))

        params = {
            "crop_range"       : crop_range,
            "smooth_window"    : int(s_w.val),
            "smooth_polyorder" : int(s_p.val),
            "despike_kernel"   : int(s_k.val),
            "despike_threshold": int(s_t.val)
        }

        processed = preprocess(spectrum, **params)
        l_prc.set_xdata(processed.spectral_axis)
        l_prc.set_ydata(normalize(processed.spectral_data))

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    for s in [s_k, s_t, s_w, s_p, s_cmin, s_cmax]:
        s.on_changed(update)

    update(None)
    plt.show()


if __name__ == '__main__':
    s = load_spectrum("./data/St kC CLs/St kC CL 0 Region 1.txt")
    interactive_preprocessing_tuner(s)