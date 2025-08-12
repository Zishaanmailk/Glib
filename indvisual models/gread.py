import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# improve visual quality
plt.rcParams['lines.antialiased'] = True
plt.rcParams['path.simplify'] = False   # avoid simplifying the polyline (higher fidelity)
plt.rcParams['figure.dpi'] = 120        # increase DPI for crisper lines

# Load data
data = pd.read_csv("data2")
x = np.array(data['x'], dtype=float)
y = np.array(data['y'], dtype=float)

def compute_gd_history(x, y, iters=200, lr=1e-5):
    """Run simple linear GD and return arrays of m and c for each iteration."""
    m = 0.0
    c = 0.0
    n = len(x)
    m_hist = []
    c_hist = []
    for _ in range(iters):
        ypred = m * x + c
        md = -(2.0 / n) * np.sum(x * (y - ypred))
        cd = -(2.0 / n) * np.sum(y - ypred)
        m -= lr * md
        c -= lr * cd
        m_hist.append(m)
        c_hist.append(c)
    return np.array(m_hist), np.array(c_hist)

def animate_high_quality(x, y, iters=200, lr=1e-5, fade=True):
    # compute parameter history
    m_hist, c_hist = compute_gd_history(x, y, iters=iters, lr=lr)

    # sort x for smooth plotting (lines must be monotonic in x)
    sort_idx = np.argsort(x)
    x_s = x[sort_idx]
    y_s = y[sort_idx]

    # precompute predicted curves on sorted x
    ypreds_sorted = [m_hist[k] * x_s + c_hist[k] for k in range(iters)]

    # prepare line segments: each element is an (N,2) array for a polyline
    segments = [np.column_stack((x_s, y_pred)) for y_pred in ypreds_sorted]

    # colors (fade older lines if desired)
    if fade:
        alphas = np.linspace(0.12, 1.0, iters)    # older -> more transparent
    else:
        alphas = np.ones(iters)
    colors = [(1.0, 0.0, 0.0, a) for a in alphas]  # RGBA red with varying alpha

    # set up figure & axes
    fig, (ax_trace, ax_live) = plt.subplots(1, 2, figsize=(12, 5))

    # scatter data (use unsorted to keep original ordering)
    ax_trace.scatter(x, y, s=18, color='blue', label='data')
    ax_trace.set_title('Trace (all past lines)')
    ax_trace.set_xlabel('x')
    ax_trace.set_ylabel('y')

    ax_live.scatter(x, y, s=18, color='blue', label='data')
    ax_live.set_title('Live (current line)')
    ax_live.set_xlabel('x')
    ax_live.set_ylabel('y')

    # autoscale limits to include all predicted curves and data
    all_ys = np.hstack([y_s] + [yp for yp in ypreds_sorted])
    ymin, ymax = np.min(all_ys), np.max(all_ys)
    yrange = ymax - ymin if ymax != ymin else 1.0
    pad = 0.05 * yrange
    xmin, xmax = x_s.min(), x_s.max()
    xpad = 0.05 * (xmax - xmin if xmax != xmin else 1.0)

    for ax in (ax_trace, ax_live):
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - pad, ymax + pad)

    # create a LineCollection for the trace (initially empty)
    trace_lc = LineCollection(segments[:1], linewidths=2.0, linestyle='solid', antialiaseds=True, colors=colors[:1])
    ax_trace.add_collection(trace_lc)

    # create the single Line2D for the live plot
    (live_line,) = ax_live.plot([], [], linewidth=2.5, solid_capstyle='round', solid_joinstyle='round')

    # initialization function for blitting
    def init():
        trace_lc.set_segments([])        # no segments initially
        trace_lc.set_color(colors[:1])
        live_line.set_data([], [])
        return trace_lc, live_line

    # update function called each frame
    def update(frame):
        # update LineCollection to include all segments up to current frame
        trace_lc.set_segments(segments[:frame + 1])
        trace_lc.set_color(colors[:frame + 1])  # update colors to get fade effect

        # update live line to current prediction
        y_curr = segments[frame][:, 1]
        live_line.set_data(x_s, y_curr)

        return trace_lc, live_line

    # FuncAnimation with blit=True (faster & smoother)
    anim = FuncAnimation(fig, update, frames=iters, init_func=init,
                         interval=40, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()
    return anim

# usage
anim = animate_high_quality(x, y, iters=200, lr=1e-5, fade=True)
