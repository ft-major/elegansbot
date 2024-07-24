import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from matplotlib.transforms import Bbox


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        else:
            return False
    except NameError:
        return False


def set_bbox_inches_tight(fig, ratio_margin=0.01):
    assert 0 <= ratio_margin <= 1
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = bbox.width, bbox.height  # inches

    bb = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in fig.axes]
    tight_bbox_raw = Bbox.union(bb)
    tight_bbox = fig.transFigure.inverted().transform_bbox(tight_bbox_raw)

    l, b, w, h = tight_bbox.bounds  # 0~1 value (relative)
    margin = ratio_margin * max(width, height)
    w_m = margin / width
    h_m = margin / height
    l -= w_m
    b -= h_m
    w += w_m * 2
    h += h_m * 2
    for ax in fig.axes:
        l2, b2, w2, h2 = ax.get_position().bounds
        r2, t2 = l2 + w2, b2 + h2
        l3 = (l2 - l) / w
        r3 = (r2 - l) / w
        b3 = (b2 - b) / h
        t3 = (t2 - b) / h
        w3 = r3 - l3
        h3 = t3 - b3
        ax.set_position((l3, b3, w3, h3))

    fig.set_size_inches((width + 2 * margin, height + 2 * margin))


def plot_outline_n_trajectory(
        ax,
        i,
        env,
        draw=False,  # whether to run fig.canvas.draw()
        show_legend=True,
        init_fig=False,
        dpi=None,
):
    """
    Drawing single snapshot image
    Body outline + head/tail track
    """

    if init_fig == True:
        xy_tip_log = env.xy_tip_log

        width = xy_tip_log[:, 0, :].max() - xy_tip_log[:, 0, :].min() + 1
        height = xy_tip_log[:, 1, :].max() - xy_tip_log[:, 1, :].min() + 1
        yx_ratio = height / width
        width = 3 / np.sqrt(yx_ratio) + 1
        width = max(width, 4.2)
        height = 3 * np.sqrt(yx_ratio)

        fig, ax = plt.subplots(dpi=dpi, figsize=(width, height))

        return fig, ax

    t_log = env.t_log
    xy_tip_log = env.xy_tip_log

    x_tip = xy_tip_log[i, 0, :]
    y_tip = xy_tip_log[i, 1, :]
    track_head = xy_tip_log[:i, :, 0]
    track_tail = xy_tip_log[:i, :, -1]
    time = t_log[i]

    fig = ax.get_figure()

    n = env.n
    thickness = (env.L / 20) * np.sin(np.arccos(np.linspace(-.98, .98, n + 1)))

    atan2 = np.arctan2(np.diff(y_tip), np.diff(x_tip))
    thetaDiff = np.diff(atan2)
    thetaDiff[thetaDiff < -np.pi] += 2 * np.pi
    thetaDiff[thetaDiff > np.pi] -= 2 * np.pi

    theta = np.cumsum(np.concatenate([[atan2[0]], thetaDiff]))

    midang = theta[:-1] + theta[1:]
    difang = np.abs(theta[:-1] - theta[1:]) < np.pi
    midang = midang * difang + (midang + 2 * np.pi) * np.logical_not(difang)
    angle = np.concatenate([[theta[0]], midang / 2, [theta[-1]]])

    sign = 1 if env.polarity_clockwise == True else -1
    dx, dy = -thickness * np.sin(angle) * sign, thickness * np.cos(angle) * sign
    x_dorsal, y_dorsal = x_tip + dx, y_tip + dy
    x_ventral, y_ventral = x_tip - dx, y_tip - dy

    if ax.lines:
        ax.lines[0].set_data(x_tip[0], y_tip[0])
        ax.lines[1].set_data(x_dorsal, y_dorsal)
        ax.lines[2].set_data(x_ventral, y_ventral)
        ax.lines[3].set_data([x_dorsal[0], x_ventral[0]], [y_dorsal[0], y_ventral[0]])
        ax.lines[4].set_data([x_dorsal[-1], x_ventral[-1]], [y_dorsal[-1], y_ventral[-1]])
        ax.lines[5].set_data(track_head[:, 0], track_head[:, 1])
        ax.lines[6].set_data(track_tail[:, 0], track_tail[:, 1])
        ax.texts[0].set_text(f'Time: {time:.3f} (sec)')

        if draw == True:
            fig.canvas.draw()
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        left = np.min(xy_tip_log[:, 0, :], axis=None) - 0.5
        right = np.max(xy_tip_log[:, 0, :], axis=None) + 0.5
        top = np.max(xy_tip_log[:, 1, :], axis=None) + 0.5
        bottom = np.min(xy_tip_log[:, 1, :], axis=None) - 0.5
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_aspect(1)

        lw = 1
        ax.plot(x_tip[0], y_tip[0], 'o', color=[1, 1, 0], label='Head position', linewidth=lw, markersize=lw * 10)
        ax.plot(x_dorsal, y_dorsal, '-', color=[.8, 0, 0], label='Dorsal', linewidth=lw)
        ax.plot(x_ventral, y_ventral, '-', color=[0, 0, 1], label="Ventral", linewidth=lw)
        ax.plot([x_dorsal[0], x_ventral[0]], [y_dorsal[0], y_ventral[0]], '-',
                color=[.5, 0, .5], linewidth=lw)
        ax.plot([x_dorsal[-1], x_ventral[-1]], [y_dorsal[-1], y_ventral[-1]],
                '-', color=[.5, 0, .5], linewidth=lw)
        ax.plot(track_head[:, 0], track_head[:, 1], '-',
                color=[1, .5, 0], linewidth=lw / 2, label="Head track")
        ax.plot(track_tail[:, 0], track_tail[:, 1], '-',
                color=[0, .5, 1], linewidth=lw / 2, label="Tail track")
        ax.text(left + 0.1, bottom + 0.1, f'Time: {time:.3f} (sec)')
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        if show_legend == True:
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

    return ax.lines + ax.texts


def play_animation(
        env,
        func_plot=plot_outline_n_trajectory,  # drawing function
        speed_playback=1,
        dpi=120,
        bbox_inches_tight=True,
        ax=None,
):
    """
    Playing animation of worm's movement from the records.


    Example usage
    -------------
    env.play_animation()


    Example usage in Jupyter notebook
    ---------------------------------
    %matplotlib notebook
    env.play_animation()
    %matplotlib inline
    """

    assert speed_playback > 0
    dt = env.dt_snapshot / speed_playback

    running_in_notebook = is_notebook()
    if not (running_in_notebook):
        plt.ion()

    if type(ax) != type(None):
        fig = ax.get_figure()
    else:
        fig, ax = func_plot(None, 0, env, init_fig=True, dpi=dpi)

    if not (running_in_notebook):
        plt.show()

    func_plot(ax, 0, env, draw=True)

    if bbox_inches_tight == True:
        set_bbox_inches_tight(fig)

    i = 0
    delay = 0
    func_plot(ax, i, env, draw=True)
    while True:
        tmp = time.time()
        step = int(delay / dt)
        if step > 0:
            i += step
            if i >= env.n_snapshot:
                break
            func_plot(ax, i, env, draw=True)
            if not (running_in_notebook):
                fig.canvas.flush_events()
            delay -= step * dt
        else:
            time.sleep(dt / 10)
        delay += time.time() - tmp

    func_plot(ax, env.n_snapshot - 1, env, draw=True)

    if not (running_in_notebook):
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)


def save_animation(
        env,
        file_name="animation.mp4",
        speed_playback=1,
        dpi=120,
        fps=60,
        bbox_inches_tight=True,
        func_plot=plot_outline_n_trajectory,  # drawing function
        ax=None,
):
    """
    Saving animation of worm's movement into MP4 video file.


    This function requires FFMPEG to be installed on your computer.
    and 'ffmpeg' command to be accessible by setting PATH environment variable properly.
    Check out 'https://ffmpeg.org'.


    Example usage
    -------------
    env.save_animation(file_name=f"example.mp4")
    """
    assert speed_playback > 0
    dt = env.dt_snapshot / speed_playback

    running_in_notebook = is_notebook()
    if not (running_in_notebook):
        plt.ion()

    if type(ax) != type(None):
        fig = ax.get_figure()
    else:
        fig, ax = func_plot(None, 0, env, init_fig=True, dpi=dpi)

    func_plot(ax, 0, env)  # Drawing object(line, text, ...) initialization

    if bbox_inches_tight == True:
        set_bbox_inches_tight(fig)

    def init():  # Video initialization
        return func_plot(ax, 0, env)

    idx_ = [0]
    i = 0
    delay = 0
    time_elapsed = 0
    while True:
        time_elapsed += 1 / fps
        delay += 1 / fps
        step = int(delay / dt)
        delay -= step * dt
        i += step
        if i < env.n_snapshot - 1:
            idx_.append(i)
        else:
            idx_.append(env.n_snapshot - 1)
            break

    def update(i):
        return func_plot(ax, idx_[i], env)

    anim = mpl_animation.FuncAnimation(
        fig,
        func=update,
        init_func=init,
        frames=len(idx_),
        interval=1000 / fps,
        blit=True,
    )
    anim.save(file_name)
    plt.close(fig)


def plot_overview(env, n_row=6, dpi=80):
    """
    Plotting overview of worm's trajectory
    """

    xy_tip_log = env.xy_tip_log
    left = np.min(xy_tip_log[:, 0, :], axis=None) - 0.5
    right = np.max(xy_tip_log[:, 0, :], axis=None) + 0.5
    top = np.max(xy_tip_log[:, 1, :], axis=None) + 0.5
    bottom = np.min(xy_tip_log[:, 1, :], axis=None) - 0.5
    width = right - left
    height = top - bottom
    yx_ratio = height / width

    fig, ax_ = plt.subplots(
        nrows=n_row,
        dpi=dpi,
        figsize=(5, n_row * 2),
    )

    for i, idx in enumerate(np.linspace(0, env.n_snapshot - 1, n_row).astype(int)):
        ax = ax_[i]
        if i == 0:
            plot_outline_n_trajectory(ax, idx, env, show_legend=True)
        else:
            plot_outline_n_trajectory(ax, idx, env, show_legend=False)

        if i < n_row - 1:
            ax_[i].set_xticklabels([])
            ax_[i].set_xlabel('')

        if yx_ratio > 1 / 1.7:  # to keep the time information in the box
            x_center = (left + right) / 2
            ax.set_xlim(x_center - height * .85, x_center + height * .85)
            ax.texts[0].set_position((x_center - height * .85 + 0.1, bottom + 0.1))

    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches((bbox.width, bbox.height))
    set_bbox_inches_tight(fig)
    plt.show()


def plot_speed_graph(env, dpi=80):
    """
    Plot speed graph


    Example usage
    -------------
    env.velocity_graph()
    """
    t_log = env.t_log
    vc_log = env.vc_log

    speed_log = np.sqrt((vc_log ** 2).sum(axis=-1))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    ax.axhline(y=0, linestyle='--', linewidth=.5, color=(.3, .3, .3))
    ax.plot(t_log, speed_log, color=[0, 0, 1], label='|v|')
    ax.plot(t_log, vc_log[:, 0], '--', color=[1, 0, 0], label=r'$v_x$')
    ax.plot(t_log, vc_log[:, 1], '--', color=[0, .5, 0], label=r'$v_y$')
    ax.set_title("worm locomotion speed")
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("worm speed (mm/sec)")
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches((bbox.width, bbox.height))
    set_bbox_inches_tight(fig)
    plt.show()
    print(f"average worm speed: {np.mean(speed_log):.5f} (mm/sec)")
