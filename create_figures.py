#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import colorsys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Patch
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatter

import seaborn as sns
import pandas as pd

import re

import numpy as np


from typing import (
    Tuple, Union, List, Dict, Optional,
)


data_folder = Path("./data")

plot_folder = Path("./figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,          # slightly larger base font
    "axes.labelsize": 8,
    "legend.fontsize": 6,
    "legend.title_fontsize": 6,
    "lines.linewidth": 1.0,
    "figure.dpi": 300,
})

# Default base colours (one per distinct b)
base_colors = ["#DDA448", "#476C9B", "#87C38F", "#DA2C38", "#43291F", "#FFD3BA"]




##### Plot functions, called in other functions below

def _force_minor_log_labels(ax: plt.Axes,numticks=6):
    def _minor_fmt(val: float, pos: int) -> str:
        # Hide label if it's exactly a power of ten (major tick)
        if np.isclose(val, 10 ** np.round(np.log10(val))):
            return ""
        return f"{val:g}"

    ax.yaxis.set_minor_formatter(FuncFormatter(_minor_fmt))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10, 3) * 0.1, numticks=numticks))

    for lab in ax.yaxis.get_minorticklabels():
        lab.set_fontsize(6)  # smaller font for minor ticks
        lab.set_rotation(25)
        lab.set_horizontalalignment('right')
        lab.set_visible(True)


def _make_shaded_palette(
    df: pd.DataFrame,
    base_colors: Optional[List[str]] = None,
    light: float = 0.45,
    dark: float = 0.85,
) -> Dict[str, str]:
    """Map each string key ``"b,d"`` → hex colour.

    * One distinct *hue* per **b** (taken from *base_colors*).
    * Within that hue, vary *lightness* linearly across **d** values.
    """
    if base_colors is None:
        base_colors = globals()["base_colors"]

    unique_b = sorted(df["b"].unique())
    if len(unique_b) > len(base_colors):
        raise ValueError("Need ≥{} base colours (have {}).".format(len(unique_b), len(base_colors)))

    b2c = dict(zip(unique_b, base_colors))
    d_min, d_max = df["d"].min(), df["d"].max()

    def _l(v: float) -> float:
        return light + (dark - light) * (v - d_min) / (d_max - d_min)

    palette: Dict[str, str] = {}
    for b_val in unique_b:
        h, _l0, s = colorsys.rgb_to_hls(*mcolors.to_rgb(b2c[b_val]))
        for d_val in sorted(df.loc[df["b"] == b_val, "d"].unique()):
            rgb = colorsys.hls_to_rgb(h, _l(d_val), s)
            palette[f"{b_val},{d_val}"] = mcolors.to_hex(rgb)
    return palette


def _add_split_legends(
    fig: plt.Figure,
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    legend_fontsize: int = 9,
) -> None:
    """Colour legend outside; marker legend inside lower‑right."""

    handles, labels = ax.get_legend_handles_labels()
    if ax.legend_:
        ax.legend_.remove()

    style_set = set(df["QueryDescr"].astype(str))
    hue_pat = re.compile(r"^\d+,\d+$")

    hue_h, hue_l = [], []
    style_h, style_l = [], []
    guide_h, guide_l = [], []

    for h, l in zip(handles, labels):
        if l in style_set:
            style_h.append(h); style_l.append(l.replace("_", " "))
        elif hue_pat.match(l):
            hue_h.append(h); hue_l.append(l)
        elif l.startswith("Guide"):
            guide_h.append(h); guide_l.append(l)

    # ---- Outside (right) colour legend --------------------------------------
    hue_sorted = sorted(zip(hue_h, hue_l), key=lambda t: tuple(map(int, t[1].split(','))))
    if hue_sorted:
        h_h, h_l = zip(*hue_sorted)
        fig.legend(
            h_h, h_l, title="Colour: (b,d)",
            loc="upper left", bbox_to_anchor=(.95, .95),
            borderaxespad=0.0, frameon=False,
            fontsize=legend_fontsize, title_fontsize=legend_fontsize,
        )

    # ---- Inside (axes) marker legend  ---------------------------------------
    if style_h:
        ax.legend(
            style_h, style_l, title="Query flavor",
            loc="lower right", frameon=False,
            fontsize=legend_fontsize, title_fontsize=legend_fontsize,
        )

    # Optional guide legend goes outside under colour legend
    if guide_h:
        fig.legend(
            guide_h, guide_l,
            loc="upper left", bbox_to_anchor=(1.00, 0.60),
            borderaxespad=0.0, frameon=False,
            fontsize=legend_fontsize,
        )


def plot_2d_distribution(data, n_quantiles=10, highlight_rectangle=None,
                         highlight_points=None, with_marginals=False, colors=base_colors,
                         fig_size=(7,3),
                         target_path="grid.pdf"):
    x, y = data[:, 0], data[:, 1]


    # Quantile calculations (right plot)
    x_q = np.quantile(x, np.linspace(0, 1, n_quantiles + 1))
    y_q = np.quantile(y, np.linspace(0, 1, n_quantiles + 1))
    x_bins = np.searchsorted(x_q, x, side='left') - 1
    y_bins = np.searchsorted(y_q, y, side='left') - 1
    x_bins = np.clip(x_bins, 0, n_quantiles - 1)
    y_bins = np.clip(y_bins, 0, n_quantiles - 1)

    x_frac = (x - x_q[x_bins]) / (x_q[x_bins + 1] - x_q[x_bins])
    y_frac = (y - y_q[y_bins]) / (y_q[y_bins + 1] - y_q[y_bins])
    x_trans, y_trans = x_bins + x_frac, y_bins + y_frac

    # Set up GridSpec layout
    fig = plt.figure(figsize=fig_size, constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig, width_ratios=[4, 0.3, 4, 0.1], height_ratios=[0.8, 4])

    # Central plot (left)
    ax_main = fig.add_subplot(gs[1, 0])
    # sns.kdeplot(x=x, y=y, ax=ax_main, fill=True, cmap="Blues", levels=30)
    ax_main.scatter(x, y, s=4, alpha=0.9, edgecolor=None, color=colors[1])  # Overlay scatter on top of KDE

    # Get the full visible plot extent
    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()
    
    ax_main.add_patch(Rectangle(
        (x_q[0], y_q[0]),
        x_q[-1] - x_q[0], y_q[-1] - y_q[0],
        color=colors[3], alpha=0.05, zorder=1.5
    ))


    ax_main.set_xlabel(r"Attribute A")
    ax_main.set_ylabel(r"Attribute B")
    ax_main.set_title(r"Original Data Space")

    # Draw quantile lines on the original (left) plot
    for x_val in x_q:
        ax_main.axvline(x_val, color='gray', linestyle='--', lw=0.6, alpha=0.4)
    for y_val in y_q:
        ax_main.axhline(y_val, color='gray', linestyle='--', lw=0.6, alpha=0.4)

    # Marginal histograms (top, right)
    if with_marginals:
        ax_xhist = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_yhist = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)

        for ax in [ax_xhist, ax_yhist]:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        ax_xhist.hist(x, bins=30, color="gray", alpha=0.4)
        ax_yhist.hist(y, bins=30, orientation='horizontal', color="gray", alpha=0.4)

    # Right plot (quantile-grid transformed scatter)
    ax_right = fig.add_subplot(gs[1, 2])
    sns.scatterplot(x=x_trans, y=y_trans, ax=ax_right, s=4, alpha=0.9, edgecolor=None)
    ax_right.set_xlim(0, n_quantiles)
    ax_right.set_ylim(0, n_quantiles)
    # Optional: transform and plot highlight points
    if highlight_points is not None:
        highlight_points = np.array(highlight_points)
        x_hp, y_hp = highlight_points[:, 0], highlight_points[:, 1]

        # Transform highlight points
        hp_x_bins = np.minimum(np.searchsorted(x_q, x_hp, side='right') - 1, n_quantiles - 1)
        hp_y_bins = np.minimum(np.searchsorted(y_q, y_hp, side='right') - 1, n_quantiles - 1)
        hp_x_frac = (x_hp - x_q[hp_x_bins]) / (x_q[hp_x_bins + 1] - x_q[hp_x_bins])
        hp_y_frac = (y_hp - y_q[hp_y_bins]) / (y_q[hp_y_bins + 1] - y_q[hp_y_bins])
        hp_x_trans, hp_y_trans = hp_x_bins + hp_x_frac, hp_y_bins + hp_y_frac

        # Plot original highlight points (left)
        ax_main.scatter(x_hp, y_hp, c='red', s=60, marker='X', edgecolor='black')

        # Plot transformed highlight points (right)
        ax_right.scatter(hp_x_trans, hp_y_trans, c='red', s=60, marker='X', edgecolor='black')

    # Grid lines on the right plot
    for i in range(n_quantiles + 1):
        ax_right.axvline(i, color='gray', linestyle='--', lw=0.8, alpha=0.7)
        ax_right.axhline(i, color='gray', linestyle='--', lw=0.8, alpha=0.7)


    if highlight_rectangle is not None:
        pt1, pt2 = highlight_rectangle
        rect_min = np.minimum(pt1, pt2)
        rect_max = np.maximum(pt1, pt2)
        width, height = rect_max - rect_min

        # Add shaded rectangle to data space plot (left)
        ax_main.add_patch(Rectangle(
            rect_min, width, height,
            facecolor='none', edgecolor='black', hatch='////', linewidth=1.0, zorder=3
        ))

        # Transform rectangle corners to quantile-transformed space
        x0_bin = np.searchsorted(x_q, rect_min[0], side='left') - 1
        x1_bin = np.searchsorted(x_q, rect_max[0], side='left') - 1
        y0_bin = np.searchsorted(y_q, rect_min[1], side='left') - 1
        y1_bin = np.searchsorted(y_q, rect_max[1], side='left') - 1

        x0_bin = np.clip(x0_bin, 0, n_quantiles - 1)
        x1_bin = np.clip(x1_bin, 0, n_quantiles - 1)
        y0_bin = np.clip(y0_bin, 0, n_quantiles - 1)
        y1_bin = np.clip(y1_bin, 0, n_quantiles - 1)

        # Fractional positions
        x0_frac = (rect_min[0] - x_q[x0_bin]) / (x_q[x0_bin + 1] - x_q[x0_bin])
        x1_frac = (rect_max[0] - x_q[x1_bin]) / (x_q[x1_bin + 1] - x_q[x1_bin])
        y0_frac = (rect_min[1] - y_q[y0_bin]) / (y_q[y0_bin + 1] - y_q[y0_bin])
        y1_frac = (rect_max[1] - y_q[y1_bin]) / (y_q[y1_bin + 1] - y_q[y1_bin])

        # Construct transformed rectangle
        x_start = x0_bin + x0_frac
        x_end = x1_bin + x1_frac
        y_start = y0_bin + y0_frac
        y_end = y1_bin + y1_frac

        # Add shaded rectangle to transformed space plot (right)
        ax_right.add_patch(Rectangle(
            (x_start, y_start), x_end - x_start, y_end - y_start,
            facecolor='none', edgecolor='black', hatch='////', linewidth=1.0, zorder=3
        ))

        # Find all points in same grid cells
        in_same_cells = (
            (x_bins >= x0_bin) & (x_bins <= x1_bin) &
            (y_bins >= y0_bin) & (y_bins <= y1_bin)
        )

        # Now determine if those points are also *outside* the rectangle range
        outside_rect = (
            (x < rect_min[0]) | (x > rect_max[0]) |
            (y < rect_min[1]) | (y > rect_max[1])
        )

        # Final selection: same cell, but outside actual query box
        highlight_mask = in_same_cells & outside_rect

        # Transformed coordinates of those points
        x_highlight = x_trans[highlight_mask]
        y_highlight = y_trans[highlight_mask]

        # Overlay these on the right plot
        ax_right.scatter(
            x_highlight, y_highlight,
            s=16, color=colors[0], marker='s', edgecolor='black', label='False Positives'
        )

        # Add legend for false positives
        
        # Create a legend entry for the rectangle
        rectangle_legend = Patch(
            facecolor='none', edgecolor='black', hatch='////', label='Query Rectangle'
        )

        # Combine with existing legend items (if any)
        handles, labels = ax_right.get_legend_handles_labels()
        handles.append(rectangle_legend)
        labels.append('Query Rectangle')
        ax_right.legend(handles=handles, labels=labels, loc='lower right', frameon=True)

    if highlight_rectangle is not None:
        pt1, pt2 = highlight_rectangle
        rect_min = np.minimum(pt1, pt2)
        rect_max = np.maximum(pt1, pt2)
        width, height = rect_max - rect_min

        # Add shaded rectangle to data space plot (left)
        ax_main.add_patch(Rectangle(
            rect_min, width, height,
            facecolor='none', edgecolor='black', hatch='////', linewidth=1.0, zorder=3
        ))

        # Transform rectangle corners to quantile-transformed space
        x0_bin = np.searchsorted(x_q, rect_min[0], side='left') - 1
        x1_bin = np.searchsorted(x_q, rect_max[0], side='left') - 1
        y0_bin = np.searchsorted(y_q, rect_min[1], side='left') - 1
        y1_bin = np.searchsorted(y_q, rect_max[1], side='left') - 1

        x0_bin = np.clip(x0_bin, 0, n_quantiles - 1)
        x1_bin = np.clip(x1_bin, 0, n_quantiles - 1)
        y0_bin = np.clip(y0_bin, 0, n_quantiles - 1)
        y1_bin = np.clip(y1_bin, 0, n_quantiles - 1)

        # Fractional positions
        x0_frac = (rect_min[0] - x_q[x0_bin]) / (x_q[x0_bin + 1] - x_q[x0_bin])
        x1_frac = (rect_max[0] - x_q[x1_bin]) / (x_q[x1_bin + 1] - x_q[x1_bin])
        y0_frac = (rect_min[1] - y_q[y0_bin]) / (y_q[y0_bin + 1] - y_q[y0_bin])
        y1_frac = (rect_max[1] - y_q[y1_bin]) / (y_q[y1_bin + 1] - y_q[y1_bin])

        # Construct transformed rectangle
        x_start = x0_bin + x0_frac
        x_end = x1_bin + x1_frac
        y_start = y0_bin + y0_frac
        y_end = y1_bin + y1_frac

        # Add shaded rectangle to transformed space plot (right)
        ax_right.add_patch(Rectangle(
            (x_start, y_start), x_end - x_start, y_end - y_start,
            facecolor='none', edgecolor='black', hatch='////', linewidth=1.0, zorder=3
        ))


    ax_right.set_xlabel(r"Quantile bins A")
    ax_right.set_ylabel(r"Quantile bins B")
    ax_right.set_title(r"Grid Space")

    # fig.suptitle(r"2D Distribution Before and After Quantization", fontsize=16)
    fig.set_constrained_layout_pads(h_pad=0.01, hspace=0.01)

    # safe figure to file:
    target_path = Path(target_path)
    fig.savefig(target_path, format="pdf", bbox_inches="tight")


def plot_grouped_scatter(df, grp1="Strategy", grp2="b", symb="s", yscale="log", 
                                ymin=None, ymax=None, 
                                path="./", exp_suffix="", show_errorbars=False, show_means="expand_some"):
    """
    Creates one plot per attribute for a dataframe with either 3-level or 4-level MultiIndex.
    Uses grp1 for x-axis, grp2 for hue, and optionally symb to assign per-point markers.
    Aggregates over the remaining level (e.g., query) for mean/std.
    """

    path = Path(path)
    assert path.exists() and path.is_dir(), f"{path} does not exist or is no folder!"

    if not isinstance(df.index, pd.MultiIndex) or len(df.index.levels) not in [3, 4]:
        raise ValueError("DataFrame must have a 3- or 4-level MultiIndex")

    df_flat = df.reset_index()
    value_columns = df.columns

    # Colors for hues (grp2)
    unique_hues = df_flat[grp2].unique()
    # palette = sns.color_palette('Set2', n_colors=len(unique_hues))
    palette = base_colors
    color_mapping = dict(zip(unique_hues, palette))

    # Marker mapping if symb is given
    if symb:
        unique_symbs = df_flat[symb].unique()
        marker_styles = ['o', 's', '^', 'v', 'D', 'X', 'P', '*', '+', 'x']
        marker_mapping = {val: marker_styles[i % len(marker_styles)] for i, val in enumerate(unique_symbs)}
    else:
        df_flat["_dummy_symb"] = "_"
        symb = "_dummy_symb"
        marker_mapping = {"_": 'o'}

    # Dodge hue values
    dodge_amount = 0.15
    hue_order = list(unique_hues)
    hue_offsets = {
        hue_val: (i - (len(hue_order) - 1) / 2) * dodge_amount
        for i, hue_val in enumerate(hue_order)
    }

    for col in value_columns:
        plt.figure(figsize=(10, 6))

        # Compute x-axis positions
        x_vals = df_flat[grp1].unique()
        x_pos_map = {val: i for i, val in enumerate(x_vals)}

        # Plot raw points manually with jitter and markers
        for (x_val, hue_val, symb_val), subdf in df_flat.groupby([grp1, grp2, symb]):
            x_base = x_pos_map[x_val] + hue_offsets[hue_val]
            jitter = np.random.uniform(-0.05, 0.05, size=len(subdf))
            plt.scatter(
                x=np.full(len(subdf), x_base) + jitter,
                y=subdf[col],
                color=color_mapping[hue_val],
                marker=marker_mapping[symb_val],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8,
                label=hue_val
            )

        if show_means:
            if type(show_means) is str:
                # filter only values for a specific grp1 value specified by the argument:
                df_flat = df_flat[df_flat[grp1] == show_means]

            # Group for mean/std
            grouped = df_flat.groupby([grp1, grp2, symb])[col]
            means = grouped.mean().reset_index()
            stds = grouped.std().reset_index()
            means = pd.merge(means, stds, on=[grp1, grp2, symb], suffixes=('', '_std'))

            for _, row in means.iterrows():
                x_val = row[grp1]
                hue_val = row[grp2]
                symb_val = row[symb]
                mean_val = row[col]
                std_val = row[f"{col}_std"]

                x_base = x_pos_map[x_val] + hue_offsets[hue_val]
                marker = marker_mapping[symb_val]
                color = color_mapping[hue_val]

                plt.scatter(
                    x=x_base,
                    y=mean_val,
                    color="grey",
                    edgecolor='black',
                    marker=marker,
                    s=50,
                    zorder=5
                )
                if show_errorbars:
                    if yscale == "log":
                        print(f"Warning: yscale is set to log, but error bars are shown for {col}!")
                    plt.errorbar(
                        x=x_base,
                        y=mean_val,
                        yerr=std_val,
                        fmt='none',
                        ecolor=color,
                        elinewidth=1.5,
                        capsize=4,
                        zorder=4
                    )

                y_offset = 0.3 * mean_val
                plt.text(
                    x=x_base,
                    y=mean_val + y_offset,
                    s=f"{mean_val:.1f}",
                    ha='center',
                    va='bottom',
                    fontsize=14,
                    color="black",
                    fontweight='bold',
                    zorder=6
                )

        # Final plot setup
        ax = plt.gca()
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([l.replace("_"," ") for l in x_vals], rotation=25, ha='right')

        # plt.title(f"{exp_suffix}: {col} over Aggregated Trials")
        plt.ylabel(col.replace("_", " "))
        plt.xlabel(grp1.replace("_", " "))

        plt.yscale(yscale)
        if ymin is not None or ymax is not None:
            plt.ylim(bottom=ymin if ymin is not None else None, top=ymax if ymax is not None else None)

        # Create legend for colors (hue)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend1 = plt.legend(by_label.values(), by_label.keys(), title=grp2,
                             loc='lower right', facecolor='white', frameon=False)
        plt.gca().add_artist(legend1)

        # Create legend for marker styles (symb)
        import matplotlib.lines as mlines
        symb_handles = [
            mlines.Line2D([], [], color='black', marker=marker_mapping[val], linestyle='None',
                          markersize=8, label=str(val))
            for val in marker_mapping
        ]
        plt.legend(handles=symb_handles, title=symb if symb != "_dummy_symb" else None,
                   loc='lower center', facecolor='white', frameon=False)
        plt.tight_layout()

        plt.savefig(path / f"{col}_{exp_suffix}.pdf")
        plt.close()


def plot_query_placement(df, reference_selectivity=None, figsize=(3.4,2.1), x_lim=(-0.000001,0.00002),
                           show_error_bars=False,
                           x_col="Selectivity", y_col="Runtime_[ms]",
                           path= "./result_cardinality_vs_runtime.pdf"):
    """
    Scatter plot of result size vs execution time with variance error bars,
    colored by 'table'
    
    Parameters:
    - df: pandas DataFrame containing at least columns ['Query', 'Table', x_col, y_col]
    - x_col: name of column for the x-axis (e.g., "result_cardinality")
    - y_col: name of column for the y-axis (e.g., "runtime")
    """
    # Aggregate mean and variance by query & table
    df = df.copy()

    agg = df.groupby(['Query', 'Table'])[[x_col, y_col]].agg(['mean', 'std'])
    # Flatten MultiIndex columns
    agg.columns = [f"{col[0]}_{col[1]}" for col in agg.columns]
    agg = agg.reset_index()
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sns.scatterplot(
        data=agg,
        x=f"{x_col}_mean",
        y=f"{y_col}_mean",
        hue='Table',
        style='Table',
        palette=base_colors,
        ax=ax,
        legend='full'
    )

    # Overlay error bars for variance
    if show_error_bars:
        ax.errorbar(
            x=agg[f"{x_col}_mean"],
            y=agg[f"{y_col}_mean"],
            xerr=agg[f"{x_col}_std"],
            yerr=agg[f"{y_col}_std"],
            fmt='none',
            alpha=0.9
        )

    # Highlight the first point
    first_point = agg.iloc[0]
    ax.scatter(
        first_point[f"{x_col}_mean"],
        first_point[f"{y_col}_mean"],
        color=base_colors[3],
        edgecolor='black',
        s=100,
        zorder=5,  # Draw on top
        # label='Original Query'
    )
    ax.annotate(
        "Example Query",
        (first_point[f"{x_col}_mean"], first_point[f"{y_col}_mean"]),
        textcoords="offset points",
        xytext=(10, 10),
        ha='left',
        color=base_colors[3],
        # fontsize=10,
        weight='bold'
    )

    # Add vertical reference line if given
    if reference_selectivity is not None:
        ax.axvline(
            x=reference_selectivity,
            color='black',
            linestyle='--',
            linewidth=2,
            # label="Independence Estimate"
        )
        ax.text(
            reference_selectivity,
            ax.get_ylim()[1] * 0.95,  # Near the top of the y-axis
            "Independence Estimate",
            color=base_colors[4],
            # fontsize=10,
            ha='center',
            va='top',
            rotation=90,
            backgroundcolor='white'
        )
    ax.legend(loc='center right',frameon=True) 

    ax.set_ylabel(y_col.replace("_"," "))
    # ax.set_title(f"")

    ax.set_xlabel(x_col.replace("_"," "))

    if x_lim is not None:
        ax.set_xlim(*x_lim)

    plt.savefig(path)


def plot_optimal_runtime(
    df: pd.DataFrame,
    *,
    x: str = "Team_Count",
    y: str = "Runtime_[ms]",
    hue: str = "Type",
    style: str | None = None,        # fall back to hue if None
    annotate: str = None,
    annotation_sfx: str = " ids",
    y2: str = None,
    group_cols: list[str] | None = None,
    agg_func: str | dict = "mean",   # can be "median", {"col": "sum"}, …
    err: str = "sd",
    pdf_path: str | Path = "plot.pdf",
    figsize: tuple[int, int] = (3.4, 2.5),
    line_kws: dict | None = None,
    secondary_line_kws: dict | None = None,
    log_y: bool = False,
):
    """
    Aggregate ``df`` and create a Seaborn line plot with:
      • primary y-axis: ``y`` with error bars (±err)
      • hue/style: ``hue`` (and optionally ``style``)
      • point annotations: values from ``annotate``
      • secondary y-axis: ``y2``

    Parameters
    ----------
    df : DataFrame
        Raw, un-aggregated data.
    x, y, hue, style, annotate, y2 : str
        Column names.
    group_cols : list[str] | None
        Extra columns to include in the group-by.  
        Default → ``[x, hue]`` (or `[x, hue, style]` if style supplied).
    agg_func : str | dict | callable
        Aggregation for all value columns.
    err : str | callable
        Error estimator (passed to seaborn, e.g. "sem", "sd", np.std, …).
    pdf_path : str | Path
        Where the PDF is written.
    line_kws / secondary_line_kws
        Keyword args forwarded to seaborn/plt plot calls.
    """

    # ---------- 1. aggregation ----------
    if group_cols is None:
        group_cols = [x, hue] + ([style] if style else [])
    cols = [*group_cols, x, y, y2, annotate]
    cols = [c for c in cols if c is not None]
    df = df[[*dict.fromkeys(cols)]].copy()
    grouped = (
        df.groupby(group_cols, dropna=False, sort=False)
          .agg(agg_func)
          .reset_index()
    )
    
    unique_hues = sorted(grouped[hue].unique())  # sort for consistent order
    # palette = sns.color_palette(n_colors=len(unique_hues))
    palette = base_colors
    color_map = dict(zip(unique_hues, palette))

    # Add one entry per hue category to the legend
    custom_handles = [
        Patch(facecolor=color_map[val], label=f"{val}")
        for val in unique_hues
    ]

    # ---------- 2. basic seaborn plot ----------
    sns.set_theme(style="ticks", context="paper")
    plt.figure(figsize=figsize, constrained_layout=True)

    line_kwargs = dict(marker="o")
    line_kwargs.update(line_kws or {})

    ax = sns.lineplot(
        data=grouped,
        x=x, y=y,
        hue=hue,
        style=style or hue,
        palette=color_map,
        errorbar=err,
        legend="brief",
        **line_kwargs,
    )

    # xmin, xmax = ax.get_xlim()
    # x_pad = 0.01 * (xmax - xmin)  # padding
    # ax.set_xlim(left=xmin - x_pad)
    ax.set_xticks(sorted(grouped[x].unique()))

    base = 10
    if log_y:
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=ymin / 1.5, top=ymax * 1.2)
        min_exp = int(np.floor(np.log10(ymin)))
        max_exp = int(np.ceil(np.log10(ymax)))
        ticks = [base ** i for i in range(min_exp, max_exp + 1)]
        
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=base, subs=[1, 2, 3, 5, 7], numticks=10))

        # Explicitly format both major and minor ticks
        ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%d'))

        ymin, ymax = ax.get_ylim()
        

    # ---------- 3. point annotations ----------
    if annotate is not None:
        for _, row in grouped.iterrows():
            ax.annotate(
                f"{int(round(row[annotate])):,}"+annotation_sfx,
                (row[x], row[y]),
                xytext=(-6, 0),  # shift
                textcoords='offset points',
                ha="right", va="center", fontsize="small",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.99),
            )

    ax.set_ylabel(y.replace("_", " "))
    ax.set_xlabel(x.replace("_", " "))

    # custom_handles = []

    # # Add one entry per hue category (with default seaborn color cycle)
    # palette = sns.color_palette()
    # unique_hues = grouped[hue].unique()
    # for idx, val in enumerate(unique_hues):
    #     color = palette[idx % len(palette)]
    #     custom_handles.append(Patch(facecolor=color, label=f"{hue}: {val}"))

    # ---------- 4. secondary y-axis ----------
    sec_kwargs = dict(marker="D", s=55, alpha=0.9)
    sec_kwargs.update(secondary_line_kws or {})


    if y2 is not None:
        ax2 = ax.twinx()
        for key, sub in grouped.groupby(hue):
            ax2.scatter(
                sub[x], sub[y2],
                label=f"{y2}",
                color=color_map[key],
                **sec_kwargs
            )

        ax2.set_ylabel(y2.replace("_", " "))
        ax2.grid(False)
        custom_handles.append(Line2D(
            [], [], linestyle='None', marker='D', color='black',
            label=f"{y2.replace('_', ' ')}"
        ))

    # combine legends: primary (hue/style) + secondary lines
    ax.legend(
        handles=custom_handles,
        loc="center right",
        frameon=True,
        framealpha=1,
        fontsize="small",
    )

    # ---------- 5. write and clean up ----------
    pdf_path = Path(pdf_path).with_suffix(".pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()

    return grouped


def plot_runtime_vs_storage(
    df_plot: pd.DataFrame,
    file_path: str,
    *,
    x: str = "Runtime_[ms]",
    y: str = "Total_Storage_Size_[GB]",
    highlights: Optional[List[Tuple[int, str]]] = None,
    base_colors: Optional[List[str]] = None,
    light: float = 0.3,
    dark: float = 0.9,
    figsize: Tuple[float, float] = (3.4, 2.5),
    legend_fontsize: int = 7,
) -> None:
    """Scatter & optional guide line with distinct legend placement."""

    if "(b,d)" in df_plot.columns:
        df_plot[["b", "d"]] = df_plot["(b,d)"].str.split(',', expand=True).astype(int)
    df_plot["(b,d)"] = df_plot["b"].astype(str) + "," + df_plot["d"].astype(str)
    pal = _make_shaded_palette(df_plot, base_colors, light, dark)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.78)  # more margin for outside legend

    # highlight guide lines
    if highlights:
        for b_sel, q_sel in highlights:
            g = df_plot.query("b == @b_sel and QueryDescr == @q_sel")
            
            if g.empty:
                continue
            means = g.sort_values(["b",y]).groupby("(b,d)")[[x, y]].mean()
            ax.plot(
                means[x], means[y], color=base_colors[3], linewidth=1.2,
                marker=None,
                zorder=-1,
            )
            ax.plot(
                means[x], means[y], color=base_colors[3], linewidth=0,
                marker=".", markersize=3,
                zorder=7,
            )

    sns.scatterplot(
        data=df_plot, x=x, y=y,
        hue="(b,d)", style="QueryDescr", palette=pal,
        s=32, legend="full", ax=ax, zorder=3,
    )

    ax.set(xscale="log", yscale="log")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))

    _add_split_legends(fig, ax, df_plot, legend_fontsize=legend_fontsize)

    fig.tight_layout()
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)


def plot_vol_vs_overhead(
    df_plot: pd.DataFrame,
    file_path: str,
    *,
    x: str = "Overhead",
    y: str = "Volume_[MB]",
    highlights: Optional[List[Tuple[int, str]]] = None,
    base_colors: Optional[List[str]] = None,
    light: float = 0.3,
    dark: float = 0.9,
    figsize: Tuple[float, float] = (3.4, 2.5),
    legend_fontsize: int = 7,
) -> None:
    """List count vs size with colour & marker legends."""

    if "(b,d)" in df_plot.columns:
        df_plot[["b", "d"]] = df_plot["(b,d)"].str.split(',', expand=True).astype(int)
    df_plot["(b,d)"] = df_plot["b"].astype(str) + "," + df_plot["d"].astype(str)

    palette = _make_shaded_palette(df_plot, base_colors, light, dark)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.78)

    sns.scatterplot(
        data=df_plot, x=x, y=y,
        hue="(b,d)", style="QueryDescr", palette=palette,
        s=32, legend="full", ax=ax,
    )

    if highlights:
        for b_sel, q_sel in highlights:
            g = df_plot.query("b == @b_sel and QueryDescr == @q_sel")
            if g.empty:
                continue
            means = g.sort_values(["b",y]).groupby("(b,d)")[[x, y]].mean()
            ax.plot(
                means[x], means[y], color=base_colors[3], linewidth=1.2,
                marker=None,
                zorder=-1,
            )
            ax.plot(
                means[x], means[y], color=base_colors[3], linewidth=0,
                marker=".", markersize=3,
                zorder=7,
            )

    ax.set(xscale="log", yscale="log")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))

    _add_split_legends(fig, ax, df_plot, legend_fontsize=legend_fontsize)

    fig.tight_layout()
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_vs_result_with_bars(
    df: pd.DataFrame,
    group_by: List[str],
    x_col: str = "result_cardinality",
    y_col: str = "runtime",
    agg_dict: Dict[str, Union[str, List[str]]] | None = None,
    color_by: List[str] | None = None,
    style_by: Optional[str] = None,
    extra_runtime: Optional[Dict[str, float]] = None,
    yscale: str = "linear",
    bar_color: str = "gray",
    bar_alpha: float = 0.3,
    size_by: str = "input_cardinality",
    min_size: float = 50,
    max_size: float = 500,
    palette: Union[str, Dict] = base_colors,
    fig_size: tuple[float, float] = (3.4, 3),
    path: str = "runtime_vs_result_bars.pdf",
):
    # -------- defaults --------
    if agg_dict is None:
        agg_dict = {
            x_col: ["mean", "std"],
            y_col: ["mean", "std"],
            size_by: "mean",
        }
    if color_by is None:
        color_by = ["Team_Count", "D"]

    # -------- aggregation --------
    agg = df.groupby(group_by).agg(agg_dict)
    agg.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in agg]
    agg = agg.reset_index()

    # Hue & style keys
    agg["_hue"] = (
        agg[color_by[0]].astype(str)
        if len(color_by) == 1
        else agg[color_by].astype(str).agg(",".join, axis=1)
    )
    markers = None
    if style_by:
        agg["_style"] = agg[style_by].astype(str)
        filled = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H"]
        markers = {s: filled[i % len(filled)] for i, s in enumerate(agg["_style"].unique())}

    x_mean = f"{x_col}_mean"
    y_mean = f"{y_col}_mean"
    size_col = f"{size_by}_mean" if f"{size_by}_mean" in agg else size_by

    # -------- figure --------
    fig, ax = plt.subplots(figsize=fig_size)
    fig.subplots_adjust(right=0.78)  # reserve space for legend

    # -------- optional line for extra_runtime --------
    if extra_runtime:
        x_line, y_line = [], []
        for hue in agg["_hue"].unique():
            if hue in extra_runtime:
                x_val = float(agg.loc[agg["_hue"] == hue, x_mean].mean())
                x_line.append(x_val)
                y_line.append(extra_runtime[hue])
        if x_line:
            x_sorted, y_sorted = zip(*sorted(zip(x_line, y_line)))
            ax.plot(x_sorted, y_sorted, color=bar_color, alpha=bar_alpha,
                    marker="o", linestyle="--", linewidth=1.2, zorder=10)

            # Add legend for extra runtime
            extra_line = mlines.Line2D([], [], color=bar_color, 
                                       linestyle="--", marker="o", alpha=bar_alpha)
            fig.legend(handles=[extra_line],
                       loc="upper right",
                       title="VA-Scan",
                       bbox_to_anchor=(1.0, 0.32), frameon=False)


    # -------- scatter --------
    sns.scatterplot(
        data=agg,
        x=x_mean,
        y=y_mean,
        hue="_hue",
        style="_style" if style_by else None,
        markers=markers,
        size=size_col,
        sizes=(min_size, max_size),
        palette=palette,
        edgecolor="w",
        linewidth=0.5,
        legend=False,
        ax=ax,
    )

    # -------- error bars --------
    if f"{x_col}_std" in agg.columns and f"{y_col}_std" in agg.columns:
        ax.errorbar(agg[x_mean], agg[y_mean],
                     xerr=agg[f"{x_col}_std"], yerr=agg[f"{y_col}_std"],
                     fmt="none", alpha=0.8)

    # -------- legends --------
    unique_hues = agg["_hue"].unique()
    hue_handles = [
        mlines.Line2D([], [], color=palette[i % len(palette)],
                      marker="o", linestyle="", markersize=6, label=h)
        for i, h in enumerate(unique_hues)
    ]
    fig.legend(hue_handles, unique_hues,
               title=",".join(color_by), loc="upper right", bbox_to_anchor=(1., 0.95), frameon=False)

    if style_by:
        style_handles = [
            mlines.Line2D([], [], color="grey", marker=markers[s], linestyle="", markersize=6, label=s)
            for s in agg["_style"].unique()
        ]
        ax.legend(style_handles, [h.get_label() for h in style_handles],
                  title=style_by.replace("_", " ").title(), 
                  loc="upper left",
                  bbox_to_anchor=(0.0, 0.9),
                  frameon=False)

    # -------- axis scaling --------
    ax.set_xscale("log")
    if yscale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=True))
        _force_minor_log_labels(ax, numticks=3)
        # ax.tick_params(axis="y", which="minor", labelsize=8, labelleft=True)
        ax.grid(True, which="both", axis="y", linestyle=":", linewidth=0.5)

    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(y_col.replace("_", " "))

    fig.tight_layout(rect=[0, 0, 0.89, 1])  # pack inside left area
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)

    return fig, ax


def dual_axis_curve_plot(
    df_primary, output_path, x_range=(0, 1), y_margin=0.05, df_secondary=None,
    figsize=(3.4, 2), x_col='Relative Size', y_col='Time per Tuple [ns]', hue_col='b', style_col='b'):
    """
    Plots a multi-line (styled) curve plot with optional dual y-axes:
    - df_primary is plotted on the primary y-axis
    - df_secondary (if provided) on the secondary y-axis

    Parameters:
    - df_primary: DataFrame for primary y-axis
    - df_secondary: DataFrame for secondary y-axis (optional)
    - output_path: path to save the plot image
    - x_range: tuple for x-axis domain
    - y_margin: float, margin ratio for y-axis limit extensions
    - figsize: size of the figure (in inches)
    - x_col, y_col: names of x and y value columns
    - hue_col, style_col: columns to use for color and line style
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # Primary axis plot
    sns.lineplot(
        data=df_primary, x=x_col, y=y_col, hue=hue_col, palette=base_colors, style=style_col,
        markers=True, dashes=True, ax=ax1
    )
    ax1.set_ylabel(y_col.replace("_"," "))
    ax1.tick_params(axis='y')
    y1_max = df_primary[y_col].max()
    ax1.set_ylim(0, y1_max * (1 + y_margin))

    # Secondary axis plot if df_secondary is provided
    if df_secondary is not None:
        ax2 = ax1.twinx()
        sns.lineplot(
            data=df_secondary, x=x_col, y=y_col, hue=hue_col, style=style_col,
            markers=True, dashes=True, ax=ax2
        )
        ax2.set_ylabel("Secondary Axis")
        ax2.tick_params(axis='y')
        y2_max = df_secondary[y_col].max()
        ax2.set_ylim(0, y2_max * (1 + y_margin))
    else:
        ax2 = None

    # Adjust x-axis with margin
    x_min, x_max = x_range
    x_span = x_max - x_min
    adjusted_x_min = x_min - x_span * y_margin
    adjusted_x_max = x_max + x_span * y_margin
    ax1.set_xlim(adjusted_x_min, adjusted_x_max)

    # Set x ticks and labels
    all_x = pd.concat([df_primary[x_col]] + ([df_secondary[x_col]] if df_secondary is not None else []))
    x_ticks = sorted(all_x.unique())
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([str(x) for x in x_ticks], rotation=25, ha='right')
    ax1.set_xlabel(x_col.replace("_"," "))

    # Consolidate legends from both axes
    handles, labels = ax1.get_legend_handles_labels()
    if ax2:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    ax1.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()




########################### Functions to create specific figures for the paper
def figure_1(target_path = plot_folder / "figure_1_grid_index.pdf"):

    rng = np.random.default_rng(42)
    n = 1000
    centroids = [[0,0],[6,4]]
    scales = [[1,1],[0.2,0.1]]
    sample_data = list()
    for cen, scale in zip(centroids,scales):
        sample_data.append(rng.normal(loc=cen, scale=scale, size=(n//2, 2)))
    sample_data = np.vstack(sample_data)

    # Run and plot
    plot_2d_distribution(sample_data, target_path=target_path,
                         n_quantiles=10, highlight_rectangle=[[1,0],[3,2]])



def figure_2_and_3(source_path = data_folder / "plan_optimization_experiment.parquet",
                   target_path = plot_folder):
    df = pd.read_parquet(source_path)

    max_runtime = df["Runtime_[ms]"].max()
    min_runtime = df["Runtime_[ms]"].min()

    backends = ["dram", "liburing"]
    datasets = ["uniform", "real"]

    attr = ["b","Query", "s","Strategy","Runtime_[ms]"]
    for be in backends:
        for ds in datasets:
            data = df.query(f"Backend == \'{be}\' and Dataset == \'{ds}\'")
            data = data[attr].groupby(["b", "s","Strategy","Query"]).mean()
            
            plot_grouped_scatter(data, ymin=min_runtime*0.9, ymax=max_runtime*1.2,
                                 path=target_path, exp_suffix=f"figure_2_and_3_{be}-{ds}",
                                 show_means="expand_some")


def figure_4(source_path = data_folder / "query_placement_experiment.parquet",
             target_path = plot_folder / "figure_4_varying_position.pdf"):
    results = pd.read_parquet(source_path)

    plot_query_placement(results, path=target_path)


def figure_5(source_path = data_folder / "volume_optimal_experiment.parquet",
             target_path = plot_folder / "figure_5_volume_optimal_runtime.pdf"):
    result = pd.read_parquet(source_path)
    # result["Total_List_Count_[K]"] = result["Total_List_Count"]/1000

    plot_optimal_runtime(
        result,
        y          = "Runtime_[ms]",
        y2         = None,
        annotate   = None,
        group_cols = ["Team_Count", "Type", "d"],
        pdf_path   = target_path,
        figsize=(3.4, 1.5)
    )


def figure_6_and_7(source_path = data_folder / "composition_variation_experiment.parquet",
                   target_path = plot_folder):

    results = pd.read_parquet(source_path)

    attr1 = ["QueryDescr","b","d","Index","Runtime_[ms]","Total_Storage_Size_[GB]"]
    attr2 = ["QueryDescr","b","d","Index","Volume_[MB]","Overhead"]
    pareto_data = results[attr1].groupby(["QueryDescr","b","d","Index"]).mean()
    vol_vs_overhead_data = results[attr2].groupby(["QueryDescr","b","d","Index"]).mean()

    plot_vol_vs_overhead(vol_vs_overhead_data.reset_index(), target_path / "figure_6_vol_vs_overhead.pdf",
                         x = "Overhead", y = "Volume_[MB]",
                         highlights=[(5,"balanced_selective"), (10,"balanced_selective"), (10,"diverse")],
                         light=0.3,dark=0.9, base_colors = base_colors,
                         figsize=(3.4, 2.5))

    plot_runtime_vs_storage(pareto_data.reset_index(), target_path / "figure_7_pareto.pdf",
                            x = "Total_Storage_Size_[GB]", y = "Runtime_[ms]",
                            highlights=[(10,"balanced"), (10,"balanced_selective"), (10,"diverse")],
                            light=0.3, dark=0.9, base_colors = base_colors,
                            legend_fontsize=7, figsize=(3.4, 2.5))


def figure_8(source_path = data_folder / "dimensional_scaling_experiment.parquet",
             target_path = plot_folder / "figure_8_dimensional_scaling.pdf"):
    
    N = 1221147850
    dimension_counts = [1, 2, 4,   4+2, 4+4, 2+4+4,   4+4+4, 4+4+4+2, 4+4+4+4]

    def va_file_scan_time_ms(D, bit_per_value=4, bandwidth_GBs=14):
        return N*bit_per_value*D/bandwidth_GBs / 1e6



    results = pd.read_parquet(source_path)

    results["Selectivity"] = results["result_cardinality"] / N
    # results["Runtime_[ms]"] = results["runtime"] / 1e6
    results["Runtime_[s]"] = results["runtime"] / 1e9
    # ideal time for scanning a VA file, using a single PCIe 5.0 SSD. Does not contain intersection time
    extra = {str(d): va_file_scan_time_ms(d)/1000 for d in dimension_counts}

    fig, ax = plot_runtime_vs_result_with_bars(
        df=results,
        x_col= "Selectivity",
        y_col= "Runtime_[s]",
        group_by=["team_count","D","query"],
        agg_dict={
        "Selectivity":["mean"],
        "Runtime_[s]":["mean"],
        "input_cardinality":"mean"
        },
        yscale="log",
        color_by=["D"],
        extra_runtime=extra,
        style_by="team_count",
        size_by=None,
        palette=base_colors,
        fig_size=(3.4, 2.1),
        path=target_path
    )


def figure_9(source_path = data_folder / "table_scaling_experiment.parquet",
                 target_path = plot_folder / "figure_9_table_scaling.pdf"):
    results = pd.read_parquet(source_path)
    
    dual_axis_curve_plot(results, target_path,
                         y_col="Time per Tuple [ns]",
                         x_col="Relative Size", hue_col="b",style_col="b")

def create_all():
    """
    Create all figures for the paper.
    """
    figure_1()
    figure_2_and_3()
    figure_4()
    figure_5()
    figure_6_and_7()
    figure_8()
    figure_9()

if __name__ == "__main__":
    create_all()
    print("All figures created successfully.")
