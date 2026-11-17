"""Six linked recovery panels with explicit dataset identities for point clicks.

This is display/selection code only. Posterior draws come from the notebook's
verified held-out quotes and pretrained estimator, never from these quantiles.
"""

import numpy as np


def recovery_figure(truth, quantiles, bounds, labels, *, color="#3274c8"):
    """Build an all-parameter Plotly figure, preserving raw recovery values."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    truth = np.asarray(truth, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if (truth.ndim != 2 or truth.shape[1] != 6 or truth.shape[0] == 0
            or quantiles.shape != (3, *truth.shape) or bounds.shape != (6, 2)
            or len(labels) != 6 or not all(np.isfinite(a).all() for a in (truth, quantiles, bounds))
            or np.any(np.diff(quantiles, axis=0) < 0) or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("six-target recovery requires finite, ordered quantiles and valid bounds")
    figure = make_subplots(rows=2, cols=3, subplot_titles=list(labels),
                           horizontal_spacing=.085, vertical_spacing=.20)
    count = len(truth)
    interval_cases = np.linspace(0, count - 1, min(32, count), dtype=int)
    show_interval = np.zeros(count, dtype=bool)
    show_interval[interval_cases] = True
    for parameter in range(6):
        row, column = divmod(parameter, 3)
        low = min(bounds[parameter, 0], truth[:, parameter].min(), quantiles[0, :, parameter].min())
        high = max(bounds[parameter, 1], truth[:, parameter].max(), quantiles[2, :, parameter].max())
        pad = .025 * (high - low)
        figure.add_trace(go.Scatter(
            x=[low, high], y=[low, high], mode="lines", hoverinfo="skip",
            line={"color": "#8a929e", "width": 1, "dash": "dash"}, showlegend=False,
        ), row=row + 1, col=column + 1)
        figure.add_trace(go.Scatter(
            x=truth[:, parameter].tolist(), y=quantiles[1, :, parameter].tolist(), mode="markers",
            customdata=[[case, parameter, quantiles[0, case, parameter], quantiles[2, case, parameter]]
                        for case in range(count)],
            ids=[str(case) for case in range(count)],
            marker={"color": color, "size": 5, "opacity": .65},
            selected={"marker": {"color": "#f08a24", "size": 10, "opacity": 1}},
            unselected={"marker": {"opacity": .4}},
            error_y={"type": "data", "symmetric": False,
                     "array": np.where(show_interval, quantiles[2, :, parameter] - quantiles[1, :, parameter], 0).tolist(),
                     "arrayminus": np.where(show_interval, quantiles[1, :, parameter] - quantiles[0, :, parameter], 0).tolist(),
                     "color": "rgba(50,116,200,.25)", "thickness": .7, "width": 0},
            hovertemplate=("Data set %{customdata[0]}<br>Truth: %{x:.4g}<br>Median: %{y:.4g}"
                           "<br>90% interval: [%{customdata[2]:.4g}, %{customdata[3]:.4g}]<extra></extra>"),
            showlegend=False, name=str(labels[parameter]),
        ), row=row + 1, col=column + 1)
        figure.update_xaxes(title_text="True value", range=[low - pad, high + pad],
                            row=row + 1, col=column + 1, nticks=5)
        figure.update_yaxes(title_text="Posterior median", range=[low - pad, high + pad],
                            row=row + 1, col=column + 1, nticks=5)
    figure.update_layout(
        height=640, margin={"l": 54, "r": 20, "t": 48, "b": 42},
        clickmode="event+select", dragmode=False, hovermode="closest", template="plotly_white",
        font={"family": "sans-serif", "size": 11},
    )
    return figure


def selected_case(points, number_of_cases, figure=None):
    """Read the original row ID, independently of panel/trace/point ordering.

    A plain click selects one data set; if shift-click selected several, the
    last returned data point is used. An empty selection shows data set zero.
    """
    if not isinstance(number_of_cases, int) or number_of_cases <= 0:
        raise ValueError("positive recovery case count required")
    if not points:
        return 0
    valid = []
    for point in points:
        data = point.get("customdata") if isinstance(point, dict) else None
        # marimo 0.25.1 retains trace/point indices but strips customdata from
        # frontend events. Recover the explicit row ID from the original figure.
        if data is None and isinstance(point, dict) and figure is not None:
            curve, index = point.get("curveNumber"), point.get("pointIndex", point.get("pointNumber"))
            if (isinstance(curve, bool) or not isinstance(curve, int)
                    or isinstance(index, bool) or not isinstance(index, int)
                    or not 0 <= curve < len(figure.data)):
                raise ValueError("invalid recovery trace/point selection")
            identities = figure.data[curve].customdata
            if identities is not None:
                if not 0 <= index < len(identities):
                    raise ValueError("invalid recovery point selection")
                data = list(identities[index])
        # A diagonal/reference trace has no dataset identity and is not selectable.
        if data is None:
            continue
        if (not isinstance(data, (tuple, list)) or len(data) != 4
                or isinstance(data[0], bool) or not isinstance(data[0], int)
                or isinstance(data[1], bool) or not isinstance(data[1], int)
                or not 0 <= data[0] < number_of_cases or not 0 <= data[1] < 6):
            raise ValueError("invalid recovery selection identity")
        valid.append(data[0])
    return valid[-1] if valid else 0
