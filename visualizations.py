# libraries
import numpy as np
import plotly.graph_objects as go


def viz_irregular_events(time_axis: np.ndarray, signal: np.ndarray, features: dict,
                         title: str = "Irregular Event Detection"):
    """
    Visualizes the results of the event-based segmentation for irregular movements.
    Plots the detrended signal, marked peaks, marked valleys, and the zero-crossing baseline.

    Args:
        time_axis (np.ndarray): Time values (x-axis).
        signal (np.ndarray): The signal used for detection (usually the DETRENDED signal).
        features (dict): Output dictionary from 'extract_irregular_movements_parameters'.
        title (str): Plot title.

    Returns:
        None
    """

    fig = go.Figure()

    # 1. Plot the Main Signal (Detrended)
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=signal,
        mode='lines',
        name='Signal (Detrended)',
        line=dict(color='black', width=1.5),
        opacity=0.7
    ))

    # 2. Plot Valid Peaks (Green Up-Triangles)
    peak_indices = features.get('valid_peaks_idx', [])
    if peak_indices:
        fig.add_trace(go.Scatter(
            x=time_axis[peak_indices],
            y=signal[peak_indices],
            mode='markers',
            name='Valid Peaks',
            marker=dict(symbol='triangle-up', size=12, color='#00CC96', line=dict(width=1, color='black')),
            hovertemplate='Time: %{x:.2f}s<br>Amp: %{y:.2f}<extra>Peak</extra>'
        ))

    # 3. Plot Valid Valleys (Red Down-Triangles)
    valley_indices = features.get('valid_valleys_idx', [])
    if valley_indices:
        fig.add_trace(go.Scatter(
            x=time_axis[valley_indices],
            y=signal[valley_indices],
            mode='markers',
            name='Valid Valleys',
            marker=dict(symbol='triangle-down', size=12, color='#EF553B', line=dict(width=1, color='black')),
            hovertemplate='Time: %{x:.2f}s<br>Amp: %{y:.2f}<extra>Valley</extra>'
        ))

    # 4. Zero-Crossing Baseline (Critical for verifying segmentation)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Zero-Crossing Baseline")

    # 5. Add Annotation for Extracted Metrics
    # (Optional: Display the calculated period/amp on the chart for quick QA)
    metrics_text = (
        f"<b>Reps:</b> {features.get('num_repetitions', 0):.1f} | "
        f"<b>Freq:</b> {features.get('repetition_freq', 0):.2f} Hz<br>"
        f"<b>Mean Amp:</b> {features.get('amplitude_mean', 0):.2f} | "
        f"<b>Mean Period:</b> {features.get('period_mean', 0):.2f}s"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text=metrics_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )

    # 6. Layout Styling
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5),
        xaxis_title="Time [s]",
        yaxis_title="Amplitude (Detrended)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    fig.show()
