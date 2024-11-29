import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import seaborn as sns
sns.set_theme()

# =============================================================================
#### PLOT TEST-SECTION
# =============================================================================

def get_colormap(predictions, global_settings):
        # Define your custom color list
        color_list = [
                'forestgreen',   # Green
                'darkorange',    # Orange
                'darkviolet',    # Purple
                'teal',          # Teal
                'saddlebrown',   # Brown
                'hotpink',       # Pink
                'gold',          # Gold
                'slategray',     # Gray
                'turquoise',     # Cyan
                'mediumpurple'   # Magenta
        ]

        colors = None
        # Create a cyclic colormap from the color list
        if global_settings['prediction_length'] != 1:
                color_map = mcolors.LinearSegmentedColormap.from_list('cyclic_custom', color_list*10, N=256)
                colors = color_map(np.linspace(0, 1, len(predictions)))

        return colors

def plot_graph(ax, dates, actual, predictions, y_err, global_settings, confidence_interval_alpha):

        colors = get_colormap(predictions, global_settings)

        # Plot the actual data
        ax.plot(dates, actual, label='Actual', color='black')

        if global_settings['prediction_length'] == 1:
                predictions = np.array(predictions).reshape(-1,)
                y_err = np.array(y_err).reshape(-1,)
                start_idx = 0 #+global_settings['seq_length']
                end_idx = len(predictions) #+  global_settings['seq_length']
                x_values = dates[start_idx:end_idx]
                if len(x_values) != len(predictions):
                                # Slice pred to match the length of x_values
                                predictions = predictions[:len(x_values)]
                                y_err = y_err[:len(x_values)]
                ax.fill_between(x_values, predictions - y_err, predictions + y_err, label ='95% confidence',
                                        alpha=confidence_interval_alpha, linewidth = 1, facecolor = (1,0.7,0,0.5), edgecolor = (1,0.7,0,0.7))
                ax.plot(x_values, predictions, label="Predicted", color='red')
        else:
                # Plot each prediction sequence with a different color
                for i, ((pred, err), color) in enumerate(zip(zip(predictions, y_err), colors)):
                        start_idx = i #+ global_settings['seq_length']
                        end_idx = i + global_settings['prediction_length'] #+ global_settings['seq_length']
                        # Use the datetime index for the x-axis
                        x_values = dates[start_idx:end_idx]
                        # Adjust the length of pred to match x_values if necessary
                        # if len(x_values) != len(pred):
                        #         # Slice pred to match the length of x_values
                        #         pred = pred[:len(x_values)]
                        #         err = err[:len(x_values)]

                        ax.fill_between(x_values, pred - err, pred + err, label ='95% confidence' if i == 0 else "",
                                        alpha=confidence_interval_alpha, linewidth = 1, facecolor = color, edgecolor = color)
                        ax.plot(x_values, pred, color=color,  linestyle='--', label='Predicted' if i == 0 else "")

        return ax

def generate_safe_zoom_ranges(start_idx, end_idx, zoom_levels):
    """
    Generate zoom ranges that stay within the specified start and end indices.

    Parameters:
    - start_idx: The starting index of the first sub-curve.
    - end_idx: The ending index of the last sub-curve.
    - zoom_levels: List of zoom window sizes (number of time steps).

    Returns:
    - List of tuples representing the zoom ranges.
    """
    zoom_ranges = []
    current_start = start_idx

    for zoom_window in zoom_levels:
        # Ensure the range does not exceed the end index
        if current_start + zoom_window > end_idx:
            break  # Stop if the next range would exceed the end boundary

        zoom_ranges.append((current_start, current_start + zoom_window))
        current_start += zoom_window  # Shift the start index for the next range

    # Adjust the last range to stay within bounds
    if zoom_ranges and zoom_ranges[-1][1] > end_idx:
        last_start, _ = zoom_ranges[-1]
        zoom_ranges[-1] = (last_start, end_idx)

    return zoom_ranges
