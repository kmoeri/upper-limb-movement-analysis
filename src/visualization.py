# src/visualization.py
import json
# libraries
import os
import shap
import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, r2_score, roc_curve, auc, mean_absolute_error

# modules
from src.config import config, project_path


class Visualizer:
    def __init__(self) -> None:
        # feature validation directory
        self.features_path = os.path.join(project_path, 'data', '04_features')
        os.makedirs(self.features_path, exist_ok=True)
        # main results directory (parent)
        self.results_path = os.path.join(project_path, 'data', '05_results')
        os.makedirs(self.results_path, exist_ok=True)
        # temporal consistency results directory (child)
        self.temp_consistency_res_path = os.path.join(self.results_path, '02_temporal_consistency')
        os.makedirs(self.temp_consistency_res_path, exist_ok=True)
        # feature extraction results directory (child)
        self.features_res_path = os.path.join(self.results_path, '03_feature_extraction')
        os.makedirs(self.features_res_path, exist_ok=True)
        # classification results directory (child)
        self.classification_res_path = os.path.join(self.results_path, '04_classification')
        os.makedirs(self.classification_res_path, exist_ok=True)
        # kinematics quality check results directory (child)
        self.kinematics_quality_res_path = os.path.join(self.results_path, '05_kinematics_quality')
        os.makedirs(self.kinematics_quality_res_path, exist_ok=True)
        # regression results directory (child)
        self.regression_res_path = os.path.join(self.results_path, '06_regression')
        os.makedirs(self.regression_res_path, exist_ok=True)

        # video frame rate
        self.fps = config['camera_param']['fps']

    # ============================================================================= #
    #                            2) TEMPORAL CONSISTENCY                            #
    # ============================================================================= #
    def vis_hand_consistency_raincloud(self, df: pd.DataFrame, states_to_plot: list[str],
                                       title: str = '', x_label: str = '', y_label: str = '') -> None:
        """
        Generates and saves a raincloud plot (combination of a half-violin, boxplot, and stripplot)
        to visualize the distribution of temporal consistency (CoV) across different body segments.

        Args:
            df (pd.DataFrame): DataFrame containing the CoV data for all segments.
            states_to_plot (list[str]): List of states to plot.
            title (str, optional): Title of the plot and base name for the saved file. Defaults to ''.
            x_label (str, optional): Label for the x-axis. Defaults to ''.
            y_label (str, optional): Label for the y-axis. Defaults to ''.

        Returns:
            None
        """

        sns.set_theme(style="whitegrid")
        link_palette = sns.color_palette("Set2", len(states_to_plot))

        if not states_to_plot:
            print("Error: states_to_plot list is empty.")
            return

        # map columns to colors
        column_colors = dict(zip(states_to_plot, link_palette))

        fig, ax = plt.subplots(figsize=(2.5 * len(states_to_plot), 6))

        # loop through each state to plot one by one
        for i, state_name in enumerate(states_to_plot):

            # extract data for the current state
            subset = df[df['State'] == state_name]['CoV'].dropna().values
            if len(subset) == 0:
                continue

            color = column_colors[state_name]

            # violin plot representing the density
            violin_plot = sns.violinplot(
                x=[i] * len(subset), y=subset,
                inner=None, cut=0, bw_method=0.2, linewidth=0,
                color=color, ax=ax, alpha=0.6,
                zorder=1
            )

            # mask left half of violin plot
            try:
                violin = ax.collections[-1]
                path = violin.get_paths()[0]
                vertices = path.vertices
                # find the mean X position of the vertices for the center line
                mean_x = np.mean(vertices[:, 0])
                # set all X-vertices to be AT LEAST the mean X, effectively clipping the left side
                vertices[:, 0] = np.maximum(vertices[:, 0], mean_x)

            except IndexError:
                pass

            # boxplot
            sns.boxplot(
                x=[i] * len(subset),
                y=subset,
                whis=1.5,
                width=0.1,
                showcaps=True,
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'zorder': 3},
                whiskerprops={'color': 'black', 'zorder': 3},
                flierprops={'marker': ''},
                medianprops={'color': 'black', 'zorder': 3},
                ax=ax
            )

            # stripplot
            # jitter = np.random.normal(loc=-0.2, scale=0.05, size=len(subset))
            jitter = np.random.uniform(low=-0.3, high=-0.1, size=len(subset))
            ax.scatter(
                np.full_like(subset, i) + jitter,
                subset,
                color=color, alpha=0.3, s=30, edgecolor='none',
                zorder=2
            )

        ax.set_xticks(range(len(states_to_plot)))
        ax.set_xticklabels([name.replace('-', ' ').title() for name in states_to_plot])
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        # y-axis limits and steps
        max_val = df[df['State'].isin(states_to_plot)]['CoV'].max()
        y_max = np.ceil(max_val * 1.15) if not np.isnan(max_val) else 10.0
        ax.set_ylim(0.0, y_max)

        fig.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()

        suffix = '.png'
        f_name: str = title.replace(' ', '_') + suffix
        f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600)

        plt.close()

    def vis_arm_consistency_raincloud(self, df: pd.DataFrame, states_to_plot: list[str],
                                      title: str = '', x_label: str = '', y_label: str = '') -> None:
        """
        Generates and saves a raincloud plot (combination of a half-violin, boxplot, and stripplot)
        to visualize the distribution of temporal consistency (CoV) across different body segments.

        Args:
            df (pd.DataFrame): DataFrame containing the CoV data for all segments.
            states_to_plot (list[str]): List of states to plot.
            title (str, optional): Title of the plot and base name for the saved file. Defaults to ''.
            x_label (str, optional): Label for the x-axis. Defaults to ''.
            y_label (str, optional): Label for the y-axis. Defaults to ''.

        Returns:
            None
        """

        sns.set_theme(style="whitegrid")
        link_palette = sns.color_palette("Set1", 5)

        if not columns_to_plot:
            print("Error: columns_to_plot list is empty.")
            return

        # prepare Data
        plot_columns = columns_to_plot[:min(len(columns_to_plot), 5)]

        # map columns to colors
        column_colors = dict(zip(plot_columns, link_palette))

        fig, ax = plt.subplots(figsize=(1.8 * len(plot_columns), 6))

        # loop through each column to plot one by one
        for i, link_name in enumerate(plot_columns):

            # extract data for the current column
            subset = link_df[link_name].dropna()
            color = column_colors[link_name]

            # violin plot representing the density
            violin_plot = sns.violinplot(
                x=[i] * len(subset), y=subset,
                inner=None, cut=0, bw_method=0.2, linewidth=0,
                color=color, ax=ax, alpha=0.6,
                zorder=1
            )

            # mask left half of violin plot
            try:
                violin = ax.collections[-1]
                path = violin.get_paths()[0]
                vertices = path.vertices
                # find the mean X position of the vertices for the center line
                mean_x = np.mean(vertices[:, 0])
                # set all X-vertices to be AT LEAST the mean X, effectively clipping the left side
                vertices[:, 0] = np.maximum(vertices[:, 0], mean_x)

            except IndexError:
                # handle case where violin plot may fail if data is too sparse
                print(f"Warning: Could not mask violin for {link_name}.")
                pass

            # boxplot
            sns.boxplot(
                x=[i] * len(subset),
                y=subset,
                whis=1.5,
                width=0.1,
                showcaps=True,
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'zorder': 3},
                whiskerprops={'color': 'black', 'zorder': 3},
                flierprops={'marker': ''},
                medianprops={'color': 'black', 'zorder': 3},
                ax=ax
            )

            # stripplot
            # jitter = np.random.normal(loc=-0.2, scale=0.05, size=len(subset))
            jitter = np.random.uniform(low=-0.3, high=-0.1, size=len(subset))
            ax.scatter(
                np.full_like(subset, i) + jitter,
                subset,
                color=color, alpha=0.3, s=30, edgecolor='none',
                zorder=2
            )

        ax.set_xticks(range(len(plot_columns)))
        ax.set_xticklabels([name.replace('-', ' ').title() for name in plot_columns])
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        # y-axis limits and steps
        y_min = 0.0
        y_max = 100.0
        y_step = 10.0

        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + y_step, y_step))

        fig.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()

        suffix = '.png'
        f_name: str = title.replace(' ', '_') + suffix
        f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600)

        plt.close()

    # def viz_comparison_subplots(self, segment_df, segment_order: list[str],
    #                             segment_col: str = 'Finger', body_part: str = 'Hand') -> None:
    #     """
    #     Generates and saves a two-panel figure containing a grouped boxplot comparing the healthy
    #     and affected sides, and a raincloud plot presenting the distribution of the affected side.
    #
    #     Args:
    #         segment_df (pd.DataFrame): DataFrame containing the CoV, Side, and segment data.
    #         segment_order (list[str]): Ordered list of segments to plot on the x-axis.
    #         segment_col (str, optional): Name of the column representing the body segments. Defaults to 'Finger'.
    #         body_part (str, optional): Name of the body part being analyzed, used for titles and filenames. Defaults to 'Hand'.
    #
    #     Returns:
    #         None
    #     """
    #     fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1]})
    #
    #     # subplot 1: grouped boxplot
    #     sns.boxplot(data=segment_df,
    #                 x=segment_col,
    #                 y='CoV',
    #                 hue='Side',
    #                 ax=axes[0],
    #                 palette='Set2',
    #                 order=segment_order)
    #
    #     axes[0].set_title('Comparison: Healthy vs. Affected Side')
    #     axes[0].set_ylabel('Mean CoV (%)')
    #     axes[0].set_xlabel(segment_col.replace('_', ' '))
    #
    #     # subplot 2: raincloud of affected side - violin/strip combination
    #     df_aff = segment_df[segment_df['Side'] == 'Affected']
    #     sns.violinplot(
    #         data=df_aff, x=segment_col, y='CoV',
    #         ax=axes[1], inner=None, color=".8", order=segment_order
    #     )
    #     sns.stripplot(
    #         data=df_aff, x=segment_col, y='CoV',
    #         ax=axes[1], alpha=0.4, order=segment_order
    #     )
    #     axes[1].set_title(f'Distribution of Affected {body_part} Stability')
    #     axes[1].set_ylabel('Mean CoV (%)')
    #     axes[1].set_xlabel(segment_col.replace('_', ' '))
    #
    #     plt.tight_layout()
    #
    #     suffix = '.png'
    #     f_name: str = f'Distribution_of_Affected_Side_Stability_{body_part}{suffix}'
    #     f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
    #     if not os.path.exists(f_path):
    #         plt.savefig(f_path, format=suffix[1:], dpi=600)
    #
    #     plt.close()

    def viz_comparison_boxplot(self, segment_df: pd.DataFrame, body_part: str = 'Hand') -> None:
        """
        Generates and saves a 4-way grouped boxplot comparing the temporal consistency (CoV)
        across Active/Passive roles and Healthy/Affected conditions.

        Args:
            segment_df (pd.DataFrame): DataFrame containing CoV, State, and segment data.
            body_part (str, optional): Name of the body part being analyzed, used for titles and filenames. Defaults to 'Hand'.

        Returns:
            None
        """

        sns.set_style('white')

        plt.figure(figsize=(10, 6))

        # custom palette for the four states
        custom_palette = {
            'Healthy (Active)': '#1f77b4',  # dark blue
            'Healthy (Passive)': '#aec7e8', # light blue
            'Affected (Active)': '#d62728', # dark red
            'Affected (Passive)': '#ff9896' # light red
        }

        # order of the legend and bars
        state_order = ['Healthy (Active)', 'Healthy (Passive)', 'Affected (Active)', 'Affected (Passive)']

        ax = sns.boxplot(data=segment_df,
                         x='State',
                         y='CoV',
                         hue='State',
                         palette=custom_palette,
                         order=state_order,
                         dodge=False)

        plt.title(f'Temporal Consistency: Active vs. Passive {body_part}', fontsize=14)
        plt.ylabel('Mean CoV (%)', fontsize=12)
        plt.xlabel('Hand State', fontsize=12)

        # legend not necessary
        if ax.legend_:
            ax.legend_.remove()

        # dynamic y-axis scaling
        max_cov: float = segment_df['CoV'].max()

        # set the ceiling 15% higher than the max value
        ceiling: float = max_cov * 1.15

        if ceiling <= 5.0:
            step = 0.5          # steps: 0.5%
        elif ceiling <= 15.0:
            step = 2.0          # steps: 2.0%
        else:
            step = 5.0          # steps: 5.0%

        # create tick marks
        ticks: np.ndarray = np.arange(0, np.ceil(ceiling) + step, step)

        # format labels
        labels: list = [f'{tick:.1f}' if step < 1 else f'{int(tick)}' for tick in ticks]

        # apply limits, ticks, and labels
        plt.ylim(0, ticks[-1])
        plt.yticks(ticks, labels=labels)

        # force horizontal grid
        ax.yaxis.grid(True, color='lightgrey', linestyle='-', linewidth=1.0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        plt.tight_layout()

        # save plot
        suffix = '.png'
        f_name: str = f'4-Way_Consistency_Boxplot_{body_part}{suffix}'
        f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600)

        plt.close()

    # def viz_finger_comparison_boxplot(self, segment_df: pd.DataFrame, segment_order: list[str],
    #                                   segment_col: str = 'Finger', body_part: str = 'Hand') -> None:
    #     """
    #     Generates and saves a 4-way grouped boxplot comparing the temporal consistency (CoV)
    #     across Active/Passive roles and Healthy/Affected conditions.
    #
    #     Args:
    #         segment_df (pd.DataFrame): DataFrame containing CoV, State, and segment data.
    #         segment_order (list[str]): Ordered list of segments to plot on the x-axis.
    #         segment_col (str, optional): Name of the column representing the body segments. Defaults to 'Finger'.
    #         body_part (str, optional): Name of the body part being analyzed, used for titles and filenames. Defaults to 'Hand'.
    #
    #     Returns:
    #         None
    #     """
    #
    #     sns.set_style('white')
    #
    #     plt.figure(figsize=(12, 6))
    #
    #     # custom palette for the four states
    #     custom_palette = {
    #         'Healthy (Active)': '#1f77b4',  # dark blue
    #         'Healthy (Passive)': '#aec7e8', # light blue
    #         'Affected (Active)': '#d62728', # dark red
    #         'Affected (Passive)': '#ff9896' # light red
    #     }
    #
    #     # order of the legend and bars
    #     hue_order = ['Healthy (Active)', 'Healthy (Passive)', 'Affected (Active)', 'Affected (Passive)']
    #
    #     ax = sns.boxplot(data=segment_df,
    #                      x=segment_col,
    #                      y='CoV',
    #                      hue='State',
    #                      palette=custom_palette,
    #                      hue_order=hue_order,
    #                      order=segment_order)
    #
    #     plt.title(f'Temporal Consistency: Active vs. Passive {body_part}', fontsize=14)
    #     plt.ylabel('Mean CoV (%)', fontsize=12)
    #     plt.xlabel(segment_col.replace('_', ' '), fontsize=12)
    #
    #     # legend placed outside the plot to prevent it from covering data
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=f'{body_part} State')
    #
    #     # dynamic y-axis scaling
    #     max_cov: float = segment_df['CoV'].max()
    #
    #     # set the ceiling 15% higher than the max value
    #     ceiling: float = max_cov * 1.15
    #
    #     if ceiling <= 5.0:
    #         step = 0.5          # steps: 0.5%
    #     elif ceiling <= 15.0:
    #         step = 2.0          # steps: 2.0%
    #     else:
    #         step = 5.0          # steps: 5.0%
    #
    #     # create tick marks
    #     ticks: np.ndarray = np.arange(0, np.ceil(ceiling) + step, step)
    #
    #     # format labels
    #     labels: list = [f'{tick:.1f}' if step < 1 else f'{int(tick)}' for tick in ticks]
    #
    #     # apply limits, ticks, and labels
    #     plt.ylim(0, ticks[-1])
    #     plt.yticks(ticks, labels=labels)
    #
    #     # force horizontal grid
    #     ax.yaxis.grid(True, color='lightgrey', linestyle='-', linewidth=1.0)
    #     ax.xaxis.grid(False)
    #     ax.set_axisbelow(True)
    #
    #     plt.tight_layout()
    #
    #     # save plot
    #     suffix = '.png'
    #     f_name: str = f'4-Way_Consistency_Boxplot_{body_part}{suffix}'
    #     f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
    #     if not os.path.exists(f_path):
    #         plt.savefig(f_path, format=suffix[1:], dpi=600)
    #
    #     plt.close()

    # def viz_normality_qq_plots(self, paired_df: pd.DataFrame, segment_order: list[str],
    #                            segment_col: str = 'Finger', body_part: str = 'Hand') -> None:
    #     """
    #     Generates and saves a grid of Q-Q plots to visually assess the normality of paired
    #     differences (Healthy vs. Affected) for each body segment, including the Shapiro-Wilk p-value.
    #
    #     Args:
    #         paired_df (pd.DataFrame): DataFrame containing paired CoV data for Healthy and Affected conditions.
    #         segment_order (list[str]): Ordered list of segments to generate subplots for.
    #         segment_col (str, optional): Name of the column representing the body segments. Defaults to 'Finger'.
    #         body_part (str, optional): Name of the body part being analyzed, used for filenames. Defaults to 'Hand'.
    #
    #     Returns:
    #         None
    #     """
    #     # create a 3x2 grid
    #     fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    #
    #     # flatten the 2D array so we can use a single index 'i'
    #     axes_flat = axes.flatten()
    #
    #     for i, segment in enumerate(segment_order):
    #         # select the specific subplot for this finger
    #         ax = axes_flat[i]
    #
    #         segment_data = paired_df[paired_df[segment_col] == segment]
    #         differences = (segment_data['Healthy'] - segment_data['Affected']).dropna()
    #         print(f'N: {len(differences)}')
    #         if len(differences) >= 3:
    #             _, p_val = stats.shapiro(differences)
    #             norm_text = f"Shapiro p={p_val:.3f}"
    #             stats.probplot(differences, dist="norm", plot=ax)
    #         else:
    #             norm_text = "N too small"
    #             ax.text(0.5, 0.5, 'Insufficient Data', ha='center')
    #
    #         ax.set_title(f'Q-Q Plot: {segment}\n{norm_text}')
    #
    #         # formatting for a 2-column layout
    #         ax.set_ylabel('Sample Quantiles' if i % 2 == 0 else '')
    #         ax.set_xlabel('Theoretical Quantiles')
    #
    #     # hide the 6th empty subplot
    #     if len(segment_order) < len(axes_flat):
    #         axes_flat[-1].axis('off')
    #
    #     plt.tight_layout()
    #
    #     suffix = '.png'
    #     f_name: str = f'QQ_Plots_{body_part}{suffix}'
    #     f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
    #     if not os.path.exists(f_path):
    #         plt.savefig(f_path, format=suffix[1:], dpi=600)
    #
    #     plt.close()

    def viz_normality_qq_plots(self, paired_df: pd.DataFrame, body_part: str = 'Hand') -> None:
        """
        Generates and saves a grid of Q-Q plots to visually assess the normality of paired
        differences (Healthy vs. Affected) for each body segment, including the Shapiro-Wilk p-value.

        Args:
            paired_df (pd.DataFrame): DataFrame containing paired CoV data for Healthy and Affected conditions.
            body_part (str, optional): Name of the body part being analyzed, used for filenames. Defaults to 'Hand'.

        Returns:
            None
        """
        # create a 3x2 grid
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        roles: list[str] = ['Healthy', 'Affected']

        # aggreate the raw segments into 1 mean value per participant per condition/role
        agg_df: pd.DataFrame = paired_df.groupby(['Participant', 'Hand_Condition', 'Hand_Role'])['CoV'].mean().reset_index()

        for i, role in enumerate(roles):
            # select the specific subplot for this finger
            ax = axes[i]

            role_data = agg_df[agg_df['Hand_Role'] == role]
            pivot_data = role_data.pivot(index='Participant', columns='Hand_Condition', values='CoV').dropna()

            if not pivot_data.empty and len(pivot_data) >= 3:
                differences = pivot_data['Healthy'] - pivot_data['Affected']
                _, p_val = stats.shapiro(differences)
                norm_text = f"Shapiro p={p_val:.3f}"
                stats.probplot(differences, dist="norm", plot=ax)
            else:
                norm_text = "N too small"
                ax.text(0.5, 0.5, 'Insufficient Data', ha='center')

            ax.set_title(f'Q-Q Plot: {role} {body_part}\n{norm_text}')

            # formatting for a 2-column layout
            ax.set_ylabel('Sample Quantiles' if i == 0 else '')
            ax.set_xlabel('Theoretical Quantiles')

        plt.tight_layout()

        suffix = '.png'
        f_name: str = f'QQ_Plots_{body_part}{suffix}'
        f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600)

        plt.close()

    def viz_lmm_interaction_trajectories(self, segment_df: pd.DataFrame, body_part: str = 'Hand') -> None:
        """
        Creates and saves an interaction trajectory showing individual participant trajectories
        from Passive to Active states, split by Healthy vs. Affected conditions, overlaid
        with the group mean to visualize the LMM interaction effect.

        Args:
            segment_df (pd.DataFrame): DataFrame containing CoV, Participant, condition, and role data.
            body_part (str, optional): Name of the body part being analyzed, used for labels and filenames. Defaults to 'Hand'.

        Returns:
            None
        """
        # aggregate the data: get the mean CoV per participant for each state (ignoring specific fingers)
        agg_df = segment_df.groupby(['Participant', 'Hand_Condition', 'Hand_Role'])['CoV'].mean().reset_index()

        # treat 'Participant' as categorical variable
        agg_df['Participant'] = agg_df['Participant'].astype(str)

        # setup seaborn FaceGrid
        sns.set_theme(style="whitegrid")
        num_participants: int = agg_df['Participant'].nunique()
        gradient_cmap = sns.color_palette("viridis", n_colors=num_participants)
        face_grid = sns.FacetGrid(agg_df, col="Hand_Condition", height=6, aspect=0.85,
                                  col_order=['Healthy', 'Affected'], hue='Participant', palette=gradient_cmap)

        # plot individual participant lines
        face_grid.map_dataframe(sns.lineplot, x="Hand_Role", y="CoV", estimator=None,
                                alpha=0.7, linewidth=1.5, marker="o")

        # plot the group mean (fixed effect)
        # ensure global mean: prevent seaborn from drawing a mean line for each participant
        for ax, condition in zip(face_grid.axes.flat, ['Healthy', 'Affected']):
            # get all participants for this specific condition
            subset = agg_df[agg_df['Hand_Condition'] == condition]

            sns.lineplot(data=subset, x="Hand_Role", y="CoV", estimator='mean', errorbar=None, color="black",
                         linewidth=3, marker="D", markersize=8, zorder=10, legend=False, ax=ax)

        # formatting
        face_grid.set_axis_labels(f'Role (State of Movement)', 'Mean CoV (%)', fontsize=12)
        face_grid.set_titles(col_template=f'{{col_name}} {body_part}', size=14, weight='bold')

        # adjust y-axis to start at 0
        face_grid.set(ylim=(0, None))

        # add the legend outside the plot
        face_grid.add_legend(title="Participant ID", bbox_to_anchor=(1.02, 0.5), loc='center left')

        plt.subplots_adjust(top=0.88)
        face_grid.figure.suptitle('Impact of Movement: Individual Trajectories (Rest vs. Active)',
                                  fontsize=16, weight='bold')

        # save the plot
        suffix = '.png'
        f_name: str = f'LMM_Interaction_Trajectories_{body_part}{suffix}'
        f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600, bbox_inches='tight')

        plt.close()

    def viz_lmm_normality_and_homoscedasticity(self, model, body_part: str = 'Hand') -> None:
        """
        Generates and saves diagnostic plots (a Q-Q plot and a Tukey-Anscombe scatter plot)
        to verify the normality of residuals and homoscedasticity assumptions of the
        fitted Linear Mixed Model.

        Args:
            model: The fitted statsmodels Linear Mixed Model (LMM) object.
            body_part (str, optional): Name of the body part being analyzed, used for filenames. Defaults to 'Hand'.

        Returns:
            None
        """
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        residuals = model.resid
        fitted = model.fittedvalues

        # plot Q-Q Plot to test normality assumption
        sm.qqplot(residuals, line='45', fit=True, ax=ax[0], alpha=0.5,
                  markerfacecolor='#1f77b4', markeredgecolor='none')

        ax[0].set_title('Q-Q Plot of Residuals (Normality Check)', fontsize=14)

        # plot Tukey-Anscombe (homoscedasticity)
        sns.scatterplot(x=fitted, y=residuals, ax=ax[1], alpha=0.6, color="#1f77b4", edgecolor='none')
        ax[1].axhline(y=0, color='red', linestyle='--', lw=2)
        ax[1].set_xlabel('Fitted Values (Predicted CoV)', fontsize=12)
        ax[1].set_ylabel('Residuals (Error)', fontsize=12)
        ax[1].set_title('Tukey-Anscombe Plot (Homoscedasticity Check)', fontsize=14)

        plt.tight_layout()

        # save the plot
        suffix = '.png'
        f_name: str = f'LMM_Assumption_Tests_{body_part}{suffix}'
        f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600, bbox_inches='tight')

        plt.close()

    # ============================================================================= #
    #                            3) PARAMETER EXTRACTION                            #
    # ============================================================================= #
    def viz_finger_alternation(self, time_axis: list, signals: list, features: list,
                               p_id: str = '', visit_id: str = '', ex_id: str = ''):
        """
        Visualizes the timeseries of the FingerAlternation exercise.
        Plots the detrended signal with strict and near valleys to highlight the tapping capacity.

        Args:
            time_axis (list[np.ndarray]): Time values (x-axis).
            signals (list[np.ndarray]): List of tapping signals for all fingers.
            features (list[dict]): Output dictionary from 'extract_irregular_movements_parameters()'.
            visit_id (str): Visit ID. Defaults to ''.
            p_id (str): Participant identifier key. Defaults to ''.
            ex_id (str): Experiment identifier key. Defaults to ''.

        Returns:
            None
        """
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 12), sharex=True)
        alt_labels = ['Index', 'Middle', 'Ring', 'Pinky']

        line_handles = []
        line_labels = []

        # t_axis is shared for all subplots
        t_axis = time_axis[0] if isinstance(time_axis, list) else time_axis

        for i, (curr_ax, sig, feat) in enumerate(zip(axes, signals, features)):
            plot_sig = np.array(sig)
            label_name = alt_labels[i]

            # plot signal line
            line_obj, = curr_ax.plot(t_axis, plot_sig, color='#43a2ca', linewidth=2, zorder=3)

            # threshold lines and zones
            curr_ax.axhspan(0.0, 0.10, facecolor='#e0f3db', alpha=1.0, zorder=1)  # strict
            curr_ax.axhspan(0.10, 0.25, facecolor='#a8ddb5', alpha=1.0, zorder=1)  # near
            curr_ax.axhline(0.10, color='black', linestyle=':', linewidth=1.0, zorder=2)  # strict
            curr_ax.axhline(0.25, color='black', linestyle=':', linewidth=1.0, zorder=2)  # near

            # plot hysteresis valleys
            strict_idc = feat.get('hysteresis_strict', [])
            near_idc = feat.get('hysteresis_near', [])

            if len(strict_idc) > 0:
                curr_ax.scatter(t_axis[strict_idc], plot_sig[strict_idc],
                                marker='o', s=70, facecolor='#e0f3db', edgecolor='black', linewidth=1.5, zorder=4)
            if len(near_idc) > 0:
                curr_ax.scatter(t_axis[near_idc], plot_sig[near_idc],
                                marker='o', s=70, facecolor='#a8ddb5', edgecolor='black', linewidth=1.5, zorder=4)

            # y-axis formatting
            curr_ax.set_ylabel('Amplitude (Normalized)', fontsize=14)
            curr_ax.set_ylim(-0.0, 0.65)
            curr_ax.set_yticks(np.arange(0.0, 0.61, 0.1))

            # print finger name centered on top of the graph
            curr_ax.text(0.50, 0.97, label_name, transform=curr_ax.transAxes, verticalalignment='top',
                         horizontalalignment='center', fontsize=16, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3), zorder=5)

            # collect legend handles from first axis
            if i == 0:
                line_handles.append(line_obj)
                line_labels.append('Tapping Distance')
                custom_strict = Line2D([0], [0], marker='o', color='w', markerfacecolor='#e0f3db',
                                       markeredgecolor='black', markersize=8)
                custom_near = Line2D([0], [0], marker='o', color='w', markerfacecolor='#a8ddb5',
                                     markeredgecolor='black', markersize=8)
                line_handles.extend([custom_strict, custom_near])
                line_labels.extend(['Strict Tapping (< 0.10)', 'Near Tapping (< 0.25)'])

            curr_ax.grid(True, linestyle=':', alpha=0.7, zorder=0)
            curr_ax.spines['top'].set_visible(False)
            curr_ax.spines['right'].set_visible(False)

        axes[-1].set_xlabel('Time [s]', fontsize=14)

        # title and legend
        axes[0].set_title(f'{p_id}_{ex_id} - Event Detection', fontweight='bold', fontsize=16, pad=45)
        axes[0].legend(line_handles, line_labels, loc='lower center', bbox_to_anchor=(0.5, 1.05),
                       ncol=3, frameon=False, fontsize=12)

        plt.tight_layout()

        # save the plot
        suffix = '.png'
        f_name: str = f'{p_id}-{visit_id}_{ex_id}_event_detection{suffix}'
        f_path: str = os.path.join(self.features_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600, bbox_inches='tight')
        plt.close(fig)

    def viz_single_signal_exercises(self, time_axis, signal, features,
                                    p_id: str = '', visit_id: str = '', ex_id: str = ''):
        """
        Visualizes the timeseries of the repetitive binary exercises:
        - FingerTapping (index finger tapping)
        - HandOpening (hand opening and closing)
        - ProSup (pronation supination)
        Plots the detrended signal, marked peaks, marked valleys, and the zero-crossing baseline.

        Args:
            time_axis (np.ndarray): Time values (x-axis).
            signal (np.ndarray): The signal used for detection.
            features (dict): Output dictionary from 'extract_irregular_movements_parameters'.
            visit_id (str): Visit ID. Defaults to ''.
            p_id (str): Participant identifier key. Defaults to ''.
            ex_id (str): Experiment identifier key. Defaults to ''.

        Returns:
            None
        """

        def adjust_lightness(color_in, amount):
            c = mc.to_rgb(color_in)
            c = colorsys.rgb_to_hls(*c)
            color_out = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
            return color_out

        # handle lists with a single element
        t_axis = time_axis[0] if isinstance(time_axis, list) else time_axis
        plot_sig = np.array(signal[0] if isinstance(signal, list) else signal)
        feat = features[0] if isinstance(features, list) else features

        fig, curr_ax = plt.subplots(figsize=(16, 6))

        color = '#43a2ca'
        dark_color = adjust_lightness(color, amount=0.8)
        light_color = adjust_lightness(color, amount=1.6)

        if 'ProSup' in ex_id:
            plot_sig = plot_sig - np.mean(plot_sig)

        # plot the main signal
        line_obj, = curr_ax.plot(t_axis, plot_sig, color=color, linewidth=1.5, alpha=0.7, zorder=2)

        # plot valid peaks (dark up-triangle)
        peak_indices = feat.get('valid_peaks_idx', [])
        if len(peak_indices) > 0:
            curr_ax.scatter(t_axis[peak_indices], plot_sig[peak_indices],
                            marker='^', s=80, facecolor=dark_color, edgecolor='black', linewidth=0.8, zorder=3)

        # plot valid valleys (light down-triangle)
        valley_indices = feat.get('valid_valleys_idx', [])
        if len(valley_indices) > 0:
            curr_ax.scatter(t_axis[valley_indices], plot_sig[valley_indices],
                            marker='v', s=80, facecolor=light_color, edgecolor='black', linewidth=0.8, zorder=3)

        # y-axis formatting and labels based on exercise
        if 'ProSup' in ex_id:
            curr_ax.set_ylabel('Rotation Angle [°] (Centered)', fontsize=12)
            curr_ax.set_ylim(-150, 150)
            curr_ax.set_yticks(np.arange(-150, 151, 30))
        else:
            curr_ax.set_ylabel('Amplitude (Normalized)', fontsize=12)
            curr_ax.set_ylim(0.0, 1.0)
            curr_ax.set_yticks(np.arange(0.0, 1.01, 0.2))

        curr_ax.grid(True, linestyle=':', alpha=0.7, zorder=0)
        curr_ax.spines['top'].set_visible(False)
        curr_ax.spines['right'].set_visible(False)
        curr_ax.set_xlabel('Time [s]', fontsize=12)

        # legend
        custom_peak = Line2D([0], [0], marker='^', color='w', markerfacecolor=dark_color, markeredgecolor='black',
                             markersize=9)
        custom_valley = Line2D([0], [0], marker='v', color='w', markerfacecolor=light_color, markeredgecolor='black',
                               markersize=9)

        # legend layout
        curr_ax.set_title(f'{p_id}_{ex_id} - Event Detection', fontweight='bold', fontsize=16, pad=45)
        curr_ax.legend([line_obj, custom_peak, custom_valley], [ex_id.split('_')[0], 'Peaks', 'Valleys'],
                       loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False, fontsize=12)

        plt.tight_layout()

        # save the plot
        suffix = '.png'
        f_name: str = f'{p_id}-{visit_id}_{ex_id}_event_detection{suffix}'
        f_path: str = os.path.join(self.features_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600, bbox_inches='tight')
        plt.close(fig)

    def feature_zscore_heatmap(self, feature_file_path: str) -> None:

        print('\nGenerating Z-Score Feature Expression Heatmap ...')

        df: pd.DataFrame = pd.read_csv(feature_file_path)

        exercises = df['ex_name'].unique()
        meta_cols: list[str] = ['p_ID', 'visit_ID', 'ex_name', 'affected_side', 'side_focus', 'side_condition', 'cam_ID', 'AHA_Score']

        for ex_name in exercises:
            # isolate current exercise
            ex_df: pd.DataFrame = df[df['ex_name'] == ex_name].copy()

            # identify numeric features
            features = [col for col in ex_df.columns if col not in meta_cols and pd.api.types.is_numeric_dtype(ex_df[col])]

            # isolate the healthy dataset to define the baseline
            healthy_mask = ex_df['side_condition'].str.lower() == 'healthy'
            healthy_df = ex_df[healthy_mask]

            if healthy_df.empty:
                print(f'Warning: No healthy side condition found for {ex_name}.')
                continue

            # calculate baseline statistics (healthy mean and std)
            healthy_mean = healthy_df[features].mean()
            healthy_std = healthy_df[features].std()

            # prevent division by zero for zero-variance features
            healthy_std.replace(0, 1e-8, inplace=True)

            # apply z-score normalization
            z_df = ex_df.copy()
            z_df[features] = (ex_df[features] - healthy_mean) / healthy_std

            # combine p_ID and visit ID to handle longitudinal data cleanly
            z_df['participant_visit'] = z_df['p_ID'] + '_' + z_df['visit_ID'].astype(str)

            # split into healthy and affected for plotting
            h_z = z_df[z_df['side_condition'].str.lower() == 'healthy']
            a_z = z_df[z_df['side_condition'].str.lower() == 'affected']

            # pivot data: rows = features and columns = participants
            h_pivot = h_z.pivot_table(index='participant_visit', values=features).T
            a_pivot = a_z.pivot_table(index='participant_visit', values=features).T

            # visualization
            fig, axes = plt.subplots(ncols=3, figsize=(24, max(4, len(features) * 0.25)),
                                     gridspec_kw={'width_ratios': [1, 1, 0.03], 'wspace': 0.1},
                                     layout='constrained')

            # color map is caped to +/- 4 Standard Deviations
            vmin, vmax = -4.0, 4.0
            cmap = 'vlag'

            # healthy heatmap (axis[0])
            sns.heatmap(h_pivot, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax, annot=False, fmt='.1f', linewidths=0.5,
                        cbar=False)
            axes[0].set_title(f'Healthy Hand ({ex_name})', fontsize=18, pad=15)
            axes[0].set_xlabel('Participant / Visit', fontsize=16, labelpad=10)
            axes[0].set_ylabel('Features', fontsize=16, labelpad=10)
            axes[0].tick_params(labelsize=12)

            # format x-ticks to hide every 2nd label
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=12)
            for i, tick in enumerate(axes[0].xaxis.get_major_ticks()):
                if i % 2 != 0:
                    tick.label1.set_visible(False)      # hides text
                    tick.tick1line.set_visible(False)   # hides tick strokes

            # affected heatmap (axis[1])
            sns.heatmap(a_pivot, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax, annot=False, fmt='.1f', linewidths=0.5,
                        cbar=True, cbar_ax=axes[2],
                        cbar_kws={'label': 'Z-Score (Standard Deviations from Healthy Mean)'})

            axes[1].set_title(f'Affected Hand ({ex_name})', fontsize=18, pad=15)
            axes[1].set_xlabel('Participant / Visit', fontsize=16, labelpad=10)
            axes[1].set_ylabel('', fontsize=16)

            # manually clear y-axis ticks
            axes[1].set_yticks([])
            axes[1].tick_params(labelsize=12)

            # format x-ticks to hide every 2nd label
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=12)
            for i, tick in enumerate(axes[1].xaxis.get_major_ticks()):
                if i % 2 != 0:
                    tick.label1.set_visible(False)      # hides text
                    tick.tick1line.set_visible(False)   # hides tick strokes

            # format colorbar
            axes[2].tick_params(labelsize=12)
            axes[2].yaxis.label.set_size(16)
            axes[2].yaxis.labelpad = 10

            suffix = '.png'
            f_name: str = f'all_extracted_features_heatmap_{ex_name}' + suffix
            f_path: str = os.path.join(self.features_path, f_name)
            if not os.path.exists(f_path):
                plt.savefig(f_path, format=suffix[1:], dpi=600)

            plt.close()

    # ============================================================================= #
    #                            4) KINEMATICS QUALITY CHECK                        #
    # ============================================================================= #
    def viz_render_dashboard(self, df:pd.DataFrame, metrics_dict: dict, p_id: str, visit_id: str, ex_id: str,
                             side_focus: str, skip_frames: int = 2) -> None:
        """
        Renders a modular quality control dashboard and saves it as a video file (mp4).

        Args:
            df (pd.DataFrame): dataframe with movement data.
            metrics_dict (dict): Dictionary containing pose, hands, and metrics.
            p_id (str): Participant identifier key.
            visit_id (str): Visit ID.
            ex_id (str): Experiment identifier key.
            side_focus (str): 'R' or 'L' indicating the active hand.
            skip_frames (int): Number of frames to step by (1 = full fps). Defaults to 2.

        Returns:
            None
        """

        # helper to remove global translation by anchoring the wrist to (0,0,0)
        def _center_hand_on_wrist(hand_dict: dict, wrist_key: str) -> None:
            if wrist_key not in hand_dict:
                return

            # extract the wrist trajectory as numpy arrays
            wrist_x = np.array(hand_dict[wrist_key][0])
            wrist_y = np.array(hand_dict[wrist_key][1])
            wrist_z = np.array(hand_dict[wrist_key][2])

            # subtract the wrist position from every joint in the hand
            for joint, data in hand_dict.items():
                hand_dict[joint][0] = (np.array(data[0]) - wrist_x).tolist()
                hand_dict[joint][1] = (np.array(data[1]) - wrist_y).tolist()
                hand_dict[joint][2] = (np.array(data[2]) - wrist_z).tolist()

        # check whether video already exists
        f_name = f'{p_id}-{visit_id}_{ex_id}_QC.mp4'
        out_path = os.path.join(self.kinematics_quality_res_path, f_name)

        if os.path.exists(out_path):
            return

        # nested dictionaries for 3D plotter
        dashboard_data = {'left_hand': {}, 'right_hand': {}, 'pose': {}}

        # helper to extract a (3, frames) list of arrays for a specific joint
        def get_joint_data(joint_name):
            if f"{joint_name}_x" in df.columns:
                return [df[f"{joint_name}_x"].tolist(), df[f"{joint_name}_y"].tolist(), df[f"{joint_name}_z"].tolist()]
            return None

        # safe identifier function
        def get_side_id(joint_name):
            for char in joint_name:
                if char.isdigit():
                    return char
            return None

        # left hand dict
        for link in config['body_parts'].get('hands_link_lst', []):
            for joint in link:
                if get_side_id(joint) == '1' and joint not in dashboard_data['left_hand']:
                    data = get_joint_data(joint)
                    if data:
                        dashboard_data['left_hand'][joint] = data

        # right hand dict
        for link in config['body_parts'].get('hands_link_lst', []):
            for joint in link:
                if get_side_id(joint) == '2' and joint not in dashboard_data['right_hand']:
                    data = get_joint_data(joint)
                    if data:
                        dashboard_data['right_hand'][joint] = data

        # pose dict
        for link in config['body_parts'].get('pose_link_lst', []):
            for joint in link:
                if joint not in dashboard_data['pose']:
                    data = get_joint_data(joint)
                    if data:
                        dashboard_data['pose'][joint] = data

        _center_hand_on_wrist(dashboard_data['left_hand'], 'wrist1')
        _center_hand_on_wrist(dashboard_data['right_hand'], 'wrist2')

        num_metrics = len(metrics_dict)

        # ensure skip_frames is at least 1 (skip_frames == 1: use every frame, skip_frames == 2: use every 2nd frame)
        skip_frames = max(1, skip_frames)

        # extract number of frames from whichever hand exists
        if dashboard_data.get('right_hand'):
            sample_lm = list(dashboard_data['right_hand'].values())[0]
        elif dashboard_data.get('left_hand'):
            sample_lm = list(dashboard_data['left_hand'].values())[0]
        else:
            return  # no hand data to plot

        n_frames = len(sample_lm[0])
        frames_to_run = np.arange(0, n_frames, skip_frames)

        # set up the dynamic GridSpec figure
        # top row is for 3D plots; subsequent rows are for 1D metrics.
        total_rows = 1 + num_metrics
        fig = plt.figure(figsize=(16, 4 + (3 * num_metrics)))
        fig.suptitle('Kinematic Quality Control Dashboard', fontsize=16, fontweight='bold')
        gs = gridspec.GridSpec(total_rows, 3, figure=fig, hspace=0.2, wspace=0.2)

        # initialize 3D plots (top row): right hand (col 1), pose (col 2), left hand (col 3)
        ax_left = fig.add_subplot(gs[0, 0], projection='3d')
        ax_pose = fig.add_subplot(gs[0, 1], projection='3d')
        ax_right = fig.add_subplot(gs[0, 2], projection='3d')

        # colors for alternating tapping
        alt_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # helper to calculate static axis limits and setup 3D plots
        def _setup_3d_axis(ax, title, lm_dict, title_color='black', side_focused=None, current_ex=''):
            ax.set_title(title, fontsize=12, fontweight='bold', color=title_color)
            if not lm_dict:
                return None, {}

            # calculate global min/max for static bounding box
            all_x = [x for lm in lm_dict.values() for x in lm[0]]
            all_y = [y for lm in lm_dict.values() for y in lm[1]]
            all_z = [z for lm in lm_dict.values() for z in lm[2]]

            pad = 0.05
            ax.set_xlim([np.min(all_x) - pad, np.max(all_x) + pad])
            ax.set_ylim([np.min(all_y) - pad, np.max(all_y) + pad])
            ax.set_zlim([np.min(all_z) - pad, np.max(all_z) + pad])

            # invert axes to transform between MediaPipe and matplotlib coordinate systems
            #ax.invert_yaxis()
            ax.invert_xaxis()
            ax.invert_zaxis()

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # initialize empty plot elements
            pts, = ax.plot([], [], [], 'ko', markersize=2, alpha=1.0)

            # dynamic exercise diagnosis
            diag = {}

            if side_focused:
                if 'FingerTapping' in current_ex:
                    diag['dot_w'], = ax.plot([], [], [], 'ro', markersize=6, zorder=10)
                    diag['dot_f'], = ax.plot([], [], [], 'bo', markersize=6, zorder=10)
                    diag['band'], = ax.plot([], [], [], '-', c='#00ff00', lw=2, zorder=5)

                elif 'FingerAlternation' in current_ex:
                    diag['dot_w'], = ax.plot([], [], [], 'ro', markersize=6, zorder=10)
                    diag['dot_t'], = ax.plot([], [], [], 'ro', markersize=6, zorder=10)
                    diag['dots_f'] = [ax.plot([], [], [], 'bo', markersize=6, zorder=10)[0] for _ in range(4)]
                    diag['bands'] = [ax.plot([], [], [], '-', c=alt_colors[color], lw=1.5, alpha=0.5, zorder=5)[0] for color in range(4)]

                elif 'HandOpening' in current_ex:
                    diag['dot_w'], = ax.plot([], [], [], 'ro', markersize=8, zorder=10)
                    diag['dots_f'] = [ax.plot([], [], [], 'bo', markersize=6, zorder=10)[0] for _ in range(4)]
                    diag['bands'] = [ax.plot([], [], [], '-', c='#00ff00', lw=2, alpha=0.6, zorder=5)[0] for _ in range(4)]

                elif 'ProSup' in current_ex:
                    # xyz coordinate axes on the wrist
                    diag['quivers'] = []
                    ax.plot([], [], [], '-', c='r', lw=2, label='X (Medial)')
                    ax.plot([], [], [], '-', c='g', lw=3, zorder=10, label='Y (Distal)')
                    ax.plot([], [], [], '-', c='b', lw=3, zorder=10, label='Z (Palm)')
                    ax.legend(loc='upper left', fontsize=8)

            return pts, diag

        # define dynamic titles and colors based on side_focus
        left_title = "Left Hand (Active)" if side_focus == 'L' else "Left Hand (Passive)"
        right_title = "Right Hand (Active)" if side_focus == 'R' else "Right Hand (Passive)"
        left_color = '#2ca02c' if side_focus == 'L' else '#7f7f7f'
        right_color = '#2ca02c' if side_focus == 'R' else '#7f7f7f'

        # set up the three axes for each 3D animation plot
        pts_left, diag_l = _setup_3d_axis(ax_left, left_title, dashboard_data['left_hand'], left_color,
                                          'L' if side_focus == 'L' else None, current_ex=ex_id)

        pts_pose, diag_p = _setup_3d_axis(ax_pose, "Pose Skeleton", dashboard_data['pose'], current_ex=ex_id)

        pts_right, diag_r = _setup_3d_axis(ax_right, right_title, dashboard_data['right_hand'], right_color,
                                           'R' if side_focus == 'R' else None, current_ex=ex_id)

        # setup 3D links between landmarks
        pose_links = config['body_parts'].get('pose_link_lst', [])
        hand_links = config['body_parts'].get('hands_link_lst', [])

        lines_left = [ax_left.plot([], [], [], '-', c='gray', lw=1.5, alpha=0.8)[0] for _ in hand_links] if pts_left else []
        lines_pose = [ax_pose.plot([], [], [], '-', c='#1f77b4', lw=2, alpha=0.8)[0] for _ in pose_links] if pts_pose else []
        lines_right = [ax_right.plot([], [], [], '-', c='gray', lw=1.5, alpha=0.8)[0] for _ in hand_links] if pts_right else []

        # initialize the 1D metric plots (bottom rows)
        tracker_lines = []
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # run for each metric (can be only one or multiple)
        for i, (m_title, (t_data, sig_data)) in enumerate(metrics_dict.items()):
            # span the metric's time series across all 3 columns
            ax_m = fig.add_subplot(gs[i + 1, :])
            ax_m.set_title(m_title, fontsize=10, fontweight='bold')

            # if the data is a list, there are multiple time series -> overlap
            if isinstance(t_data, list) and isinstance(t_data[0], (list, np.ndarray)):
                min_x, max_x = float('inf'), float('-inf')

                # loop through the multiple signals
                for j in range(len(t_data)):
                    t_arr = np.array(t_data[j])
                    sig_arr = np.array(sig_data[j])

                    c = color_palette[j % len(color_palette)]
                    digit_label = f'Digit {j+2}'    # maps 0, 1, 2, 3 -> Digits 2, 3, 4, 5

                    # plot the static signal
                    ax_m.plot(t_arr, sig_arr, color=c, lw=1.5, alpha=0.8, label=digit_label)
                    min_x = min(min_x, float(t_arr[0]))
                    max_x = max(max_x, float(t_arr[-1]))

                    if j == 0:
                        # initialize a red vertical tracker line at position x = 0
                        vline = ax_m.axvline(x=t_arr[0], color='red', lw=2, zorder=5)
                        tracker_lines.append((vline, t_arr))

                ax_m.set_xlim((min_x, max_x))
                ax_m.legend(loc='upper right', fontsize=8)

            else:
                # single time series
                t_arr = np.array(t_data)
                sig_arr = np.array(sig_data)
                ax_m.plot(t_arr, sig_arr, color='black', lw=1.5)
                ax_m.set_xlim((float(t_arr[0]), float(t_arr[-1])))

                vline = ax_m.axvline(x=t_arr[0], color='red', lw=2, zorder=5)
                tracker_lines.append((vline, t_arr))

            ax_m.set_ylabel('Amplitude', fontsize=9)
            ax_m.grid(True, linestyle=':', alpha=0.6)

            if i == num_metrics - 1:
                ax_m.set_xlabel('Time [s]', fontsize=10)

        # defines the update function
        def update(frame):
            updated_artists = []

            # helper to update the 3D skeletons
            def update_skeleton(ax, lm_dict, pts, lines, links, diag=None, side_focused=None, current_ex=''):

                # return for missing data
                if not lm_dict or not pts:
                    return

                # update the base skeleton points
                xs = [lm_dict[key][0][frame] for key in lm_dict.keys()]
                ys = [lm_dict[key][1][frame] for key in lm_dict.keys()]
                zs = [lm_dict[key][2][frame] for key in lm_dict.keys()]
                pts.set_data(xs, ys)
                pts.set_3d_properties(zs)
                updated_artists.append(pts)

                # update the base skeleton lines
                for line, segment in zip(lines, links):
                    lm1, lm2 = segment
                    if lm1 in lm_dict and lm2 in lm_dict:
                        line.set_data([lm_dict[lm1][0][frame], lm_dict[lm2][0][frame]],
                                      [lm_dict[lm1][1][frame], lm_dict[lm2][1][frame]])
                        line.set_3d_properties([lm_dict[lm1][2][frame], lm_dict[lm2][2][frame]])
                        updated_artists.append(line)

                # update diagnostic markers
                if diag and side_focused:
                    side_idx = '1' if side_focused == 'L' else '2'
                    w_k = f'wrist{side_idx}'

                    if 'FingerTapping' in current_ex:
                        f_k = f'ftip{side_idx}2'
                        if w_k in lm_dict and f_k in lm_dict:
                            diag['dot_w'].set_data([lm_dict[w_k][0][frame]], [lm_dict[w_k][1][frame]])
                            diag['dot_w'].set_3d_properties([lm_dict[w_k][2][frame]])
                            diag['dot_f'].set_data([lm_dict[f_k][0][frame]], [lm_dict[f_k][1][frame]])
                            diag['dot_f'].set_3d_properties([lm_dict[f_k][2][frame]])
                            diag['band'].set_data([lm_dict[w_k][0][frame], lm_dict[f_k][0][frame]],
                                                  [lm_dict[w_k][1][frame], lm_dict[f_k][1][frame]])
                            diag['band'].set_3d_properties([lm_dict[w_k][2][frame], lm_dict[f_k][2][frame]])
                            updated_artists.extend([diag['dot_w'], diag['dot_f'], diag['band']])

                    elif 'FingerAlternation' in current_ex:
                        t_k = f'ftip{side_idx}1'
                        f_keys = [f'ftip{side_idx}{d}' for d in range(2, 6)]
                        if w_k in lm_dict and t_k in lm_dict and all(fk in lm_dict for fk in f_keys):
                            diag['dot_w'].set_data([lm_dict[w_k][0][frame]], [lm_dict[w_k][1][frame]])
                            diag['dot_w'].set_3d_properties([lm_dict[w_k][2][frame]])
                            diag['dot_t'].set_data([lm_dict[t_k][0][frame]], [lm_dict[t_k][1][frame]])
                            diag['dot_t'].set_3d_properties([lm_dict[t_k][2][frame]])
                            updated_artists.extend([diag['dot_w'], diag['dot_t']])

                            for index, fk in enumerate(f_keys):
                                diag['dots_f'][index].set_data([lm_dict[fk][0][frame]], [lm_dict[fk][1][frame]])
                                diag['dots_f'][index].set_3d_properties([lm_dict[fk][2][frame]])
                                diag['bands'][index].set_data([lm_dict[t_k][0][frame], lm_dict[fk][0][frame]],
                                                              [lm_dict[t_k][1][frame], lm_dict[fk][1][frame]])
                                diag['bands'][index].set_3d_properties([lm_dict[t_k][2][frame], lm_dict[fk][2][frame]])
                                updated_artists.extend([diag['dots_f'][index], diag['bands'][index]])

                    elif 'HandOpening' in current_ex:
                        f_keys = [f'ftip{side_idx}{d}' for d in range(2, 6)]
                        if w_k in lm_dict and all(fk in lm_dict for fk in f_keys):
                            diag['dot_w'].set_data([lm_dict[w_k][0][frame]], [lm_dict[w_k][1][frame]])
                            diag['dot_w'].set_3d_properties([lm_dict[w_k][2][frame]])
                            updated_artists.append(diag['dot_w'])

                            for index, fk in enumerate(f_keys):
                                diag['dots_f'][index].set_data([lm_dict[fk][0][frame]], [lm_dict[fk][1][frame]])
                                diag['dots_f'][index].set_3d_properties([lm_dict[fk][2][frame]])
                                diag['bands'][index].set_data([lm_dict[w_k][0][frame], lm_dict[fk][0][frame]],
                                                              [lm_dict[w_k][1][frame], lm_dict[fk][1][frame]])
                                diag['bands'][index].set_3d_properties([lm_dict[w_k][2][frame], lm_dict[fk][2][frame]])
                                updated_artists.extend([diag['dots_f'][index], diag['bands'][index]])

                    elif 'ProSup' in current_ex:
                        index_name = f'mcp{side_idx}2'
                        pinky_name = f'mcp{side_idx}5'

                        if w_k in lm_dict and index_name in lm_dict and pinky_name in lm_dict:
                            wrist_pos = np.array([lm_dict[w_k][0][frame],
                                                  lm_dict[w_k][1][frame],
                                                  lm_dict[w_k][2][frame]])
                            index_pos = np.array([lm_dict[index_name][0][frame],
                                                  lm_dict[index_name][1][frame],
                                                  lm_dict[index_name][2][frame]])
                            pinky_pos = np.array([lm_dict[pinky_name][0][frame],
                                                  lm_dict[pinky_name][1][frame],
                                                  lm_dict[pinky_name][2][frame]])

                            # calculate the coordinate triangle from the 3D landmarks
                            y_vec = 0.5 * (index_pos + pinky_pos) - wrist_pos
                            z_vec = np.cross((index_pos - wrist_pos), (pinky_pos - wrist_pos))
                            if side_idx == '1':
                                z_vec = z_vec * (-1)
                            x_vec = np.cross(y_vec, z_vec)

                            # normalize the vectors
                            x_n = x_vec / (np.linalg.norm(x_vec) + 1e-8)
                            y_n = y_vec / (np.linalg.norm(y_vec) + 1e-8)
                            z_n = z_vec / (np.linalg.norm(z_vec) + 1e-8)

                            # remove old arrows
                            for q in diag['quivers']:
                                q.remove()
                            diag['quivers'].clear()

                            # draw new 3D arrows originating at the wrist
                            ax_length = 0.08
                            q_x = ax.quiver(wrist_pos[0], wrist_pos[1], wrist_pos[2], x_n[0], x_n[1], x_n[2], color='r',
                                            length=ax_length, normalize=True)
                            q_y = ax.quiver(wrist_pos[0], wrist_pos[1], wrist_pos[2], y_n[0], y_n[1], y_n[2], color='g',
                                            length=ax_length, normalize=True)
                            q_z = ax.quiver(wrist_pos[0], wrist_pos[1], wrist_pos[2], z_n[0], z_n[1], z_n[2], color='b',
                                            length=ax_length, normalize=True)

                            diag['quivers'].extend([q_x, q_y, q_z])
                            updated_artists.extend([q_x, q_y, q_z])

            # update the skeleton positions for the animation

            update_skeleton(ax_left, dashboard_data['left_hand'], pts_left, lines_left, hand_links, diag_l,
                            'L' if side_focus == 'L' else None, current_ex=ex_id)

            update_skeleton(ax_pose, dashboard_data['pose'], pts_pose, lines_pose, pose_links, current_ex=ex_id)

            update_skeleton(ax_right, dashboard_data['right_hand'], pts_right, lines_right, hand_links, diag_r,
                            'R' if side_focus == 'R' else None, current_ex=ex_id)

            # update the 1D red tracker lines
            for v_line, t_array in tracker_lines:
                # grab the time at the current frame (in case arrays differ slightly in length)
                idx = min(frame, len(t_array) - 1)
                v_line.set_xdata([t_array[idx], t_array[idx]])
                updated_artists.append(v_line)

            return updated_artists

        # render to videos (mp4)
        #print(f"Rendering the quality check dashboard to {out_path}...")
        interval_ms = (1000.0 / self.fps) * skip_frames

        # run the animation
        ani = animation.FuncAnimation(fig, update, frames=frames_to_run, interval=interval_ms, blit=True)

        # save the video file using ffmpeg. Higher bitrates ensure less blurry frames.
        ani.save(out_path, writer='ffmpeg', fps=(self.fps / skip_frames), bitrate=2000)

        plt.close(fig)

    # ============================================================================= #
    #                           5) CLASSIFICATION                                   #
    # ============================================================================= #
    def corr_matrix_heatmap(self, matrix_data: pd.DataFrame, mask: np.ndarray, ex_name: str, labels: list[str]) -> None:

        plt.figure(figsize=(12, 10))

        sns.heatmap(matrix_data, mask=mask, cmap='coolwarm', vmin=0, vmax=1, xticklabels=labels, yticklabels=labels,
                    annot=True, annot_kws={"size": 14}, square=True, linewidths=.5, fmt='.2f')

        plt.title(f'Spearman Correlation Matrix - {ex_name}', fontsize=16)
        plt.tight_layout()

        suffix = '.png'
        f_name: str = ex_name + '_feature_heatmap' + suffix
        f_path: str = os.path.join(self.classification_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600)

        plt.close()

    def viz_classification_confusion_matrix(self, df: pd.DataFrame, model_algo: str) -> None:

        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(df['Real_Score'], (df['Predicted_Score'] > 0.5).astype('int'))
        hm = sns.heatmap(cm, annot=True, annot_kws={"size": 20}, fmt='d', cmap='Blues', cbar=False,
                         xticklabels=['Non-Paretic (0)', 'Paretic (1)'],
                         yticklabels=['Non-Paretic (0)', 'Paretic (1)'],)
        hm.set_xticklabels(hm.get_xticklabels(), fontsize=16)
        hm.set_yticklabels(hm.get_yticklabels(), fontsize=16)

        # figure labels
        title: str = ''
        if model_algo.upper() == 'RF':
            title = 'Random Forest'
        elif model_algo.upper() == 'XGBOOST':
            title = 'XGBoost'
        elif model_algo.upper() == 'CATBOOST':
            title = 'CatBoost'
        plt.title(f'{title}', fontsize=18)
        plt.xlabel('Predicted Label', fontsize=18)
        plt.ylabel('True Label', fontsize=18)
        plt.tight_layout()

        # save figure
        model_dir: str = os.path.join(self.classification_res_path, f'{model_algo}_classification')
        plt.savefig(os.path.join(model_dir, f'{model_algo}_confusion_matrix.png'), dpi=600)
        plt.close()

    @staticmethod
    def viz_roc_curve(df: pd.DataFrame, model_algo: str, target: str, out_dir: str, n_bootstraps: int = 1000) -> None:
        """
        Plots the Receiver Operating Characteristic (ROC) curve for pooled Out-Of-Fold predictions
        with AUC 95% Confidence Intervals.
        """
        if 'Predicted_Score' not in df.columns or 'Real_Score' not in df.columns:
            print("Required columns missing for ROC curve.")
            return

        y_true = df['Real_Score'].values
        y_prob = df['Predicted_Score'].values

        # 1) Calculate the main ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # 2) Bootstrap the 95% Confidence Interval for the band AND the AUC score
        fpr_grid = np.unique(np.concatenate(([0.0, 1.0], fpr)))
        tpr_bootstraps = []
        auc_bootstraps = []

        for _ in range(n_bootstraps):
            indices = resample(np.arange(len(y_true)), replace=True)
            y_true_b = y_true[indices]
            y_prob_b = y_prob[indices]

            if len(np.unique(y_true_b)) < 2:
                continue

            fpr_b, tpr_b, _ = roc_curve(y_true_b, y_prob_b)

            auc_bootstraps.append(auc(fpr_b, tpr_b))

            idx = np.searchsorted(fpr_b, fpr_grid, side='right') - 1
            idx = np.clip(idx, 0, len(tpr_b) - 1)
            tpr_interp = tpr_b[idx]
            tpr_bootstraps.append(tpr_interp)

        # Calculate percentiles for the Band
        tpr_lower = np.percentile(tpr_bootstraps, 2.5, axis=0)
        tpr_upper = np.percentile(tpr_bootstraps, 97.5, axis=0)

        auc_lower = np.percentile(auc_bootstraps, 2.5)
        auc_upper = np.percentile(auc_bootstraps, 97.5)

        # Bypasses the buggy step='post' in fill_between by manually duplicating
        # the coordinates to create perfect rigid geometric steps.
        fpr_step = np.repeat(fpr_grid, 2)[1:]
        tpr_lower_step = np.repeat(tpr_lower, 2)[:-1]
        tpr_upper_step = np.repeat(tpr_upper, 2)[:-1]

        # 3) Plotting
        plt.figure(figsize=(8, 6))

        # Plot the CI band using the manual step arrays
        plt.fill_between(fpr_step, tpr_lower_step, tpr_upper_step, color='#AED6F6', alpha=0.5, label='95% CI')

        # Plot the main curve
        plt.plot(fpr, tpr, color='#1B4F72', lw=2.5,
                 label=f'Pooled OOF ROC (AUC = {roc_auc:.2f} [{auc_lower:.2f} - {auc_upper:.2f}])',
                 drawstyle='steps-post')

        plt.plot([0, 1], [0, 1], color='#7F8C8D', lw=2, linestyle='--', alpha=0.8, label='Chance (AUC = 0.5)')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve: {model_algo.upper()} ({target})', fontsize=16, pad=15)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f'{model_algo}_roc_curve_{target}.png')
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"ROC Curve saved to {out_path}")

    @staticmethod
    def viz_combined_roc_curve(predictions_dict: dict, target: str, out_dir: str,
                               n_bootstraps: int = 1000) -> None:
        """
        Plots a combined Receiver Operating Characteristic (ROC) curve for multiple models
        with highly transparent overlapping 95% Confidence Intervals.

        Args:
            predictions_dict (dict): Dictionary mapping model display names to their out-of-fold prediction DataFrames.
                                     Example: {'Random Forest': df_rf, 'XGBoost': df_xgb, 'CatBoost': df_cat}
        """
        plt.figure(figsize=(9, 7))

        # High-contrast color palette for scientific publication
        line_colors = ['#1B4F72', '#D35400', '#27AE60']  # Navy, Rust Orange, Emerald Green
        #line_colors = ['#1B4F72', '#2980B9', '#AED6F6']
        band_colors = ['#AED6F6', '#F5CBA7', '#ABEBC6']  # Light Blue, Light Orange, Light Green
        line_styles = ['-', '--', ':']

        for idx, (model_name, df) in enumerate(predictions_dict.items()):
            if 'Predicted_Score' not in df.columns or 'Real_Score' not in df.columns:
                print(f"Required columns missing for {model_name}. Skipping.")
                continue

            y_true = df['Real_Score'].values
            y_prob = df['Predicted_Score'].values

            # 1) Calculate the main ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            # 2) Bootstrap the 95% Confidence Interval for the band AND the AUC score
            fpr_grid = np.unique(np.concatenate(([0.0, 1.0], fpr)))
            tpr_bootstraps = []
            auc_bootstraps = []

            for _ in range(n_bootstraps):
                indices = resample(np.arange(len(y_true)), replace=True)
                y_true_b = y_true[indices]
                y_prob_b = y_prob[indices]

                if len(np.unique(y_true_b)) < 2:
                    continue

                fpr_b, tpr_b, _ = roc_curve(y_true_b, y_prob_b)
                auc_bootstraps.append(auc(fpr_b, tpr_b))

                idx_search = np.searchsorted(fpr_b, fpr_grid, side='right') - 1
                idx_search = np.clip(idx_search, 0, len(tpr_b) - 1)
                tpr_interp = tpr_b[idx_search]
                tpr_bootstraps.append(tpr_interp)

            # Calculate percentiles
            tpr_lower = np.percentile(tpr_bootstraps, 2.5, axis=0)
            tpr_upper = np.percentile(tpr_bootstraps, 97.5, axis=0)
            auc_lower = np.percentile(auc_bootstraps, 2.5)
            auc_upper = np.percentile(auc_bootstraps, 97.5)

            # 3) Manual Step Conversion (Fixes the polygon rendering bug)
            fpr_step = np.repeat(fpr_grid, 2)[1:]
            tpr_lower_step = np.repeat(tpr_lower, 2)[:-1]
            tpr_upper_step = np.repeat(tpr_upper, 2)[:-1]

            # 4) Plot this model's band and line
            c_idx = idx % len(line_colors)
            ls = line_styles[c_idx]

            plt.fill_between(fpr_step, tpr_lower_step, tpr_upper_step, color=band_colors[c_idx], alpha=0.2)

            plt.plot(fpr, tpr, color=line_colors[c_idx], lw=2.5, linestyle=ls,
                     label=f'{model_name} (AUC = {roc_auc:.2f} [{auc_lower:.2f} - {auc_upper:.2f}])',
                     drawstyle='steps-post')

        # Add the chance line
        plt.plot([0, 1], [0, 1], color='#7F8C8D', lw=2, linestyle='--', alpha=0.8, label='Chance (AUC = 0.5)')

        # Formatting
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'Model Consensus: ROC Curves ({target})', fontsize=18, pad=15)

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [2,1,0,3]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                   loc="lower right", fontsize=11, framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f'combined_roc_curve_{target}.png')
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Combined ROC Curve saved to {out_path}")

    # ============================================================================= #
    #                           6) REGRESSION                                       #
    # ============================================================================= #
    def viz_regression_target_distribution(self, df_baseline: pd.DataFrame, target_col: str, model_algo: str) -> None:
        """
        Creates a histogram of the target score (e.g., Jebsen Taylor Hand Function Test) distribution for
        the baseline visits, color-coded by severity bins (terciles), with vertical lines marking the splits.

        Args:
            df_baseline (pd.DataFrame): Dataset including all features and target variables of the 1st visit (T1)
            target_col (str): The target column with the scores to be predicted (also used for training).
            model_algo (str): The used model algorithm (xgboost, catboost, or random forest)

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))

        # define a clinical traffic-light palette
        palette = {'Mild': '#2ca02c', 'Moderate': '#ff7f0e', 'Severe': '#d62728'}

        # plot a stacked histogram
        sns.histplot(data=df_baseline, x=target_col, hue='severity_bin',
                     palette=palette, multiple='stack', bins=15, edgecolor='black', alpha=0.85)

        # calculate the exact numerical thresholds where the percentiles split
        mild_max = df_baseline[df_baseline['severity_bin'] == 'Mild'][target_col].max()
        mod_max = df_baseline[df_baseline['severity_bin'] == 'Moderate'][target_col].max()

        # overlay vertical dashed lines for the tercile boundaries
        plt.axvline(mild_max, color='black', linestyle='--', lw=2, label=f'33rd Percentile ({mild_max:.2f})')
        plt.axvline(mod_max, color='black', linestyle='--', lw=2, label=f'66th Percentile ({mod_max:.2f})')

        # layout and labeling
        # clean up the title based on the target name
        clean_target = target_col.replace('Asymmetry_JT_Ratio_', 'JT Subtest: ').replace('_', ' ')
        plt.title(f'Target Distribution & Stratification Splits\n{clean_target}', fontsize=16, pad=15)
        plt.xlabel('Target Score (Log-Ratio)', fontsize=14)
        plt.ylabel('Count (Baseline Participants)', fontsize=14)

        # fix the legend to include both the bins and the vertical lines
        plt.legend(title='Severity Bin & Splits', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # save figure
        model_dir: str = os.path.join(self.regression_res_path, f'{model_algo}_regression')
        os.makedirs(model_dir, exist_ok=True)

        f_basename: str = f'{model_algo}_target_distribution_{target_col}.png'
        plt.savefig(os.path.join(model_dir, f_basename), dpi=600, bbox_inches='tight')
        plt.close()

    def viz_regression_identity_plot(self, df_subtests: pd.DataFrame, df_total: pd.DataFrame, model_algo: str) -> None:

        model_dir: str = os.path.join(self.regression_res_path, f'{model_algo}_regression')
        os.makedirs(model_dir, exist_ok=True)
        df_combined = pd.concat([df_subtests, df_total], ignore_index=True)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_combined, x='Real_Score', y='Predicted_Score', hue='Target', s=80, alpha=0.7)

        # prediction line
        min_val: float = float(min(df_combined['Real_Score'].min(), df_combined['Predicted_Score'].min())) * 0.9
        max_val: float = float(max(df_combined['Real_Score'].max(), df_combined['Predicted_Score'].max())) * 1.1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect (y=x)')

        # figure layout
        plt.title(f'Real vs. Predicted JTHFT Asymmetry Ratio', fontsize=18)
        plt.xlabel('Real Log-Ratio (Stopwatch)', fontsize=16)
        plt.ylabel('Predicted Log-Ratio (Kinematics)', fontsize=16)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend(loc='upper left')
        plt.tight_layout()

        # save figure
        model_dir: str = os.path.join(self.regression_res_path, f'{model_algo}_regression')
        plt.savefig(os.path.join(model_dir, f'{model_algo}_identity_plot_combined.png'), dpi=600)
        plt.close()

        # plot total asymmetry ratio separately
        plt.figure(figsize=(8, 7))

        y_true = df_total['Real_Score'].values
        y_pred = df_total['Predicted_Score'].values

        # calculate metrics
        r2 = r2_score(y_true, y_pred)
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)

        # draw the Scatter Plot (Updated Color)
        sns.scatterplot(data=df_total, x='Real_Score', y='Predicted_Score',
                        color='#2980B9', s=120, alpha=0.8, edgecolor='white')

        # calculate axis limits for drawing the lines
        min_val_tot: float = float(min(y_true.min(), y_pred.min())) * 0.95
        max_val_tot: float = float(max(y_true.max(), y_pred.max())) * 1.05
        x_line = np.array([min_val_tot, max_val_tot])

        # plot the perfect identity line (y=x)
        plt.plot(x_line, x_line, 'k--', alpha=0.5, label='Perfect Prediction (y=x)')

        # plot the line of best fit
        y_fit = slope * x_line + intercept
        fit_legend = f"$R^2$ = {r2:.2f}, Slope = {slope:.2f}, Intercept = {intercept:.2f}"
        plt.plot(x_line, y_fit, color='#1B4F72', lw=2.5, label=fit_legend)

        # figure layout
        #plt.title(f'Prediction Accuracy: Total Asymmetry Ratio', fontsize=18, pad=15)
        plt.xlabel('Real Log-Ratio (Stopwatch)', fontsize=16)
        plt.ylabel('Predicted Log-Ratio (Kinematics)', fontsize=16)
        plt.legend(loc='upper left', fontsize=10, framealpha=0.9, borderpad=1, labelspacing=0.8)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()

        plt.savefig(os.path.join(model_dir, f'{model_algo}_identity_plot_total_only.png'), dpi=600, bbox_inches='tight')
        plt.close()

    def viz_regression_bland_altman(self, df: pd.DataFrame, model_algo: str) -> None:

        mean_scores = (df['Real_Score'] + df['Predicted_Score']) / 2
        diff_scores = df['Predicted_Score'] - df['Real_Score']

        bias = diff_scores.mean()
        std_diff = diff_scores.std()
        upper_loa = bias + 1.96 * std_diff
        lower_loa = bias - 1.96 * std_diff

        plt.figure(figsize=(10, 8))
        severity_order: list[str] = ['Mild', 'Moderate', 'Severe']
        sns.scatterplot(x=mean_scores, y=diff_scores, hue=df['Severity_Class'], hue_order=severity_order, s=200, palette='viridis')

        """
        # Add Participant IDs to the dots
        for i in range(len(df)):
            # Extract coordinates and ID
            x_coord = mean_scores.iloc[i]
            y_coord = diff_scores.iloc[i]
            p_id = df['p_ID'].iloc[i]

            # Annotate the plot
            plt.annotate(
                p_id,
                (x_coord, y_coord),
                textcoords="offset points",
                xytext=(0, 6),  # Shift the text 6 pixels above the dot
                ha='center',  # Horizontally center the text
                fontsize=8,  # Keep the font small to prevent overlapping
                alpha=0.75,  # Slight transparency so dots underneath remain visible
                color='black'
            )
        """

        # horizontal bias and thresh
        plt.axhline(bias, color='black', linestyle='-', linewidth=2, label=f'Bias: {bias:.3f}')
        plt.axhline(upper_loa, color='red', linestyle='--', label=f'+1.96 SD: {upper_loa:.3f}')
        plt.axhline(lower_loa, color='red', linestyle='--', label=f'-1.96 SD: {lower_loa:.3f}')

        # figure layout
        #plt.title(f'JTHFT - Total Asymmetry', fontsize=18)
        plt.xlabel('Average of Real and Predicted Score', fontsize=16)
        plt.xticks(fontsize=16)
        plt.ylabel('Difference (Predicted - Real)', fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16, loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # save figure
        model_dir: str = os.path.join(self.regression_res_path, f'{model_algo}_regression')
        plt.savefig(os.path.join(model_dir, f'{model_algo}_bland_altman.png'), dpi=600)
        plt.close()

    def viz_regression_shap_bar(self, df_shap: pd.DataFrame, df_feat: pd.DataFrame, common_cols: list[str],
                                model_algo: str, f_path: str, task_type: str) -> None:

        mean_abs_shap = df_shap[common_cols].abs().mean(axis=0)
        total_impact = mean_abs_shap.sum()

        top_cols = mean_abs_shap.nlargest(10).index.tolist()
        #rest_cols = [c for c in common_cols if c not in top_cols]

        plot_cols = []
        plot_values = []
        pct_dict = {}

        for col in top_cols:
            plot_cols.append(col)
            plot_values.append(mean_abs_shap[col])
            pct_dict[col] = f'{(mean_abs_shap[col] / total_impact) * 100:.1f}%'

        """
        if rest_cols:
            rest_impact = mean_abs_shap[rest_cols].sum()
            rest_label = f'Sum of {len(rest_cols)} other features'
            plot_cols.append(rest_label)
            plot_values.append(rest_impact)
            pct_dict[rest_label] = f'{(rest_impact / total_impact) * 100:.1f}%'
        """

        # Reverse lists because horizontal bar charts plot from bottom to top
        plot_cols.reverse()
        plot_values.reverse()

        plt.figure(figsize=(5, 8))
        ax = plt.gca()

        # Draw native matplotlib bars (Color matched to ROC Curve CI band)
        bars = ax.barh(plot_cols, plot_values, color='#6C3BAA', alpha=0.9, height=0.6)

        # --- PERCENTAGE LABELS (COMMENTED OUT FOR MANUSCRIPT) ---
        # x_offset = max(plot_values) * 0.015
        # for bar, col in zip(bars, plot_cols):
        #     y_center = bar.get_y() + bar.get_height() / 2
        #     ax.text(x_offset, y_center, pct_dict[col],
        #             va='center', ha='left', color='black', fontsize=12)
        # --------------------------------------------------------

        # write the absolute SHAP value to the right of each bar
        max_val = max(plot_values)
        x_offset = max_val * 0.03  # add a gap between the bar end and the text

        for bar, val in zip(bars, plot_values):
            y_center = bar.get_y() + bar.get_height() / 2
            bar_width = bar.get_width()

            # write the value formatted to 3 decimal places
            ax.text(bar_width + x_offset, y_center, f'{val:.3f}',
                    va='center', ha='left', color='black', fontsize=12)

        # extend the x-axis by 25% so the text doesn't get clipped
        ax.set_xlim(0, max_val * 1.25)

        target_exercise_name: str = os.path.basename(f_path).split('_shap_vals_')[-1].replace('.csv', '')

        #plt.title(f'SHAP Global Feature Importance: {target_exercise_name} ({model_algo.upper()})', fontsize=18, pad=45)

        # x-axis reflect SHAP mathematics
        plt.xlabel('Mean |SHAP value| (Impact on Model Output)', fontsize=14)

        # hides y-axis labels
        ax.set_yticklabels([])

        #plt.tick_params(axis='y', labelsize=12)
        plt.tick_params(axis='x', labelsize=11)

        # Clean borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        # save plot
        if task_type == 'classification':
            model_dir: str = os.path.join(self.classification_res_path, f'{model_algo}_{task_type}')
        else:
            model_dir: str = os.path.join(self.regression_res_path, f'{model_algo}_{task_type}')

        f_basename: str = f'{model_algo}_shap_bar_{target_exercise_name}.png'
        plt.savefig(os.path.join(model_dir, f_basename), dpi=600, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    @staticmethod
    def viz_regression_combined_shap_bar(shap_dict: dict, target: str, out_dir: str, top_n: int = 10) -> None:
        """
        Creates a grouped SHAP bar chart comparing feature importance across multiple models.
        Converts absolute SHAP to Relative Percentage Impact to safely compare different architectures.

        Args:
            shap_dict (dict): Dictionary format {'Model Name': df_shap} e.g., {'XGBoost': df_xgb, ...}

        """
        model_names = list(shap_dict.keys())

        # calculate relative impact in percentages
        relative_impacts = {}
        for name, df in shap_dict.items():
            mean_abs_shap = df.abs().mean(axis=0)
            rel_pct = (mean_abs_shap / mean_abs_shap.sum()) * 100
            relative_impacts[name] = rel_pct

        # combine and fill missing features with 0.0%
        df_compare = pd.DataFrame(relative_impacts).fillna(0.0)
        df_compare['Ensemble_Mean'] = df_compare.mean(axis=1)

        # export full ranked dataset
        df_export = df_compare.copy()

        # add rank columns for each model (1 is highest impact)
        for name in model_names:
            df_export[f'{name}_Rank'] = df_export[name].rank(ascending=False, method='min').astype(int)
        df_export['Ensemble_Rank'] = df_export['Ensemble_Mean'].rank(ascending=False, method='min').astype(int)

        # sort the export file by the 'Ensemble_Mean' from highest to lowest
        df_export = df_export.sort_values(by='Ensemble_Mean', ascending=False)

        # save as CSV
        df_export.to_csv(os.path.join(out_dir, f'combined_shap_consensus_data_{target}.csv'))

        # extract top n for plotting
        df_top = df_compare.nlargest(top_n, 'Ensemble_Mean')
        df_top = df_top.sort_values(by='Ensemble_Mean', ascending=True)
        df_top = df_top.drop(columns=['Ensemble_Mean'])
        features = df_top.index.tolist()

        # plot config
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        y = np.arange(len(features))
        bar_width = 0.25
        colors = ['#1B4F72', '#2980B9', '#AED6F6']

        # draw the grouped bars
        for idx, model_name in enumerate(model_names):
            y_pos = y + (idx - 1) * bar_width
            ax.barh(y_pos, df_top[model_name], height=bar_width,
                    color=colors[idx % len(colors)], alpha=0.9, label=model_name)

        # formatting
        ax.set_yticks(y)
        ax.set_yticklabels(features, fontsize=16)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.0f}%'))
        ax.tick_params(axis='x', labelsize=16)

        plt.xlabel('Relative Feature Impact [%]', fontsize=16)
        plt.title(f'Top {top_n} Features Across Models', fontsize=18, pad=20)
        handles, labels = ax.get_legend_handles_labels()
        order = [2,1,0]
        plt.legend(handles=[handles[idx] for idx in order], labels=[labels[idx] for idx in order],
                   loc='lower right', fontsize=14, framealpha=0.9)

        plt.grid(True, axis='x', linestyle=':', alpha=0.6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f'combined_shap_consensus_{target}.png')
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Combined Consensus SHAP Plot saved to {out_path}")

    def viz_regression_shap_beeswarm(self, df_shap: pd.DataFrame, df_feat: pd.DataFrame, common_cols: list[str],
                                     model_algo: str, f_path: str, task_type: str) -> None:

        # new imports needed for coordinate blending and custom ticks
        import matplotlib.transforms as mtransforms
        from matplotlib.ticker import MultipleLocator

        mean_abs_shap = df_shap[common_cols].abs().mean(axis=0)
        top_cols = mean_abs_shap.nlargest(10).index.tolist()
        #rest_cols = [c for c in common_cols if c not in top_cols]

        df_shap_plot = pd.DataFrame()
        df_feat_plot = pd.DataFrame()
        plot_cols = []

        # add top features first
        for col in top_cols:
            df_shap_plot[col] = df_shap[col]
            df_feat_plot[col] = df_feat[col]
            plot_cols.append(col)
        """
        # add the remaining features aggregate at the end
        if rest_cols:
            rest_label = f'Sum of {len(rest_cols)} other features'
            df_shap_plot[rest_label] = df_shap[rest_cols].sum(axis=1)
            df_feat_plot[rest_label] = 0.0
            plot_cols.append(rest_label)
        """
        plt.figure(figsize=(5, 8))

        shap.summary_plot(df_shap_plot[plot_cols].values,
                          features=df_feat_plot[plot_cols],
                          feature_names=plot_cols,
                          max_display=len(plot_cols),
                          sort=False,
                          color_bar=False,
                          show=False,
                          plot_size=(5, 7))

        ax = plt.gca()

        # hide y-labels
        ax.tick_params(axis='y', labelleft=False, left=False)

        # set x-axis ticks to steps of 0.1
        ax.xaxis.set_major_locator(MultipleLocator(0.1))

        # Safely ensure the plot spans at least -0.1 to 0.1 so the colorbar isn't cut off
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(min(x_min, -0.1), max(x_max, 0.1))

        # blended coordinate colorbar
        cmap = shap.plots.colors.red_blue
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])

        # Create the blended transform: X is tied to Data, Y is tied to the Axes Box
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

        # Bounds: [x_start, y_start, width, height]
        # Start at exactly x = -0.1, make width 0.2 (so it ends at exactly x = 0.1)
        cb_ax = ax.inset_axes([-0.1, 1.04, 0.2, 0.04], transform=trans)

        cb = plt.colorbar(sm, cax=cb_ax, orientation='horizontal')
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['Low', 'High'])
        cb.set_label('Feature Value', fontsize=12)

        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        # ---------------------------------------------------------

        target_name: str = os.path.basename(f_path).split('_shap_vals_')[-1].replace('.csv', '')

        #plt.title(f'SHAP Feature Impact: {target_name} ({model_algo.upper()})', fontsize=18)
        plt.xlabel('SHAP value (Impact on prediction)', fontsize=14)
        plt.tight_layout()

        if task_type == 'classification':
            model_dir: str = os.path.join(self.classification_res_path, f'{model_algo}_{task_type}')
        else:
            model_dir: str = os.path.join(self.regression_res_path, f'{model_algo}_{task_type}')

        f_basename: str = f'{model_algo}_shap_beeswarm_{target_name}.png'
        plt.savefig(os.path.join(model_dir, f_basename), dpi=600, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    # def viz_regression_shap_beeswarm_old(self, df_shap: pd.DataFrame, df_feat: pd.DataFrame, common_cols: list[str],
    #                                      model_algo: str, f_path: str, task_type: str) -> None:
    #
    #     # aggregate the 'rest of the features'
    #     mean_abs_shap = df_shap[common_cols].abs().mean(axis=0)
    #     top_6_cols = mean_abs_shap.nlargest(10).index.tolist()
    #     rest_cols = [c for c in common_cols if c not in top_6_cols]
    #
    #     if rest_cols:
    #         # create subset with top 6
    #         df_shap_plot = df_shap[top_6_cols].copy()
    #         df_feat_plot = df_feat[top_6_cols].copy()
    #
    #         # sum the remaining SHAP values per row
    #         rest_label = f'Sum of {len(rest_cols)} other features'
    #         df_shap_plot[rest_label] = df_shap[rest_cols].sum(axis=1)
    #         df_feat_plot[rest_label] = 0.0
    #
    #         plot_cols = top_6_cols + [rest_label]
    #         display_count = 11
    #     else:
    #         df_shap_plot = df_shap[common_cols]
    #         df_feat_plot = df_feat[common_cols]
    #         plot_cols = common_cols
    #         display_count = 10
    #
    #     plt.figure(figsize=(10, 6))
    #     shap.summary_plot(df_shap_plot.values, features=df_feat_plot, feature_names=plot_cols,
    #                       max_display=display_count, sort=False, show=False)
    #
    #     target_exercise_name: str = os.path.basename(f_path).split('_shap_vals_')[-1].replace('.csv', '')
    #     plt.title(f'SHAP Feature Impact: {target_exercise_name} ({model_algo.upper()})', fontsize=18, pad=20)
    #     plt.tight_layout()
    #
    #     # save plot
    #     if task_type == 'classification':
    #         model_dir: str = os.path.join(self.classification_res_path, f'{model_algo}_{task_type}')
    #     else:
    #         # regression
    #         model_dir: str = os.path.join(self.regression_res_path, f'{model_algo}_{task_type}')
    #
    #     f_basename: str = f'{model_algo}_shap_beeswarm_{target_exercise_name}.png'
    #     plt.savefig(os.path.join(model_dir, f_basename), dpi=600, bbox_inches='tight', pad_inches=0.2)
    #     plt.close()

    def viz_regression_feature_stability_boxplot(self, model_algo: str, f_path: str, task_type: str) -> None:

        if not os.path.exists(f_path):
            print(f'SHAP file missing: {f_path}')
            return

        # clean IDs before calculation
        df_shap = pd.read_csv(f_path)
        if 'Fold' not in df_shap.columns:
            print('Fold column missing from SHAP data. Feature stability plotting failed ...')
            return

        if 'p_ID' in df_shap.columns:
            df_shap = df_shap.drop(columns=['p_ID'])

        # transform from wide to long format: [Fold, Feature, SHAP_Value]
        df_melted = df_shap.melt(id_vars=['Fold'], var_name='Feature', value_name='SHAP_Value')

        # calculate the absolute SHAP value (impact magnitude)
        df_melted['Abs_SHAP'] = df_melted['SHAP_Value'].abs()

        # calculate the mean absolute SHAP for each feature within each fold
        fold_importance = df_melted.groupby(['Fold', 'Feature'])['Abs_SHAP'].mean().reset_index()

        # calculate the overall mean across all folds to identify the top 10
        overall_importance = fold_importance.groupby('Feature')['Abs_SHAP'].mean().sort_values(ascending=False)
        top_10_features = overall_importance.head(10).index.tolist()

        # filter the plot data to only include the top 10 features
        plot_data = fold_importance[fold_importance['Feature'].isin(top_10_features)].copy()

        # enforce categorical ordering so the best feature is always at the top (y-axis)
        plot_data['Feature'] = pd.Categorical(plot_data['Feature'], categories=top_10_features, ordered=True)

        # plotting
        plt.figure(figsize=(10, 7))

        # boxplot
        sns.boxplot(data=plot_data, x='Abs_SHAP', y='Feature', color='#B1E0D9', showmeans=True,
                    meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':'6'})

        # overlay the 5 specific fold dots on top of the boxes
        sns.stripplot(data=plot_data, x='Abs_SHAP', y='Feature', color='#2C3E50', alpha=0.7, size=6, jitter=False)

        target_name: str = os.path.basename(f_path).split('_shap_vals_')[-1].replace('.csv', '')

        plt.title(f'Feature Stability Across Folds: {target_name} ({model_algo.upper()})', fontsize=18, pad=15)
        plt.xlabel('Mean |SHAP value| per fold', fontsize=14)
        plt.ylabel('', fontsize=14)
        plt.grid(True, axis='x', linestyle=':', alpha=0.6)

        sns.despine()
        plt.tight_layout()

        # save data
        if task_type == 'classification':
            model_dir: str = os.path.join(self.classification_res_path, f'{model_algo}_{task_type}')
        else:
            model_dir: str = os.path.join(self.regression_res_path, f'{model_algo}_{task_type}')

        f_basename: str = f'{model_algo}_shap_stability_{target_name}.png'
        plt.savefig(os.path.join(model_dir, f_basename), dpi=600, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    # def viz_regression_generalization_gap(self, res_dir: str, model_algo: str, task_type: str) -> None:
    #
    #     # read the compiled metrics table
    #     metrics_file = os.path.join(res_dir, f'{model_algo}_metrics.csv')
    #     if not os.path.exists(metrics_file):
    #         print(f"Metrics file missing for {model_algo}. Cannot plot generalization gap.")
    #         return
    #
    #     df_metrics = pd.read_csv(metrics_file)
    #
    #     # ensure training metrics were tracked
    #     if 'Train_R2 (Mean±STD)' not in df_metrics.columns:
    #         print("Training metrics missing. Cannot plot generalization gap.")
    #         return
    #
    #     # clean subtest names for a cleaner y-axis
    #     df_metrics['Subtest'] = df_metrics['Target'].str.replace('Asymmetry_JT_Ratio_', '')
    #
    #     # filter out N/A rows and extract the numeric train R2 mean from the string "0.750 ± 0.020"
    #     df_metrics = df_metrics[df_metrics['Train_R2 (Mean±STD)'] != 'N/A'].copy()
    #     if df_metrics.empty:
    #         return
    #
    #     df_metrics['Train_R2'] = df_metrics['Train_R2 (Mean±STD)'].apply(lambda x: float(str(x).split(' ')[0]))
    #     df_metrics['Val_R2'] = df_metrics['R2']
    #
    #     # sort by validation R2 to make the chart flow
    #     df_metrics = df_metrics.sort_values(by='Val_R2', ascending=True)
    #
    #     # plotting
    #     plt.figure(figsize=(10, 7))
    #
    #     # draw the connecting lines representing the "gap"
    #     for i, row in df_metrics.iterrows():
    #         plt.plot([row['Train_R2'], row['Val_R2']], [row['Subtest'], row['Subtest']],
    #                  color='grey', zorder=1, linestyle='-', alpha=0.5, linewidth=2)
    #
    #     # plot the train and validation dots
    #     plt.scatter(df_metrics['Train_R2'], df_metrics['Subtest'], color='#2C3E50',
    #                 label='Train R² (Mean)', zorder=2, s=120, edgecolors='black')
    #     plt.scatter(df_metrics['Val_R2'], df_metrics['Subtest'], color='#E74C3C',
    #                 label='Validation R² (Pooled OOF)', zorder=3, s=120, edgecolors='black')
    #
    #     plt.title(f'Generalization Gap: Train vs. Validation R² ({model_algo.upper()})', fontsize=18, pad=15)
    #     plt.xlabel('R² Score (Higher is better)', fontsize=14)
    #     plt.ylabel('', fontsize=14)
    #     plt.legend(loc='lower right', framealpha=1.0, fontsize=12)
    #     plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    #
    #     # remove top and right borders
    #     plt.gca().spines['top'].set_visible(False)
    #     plt.gca().spines['right'].set_visible(False)
    #
    #     plt.tight_layout()
    #
    #     # save data
    #     if task_type == 'classification':
    #         model_dir: str = os.path.join(self.classification_res_path, f'{model_algo}_{task_type}')
    #     else:
    #         model_dir: str = os.path.join(self.regression_res_path, f'{model_algo}_{task_type}')
    #
    #     out_path = os.path.join(model_dir, f'{model_algo}_generalization_gap.png')
    #     plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
    #     plt.close()

    # ============================================================================= #
    #                           7) Utility                                          #
    # ============================================================================= #

    def viz_regression_generalization_gap(self, res_dir: str, model_algo: str, task_type: str) -> None:

        # dynamically find all targets evaluated in this folder
        targets = [f.split(f'{model_algo}_predictions_')[1].replace('.csv', '')
                   for f in os.listdir(res_dir) if f.startswith(f'{model_algo}_predictions_')]

        if not targets:
            print(f"No prediction files found for {model_algo}. Cannot plot generalization gap.")
            return

        plt.figure(figsize=(10, 8))

        # store averages for sorting the y-axis later
        plot_data = []

        # extract fold-level data for each target
        for target in targets:
            subtest_name = target.replace('Asymmetry_JT_Ratio_', '')

            train_file = os.path.join(res_dir, f'{model_algo}_fold_train_metrics_{target}.csv')
            pred_file = os.path.join(res_dir, f'{model_algo}_predictions_{target}.csv')

            if not os.path.exists(train_file) or not os.path.exists(pred_file):
                continue

            df_train = pd.read_csv(train_file)
            df_pred = pd.read_csv(pred_file)

            fold_train_r2 = []
            fold_val_r2 = []

            # calculate metrics per fold
            for fold in sorted(df_pred['Fold'].unique()):
                df_fold_preds = df_pred[df_pred['Fold'] == fold]

                # calculate Test R2 for this specific fold
                val_r2 = r2_score(df_fold_preds['Real_Score'], df_fold_preds['Predicted_Score'])
                fold_val_r2.append(val_r2)

                # extract Train R2 for this specific fold
                t_r2 = df_train[df_train['Fold'] == fold]['Train_R2'].values[0]
                fold_train_r2.append(t_r2)

            mean_tr2 = np.mean(fold_train_r2)
            mean_vr2 = np.mean(fold_val_r2)

            plot_data.append({
                'subtest': subtest_name,
                'train_r2_list': fold_train_r2,
                'val_r2_list': fold_val_r2,
                'mean_train_r2': mean_tr2,
                'mean_val_r2': mean_vr2
            })

        # sort by mean validation R2 for a clean visual flow
        plot_data = sorted(plot_data, key=lambda x: x['mean_val_r2'])

        # plotting
        for y_pos, data in enumerate(plot_data):

            # 1) Plot the faint individual folds (alpha=0.3)
            for tr2, vr2 in zip(data['train_r2_list'], data['val_r2_list']):
                plt.plot([tr2, vr2], [y_pos, y_pos], color='grey', alpha=0.3, linewidth=1, zorder=1)
                plt.scatter(tr2, y_pos, color='#2C3E50', alpha=0.3, s=40, zorder=2)
                plt.scatter(vr2, y_pos, color='#E74C3C', alpha=0.3, s=40, zorder=2)

            # 2) Plot the bold average line
            plt.plot([data['mean_train_r2'], data['mean_val_r2']], [y_pos, y_pos],
                     color='black', alpha=0.8, linewidth=2.5, zorder=3)

            # 3) Plot the bold average dots
            plt.scatter(data['mean_train_r2'], y_pos, color='#2C3E50',
                        s=140, edgecolors='black', zorder=4, label='Train R² (Mean)' if y_pos == 0 else "")
            plt.scatter(data['mean_val_r2'], y_pos, color='#E74C3C',
                        s=140, edgecolors='black', zorder=4, label='Test R² (Mean)' if y_pos == 0 else "")

        # formatting
        plt.yticks(range(len(plot_data)), [d['subtest'] for d in plot_data])
        plt.title(f'Generalization Gap by Fold: Train vs. Test R² ({model_algo.upper()})', fontsize=16, pad=15)
        plt.xlabel('R² Score (Higher is better)', fontsize=14)
        plt.legend(loc='upper left', framealpha=1.0, fontsize=12)
        plt.grid(True, axis='x', linestyle=':', alpha=0.6)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()

        # save data
        out_dir = os.path.join(self.regression_res_path if task_type == 'regression' else self.classification_res_path,
                               f'{model_algo}_{task_type}')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{model_algo}_generalization_gap_folds.png'), dpi=600, bbox_inches='tight')
        plt.close()

    def viz_regression_split_distribution(self, df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str,
                                          model_algo: str, task_type: str) -> None:

        plt.figure(figsize=(10, 7))

        # plot training distribution (from the main df target column)
        sns.histplot(df_train[target_col], color='#2C3E50', label='Training Set Pool',
                     stat='density', kde=True, alpha=0.4, bins=15, edgecolor='black')

        # plot test (OOF) distribution (from the OOF results 'Real_Score' column)
        sns.histplot(df_test['Real_Score'], color='#E74C3C', label='Test Set (OOF)',
                     stat='density', kde=True, alpha=0.5, bins=15, edgecolor='black')

        # clean target name for the title
        clean_target = target_col.replace('Asymmetry_JT_Ratio_', '')

        plt.title(f'Target Distribution Match: {clean_target} ({model_algo.upper()})', fontsize=18, pad=15)
        plt.xlabel('JTHFT Asymmetry Ratio', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, axis='both', linestyle=':', alpha=0.6)

        # remove top and right borders
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.tight_layout()

        if task_type == 'classification':
            model_dir: str = os.path.join(self.classification_res_path, f'{model_algo}_{task_type}')
        else:
            model_dir: str = os.path.join(self.regression_res_path, f'{model_algo}_{task_type}')

        f_basename = f'{model_algo}_distribution_check_{target_col}.png'
        out_path = os.path.join(model_dir, f_basename)
        plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    @staticmethod
    def viz_regression_fold_distributions(df_oof: pd.DataFrame, target_col: str, model_algo: str,
                                          out_dir: str) -> None:
        # Verify the Fold column exists
        if 'Fold' not in df_oof.columns:
            print("Fold column missing from OOF data. Cannot plot fold distributions.")
            return

        folds = sorted(df_oof['Fold'].unique())
        n_folds = len(folds)

        # Create a 1x5 grid of subplots
        fig, axes = plt.subplots(1, n_folds, figsize=(4 * n_folds, 5), sharey=True, sharex=True)

        for i, fold in enumerate(folds):
            ax = axes[i]

            # Test data: The patients in THIS fold
            test_reals = df_oof[df_oof['Fold'] == fold]['Real_Score']

            # Train data: The patients in ALL OTHER folds
            train_reals = df_oof[df_oof['Fold'] != fold]['Real_Score']

            # Plot KDEs
            sns.kdeplot(train_reals, ax=ax, color='#2C3E50', fill=True, alpha=0.4, label='Train', linewidth=2)
            sns.kdeplot(test_reals, ax=ax, color='#E74C3C', fill=True, alpha=0.5, label='Test', linewidth=2)

            ax.set_title(f'Fold {fold} (Test n={len(test_reals)})', fontsize=12)
            ax.set_xlabel('JTHFT Asymmetry Ratio', fontsize=12)

            # Formatting
            if i == 0:
                ax.set_ylabel('Density', fontsize=12)
            if i == n_folds - 1:
                ax.legend(loc='upper right')

            ax.grid(True, linestyle=':', alpha=0.6)

        plt.suptitle(f'Train vs. Test Fold Distribution of JTHFT Asymmetry Ratio', fontsize=16, y=1.05)
        sns.despine()
        plt.tight_layout()

        out_path = os.path.join(out_dir, f'{model_algo}_fold_distributions_{target_col}.png')
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()

    @staticmethod
    def viz_regression_learning_curves(log_filepath: str, out_dir: str) -> None:
        """
        Reads the DataLogger JSON file and plots the train vs validation loss curves for each exercise across all folds.
        """
        if not os.path.exists(log_filepath):
            print(f'Log file not found: {log_filepath}')
            return

        with open(log_filepath, 'r') as f:
            data = json.load(f)

        model_algo = data.get('Metadata', {}).get('Algorithm', 'Unknown')
        target = data.get('Metadata', {}).get('Target', 'Unknown')
        folds_data = data.get('Outer_Folds', {})

        # extract unique exercises dynamically
        exercises = set()
        for fold_val in folds_data.values():
            exercises.update(fold_val.get('Exercises', {}).keys())

        exercises = sorted(list(exercises))
        if not exercises:
            print('No learning curve data found in log.')
            return

        # set up a 2x2 grid of subplots with shared axes
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        axes_flat = axes.flatten()

        for i, ax in enumerate(axes_flat):
            # Hide any extra subplots if there are fewer than 4 exercises
            if i >= len(exercises):
                ax.set_visible(False)
                continue

            ex_name = exercises[i]
            all_train = []
            all_val = []
            metric = 'Loss'

            # get arrays from all folds
            for fold_val in folds_data.values():
                lc = fold_val.get('Exercises', {}).get(ex_name, {}).get('Learning_Curve')
                if lc:
                    all_train.append(lc['Train_Loss'])
                    all_val.append(lc['Val_Loss'])
                    metric = lc.get('Metric', 'Loss')

            if not all_train:
                ax.set_title(f'{ex_name} (No Data)')
                continue

            # handle varying tree counts (Optuna picks different n_estimators per fold)
            # forward-fill the last loss value to match the longest fold
            max_len = max(len(arr) for arr in all_train)
            padded_train = [arr + [arr[-1]] * (max_len - len(arr)) for arr in all_train]
            padded_val = [arr + [arr[-1]] * (max_len - len(arr)) for arr in all_val]

            # calculate means
            mean_train = np.mean(padded_train, axis=0)
            mean_val = np.mean(padded_val, axis=0)
            x_axis = range(1, max_len + 1)

            # plot faint individual fold lines
            for t_arr, v_arr in zip(padded_train, padded_val):
                ax.plot(x_axis, t_arr, color='#2C3E50', alpha=0.15, linewidth=1)
                ax.plot(x_axis, v_arr, color='#E74C3C', alpha=0.15, linewidth=1)

            # plot bold mean lines
            ax.plot(x_axis, mean_train, label=f'Train {metric}', color='#2C3E50', linewidth=2.5)
            ax.plot(x_axis, mean_val, label=f'Val {metric}', color='#E74C3C', linewidth=2.5)

            # formatting
            ax.set_title(f'{ex_name}', fontsize=14)

            # Only set X labels for the bottom row (indices 2 and 3)
            if i >= 2:
                ax.set_xlabel('Number of Trees', fontsize=12)

            # Only set Y labels for the left column (indices 0 and 2)
            if i % 2 == 0:
                ax.set_ylabel(f'Loss ({metric})', fontsize=12)

            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle(f'Optimization Learning Curves: JTHFT Asymmetry Ratio', fontsize=16, y=1.02)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f'{model_algo.lower()}_learning_curves_{target}.png')
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Learning curves saved to {out_path}")

    @staticmethod
    def viz_regression_feature_distributions(res_dir: str, model_algo: str, target: str, top_n: int = 10) -> None:
        """
        Plots horizontal boxplots of the raw log-ratios for the Top N features.
        Draws a baseline at 1.0 (Perfect Symmetry) and annotates FDR p-values from the stats table.
        """
        # file paths
        shap_vals_file = os.path.join(res_dir, f'{model_algo}_shap_vals_{target}.csv')
        shap_feats_file = os.path.join(res_dir, f'{model_algo}_shap_feats_{target}.csv')
        stats_csv_file = os.path.join(res_dir, f'{model_algo}_top_{top_n}_feature_statistics_{target}.csv')

        if not os.path.exists(shap_feats_file) or not os.path.exists(stats_csv_file):
            print("Required files not found. Ensure calc_feature_statistics is run first.")
            return

        # load data
        df_shap = pd.read_csv(shap_vals_file)
        df_feat = pd.read_csv(shap_feats_file)
        df_stats = pd.read_csv(stats_csv_file)

        # remove unnecessary columns
        cols_to_drop = ['p_ID', 'Side', 'Fold', 'Target', 'Severity_Class']
        df_shap = df_shap.drop(columns=[c for c in cols_to_drop if c in df_shap.columns])
        df_feat = df_feat.drop(columns=[c for c in cols_to_drop if c in df_feat.columns])

        # get top features
        top_features = df_stats['Feature'].tolist()

        # isolate data and melt for seaborn
        df_top_feats = df_feat[top_features]
        df_melted = df_top_feats.melt(var_name='Feature', value_name='Ratio_Value')

        # plotting setup
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # draw the boxplots
        sns.boxplot(data=df_melted, y='Feature', x='Ratio_Value', order=top_features,
                    color='#6C3BAA', width=0.4, boxprops={'alpha': 0.6},
                    flierprops={'marker': 'o', 'markersize': 5, 'alpha': 1.0})

        ax.set_xlim(left=-0.2, right=5.0)

        # draw the baseline of perfect symmetry (1.0)
        plt.axvline(x=1.0, color='#FF0055', linestyle='--', linewidth=2.5, zorder=0,
                    label='Perfect Symmetry (Ratio = 1.0)')

        # format y-axis labels with FDR p-values
        new_yticklabels = []
        for feat in top_features:
            # extract the p-value for this specific feature
            p_val = df_stats.loc[df_stats['Feature'] == feat, 'FDR_Corrected_p_value'].values[0]

            # formatting logic for the label
            p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"
            sig_marker = "*" if p_val < 0.05 else ""

            # combine feature name and p-value on two lines for a clean look
            new_label = f"{feat}\n(FDR p {p_str}){sig_marker}"
            new_yticklabels.append(new_label)

        # apply the new labels
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(new_yticklabels, fontsize=12)

        # figure layout & formatting
        plt.xlabel('Kinematic Asymmetry Ratio', fontsize=16)
        #plt.title(f'Clinical Distribution of Top {top_n} Predictive Features', fontsize=18, pad=15)
        ax.set_ylabel('')
        # legend for the baseline
        plt.legend(loc='upper right', fontsize=12, framealpha=0.9)

        plt.grid(True, axis='x', linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # save
        out_path = os.path.join(res_dir, f'{model_algo}_top_{top_n}_feature_distributions_{target}.png')
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()

        print(f"Feature distribution boxplots saved to: {out_path}")
