# src/visualization.py

# libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

# modules
from src.config import config, project_path


class Visualizer:
    def __init__(self) -> None:
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

    # ============================================================================= #
    #                            2) TEMPORAL CONSISTENCY                            #
    # ============================================================================= #
    def vis_consistency_raincloud(self, link_df: pd.DataFrame, columns_to_plot: list[str],
                                   title: str = '', x_label: str = '', y_label: str = '') -> None:

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

    def viz_comparison_subplots(self, finger_df, finger_order: list[str]) -> None:

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1]})

        # subplot 1: grouped boxplot
        sns.boxplot(data=finger_df,
                    x='Finger',
                    y='CoV',
                    hue='Side',
                    ax=axes[0],
                    palette='Set2',
                    order=finger_order)

        axes[0].set_title('Comparison: Healthy vs. Affected Side')
        axes[0].set_ylabel('Mean CoV (%)')

        # subplot 2: raincloud of affected side - violin/strip combination
        df_aff = finger_df[finger_df['Side'] == 'Affected']
        sns.violinplot(
            data=df_aff, x='Finger', y='CoV',
            ax=axes[1], inner=None, color=".8", order=['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        )
        sns.stripplot(
            data=df_aff, x='Finger', y='CoV',
            ax=axes[1], alpha=0.4, order=['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        )
        axes[1].set_title('Distribution of Affected Side Stability')
        axes[1].set_ylabel('Mean CoV (%)')

        plt.tight_layout()

        suffix = '.png'
        f_name: str = 'Distribution_of_Affected_Side_Stability' + suffix
        f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600)

        plt.close()

    def viz_comparison_boxplot(self, finger_df: pd.DataFrame, finger_order: list[str]) -> None:

        plt.figure(figsize=(12, 6))

        # custom palette for the four states
        custom_palette = {
            'Healthy (Active)': '#1f77b4',  # dark blue
            'Healthy (Passive)': '#aec7e8', # light blue
            'Affected (Active)': '#d62728', # dark red
            'Affected (Passive)': '#ff9896' # light red
        }

        # order of the legend and bars
        hue_order = ['Healthy (Active)', 'Healthy (Passive)', 'Affected (Active)', 'Affected (Passive)']

        sns.boxplot(data=finger_df,
                    x='Finger',
                    y='CoV',
                    hue='State',
                    palette=custom_palette,
                    hue_order=hue_order,
                    order=finger_order)

        plt.title('Temporal Consistency: Active vs. Passive Hand', fontsize=14)
        plt.ylabel('Mean CoV (%)', fontsize=12)
        plt.xlabel('Finger', fontsize=12)

        # legend placed outside the plot to prevent it from covering data
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Hand State')

        plt.yticks(np.arange(0, 100 + 10, 10))

        plt.tight_layout()

        suffix = '.png'
        f_name: str = '4-Way_Consistency_Boxplot' + suffix
        f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600)

        plt.close()

    def viz_normality_qq_plots(self, paired_df: pd.DataFrame, finger_order: list[str]) -> None:

        # create a 3x2 grid
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))

        # flatten the 2D array so we can use a single index 'i'
        axes_flat = axes.flatten()

        for i, finger in enumerate(finger_order):
            # select the specific subplot for this finger
            ax = axes_flat[i]

            finger_data = paired_df[paired_df['Finger'] == finger]
            differences = (finger_data['Healthy'] - finger_data['Affected']).dropna()
            print(f'N: {len(differences)}')
            if len(differences) >= 3:
                _, p_val = stats.shapiro(differences)
                norm_text = f"Shapiro p={p_val:.3f}"
                stats.probplot(differences, dist="norm", plot=ax)
            else:
                norm_text = "N too small"
                ax.text(0.5, 0.5, 'Insufficient Data', ha='center')

            ax.set_title(f'Q-Q Plot: {finger}\n{norm_text}')

            # formatting for a 2-column layout
            ax.set_ylabel('Sample Quantiles' if i % 2 == 0 else '')
            ax.set_xlabel('Theoretical Quantiles')

        # hide the 6th empty subplot
        if len(finger_order) < len(axes_flat):
            axes_flat[-1].axis('off')

        plt.tight_layout()

        suffix = '.png'
        f_name: str = 'QQ_Plots' + suffix
        f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600)

        plt.close()

    def viz_lmm_interaction_trajectories(self, finger_df: pd.DataFrame) -> None:
        """
        Creates a 'Spaghetti Plot' showing individual participant trajectories
        from Passive to Active states, split by Healthy vs. Affected hands,
        overlaid with the group mean to visualize the LMM interaction effect.
        """

        # aggregate the data: get the mean CoV per participant for each state (ignoring specific fingers)
        agg_df = finger_df.groupby(['Participant', 'Hand_Condition', 'Hand_Role'])['CoV'].mean().reset_index()

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
            # Get ALL participants for this specific condition
            subset = agg_df[agg_df['Hand_Condition'] == condition]

            sns.lineplot(data=subset, x="Hand_Role", y="CoV", estimator='mean', errorbar=None, color="black",
                         linewidth=3, marker="D", markersize=8, zorder=10, legend=False, ax=ax)

        # formatting
        face_grid.set_axis_labels('Hand Role (State of Movement)', 'Mean CoV (%)', fontsize=12)
        face_grid.set_titles(col_template="{col_name} Hand", size=14, weight='bold')

        # adjust y-axis to start at 0
        face_grid.set(ylim=(0, None))

        # add the legend outside the plot
        face_grid.add_legend(title="Participant ID", bbox_to_anchor=(1.02, 0.5), loc='center left')

        plt.subplots_adjust(top=0.88)
        face_grid.figure.suptitle('Impact of Movement: Individual Trajectories (Rest vs. Active)',
                                  fontsize=16, weight='bold')

        # Save the plot
        suffix = '.png'
        f_name: str = 'LMM_Interaction_Trajectories' + suffix
        f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600, bbox_inches='tight')

        plt.close()

    def viz_lmm_normality_and_homoscedasticity(self, model) -> None:

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

        # Save the plot
        suffix = '.png'
        f_name: str = 'LMM_Assumption_Tests' + suffix
        f_path: str = os.path.join(self.temp_consistency_res_path, f_name)
        if not os.path.exists(f_path):
            plt.savefig(f_path, format=suffix[1:], dpi=600, bbox_inches='tight')

        plt.close()

    # ============================================================================= #
    #                            3) PARAMETER EXTRACTION                            #
    # ============================================================================= #

    # ============================================================================= #
    #                            4) HELPER FUNCTION                                 #
    # ============================================================================= #
