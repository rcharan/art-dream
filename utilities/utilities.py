import seaborn           as sns
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from datetime        import datetime
from math            import ceil, sqrt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report
)

from dream_image    import DreamImage
from deep_dream     import DeepDream
from images         import (
    show_image,
    load_image,
    vgg19_process_image,
    vgg19_deprocess_image
)
from progress_bar import ProgressBar

################################################################################
#
# Table of Contents
#
#  Part 1: Timing
#  Part 2: Plotting, especially for classification tasks
#
################################################################################

################################################################################
#
# Part 1 : Super Simple Timer
#
################################################################################

# Super simple timer
#  Timing implemented as class methods
#  to avoid having to instantiate
class Timer:

    @classmethod
    def start(cls):
        cls.start_time = datetime.now()

    @classmethod
    def end(cls):
        delta     = datetime.now() - cls.start_time
        sec       = delta.seconds
        ms        = delta.microseconds // 1000
        cls.time  = f'{sec}.{ms}'
        print(f'{sec}.{ms} seconds elapsed')


################################################################################
#
# Part 2: Plotting conveniences primarily for classification tasks
#  - Confusion Matrix
#  - Correlation Matrix
#  - ROC Curve
#  - Multi-class ROC curve (experimental)
#  - FeaturePlot (mutli-plot plotting utility)
#
################################################################################


# Slight modification of sklearn's version to avoid re-predicting
#  See documentation there
def plot_confusion_matrix(estimator, y_pred, y_true, labels=None,
                          sample_weight=None, normalize=None,
                          display_labels=None, include_values=True,
                          xticks_rotation='vertical',
                          values_format=None,
                          cmap=sns.cubehelix_palette(light=1, as_cmap=True),
                          ax=None):
    if normalize not in {'true', 'pred', 'all', None}:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
                          labels=labels, normalize=normalize)

    if display_labels is None:
        if labels is None:
            display_labels = estimator.classes_
        else:
            display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    return disp.plot(include_values=include_values,
                     cmap=cmap, ax=ax, xticks_rotation=xticks_rotation)


# From the seaborn gallery
def plot_correlation_matrix(corr):

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=0, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax = ax)

    return fig

# From FangFang Lee
def plot_roc_curve(fpr, tpr, ax = None):
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots()
    else:
        return_fig = False
    ax.plot(fpr, tpr, label = 'ROC')
    xs = np.linspace(0, 1, len(fpr))
    ax.plot(xs, xs, label = 'Diagonal')
    ax.set_xlim([-0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.grid(True)
    ax.set_aspect(1)
    ax.legend()

    if return_fig:
        return fig

def plot_multiclass_roc_curve(pred_scores, class_names, y_true,
                              y_pred,
                              f1_level_curves = None):
    df  = pd.DataFrame(pred_scores, columns = class_names)
    fp  = FeaturePlot(df, axsize = 6)
    for col, scores, ax in fp:
        bin_true = (y_true == col).astype(int)
        bin_pred = (y_pred == col).astype(int)
        fpr, tpr, _ = roc_curve(bin_true, scores)
        ax.plot(fpr, tpr, label = 'ROC', linewidth = 5)
        xs = np.linspace(0, 1, len(fpr))
        ax.plot(xs, xs, label = 'Diagonal', linewidth = 5)
        offset = 0.02
        ax.set_xlim([-offset, 1+offset])
        ax.set_ylim([-offset, 1+offset])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_title(col.title())
        # ax.set_xlabel('False Positive Rate (1 - Specificity)')
        # ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.grid(True)
        ax.set_aspect(1)
        # ax.legend()

        # Plot Actual Location
        report = classification_report(bin_true, bin_pred, output_dict=True)
        tpr_actual =     report['1']['recall']
        fpr_actual = 1 - report['0']['recall']

        # Plot f1 level curves
        if f1_level_curves:
            class_prevalence = report['1']['support']/len(y_true)
            def prevalence_factor(class_prevalence):
                return (1-class_prevalence)/class_prevalence
            def f_factor(f):
                return f/(2-f)
            xs = np.linspace(-offset,1+offset, len(fpr))
            def get_level_curve(f):
                return f_factor(f) + \
                       f_factor(f) * prevalence_factor(class_prevalence) * xs
            for f in f1_level_curves:
                ax.plot(xs, get_level_curve(f), label = f'f1 = {f}')

        ax.plot([fpr_actual], [tpr_actual], marker='o', markersize=10, color="red", label = 'actual')

    list(fp.axes.values())[-1].legend(bbox_to_anchor=(1.05, 1))
    fp.fig.suptitle('ROC Curves for each Class\n with f1-score level curves')


    return fp.fig


class FeaturePlot:
    '''
        Manages a figure containing plots of many unrelated variables
        that would be unsuitable for a FacetGrid
        To use: this is an iterable that will yield (col_name, data, axis)
        for each variable it contains. For overlays, call overlay
    '''
    def __init__(self, *data, axsize = 4):
        self.data     = pd.concat(data, axis = 'columns')
        self.columns  = self.data.columns
        self.num_cols = len(self.columns)
        self._make_figure(axsize)

    def clone(self):
        return FeaturePlot(self.data)

    def _make_figure(self, axsize):
        '''
           Makes the main figure
        '''

        # Compute the size and get fig, axes
        s = ceil(sqrt(self.num_cols))
        fig, axes = plt.subplots(s, s, figsize = (axsize*s, axsize*s));
        axes = axes.ravel()

        # Delete excess axes
        to_delete = axes[self.num_cols:]
        for ax in to_delete:
            ax.remove()

        # Retain references
        self.fig  = fig
        self.axes = dict(zip(self.columns, axes))

        # Add titles
        for col, ax in self.axes.items():
            ax.set_title(col)

        self.grid_size = s

    def overlay(self, label, sharex = False, sharey = False):
        '''
            Adds a new layer of axes on top of an existing figure

            - Is a generator in similar style to self.__iter__ below.
            - A reference to the newly created axes is not maintained
                 by the class - the axes are intended to be single use.
                 If you want to access the axes later, either use the
                 matplotlib figure object or retain a reference
        '''
        for index, col in enumerate(self.columns):
            base_ax = self.axes[col]
            ax = self.fig.add_subplot(self.grid_size, self.grid_size, index + 1,
                                      sharex = base_ax if sharex else None,
                                      sharey = base_ax if sharey else None,
                                      label  = label,
                                      facecolor = 'none')

            for a in [ax, base_ax]:
                if not sharex:
                    a.tick_params(bottom = False,
                                  top = False,
                                  labelbottom = False,
                                  labeltop    = False)
                if not sharey:
                    a.tick_params(left = False,
                                  right = False,
                                  labelleft = False,
                                  labelright = False)


            yield col, self.data[col].values, ax

    def __iter__(self):
        for col in self.columns:
            yield col, self.data[col].values, self.axes[col]

################################################################################
#
# Part 4 : Dreaming
#
################################################################################

# Re-sampling
# big = skimage.transform.pyramids.pyramid_expand(dream_img.numpy(), multichannel=True)
# big = tf.convert_to_tensor(big)
# show_image(postprocess_image(big))
