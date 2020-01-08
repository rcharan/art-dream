import datetime
import functools

import numpy as np
import pandas as pd
import seaborn as sns

import os
from joblib import dump, load


################################################################################
#
# Part 1 : Some simple convenience functions
#
################################################################################

def compose(*funcs):
    outer = funcs[:-1][::-1] # Python only provides left folds
    def composed_func(*args, **kwargs):
        inner = funcs[-1](*args, **kwargs)
        return functools.reduce(lambda val, func : func(val), outer, inner)
    return composed_func

# Aliases for filters and maps
lfilter        = compose(list, filter)
lmap           = compose(list, map)
afilter        = compose(np.array, list, filter)
filternull     = functools.partial(filter, None)
lfilternull    = compose(list, filternull)
filternullmap  = compose(filternull, map)
lfilternullmap = compose(lfilternull, map)
def lmap_filter(map_, predicate_):
    def f(list_):
        return lmap(map_, filter(predicate_, list_))
    return f

def list_diff(a, b):
    return list(set(a).difference(b))

def print_dict(d):
    key_len = max(map(len, d.keys()))
    for k, v in d.items():
        print(f'{k.ljust(key_len)} : {v}')

def drop_col(df, *cols):
    df.drop(columns = list(cols), inplace = True)

################################################################################
#
# Part 2: Plotting
#
################################################################################

# Convenience to deal with turning off interactive mode
def plot(fn, *args, **kwargs):
    if 'figsize' in kwargs:
        fig, ax = plt.subplots(figsize = kwargs['figsize'])
        del kwargs['figsize']
    else:
        fig, ax = plt.subplots()
    kwargs['ax'] = ax
    fn(*args, **kwargs)
    return fig

# Minimal implementation to deal with use cases encountered
#  does not implement the full seaborn API
# For simplicity, pass a list of series with the data to be counted
def stacked_countplot(*series, normalize = False, dropna = False):
    df   = pd.concat(map(lambda s : s.value_counts(normalize = normalize,
                                                   dropna    = dropna),
                         series),
                     axis = 'columns',
                     sort = True
                    ).fillna(0)
    temp = df.sort_values(df.columns[0]).values.cumsum(axis = 0)
    df = pd.DataFrame(temp, index = df.index, columns=df.columns
                     ).sort_values(df.columns[0], ascending = False)

    fig, ax = plt.subplots()
    cmp = sns.color_palette('muted', n_colors = 9)

    for i in range(len(df)):
        sns.barplot(x = 'variable', y = 'value',
                    data = df.iloc[[i]].melt(),
                    ax = ax,
                    label = df.index[i],
                    color = cmp.pop())
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_xlabel('')
    if normalize:
        ax.set_ylabel('Frequency')
    else:
        ax.set_ylabel('Count')
    return fig, ax

def rotate_labels(axes):
  labels = axes.get_xticklabels()
  for l in labels:
    l.set_rotation('vertical')


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

################################################################################
#
# Part 2 : Timing
#
################################################################################

# Super simple timer
#  Timing implemented as class methods
#  to avoid having to instantiate
class Timer:

    @classmethod
    def start(cls):
        cls.start_time = datetime.datetime.now()

    @classmethod
    def end(cls):
        delta     = datetime.datetime.now() - cls.start_time
        sec       = delta.seconds
        ms        = delta.microseconds // 1000
        cls.time  = f'{sec}.{ms}'
        print(f'{sec}.{ms} seconds elapsed')


################################################################################
#
# Part 3 : Persistence and timing conveniences
#
################################################################################

def persist(obj_name, model, method, *data, task = 'model', force_fit = False, **kwargs):
    '''
        Persist a call (e.g. fit, fit_transform, score)

        Attempts to load from disk, otherwise makes the call and saves it

        task: either model, data, or both
    '''
    file_name = f'./models/{obj_name}.joblib'

    if os.path.exists(file_name) and not force_fit:
        print(f'Loading {obj_name} from disk')
        obj      = load(file_name)
        job_time = get_fit_time(obj_name)
        return obj

    else:
        assert task in ['model', 'data', 'both']
        print(f'Running {method}')
        Timer.start()
        return_val = model.__getattribute__(method)(*data, **kwargs)
        Timer.end()
        if task == 'model':
            out = model
        elif task == 'data':
            out = return_val
        elif task == 'both':
            out = (return_val, model)

        dump(out, file_name)
        write_fit_time(obj_name, Timer.time)
        return out


fit_time_fname = './models/fit_times.joblib'

def _fit_time_interface(model_name, write = None, verbose = True):
    if os.path.exists(fit_time_fname):
        fit_time_dict = load(fit_time_fname)
    else:
        if not write:
            print('No job time info found at all')
            return None
        fit_time_dict = {}

    if write:
        fit_time_dict[model_name] = write
        dump(fit_time_dict, fit_time_fname)
    else:
        if model_name in fit_time_dict:
            job_time = fit_time_dict[model_name]
            if verbose:
                print(f'{job_time} seconds elapsed in original job')
            return job_time
        else:
            print(f'No job time found for {model_name}')
            return None

def get_fit_time(model_name, verbose = True):
    return _fit_time_interface(model_name, verbose = verbose)

def write_fit_time(model_name, fit_time):
    return _fit_time_interface(model_name, fit_time)



def get_learner(pipe_):
    try:
        return pipe_.named_steps['learn']
    except:
        return pipe_.steps[-1][1]

def get_best_learner(pipe_):
    return get_learner(pipe_).best_estimator_

def get_best_params(pipe_):
    return get_best_learner(pipe_).get_params()

import sklearn.model_selection as cv
def print_best_params(model):
    if isinstance(model, cv.GridSearchCV):
        dict_ =  model.best_estimator_.get_params()
        dict_ = {k : v for k, v in dict_.items() if k in model.param_grid}
        return print_dict(dict_)
    else:
        return print_dict(get_best_params(model))

import scipy.stats as st
def cv_results(grid_search, p_thresh = 0.01):
    df = pd.DataFrame(grid_search.cv_results_)

    # Compute the statistical significance of the deviation from the leader
    nobs = grid_search.cv.get_n_splits()

    df.sort_values('rank_test_score', inplace = True)
    # df = df[cols].sort_values('rank_test_score')


    @np.vectorize
    def get_tvalue(index):
        scores = df.iloc[[0, index]].loc[:,'split0_test_score' : f'split{nobs-1}_test_score']
        a = scores.transpose().iloc[:,0]
        b = scores.transpose().iloc[:,1]
        p = st.ttest_rel(a, b).pvalue
        return p < p_thresh

    temp = pd.Series(get_tvalue(range(len(df))), index = df.index, name = 'statistically_different')

    if isinstance(grid_search.param_grid, list):
        keys = list(set(sum((list(d.keys()) for d in grid_search.param_grid), [])))
    else:
        keys = grid_search.param_grid.keys()
    param_cols = lmap(lambda k : f'param_{k}', keys)
    cols = param_cols + ['mean_test_score', 'std_test_score', 'rank_test_score', 'statistically_different']

    return pd.concat([df, temp], axis = 'columns')[cols]

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



from math import ceil, sqrt
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


import sklearn.metrics as metr

def plot_multiclass_roc_curve_from_dict(model_dict, y_true, f1_level_curves = None):
    return plot_multiclass_roc_curve(
        model_dict['predicted_probabilities'], model_dict['model'].classes_,
        y_true, model_dict['predictions'], f1_level_curves

    )

def plot_multiclass_roc_curve(pred_scores, class_names, y_true,
                              y_pred,
                              f1_level_curves = None):
    df  = pd.DataFrame(pred_scores, columns = class_names)
    fp  = FeaturePlot(df, axsize = 6)
    for col, scores, ax in fp:
        bin_true = (y_true == col).astype(int)
        bin_pred = (y_pred == col).astype(int)
        fpr, tpr, _ = metr.roc_curve(bin_true, scores)
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
        report = metr.classification_report(bin_true, bin_pred, output_dict=True)
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
