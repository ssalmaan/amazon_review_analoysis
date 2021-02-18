import seaborn as sns
from matplotlib import pyplot as plt
from load_data import *

def get_label(column):
    """
    Get display label for visualisation of some dataframe column name
    """
    try:
        if '_' in column:
            field, agg = column.split('_')
            return f'{agg.title()} {DISPLAY_NAMES_DICT[field]}'
        return DISPLAY_NAMES_DICT[column]
    except KeyError:
        return column


def plot_histogram(filename, df, column, title=None, log=False, max_samples=None,
                   fill=False, bins='auto'):
    """
    Plot histogram from a dataframe column and optionally save the plot
    """
    if max_samples is not None:
        df = df.sample(min(len(df.index), max_samples))

    ax = sns.histplot(data=df,
                      x=column,
                      fill=fill,
                      bins=bins
                      )
    if log:
        ax.set_yscale("log")

    xlabel = get_label(column)
    plt.xlabel(xlabel)

    if title is not None:
        plt.title(title)
    else:
        plt.title(f'Distribution : {xlabel}')

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


def print_correlation(df, x, y, x_name, y_name):
    """
    Print correlation between 2 columns of a dataframe
    """
    df = df[df[x].notna() & df[x].notna()]
    correlation = round(100 * df[x].corr(df[y]), 1)
    print(f"Correlation between {x_name} and {y_name} = {correlation}%")


def joint_plot(filename, df, x, y, max_samples=None, alpha=0.5, xlim=None, ylim=None):
    """
    Plot scatterplot from a dataframe column pair and optionally save the plot
    """
    if max_samples is not None:
        df = df.sample(min(len(df.index), max_samples))

    g = sns.JointGrid()
    x, y = df[x], df[y]
    sns.scatterplot(x=x,
                    y=y,
                    alpha=alpha,
                    marker='+',
                    ax=g.ax_joint)

    sns.kdeplot(x=x, ax=g.ax_marg_x)
    sns.kdeplot(y=y, ax=g.ax_marg_y)

    xlabel = get_label(x.name)
    ylabel = get_label(y.name)

    g.set_axis_labels(xlabel, ylabel)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
