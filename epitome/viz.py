#####################################################################
################### Visualization functions #########################
#####################################################################

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends import backend_agg
from matplotlib import figure


def joint_plot(dict_model1, 
               dict_model2, 
               metric = "AUC", 
               model1_name = "model1", 
               model2_name = "model2",
               outlier_filter = None):
    """
    Returns seaborn joint plot of two models.
    
    :param dict_model1: dictionary of TF: metrics for first model. Output from run_predictions().
    :param dict_model2: dictionary of TF: metrics for second model. Output from run_predictions().
    :param metric: metric in dicts. Should be auPRC, AUC, or GINI.
    :param model1_name: string for model 1 name, shown on x axis.
    :param model2_name: string for model 2 name, shown on y axis.
    :param outlier_filter: string filter to label. Defaults to no labels.
    """
    df1 = pd.DataFrame.from_dict(dict_model1).T
    df2 = pd.DataFrame.from_dict(dict_model2).T

    dataframe_joined = pd.concat([df1, df2], axis=1, sort=False)
    # select metric
    dataframe = dataframe_joined[metric]
    dataframe.columns=[model1_name, model2_name]

    ax = sns.jointplot(model1_name, model2_name, data=dataframe, kind="reg", color='k', stat_func=None)
    ax.ax_joint.cla()
    ax.set_axis_labels(model1_name, model2_name)
    ax.fig.suptitle(metric) 

    def ann(row):
        ind = row[0]
        r = row[1]
        plt.gca().annotate(ind, xy=(r[model1_name], r[model2_name]), 
                xytext=(2,2) , textcoords ="offset points", )
        
    # filter for labels
    if (outlier_filter is not None):
        label_df = dataframe.query(outlier_filter)

        for index, row in label_df.iterrows():
            if (not np.isnan(row[model1_name]) and not np.isnan(row[model2_name])):
                ann((index, row))

    for i,row in dataframe.iterrows():
        color = "blue" if row[model1_name] <  row[model2_name] else "red"
        ax.ax_joint.plot(row[model1_name], row[model2_name], color=color, marker='o')
        
    return ax


def plot_assay_heatmap(assaymap, matrix, cellmap):
    nv_assaymap = {v: k for k, v in assaymap.items()}

    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.xticks(np.arange(len(assaymap)), rotation = 90)
    ax.set_xticklabels(assaymap.keys())
    plt.yticks(np.arange(len(cellmap)))
    ax.set_yticklabels(cellmap.keys())

    plt.imshow(matrix!=-1)
    
    
    
    
#####################################################################    
############## Network visualization helper functions ###############
#####################################################################

def number_to_bp(n):
    """ converts bp number to short string
    :param n: number in base pairs
    :return string of number with kbp or Mbp suffix
    """
    n = str(n)
    
    if len(n) < 4:
        return n
    elif len(n) == 4:
        return "%skbp" % n[0]
    elif len(n) == 5:
        return "%skbp" % n[0:1]
    elif len(n) == 6:
        return "%skbp" % n[0:2]    
    elif len(n) == 7:
        return "%sMbp" % n[0]     
    else:
        raise
        
        
        
def heatmap_aggreement_from_model_weights(model):
    """ 
    Plots seaborn heatmap for DNase weights of first layer in network.
    Plots one heatmap for each celltype used in the features for training (about 10-13).
    
    :param model: an Epitome model
    """
    
    # get weights
    with model.graph.as_default():
        with tf.variable_scope('layer0', reuse=True):
            w = tf.get_variable('kernel')
            weights = model.sess.run(w)
            
    dnases = weights[len(model.assaymap) * (len(model.cellmap) - 1 - len(model.test_celltypes)): , :]

    num_radii = 2*len(model.radii)

    xtick_labels = [x * 200 for x in model.radii]
    xtick_labels = [number_to_bp(x) for x in xtick_labels]

    for i in range((len(model.cellmap) - 1 - len(model.test_celltypes))):
        # dnases are just at the end
        dnases = x[len(model.assaymap) * (len(model.cellmap) - 2): , :]

        heatmap = dnases[i*num_radii + int(num_radii/2):i*num_radii + num_radii, :].T
        ax = sns.heatmap(heatmap, cmap='bwr_r', vmin=np.min(dnases), vmax=np.max(dnases))
        ax.set_xlabel("Distance from peak (bp)")
        ax.set_ylabel("Unit")
        ax.set_xticklabels(xtick_labels, rotation=-60)
        plt.show()
        
        

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

def calibration_plot(truth, preds, assay_dict, list_assaymap):
    """
    Creates an xy scatter plot for predicted probability vs true probability.
    Adds a separate set of points for each transcription factor. 
    
    Args:
        :param truth: matrix of n samples by t TFs
        :param preds: matrix same size as truth
        :param assay_dict: dictionary of TFs and scores
        :param: list_assaymap: list of assay names for data matrix
    """

    fig, ax = plt.subplots()
    # only these two lines are calibration curves
    for i in range(truth.shape[1]):
        logreg_y, logreg_x = calibration_curve(truth[:,i], preds[:,i], n_bins=10)

        if (not np.isnan(assay_dict[list_assaymap[i+1]]["AUC"])): # test cell type does not have all factors!
            plt.plot(logreg_x,logreg_y, marker='o', linewidth=1, label=list_assaymap[i+1])


    # reference line, legends, and axis labels
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle('Calibration plot for test regions')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability in each bin')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              ncol=2, fancybox=True, shadow=True)

    plt.show()
    
    
    
    
############################################################################
###################### Visualizing model internals #########################
############################################################################

def plot_weight_posteriors(names, qm_vals, qs_vals, fname=None):
    """Save a PNG plot with histograms of weight means and stddevs.
    From https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py
    Args:
    names: A Python `iterable` of `str` variable names.
    qm_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
    """
    fig = figure.Figure(figsize=(6, 3))
    canvas = backend_agg.FigureCanvasAgg(fig)

    ax = fig.add_subplot(1, 2, 1)
    for n, qm in zip(names, qm_vals):
        sns.distplot(tf.reshape(qm, [-1]), ax=ax, label=n)
    ax.set_title("weight means")
    ax.set_xlim([-1.5, 1.5])
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    for n, qs in zip(names, qs_vals):
        sns.distplot(tf.reshape(qs, [-1]), ax=ax)
    ax.set_title("weight stddevs")
    ax.set_xlim([0, 0.2])

    fig.tight_layout()
    if fname != None:
        canvas.print_figure(fname, format="png")
        print("saved {}".format(fname))
        
    fig.show()
