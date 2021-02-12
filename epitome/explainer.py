import random
import os
from epitome.models import *
from epitome.functions import *
from epitome.viz import *
import sys
import numpy as np
import argparse
import lime
from lime import lime_tabular

# TODO: you are getting a probability warning in run().
# It prints everytime you generate an explanation.
import warnings
warnings.filterwarnings("ignore")

# getting an error with tf.function. TODO: should fix this
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class EpitomeExplainer:
    '''
    Uses Lime to explain feature importance for an Epitome model

    To use:
    # create an epitome model, then train  
    > explainer = EpitomeExplainer(model, train_instances = 15000)
    # explain n instances
    > results = explainer.explain(predict_n=7000)
    '''
    
    def __init__(self, model, train_instances = 20000, test_celltype = None):
        
        self.model = model
        
        # remove 1 celltype for test
        if test_celltype is None:
            self.test_celltype = self.model.eval_cell_types[0]
        else:
            assert test_celltype in self.model.eval_celltypes
            self.test_celltype = test_celltype

            
        self.eval_cell_types = self.model.eval_cell_types.copy()
        self.eval_cell_types.remove(self.test_celltype)

        self.train_instances = train_instances
        
        # get feature names from the generator
        self.feature_names = list(load_data(self.model.dataset.get_data(Dataset.TRAIN),
                                                        self.eval_cell_types, # dont train on the held out celltype
                                                        model.eval_cell_types,
                                                        model.dataset.matrix,
                                                        model.dataset.targetmap,
                                                        model.dataset.cellmap,
                                                        radii = model.radii,
                                                        indices = np.arange(0,1),
                                                        return_feature_names = True,
                                                        similarity_targets = model.dataset.similarity_targets, 
                                                        mode = Dataset.TRAIN)())[0][1][0]
        
        # Generate training data for lime
        # Define a new train index for validation so you dont train and validate on 
        # the same instances from the train set
        t_idx = int(model.dataset.get_data(Dataset.TRAIN).shape[1]/2)
        

        ######################## randomly choose indices for the train iterator #######################
        feature_indices = np.concatenate(list(map(lambda c: get_y_indices_for_cell(model.dataset.matrix, model.dataset.cellmap, c),
                                 list(model.dataset.cellmap))))
        feature_indices = feature_indices[feature_indices != -1]

        # need to re-proportion the indices to oversample underrepresented labels
        feature_assays = self.model.dataset.predict_targets

        if (len(list(model.dataset.targetmap)) > 2):
            # configure y: label matrix of ChIP for all assays from all cell lines in train
            indices = np.concatenate([EpitomeDataset.get_y_indices_for_target(model.dataset.matrix, model.dataset.targetmap, assay) for assay in model.dataset.predict_targets])
            indices = indices[indices != -1]
            y = model.data[Dataset.TRAIN][indices, :].T
            m = MLSMOTE(y)
            indices = m.fit_resample()
        else:
                # single TF model
            # get indices for DNAse and chip for this mark
            feature_indices = np.concatenate(list(map(lambda c: get_y_indices_for_cell(model.dataset.matrix, model.dataset.cellmap, c),
                                                 list(model.dataset.cellmap))))

            # chop off targets being used in similarity metric
            not_similarity_indices = np.array([v for k,v in model.dataset.targetmap.items() if k not in model.dataset.similarity_targets])
            TF_indices = feature_indices.reshape([len(model.dataset.cellmap),len(model.dataset.targetmap)])[:,not_similarity_indices]

            TF_indices =  TF_indices[TF_indices != -1]
            feature_indices = feature_indices[feature_indices != -1]

            # sites where TF is bound in at least 2 cell line
            positive_indices = np.where(np.sum(model.dataset.get_data(Dataset.TRAIN)[TF_indices,:], axis=0) > 1)[0]

            indices_probs = np.ones([model.dataset.get_data(Dataset.TRAIN).shape[1]])
            indices_probs[positive_indices] = 0
            indices_probs = indices_probs/np.sum(indices_probs, keepdims=1)

            # randomly select 10 fold sites where TF is not in any cell line
            negative_indices = np.random.choice(np.arange(0,model.dataset.get_data(Dataset.TRAIN).shape[1]),
                                                positive_indices.shape[0] * 10,
                                                p=indices_probs)
            indices = np.sort(np.concatenate([negative_indices, positive_indices]))


        # only use indices > t_idx
        indices = indices[indices > t_idx]
        random.shuffle(indices)
        
        self.train_generator = load_data(self.model.dataset.get_data(Dataset.TRAIN),
                                                                self.eval_cell_types,
                                                                model.eval_cell_types,
                                                                model.dataset.matrix,
                                                                model.dataset.targetmap,
                                                                model.dataset.cellmap,
                                                                indices = indices, 
                                                                similarity_targets = model.dataset.similarity_targets, 
                                                                radii = model.radii, mode = Dataset.TRAIN)()
        
        self.test_generator = load_data(self.model.dataset.get_data(Dataset.TRAIN),
                                                                [self.test_celltype],
                                                                model.eval_cell_types,
                                                                model.dataset.matrix,
                                                                model.dataset.targetmap,
                                                                model.dataset.cellmap,
                                                                indices = indices, 
                                                                similarity_targets = model.dataset.similarity_targets, 
                                                                radii = model.radii, mode = Dataset.TRAIN)()
        
        train_data = np.array([np.concatenate(next(self.train_generator)[0:-2]) for _ in range(self.train_instances)])

        # create the explainer
        self.explainer = lime_tabular.LimeTabularExplainer(train_data,
                                                      feature_names=self.feature_names, 
                                                      class_names=self.model.dataset.predict_targets, 
                                                      discretize_continuous=False)



    def explain_region_i(self, arr):
        """
        Explain a single vector.
        """

        def predict_fn(arr):
            return self.model._predict(arr).numpy()


        explanation = self.explainer.explain_instance(arr, 
                                                 predict_fn,
                                                 num_samples=1000, 
                                                 num_features=arr.shape[-1], top_labels=len(self.feature_names))

        return explanation

    def exp_to_matrix(self, exp):
        """ Converts explanation into a 2D matrix where rows are labels and 
        columns are features.

        :param exp: single lime explained instance.
        """
        # return matrix of cell line assays by predicted assay

        # sort by feature name and stack 
        # label rows are ordered by label_names()
        # need to select exp output by name, otherwise the ordering is with respect to importance!
        d = dict(exp.as_list(label=0))
        return np.array([d[i] for i in self.feature_names])

    def explain(self, predict_n=1000, explain_n=20):
        '''
        Get explanation for n instances.
        This takes a REALLY long time. 

        :param n: number of instances to run
        :return: dict (Epitome predictions, explanations, numpy explanations, features)
        :rtype: dict
        '''

        # tested features
        features = []
        truth=[]
        # explanation results in numpy form
        explanation_values = []
        explanations = []
        # Epitome predictions
        predictions = []


        # generate features
        for i in range(predict_n):
            tmp = next(self.test_generator)
            features.append(tmp[0])
            truth.append(tmp[1])

        predictions = self.model._predict(np.vstack(features)).numpy().flatten()
        truth = np.array(truth).flatten()

        # indices of true positives. just take a couple
        tp = np.where((truth == 1) & (predictions >0.5))[0]
        tn = np.where((truth == 0) & (predictions <0.1))[0]
        fp = np.where((truth == 0) & (predictions >0.5))[0]
        fn = np.where((truth == 1) & (predictions <0.1))[0]
        
        print("True positives: %s\nTrue negatives: %i\nFalse positives: %i\nFalse negatives: %i" % (tp.shape[0], tn.shape[0], fp.shape[0], fn.shape[0]))
        
        tp = tp[:explain_n]
        tn = tn[:explain_n]
        fp = fp[:explain_n]
        fn = fn[:explain_n]

        indices = np.concatenate([tp, tn, fp, fn])
        
        for idx in indices:

            v = features[idx]

            exp = self.explain_region_i(v)
            explanations.append(exp)
            # get matrix
            m = self.exp_to_matrix(exp)
            explanation_values.append(m)



        # reshape matrices
        np_predictions = np.array(predictions).squeeze()
        np_explanations = np.array(explanation_values)
        np_features = np.array(features).squeeze()
        
        return {"preds": np_predictions[indices], 
                "truth": truth[indices],
                "lime_exp": explanations, 
                "exp": np_explanations,
                "features": np_features[indices]}



