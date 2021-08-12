"""
Functions for explaining classifiers that use Image data.
"""
import copy
from functools import partial

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm
import random
import torch
from fps import farthestPointSampling


from . import lime_base

def cal_dis(dataset,center):
    center_mat = np.tile(center,(dataset.shape[0],1))
    distance_mat = np.square(dataset[:,0] - center_mat[:,0])+np.square(dataset[:,1] - center_mat[:,1])+np.square(dataset[:,2] - center_mat[:,2])
    return distance_mat
                        

def kmeans(dataset, num_clusters, max_iter, random_seed = None):
    #Set initial center
    num_points = dataset.shape[0]
    if random_seed != None:
        random.seed(random_seed)
    if num_clusters < 200:
        centers_idx = farthestPointSampling(dataset,num_clusters)
    else:
        centers_idx = random.sample(range(0,num_points),num_clusters)
    iteration = 0
    while(1):
        dis_mat = []
        for i in range(num_clusters):
            center_tmp = np.expand_dims(dataset[centers_idx[i]],0) #dataset[centers_idx[i]]->(3,)
            dis = cal_dis(dataset,center_tmp)
            dis_mat.append(dis)
        dis_mat = np.array(dis_mat)
        min_dis = np.argmin(dis_mat,axis=0)
        if iteration == max_iter:
            return min_dis
        new_center = []
        for c in range(num_clusters):
            c_index = np.where(min_dis==c)
            cur_data = dataset[c_index]
            mean_data = np.mean(cur_data,axis=0)
            new_center.append(np.argmin(cal_dis(dataset,np.expand_dims(mean_data,0))))
        centers_idx = new_center
        iteration += 1

class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask


class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, num_cluster=20,kernel_width=0.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
# =============================================================================
#         if kernel_width is None:
#             kernel_width = (1 / np.sqrt(num_cluster)) * .75
# =============================================================================
        kernel_width = float(kernel_width)
        
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,),
                         top_labels=5, num_features=50, num_samples=1000,
                         batch_size=1,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
# =============================================================================
#         if len(image.shape) == 2:
#             image = gray2rgb(image)
# =============================================================================
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        segments = kmeans(image,num_clusters=num_features,max_iter=10,random_seed=0)
        
        top = labels

        data, labels = self.data_labels(image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size)
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
            
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred, ret_exp.mean_diff, ret_exp.weighted_mean_diff, ret_exp.L1_loss, ret_exp.weighted_L1_loss, ret_exp.L2_loss, ret_exp.weighted_L2_loss, ret_exp.adjusted_R2) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=1):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0] #20
        
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        show = 1
        for row in tqdm(data): #100
            print("\rProcessing number ", show, "perturbation instance ...",end="")
            show += 1
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0] #Which segments to mask
            mask = np.zeros(segments.shape).astype(bool)    #1024-lenth bool matrix
            for z in zeros:
                mask[segments == z] = True
            temp = np.delete(temp,mask==True,0)
            imgs.append(temp)
            if len(imgs) == batch_size:
                imgs = torch.from_numpy(np.expand_dims(np.transpose(imgs[0],(1,0)),0)).float()
                if torch.cuda.is_available() == True:
                    imgs = imgs.cuda()
                    classifier_fn = classifier_fn.cuda()
                preds, _, _  = classifier_fn(imgs)
                if torch.cuda.is_available() == True:
                    preds = preds.detach().cpu().numpy()
                else:
                    preds = preds.detach().numpy()
                labels.extend(preds)
                imgs = []
        labels = np.array(labels)
        return data, labels
