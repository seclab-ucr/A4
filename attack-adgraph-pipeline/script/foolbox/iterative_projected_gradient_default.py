from __future__ import division
import numpy as np
from abc import abstractmethod
import logging
import warnings
# for computing distance after each projection
from scipy.spatial import distance

from .base import Attack
from .base import call_decorator
from .. import distances
from ..utils import crossentropy
from .. import nprng

import matplotlib.pyplot as plt
import math


class IterativeProjectedGradientBaseAttack(Attack):
    """Base class for iterative (projected) gradient attacks.

    Concrete subclasses should implement __call__, _gradient
    and _clip_perturbation.

    TODO: add support for other loss-functions, e.g. the CW loss function,
    see https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    """

    @abstractmethod
    def _gradient(self, a, x, class_, strict=True):
        raise NotImplementedError

    @abstractmethod
    def _clip_perturbation(self, a, noise, epsilon):
        raise NotImplementedError

    @abstractmethod
    def _check_distance(self, a):
        raise NotImplementedError

    def _compute_l2_distance(self, x1, x2):
        return distance.cdist(x1, x2, 'euclidean')

    def _compute_lp_distance(self, x1, x2):
        diff = np.array(x1) - np.array(x2)
        value = np.max(np.abs(diff)).astype(np.float64)
        return value
    
    def _generate_constrained_perturbation(self, perturbation, 
                                           perturbable_idx_set, 
                                           only_increase_idx_set,
                                           debug=False):
        
        for i in range(len(perturbation)):
            if i not in perturbable_idx_set:
                perturbation[i] = 0.0
            if i in only_increase_idx_set and perturbation[i] < 0.0:
                perturbation[i] = -perturbation[i]
        return perturbation

    def _unscale_feature(self, val, stats, is_float=False):
        [maxn, minn] = stats
        maxn, minn = float(maxn), float(minn)
        if is_float:
            return val * (maxn + minn) + minn
        else:
            return int(round(val * (maxn + minn) + minn))

    def _rescale_feature(self, val, stats):
        [maxn, minn] = stats
        maxn, minn = float(maxn), float(minn)
        if maxn == minn:
            return val
        return (val - minn) / (maxn - minn)

    def _calculate_diff(self, ori, per, stats):
        return self._unscale_feature(per, stats) - self._unscale_feature(ori, stats)

    def _recalculate_related_features(self, candidate, 
                                      original,
                                      normalization_ratios,
                                      perturbable_idx_set,
                                      debug=False):

        recalculated_candidate = np.copy(candidate)

        should_process_global_ratio = False
        should_process_connection_cnt = False
        should_match_edge_to_node = False
        should_process_sibling_cnt = False

        if perturbable_idx_set is not None:
            if 0 in perturbable_idx_set or 1 in perturbable_idx_set:
                should_process_global_cnt = True
            if 0 in perturbable_idx_set and 1 not in perturbable_idx_set:
                should_match_edge_to_node = True
            if 4 in perturbable_idx_set or 5 in perturbable_idx_set:
                should_process_connection_cnt = True
            if 22 in perturbable_idx_set:
                should_process_sibling_cnt = True

        if should_process_sibling_cnt:
            sibling_cnt_diff = self._calculate_diff(original[22], recalculated_candidate[22], normalization_ratios[25]['val'])
            node_cnt_diff = self._calculate_diff(original[0], recalculated_candidate[0], normalization_ratios[2]['val'])
            if node_cnt_diff <= sibling_cnt_diff:
                recalculated_candidate[0] += self._rescale_feature(sibling_cnt_diff, normalization_ratios[25]['val'])

        if should_process_connection_cnt:
            original_5th_feature = self._unscale_feature(recalculated_candidate[4], normalization_ratios[6]['val'], False)
            original_6th_feature = self._unscale_feature(recalculated_candidate[5], normalization_ratios[7]['val'], False)
            recalculated_candidate[6] = self._rescale_feature(original_5th_feature + original_6th_feature, normalization_ratios[8]['val'])

        if should_match_edge_to_node:
            node_cnt_diff = self._calculate_diff(original[0], recalculated_candidate[0], normalization_ratios[2]['val'])
            recalculated_candidate[1] += self._rescale_feature(node_cnt_diff * 2, normalization_ratios[3]['val'])

        if should_process_global_ratio:
            original_1st_feature = self._unscale_feature(recalculated_candidate[0], normalization_ratios[2]['val'], False)
            original_2nd_feature = self._unscale_feature(recalculated_candidate[1], normalization_ratios[3]['val'], False)
            if original_1st_feature != 0.0 and original_2nd_feature != 0.0:
                recalculated_candidate[2] = self._rescale_feature(original_1st_feature / original_2nd_feature, normalization_ratios[4]['val'])
                recalculated_candidate[3] = self._rescale_feature(original_2nd_feature / original_1st_feature, normalization_ratios[5]['val'])

        return recalculated_candidate

    def _reject_imperturbable_features(self, candidate, 
                                       original, perturbable_idx_set,
                                       debug=False):
        
        assert len(candidate) == len(original), "[ERROR] Lengths of two input arrays not equal!"

        rejected_candidate = []
        
        for i in range(len(candidate)):
            if i in perturbable_idx_set:
                rejected_candidate.append(candidate[i])
            else:
                rejected_candidate.append(original[i])

        return np.array(rejected_candidate)

    def _reject_only_increase_features(self, candidate, 
                                       original, only_increase_idx_set, 
                                       debug=False):
        
        assert len(candidate) == len(original), "[ERROR] Lengths of two input arrays not equal!"

        rejected_candidate = []

        for i in range(len(candidate)):
            if i in only_increase_idx_set and candidate[i] < original[i]:
                rejected_candidate.append(original[i])
            else:
                rejected_candidate.append(candidate[i])

        return np.array(rejected_candidate)

    def _legalize_candidate(self, candidate, 
                            feature_types, 
                            debug=False):
        if debug:
            print("[INFO] Entered candidate legalization method!")
        adv_x = np.copy(candidate)
        processed_adv_x = np.copy(candidate)
            
        for i in range(len(adv_x)):
            adv_val = adv_x[i]
            processed_adv_val = None
            if feature_types[i] == 'f':
                processed_adv_val = adv_val
                processed_adv_val = max(processed_adv_val, 0.0)
                processed_adv_val = min(processed_adv_val, 1.0)
            if feature_types[i] == 'b':
                if abs(adv_val - 1.0) > abs(adv_val - 0.0):
                    processed_adv_val = 0
                else:
                    processed_adv_val = 1
            
            if processed_adv_val is not None:
                # print(i, feature_types[i], processed_adv_val)
                # input("Press ENTER to continue...")
                processed_adv_x[i] = processed_adv_val

        lookahead_cnt = 0
        for i in range(len(adv_x)):
            if lookahead_cnt - 1 > 0:
                lookahead_cnt -= 1
                continue
            if feature_types[i] == 'c':
                j = i
                while feature_types[j] == 'c' and j + 1 < len(adv_x):
                    j += 1
                categorical_interval_end = j
                maxn = -10000
                maxn_idx = i
                for j in range(i, categorical_interval_end):
                    if adv_x[j] > maxn:
                        maxn = adv_x[j]
                        maxn_idx = j
                for j in range(i, categorical_interval_end):
                    if j == maxn_idx:
                        processed_adv_val = 1
                    else:
                        processed_adv_val = 0
                    processed_adv_x[j] = processed_adv_val
                    lookahead_cnt += 1
             
        legalized_candidate = processed_adv_x
    
        return legalized_candidate

    def _get_mode_and_class(self, a):
        # determine if the attack is targeted or not
        target_class = a.target_class()
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_

    def _run(self, a, binary_search,
             epsilon, stepsize, iterations,
             random_start, return_early,
             perturbable_idx_set, only_increase_idx_set,
             feature_defs, normalization_ratios,
             enforce_interval):
        if not a.has_gradient():
            warnings.warn('applied gradient-based attack to model that'
                          ' does not provide gradients')
            return

        self._check_distance(a)

        targeted, class_ = self._get_mode_and_class(a)

        if binary_search:
            if isinstance(binary_search, bool):
                k = 20
            else:
                k = int(binary_search)
            return self._run_binary_search(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, k=k,
                perturbable_idx_set=perturbable_idx_set, 
                only_increase_idx_set=only_increase_idx_set,
                feature_defs=feature_defs,
                normalization_ratios=normalization_ratios,
                enforce_interval=enforce_interval)
        else:
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early,
                perturbable_idx_set, only_increase_idx_set, feature_defs,
                normalization_ratios, enforce_interval)

    def _get_geometric_enforce_interval(self, base, iteations, i):
        progress = float(i + 1) / iteations
        if progress < 0.1:
            growth_ratio = 1
        elif progress >= 0.1 and progress < 0.5:
            growth_ratio = 3
        else:
            growth_ratio = 10
        grown = base * growth_ratio
        return grown

    def _run_binary_search(self, a, epsilon, stepsize, iterations,
                           random_start, targeted, class_, return_early, k,
                           perturbable_idx_set, only_increase_idx_set, feature_defs,
                           normalization_ratios, enforce_interval):

        factor = stepsize / epsilon

        def try_epsilon(epsilon):
            stepsize = factor * epsilon
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early,
                perturbable_idx_set, only_increase_idx_set, feature_defs,
                normalization_ratio, enforce_interval)

        for i in range(k):
            if try_epsilon(epsilon):
                logging.info('successful for eps = {}'.format(epsilon))
                break
            logging.info('not successful for eps = {}'.format(epsilon))
            epsilon = epsilon * 1.5
        else:
            logging.warning('exponential search failed')
            return

        bad = 0
        good = epsilon

        for i in range(k):
            epsilon = (good + bad) / 2
            if try_epsilon(epsilon):
                good = epsilon
                logging.info('successful for eps = {}'.format(epsilon))
            else:
                bad = epsilon
                logging.info('not successful for eps = {}'.format(epsilon))

    def _run_one(self, a, epsilon, stepsize, iterations,
                 random_start, targeted, class_, return_early,
                 perturbable_idx_set=None, only_increase_idx_set=None,
                 feature_defs=None, normalization_ratios=None,
                 enforce_interval=1, dynamic_interval=False, debug=False, 
                 draw=False):
        if draw:
            x_axis, y_l2, y_lf = [], [], []

        min_, max_ = a.bounds()
        s = max_ - min_

        original = a.original_image.copy()

        if random_start:
            # using uniform noise even if the perturbation clipping uses
            # a different norm because cleverhans does it the same way
            noise = nprng.uniform(
                -epsilon * s, epsilon * s, original.shape).astype(
                    original.dtype)
            x = original + self._clip_perturbation(a, noise, epsilon)
            strict = False  # because we don't enforce the bounds here
        else:
            x = original
            strict = True

        success = False

        if enforce_interval == iterations - 1:
            only_post_process = True
        else:
            only_post_process = False

        for i in range(iterations):
            if dynamic_interval:
                curr_interval = self._get_geometric_enforce_interval(enforce_interval, iterations, i)
            else:
                curr_interval = enforce_interval
            if i % curr_interval == 0 or i == iterations - 1:
                should_enforce_policy = True
            else:
                should_enforce_policy = False

            gradient = self._gradient(a, x, class_, strict=strict)
            # non-strict only for the first call and
            # only if random_start is True
            strict = True
            if targeted:
                gradient = -gradient

            # untargeted: gradient ascent on cross-entropy to original class
            # targeted: gradient descent on cross-entropy to target class
            x = x + stepsize * gradient

            x = original + self._clip_perturbation(a, x - original, epsilon)
            
            if should_enforce_policy:
                x_before_projection = x.copy()

            # phase 1: reject disallowed perturbations by changing feature values
            # back to original
            if only_post_process:
                if should_enforce_policy and perturbable_idx_set is not None:
                    x = self._reject_imperturbable_features(
                        x, 
                        original, 
                        perturbable_idx_set)
            else:
                if perturbable_idx_set is not None:
                    x = self._reject_imperturbable_features(
                        x, 
                        original, 
                        perturbable_idx_set)

            # phase 1: reject decreasing perturbations by changing only-increase
            # feature values back to original
            if should_enforce_policy and only_increase_idx_set is not None:
                x = self._reject_only_increase_features(
                    x,
                    original,
                    only_increase_idx_set)

            if only_post_process:
                if should_enforce_policy and normalization_ratios is not None:
                    x = self._recalculate_related_features(
                        x,
                        original,
                        normalization_ratios,
                        perturbable_idx_set)
            else:
                if normalization_ratios is not None:
                    x = self._recalculate_related_features(
                        x,
                        original,
                        normalization_ratios,
                        perturbable_idx_set)

            # phase 2: change values back to allowed ranges
            if should_enforce_policy and feature_defs is not None:
                x = self._legalize_candidate(
                    x,
                    feature_defs)

            # Call order changed to ensure the perturbed x is
            # still within the bounds
            x = np.clip(x, min_, max_)

            if should_enforce_policy:
                l2_dist = self._compute_l2_distance([x_before_projection], [x])
                lf_dist = self._compute_lp_distance([x_before_projection], [x])
                if draw:
                    x_axis.append(i)
                    y_l2.append(l2_dist[0])
                    y_lf.append(lf_dist)
                if debug:
                    print("Step #%d, L2 distance: %f | LP distance: %f" % (i, l2_dist, lf_dist))

            if should_enforce_policy:
                logits, is_adversarial = a.predictions(x)
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if targeted:
                        ce = crossentropy(a.original_class, logits)
                        logging.debug('crossentropy to {} is {}'.format(
                            a.original_class, ce))
                    ce = crossentropy(class_, logits)
                    logging.debug('crossentropy to {} is {}'.format(class_, ce))
                if is_adversarial:
                    if return_early:
                        return True
                    else:
                        success = True
        if draw:
            plt.plot(x_axis, y_l2, linewidth=3)
            plt.plot(x_axis, y_lf, linewidth=3)
            plt.show()
        return success


class LinfinityGradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L1GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.mean(np.abs(gradient))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L2GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.sqrt(np.mean(np.square(gradient)))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class LinfinityClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        min_, max_ = a.bounds()
        s = max_ - min_
        clipped = np.clip(perturbation, -epsilon * s, epsilon * s)
        return clipped


class L1ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.mean(np.abs(perturbation))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class L2ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.sqrt(np.mean(np.square(perturbation)))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class LinfinityDistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.Linfinity):
            logging.warning('Running an attack that tries to minimize the'
                            ' Linfinity norm of the perturbation without'
                            ' specifying foolbox.distances.Linfinity as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class L1DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MAE):
            logging.warning('Running an attack that tries to minimize the'
                            ' L1 norm of the perturbation without'
                            ' specifying foolbox.distances.MAE as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class L2DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MSE):
            logging.warning('Running an attack that tries to minimize the'
                            ' L2 norm of the perturbation without'
                            ' specifying foolbox.distances.MSE as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class LinfinityBasicIterativeAttack(
        LinfinityGradientMixin,
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Basic Iterative Method introduced in [1]_.

    This attack is also known as Projected Gradient
    Descent (PGD) (without random start) or FGMS^k.

    References
    ----------
    .. [1] Alexey Kurakin, Ian Goodfellow, Samy Bengio,
           "Adversarial examples in the physical world",
            https://arxiv.org/abs/1607.02533

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


BasicIterativeMethod = LinfinityBasicIterativeAttack
BIM = BasicIterativeMethod


class L1BasicIterativeAttack(
        L1GradientMixin,
        L1ClippingMixin,
        L1DistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """Modified version of the Basic Iterative Method
    that minimizes the L1 distance.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


class L2BasicIterativeAttack(
        L2GradientMixin,
        L2ClippingMixin,
        L2DistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """Modified version of the Basic Iterative Method
    that minimizes the L2 distance.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


class ProjectedGradientDescentAttack(
        LinfinityGradientMixin,
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Projected Gradient Descent Attack
    introduced in [1]_ without random start.

    When used without a random start, this attack
    is also known as Basic Iterative Method (BIM)
    or FGSM^k.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. seealso::

       :class:`LinfinityBasicIterativeAttack` and
       :class:`RandomStartProjectedGradientDescentAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.01,
                 iterations=40,
                 random_start=False,
                 return_early=True,
                 perturbable_idx_set=None,
                 only_increase_idx_set=None,
                 feature_defs=None,
                 normalization_ratios=None,
                 enforce_interval=1):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        print("Parameters:", epsilon, stepsize, iterations, enforce_interval)
        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early, 
                  perturbable_idx_set, only_increase_idx_set,
                  feature_defs, normalization_ratios,
                  enforce_interval)


ProjectedGradientDescent = ProjectedGradientDescentAttack
PGD = ProjectedGradientDescent


class RandomStartProjectedGradientDescentAttack(
        LinfinityGradientMixin,
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Projected Gradient Descent Attack
    introduced in [1]_ with random start.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.01,
                 iterations=40,
                 random_start=True,
                 return_early=True):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


RandomProjectedGradientDescent = RandomStartProjectedGradientDescentAttack
RandomPGD = RandomProjectedGradientDescent


class MomentumIterativeAttack(
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Momentum Iterative Method attack
    introduced in [1]_. It's like the Basic
    Iterative Method or Projected Gradient
    Descent except that it uses momentum.

    References
    ----------
    .. [1] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su,
           Jun Zhu, Xiaolin Hu, Jianguo Li, "Boosting Adversarial
           Attacks with Momentum",
           https://arxiv.org/abs/1710.06081

    """

    def _gradient(self, a, x, class_, strict=True):
        # get current gradient
        gradient = a.gradient(x, class_, strict=strict)
        gradient = gradient / max(1e-12, np.mean(np.abs(gradient)))

        # combine with history of gradient as new history
        self._momentum_history = \
            self._decay_factor * self._momentum_history + gradient

        # use history
        gradient = self._momentum_history
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient

    def _run_one(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._momentum_history = 0
        return super(MomentumIterativeAttack, self)._run_one(*args, **kwargs)

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.06,
                 iterations=10,
                 decay_factor=1.0,
                 random_start=False,
                 return_early=True):

        """Momentum-based iterative gradient attack known as
        Momentum Iterative Method.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        decay_factor : float
            Decay factor used by the momentum term.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._decay_factor = decay_factor

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


MomentumIterativeMethod = MomentumIterativeAttack
