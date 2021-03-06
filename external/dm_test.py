# Author   : John Tsang
# Date     : December 7th, 2017
# Purpose  : Implement the Diebold-Mariano Test (DM test) to compare
#            forecast accuracy
# Input    : 1) actual_lst: the list of actual values
#            2) pred1_lst : the first list of predicted values
#            3) pred2_lst : the second list of predicted values
#            4) h         : the number of steps ahead
#            5) crit      : a string specifying the criterion
#                             i)  MSE : the mean squared error
#                            ii)  MAD : the mean absolute deviation
#                           iii) MAPE : the mean absolute percentage error
#                            iv) poly : use power function to weigh the errors
#            6) poly      : the power for crit power
#                           (it is only meaningful when crit is "poly")
# Condition: 1) length of actual_lst, pred1_lst and pred2_lst is equal
#            2) h must be an integer and it must be greater than 0 and less than
#               the length of actual_lst.
#            3) crit must take the 4 values specified in Input
#            4) Each value of actual_lst, pred1_lst and pred2_lst must
#               be numerical values. Missing values will not be accepted.
#            5) power must be a numerical value.
# Return   : a named-tuple of 2 elements
#            1) p_value : the p-value of the DM test
#            2) DM      : the test statistics of the DM test
##########################################################
# References:
#
# Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of
#   prediction mean squared errors. International Journal of forecasting,
#   13(2), 281-291.
#
# Diebold, F. X. and Mariano, R. S. (1995), Comparing predictive accuracy,
#   Journal of business & economic statistics 13(3), 253-264.
#
##########################################################
from typing import NamedTuple, Union
from scipy.stats import t
import collections
import pandas as pd
import numpy as np


def dm_test(actual_lst=None, pred1_lst=None, pred2_lst=None, e_1=None, e_2=None, h=1, crit="MSE", power=2,
            alternative='two_sided') -> Union[NamedTuple, None]:
    """
    This function implements the modified test proposed by Harvey, Leybourne and Newbold (1997).

    The null hypothesis is that the two methods have the same forecast accuracy.
    For alternative="less", the alternative hypothesis is that method 2 is less accurate than method 1.
    For alternative="greater", the alternative hypothesis is that method 2 is more accurate than method 1.
    For alternative="two.sided", the alternative hypothesis is that method 1 and method 2 have different levels of accuracy.

    References
    :param e_1:
    :param e_2:
    :param actual_lst: List of actual values
    :param pred1_lst: First list of predicted values
    :param pred2_lst: Second list of predicted values (series that alternative hypothesis pertains to)
    :param h: Number of steps ahead
    :param crit: String specifying the error criterion (MSE, MAD, MAPE, poly)
    :param power: Power for crit power (only meaningful when crit is `poly`)
    :param alternative: String specifying type of alternative hypothesis (`two_sided`, `greater` or `less`)
    :return:
    """

    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt, msg)
        # Check the range of h
        if h < 1:
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt, msg)
        len_act = len(actual_lst)
        len_p1 = len(pred1_lst)
        len_p2 = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2:
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt, msg)
        # Check range of h
        if h >= len_act:
            rt = -1
            msg = "The number of steps ahead is too large."
            return rt, msg
        # Check if criterion supported
        if crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly":
            rt = -1
            msg = "The criterion is not supported."
            return (rt, msg)
            # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")

        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True

        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if not (is_actual_ok and is_pred1_ok and is_pred2_ok):
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return rt, msg
        return rt, msg

    # Error check
    if e_1 is None and e_2 is None:
        error_code = error_check()
        # Raise error if cannot pass error check
        if error_code[0] == -1:
            raise SyntaxError(error_code[1])

    # Initialize lists
    e1_lst = []
    e2_lst = []
    d_lst = []

    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()

    # Length of lists (as real numbers)
    if actual_lst:
        series_length = float(len(actual_lst))
    else:
        series_length = len(e_1)

    # construct d according to crit
    if crit == "MSE":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append((actual - p1) ** 2)
            e2_lst.append((actual - p2) ** 2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif crit == "MAD":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif crit == "MAPE":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs((actual - p1) / actual))
            e2_lst.append(abs((actual - p2) / actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif crit == "poly":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(((actual - p1)) ** (power))
            e2_lst.append(((actual - p2)) ** (power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)

    if e_1 is not None and e_2 is not None:
        d_lst = np.array(e_1) - np.array(e_2)

    # Mean of d
    mean_d = d_lst.mean()

    # Find autocovariance and construct DM test statistics

    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))  # 0, 1, 2
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / series_length
    DM_stat = V_d ** (-0.5) * mean_d
    harvey_adj = ((series_length + 1 - 2 * h + h * (h - 1) / series_length) / series_length) ** (0.5)
    DM_stat = harvey_adj * DM_stat

    # Find p-value
    p_value = np.nan
    if alternative == 'two_sided':
        p_value = 2 * t.cdf(-abs(DM_stat), df=series_length - 1)
    elif alternative == 'greater':
        p_value = t.cdf(-DM_stat, df=series_length - 1)
    elif alternative == 'less':
        p_value = t.cdf(DM_stat, df=series_length - 1)

    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', ['DM', 'p_value'])

    rt = dm_return(DM=DM_stat, p_value=p_value)

    return rt


def autocovariance(Xi, N, k, Xs):
    autoCov = 0
    len = float(N)
    for i in np.arange(0, N - k):
        autoCov += ((Xi[i + k]) - Xs) * (Xi[i] - Xs)
    return (1 / (len)) * autoCov
