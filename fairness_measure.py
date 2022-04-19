
import numpy as np
import pandas as pd
from sklearn import metrics



def discrimination(df):
    '''
    Measure the exsting descrimination in the original dataset

    :param df: the original dataset 

    :return the descrimination measure  | P(Y = 1 | S = 0) - P(Y = 1 | S = 1) | 

    '''

    p_y1_s0 = len(df[(df['label'] == 1) & (df['sensitive'] == 0)])
    p_s0 = len(df[df['sensitive'] == 0])

    p_y1_s1 = len(df[(df['label'] == 1) & (df['sensitive'] == 1)])
    p_s1 = len(df[df['sensitive'] == 1])

    descrimination = np.abs((p_y1_s0 / p_s0) - (p_y1_s1 / p_s1))

    return descrimination


def statistical_parity(X_test, pred):
    '''
    Measure the statistical parity from the predictions 

    :param df: the test dataset along with the y and pred 

    :return the statistical measure  | P(Y` = 1 | S = 0) - P(Y` = 1 | S = 1) | 

    '''

    df = X_test.copy()
    df['pred'] = pred

    p_yhat1_s0 = len(df[(df['pred'] == 1) & (df['sensitive'] == 0)])
    p_s0 = len(df[df['sensitive'] == 0])

    p_yhat1_s1 = len(df[(df['pred'] == 1) & (df['sensitive'] == 1)])
    p_s1 = len(df[df['sensitive'] == 1])

    demographic_parity = np.abs((p_yhat1_s0 / p_s0) - (p_yhat1_s1 / p_s1))

    return demographic_parity

    
    
    return descrimination

    
def equal_opportunity(X_test, y, pred):
    '''
    Measure the equal opportunity from the predictions aka "Delta TPR" 

    :param df: the test dataset along with the y and pred 

    :return the delta TPR  | P(Y` = 1 | Y = 1, S = 0) - P(Y` = 1 | Y = 1, S = 1) | 

    '''
    df = X_test.copy()
    df['y'] = y 
    df['pred'] = pred

    p_yhat1_y1_s0 = len(df[(df['pred'] == 1) & (df['y'] == 1) & (df['sensitive'] == 0)])
    p_y1_s0 = len(df[(df['y'] == 1) & (df['sensitive'] == 0)])


    p_yhat1_y1_s1 = len(df[(df['pred'] == 1) & (df['y'] == 1) & (df['sensitive'] == 1)])
    p_y1_s1 = len(df[(df['y'] == 1) & (df['sensitive'] == 1)])

    delta_TPR = np.abs((p_yhat1_y1_s0 / p_y1_s0) - (p_yhat1_y1_s1 / p_y1_s1))

    return delta_TPR



def predictive_equality(X_test, y, pred):
    '''
    Measure the predictive equality from the predictions aka "Delta FPR" 

    :param df: the test dataset along with the y and pred 

    :return the delta FPR  | P(Y` = 1 | Y = 0, S = 0) - P(Y` = 1 | Y = 0, S = 1) | 

    '''

    df = X_test.copy()
    df['y'] = y 
    df['pred'] = pred

    p_yhat1_y0_s0 = len(df[(df['pred'] == 1) & (df['y'] == 0) & (df['sensitive'] == 0)])
    p_y0_s0 = len(df[(df['y'] == 0) & (df['sensitive'] == 0)])


    p_yhat1_y0_s1 = len(df[(df['pred'] == 1) & (df['y'] == 0) & (df['sensitive'] == 1)])
    p_y0_s1 = len(df[(df['y'] == 0) & (df['sensitive'] == 1)])

   
 
    delta_FPR = np.abs((p_yhat1_y0_s0 / p_y0_s0) - (p_yhat1_y0_s1 / p_y0_s1))

    return delta_FPR


def TNR_diff(X_test, y, pred):
    '''
    Measure the "Delta TNR" 

    :param df: the test dataset along with the y and pred 

    :return the delta TNR  | P(Y` = 0 | Y = 0, S = 0) - P(Y` = 0 | Y = 0, S = 1) | 

    '''

    df = X_test.copy()
    df['y'] = y 
    df['pred'] = pred

    p_yhat0_y0_s0 = len(df[(df['pred'] == 0) & (df['y'] == 0) & (df['sensitive'] == 0)])
    p_y0_s0 = len(df[(df['y'] == 0) & (df['sensitive'] == 0)])


    p_yhat0_y0_s1 = len(df[(df['pred'] == 0) & (df['y'] == 0) & (df['sensitive'] == 1)])
    p_y0_s1 = len(df[(df['y'] == 0) & (df['sensitive'] == 1)])

 
    delta_TNR = np.abs((p_yhat0_y0_s0 / p_y0_s0) - (p_yhat0_y0_s1 / p_y0_s1))

    return delta_TNR


def equality_disincentives(X_test, y, pred):
    '''
        Meassure Δ_0 = |TPR_0 - FPR_0| and  Δ_1 = |TPR_1 - FPR_1|
        Δ =  |Δ_0 - Δ_1|
    
    '''

    ## Δ_0 = |TPR_0 - FPR_0|

    df = X_test.copy()
    df['y'] = y 
    df['pred'] = pred

    # TPR_0
    p_yhat1_y1_s0 = len(df[(df['pred'] == 1) & (df['y'] == 1) & (df['sensitive'] == 0)])
    p_y1_s0 = len(df[(df['y'] == 1) & (df['sensitive'] == 0)])
    tpr_s0 = p_yhat1_y1_s0 / p_y1_s0

    # FPR_0
    p_yhat1_y0_s0 = len(df[(df['pred'] == 1) & (df['y'] == 0) & (df['sensitive'] == 0)])
    p_y0_s0 = len(df[(df['y'] == 0) & (df['sensitive'] == 0)])
    fpr_s0 = p_yhat1_y0_s0 / p_y0_s0

    delta_0 = np.abs(tpr_s0 - fpr_s0)


    ## Δ_1 = |TPR_1 - FPR_1|


    # TPR_1
    p_yhat1_y1_s1 = len(df[(df['pred'] == 1) & (df['y'] == 1) & (df['sensitive'] == 1)])
    p_y1_s1 = len(df[(df['y'] == 1) & (df['sensitive'] == 1)])
    tpr_s1 = p_yhat1_y1_s1 / p_y1_s1
    
    # FPR_1
    p_yhat1_y0_s1 = len(df[(df['pred'] == 1) & (df['y'] == 0) & (df['sensitive'] == 1)])
    p_y0_s1 = len(df[(df['y'] == 0) & (df['sensitive'] == 1)])
    fpr_s1 = p_yhat1_y0_s1 / p_y0_s1

    delta_1 = np.abs(tpr_s1 - fpr_s1)


    ## Δ =  |Δ_0 - Δ_1|

    delta = np.abs(delta_0 - delta_1)

    return delta



def all_fairness(X_test, y, pred):
    s_p = statistical_parity(X_test, pred)
    delta_TPR = equal_opportunity(X_test, y, pred)
    delta_FPR = predictive_equality(X_test, y, pred)
    delta_TNR = TNR_diff(X_test, y, pred)

    gap_tpr_tnr =  (delta_TPR + delta_TNR) / 2 

    EoD = (delta_TPR + delta_FPR) / 2

    return np.array([s_p, delta_TPR, delta_FPR, EoD])


def temporal_bias(bias):
    
    '''
    Temporal Bias Metrics 
    
    '''
    bias = np.array(bias)
    
   
    # 1) Rate of change 
    roc = (bias[len(bias)-1] - bias[0]) / len(bias)
    
    # 2) Root Mean Square Bias
    rmsb = np.sqrt(np.mean(np.square(bias)))

    # 3) STD
    sd = np.std(bias)
    
    # 4) Max and Min bias range (Bound)
    range_bias = np.max(bias)
    
    # 5) Temporal Stability
    ts = np.mean(np.abs(np.diff(bias)))

    # 6) Maximum Bais Difference betweeen consecutive bias time steps
    mb = np.max(np.abs(np.diff(bias)))

    # 7) Max Absoulte Deviation from the mean 
    mabd = np.max(np.abs(bias - np.mean(bias)))

    # 8) Average Absoulte Deviation from the minimum value
    aadm = np.mean(np.abs(bias - np.min(bias)))

    # 10) Mean of the sum
    mean_sum = np.mean(np.abs(bias - np.mean(bias)))
    
    # 11) CUSUM bias from the mean
    difference = np.abs(bias - np.mean(bias))
    cumsum_plain = np.cumsum(difference)

    return np.array([roc, rmsb, sd, range_bias, ts, mb, mabd, aadm, mean_sum, cumsum_plain])