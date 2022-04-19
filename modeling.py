import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from fairness_measure import all_fairness
from pre_process import pre_process_raw_data, preparing_dataframe
import matplotlib.pyplot as plt

def create_model(df):

    '''
        Build a basic classifier and do the measure by AUC
    '''

    X_train, X_test, y_train, y_test = splitting(df)

    model = RandomForestClassifier().fit(X_train, y_train)


    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    print(f"AUC {auc}")

    pred = model.predict(X_test)

    print(all_fairness(X_test, y_test, pred))


def splitting(df):

    X = df.drop(['LoanDate','label'], axis = 1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


    return X_train, X_test, y_train, y_test

def get_results(clf, X_test, y_test):
    
    '''
         Return Metrics Results
    '''
    
    pred = clf.predict(X_test)
    
    Acc = metrics.accuracy_score(y_test, pred)
    Auc = metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    f1_weighted = metrics.f1_score(y_test, pred, average = 'weighted')
    precision = metrics.precision_score(y_test, pred)
    recall = metrics.recall_score(y_test, pred)
    
    return Auc




def datasets(name):

    ## Jigsaw
    if name == 'jigsaw':

        jigsaw = pd.read_csv('./data/jigsaw_cleaned_embedding.csv')
        return jigsaw


    ## Loan 
    elif name == 'loan':

        raw = pd.read_csv('./data/LoanData.csv')
        loan = pre_process_raw_data(raw)
        loan = loan[loan['year'] != 2021]
        loan.drop(['LoanDate','month'], axis = 1, inplace=True)
        loan.rename({'year':'date'}, axis = 1, inplace = True)
        loan = preparing_dataframe(loan)
        
        return loan

    ## Funding
    elif name == 'funding':

        funding = pd.read_csv('./data/cleaned_Fundings.csv')
        funding.drop(['year'], axis = 1, inplace=True)
        funding.rename({'date_posted':'date'}, axis = 1, inplace = True)
        funding.rename({'fully_funded': 'label'}, axis = 1, inplace = True)
        funding.rename({'poverty_level': 'sensitive'}, axis = 1, inplace = True)
        return funding

    ## New Adult 
    elif name == 'adult':
        adult = pd.read_csv('./data/adult_combined.csv')
        return adult


def weight_estimation(df, sensitive, label):

    '''
        Kamiran Paper
    '''
    
    deprived = (df[df[sensitive] == 0]).shape[0] # P(S=0)
    favored = (df[df[sensitive] == 1]).shape[0]  # P(S=1)
    
    DP = df[(df[sensitive] == 0) & (df[label] == 1)].shape[0]  # P(S=0, Y=+)
    DN = df[(df[sensitive] == 0) & (df[label] == 0)].shape[0]  # P(S=0, Y=-)
    FP = df[(df[sensitive] == 1) & (df[label] == 1)].shape[0]  # P(S=1, Y=+)
    FN = df[(df[sensitive] == 1) & (df[label] == 0)].shape[0]  # P(S=1, Y=-)
    
    positve = df[df[label] == 1].shape[0]   # P(Y=+)
    negative = df[df[label] == 0].shape[0]  # P(Y=-)
    n = df.shape[0]
    
    ## Get the weights 
    
    W_DP = (deprived * positve) / (n * DP)
    W_DN = (deprived * negative) / (n * DN)
    W_FP = (favored * positve) / (n * FP)
    W_FN = (favored * negative) / (n * FN)
    
    # Return weights
    return np.array([W_DP, W_DN, W_FP, W_FN]) 


def inverse_weighting(df, sensitive, label):
    '''
        Get weight for S = s, Y = y and assign the inverse weight as 1 / P(S=s, Y=y)
    '''
    DP = df[(df[sensitive] == 0) & (df[label] == 1)].shape[0]  # P(S=0, Y=+)
    DN = df[(df[sensitive] == 0) & (df[label] == 0)].shape[0]  # P(S=0, Y=-)
    FP = df[(df[sensitive] == 1) & (df[label] == 1)].shape[0]  # P(S=1, Y=+)
    FN = df[(df[sensitive] == 1) & (df[label] == 0)].shape[0]  # P(S=1, Y=-)
    
    n = df.shape[0]
    
    ## Get the weights 
    
    W_DP = n / DP
    W_DN = n / DN
    W_FP = n / FP
    W_FN = n / FN
    
    # Return weights
    return np.array([W_DP, W_DN, W_FP, W_FN]) 

def reweighing(df, sensitive, label, weights):
    
    df_ = df.copy()

    W_DP, W_DN, W_FP, W_FN = weights
    
     ## Add weight to the dataframe 
    
    df_['weight'] = np.ones(len(df_))
    
    df_['weight'] = np.where((df_.sensitive == 0) & (df_.label == 1), W_DP, df_['weight'])
    df_['weight'] = np.where((df_.sensitive == 0) & (df_.label == 0), W_DN, df_['weight'])
    df_['weight'] = np.where((df_.sensitive == 1) & (df_.label == 1), W_FP, df_['weight'])
    df_['weight'] = np.where((df_.sensitive == 1) & (df_.label == 0), W_FN, df_['weight'])
    
    return df_

def baseline_0(df):
    '''
    
    (1) Train once + test on entire future (sequentially) (average) (no fairness)
    
    '''

    aucs = []
    fairness = []


    batches = sorted(df.date.unique())
    b1 = df[df.date == batches[0]]
    print(f"Train on {batches[0]}")
    b1.drop('date', axis = 1, inplace = True)
    W = weight_estimation(b1, 'sensitive', 'label')
    b1_reweighted = reweighing(b1, 'sensitive', 'label', W)

    # Training 
     
    weights = np.array(b1_reweighted['weight'])
    
    y_b1 = b1_reweighted['label']
    X_b1 = b1_reweighted.drop(['weight','label'], axis = 1)

    model = LogisticRegression(class_weight='balanced').fit(X_b1, y_b1)

    # Testing 

    for i in batches[1:]:
        print(f"Test on {i}")
        rest_b = df[df.date == i]
        rest_b.drop('date', axis = 1, inplace = True)
        y_b = rest_b['label']
        X_b = rest_b.drop('label', axis = 1)

        pred = model.predict(X_b)

        auc = get_results(model, X_b, y_b)
        fair = all_fairness(X_b, y_b, pred)

        aucs.append(auc)
        fairness.append(fair)
    
    results = pd.DataFrame(fairness, columns=['S.P', 'TPR', 'FPR','GAP'])
    results['auc'] = aucs
    results = results[['auc', 'S.P', 'TPR', 'FPR', 'GAP']]

    return results.mean(axis = 0), results, batches


def baseline_1(df):
    '''
    
    (1) Train once + test on entire future (sequentially) (average)
    
    '''

    aucs = []
    fairness = []


    batches = sorted(df.date.unique())
    b1 = df[df.date == batches[0]]
    print(f"Train on {batches[0]}")
    b1.drop('date', axis = 1, inplace = True)
    W = weight_estimation(b1, 'sensitive', 'label')
    b1_reweighted = reweighing(b1, 'sensitive', 'label', W)

    # Training 
     
    weights = np.array(b1_reweighted['weight'])
    
    y_b1 = b1_reweighted['label']
    X_b1 = b1_reweighted.drop(['weight','label'], axis = 1)

    model = LogisticRegression(class_weight='balanced').fit(X_b1, y_b1, sample_weight= weights)

    # Testing 

    for i in batches[1:]:
        print(f"Test on {i}")
        rest_b = df[df.date == i]
        rest_b.drop('date', axis = 1, inplace = True)
        y_b = rest_b['label']
        X_b = rest_b.drop('label', axis = 1)

        pred = model.predict(X_b)

        auc = get_results(model, X_b, y_b)
        fair = all_fairness(X_b, y_b, pred)

        aucs.append(auc)
        fairness.append(fair)
    
    results = pd.DataFrame(fairness, columns=['S.P', 'TPR', 'FPR','GAP'])
    results['auc'] = aucs
    results = results[['auc', 'S.P', 'TPR', 'FPR', 'GAP']]

    return results.mean(axis = 0), results, batches

def baseline_2(df):

    '''
    (2) Train on immediate past + test on immediate future (+ average)

    '''
    aucs = []
    fairness = []

    batches = sorted(df.date.unique())
    for i in range(1, len(batches)):
        
        ## Training 
        
        b_train = df[df.date == batches[i-1]]
        b_train.drop('date', axis = 1, inplace = True)
        
        W = weight_estimation(b_train, 'sensitive', 'label') 
        b_train_reweighted = reweighing(b_train, 'sensitive', 'label', W)
        
        
        weights = np.array(b_train_reweighted['weight'])
        y_b_train = b_train_reweighted['label']
        X_b_train = b_train_reweighted.drop(['weight','label'], axis = 1)
        
        model = LogisticRegression(class_weight='balanced').fit(X_b_train, y_b_train, sample_weight = weights)
        
        ## Testing 
        b_test = df[df.date == batches[i]]
        b_test.drop('date', axis = 1, inplace = True)
        y_b_test = b_test['label']
        X_b_test = b_test.drop('label', axis = 1)
        
        pred = model.predict(X_b_test)
        
        auc = get_results(model, X_b_test, y_b_test)
        fair = all_fairness(X_b_test, y_b_test, pred)
        
        aucs.append(auc)
        fairness.append(fair)
    
    results = pd.DataFrame(fairness, columns=['S.P', 'TPR', 'FPR','GAP'])
    results['auc'] = aucs
    results = results[['auc', 'S.P', 'TPR', 'FPR', 'GAP']]

    return results.mean(axis = 0)


def baseline_3(df):

    '''
    3) Train on entire past + test on immediate future (+ average)
    
    '''
    aucs = []
    fairness = []

    batches = sorted(df.date.unique())
    for i in range(1, len(batches)):
        
        ## Training 
        print(batches[0:i])
        
        b_train = df[df['date'].isin(batches[0:i])]
        b_train.drop('date', axis = 1, inplace = True)
        
        W = weight_estimation(b_train, 'sensitive', 'label') 
        b_train_reweighted = reweighing(b_train, 'sensitive', 'label', W)
        
        weights = np.array(b_train_reweighted['weight'])
        y_b_train = b_train_reweighted['label']
        X_b_train = b_train_reweighted.drop(['weight','label'], axis = 1)
        
        model = LogisticRegression(class_weight='balanced').fit(X_b_train, y_b_train, sample_weight = weights)
        
        ## Testing 
        print(batches[i])
        b_test = df[df.date == batches[i]]
        b_test.drop('date', axis = 1, inplace = True)
        y_b_test = b_test['label']
        X_b_test = b_test.drop('label', axis = 1)

        pred = model.predict(X_b_test)
        
        auc = get_results(model, X_b_test, y_b_test)
        fair = all_fairness(X_b_test, y_b_test, pred)
        
        aucs.append(auc)
        fairness.append(fair)

    results = pd.DataFrame(fairness, columns=['S.P', 'TPR', 'FPR','GAP'])
    results['auc'] = aucs
    results = results[['auc', 'S.P', 'TPR', 'FPR', 'GAP']]

    return results.mean(axis=0), results, batches


def ours_reset(name, df, alpha):

    aucs = []
    fairness = []

    predictions = time_series_forecast(name, df)
    
    batches = sorted(df.date.unique())
    if name == 'loan':
        batches = batches[2:]

    elif name == 'funding':
        batches = batches[2:]


    for i in range(1, len(batches)):

        ## Training 
        print(f"Train on {batches[i-1]}")

        b_train = df[df.date == batches[i-1]]
        b_train.drop('date', axis = 1, inplace = True)

        print(f"Use estimation from {batches[i]}")
        prediction = predictions[predictions['date'] == batches[i]]

        anticipated_weights = reweighing_anticipated(prediction)

        b1_new_weight = complex_reweighing(b_train, 'sensitive', 'label', anticipated_weights, alpha)

        b1_weighted_anticipated = reweighing(b_train, 'sensitive','label', b1_new_weight)

        weights = np.array(b1_weighted_anticipated['weight'])
        y_b1_weighted_anticipated = b1_weighted_anticipated['label']
        X_b1_weighted_anticipated = b1_weighted_anticipated.drop(['weight','label'], axis = 1)

        model = LogisticRegression(class_weight = 'balanced').fit(X_b1_weighted_anticipated, y_b1_weighted_anticipated, sample_weight = weights)


        ## Testing 
        print(f"Test on {batches[i]}")

        b_test = df[df['date'] == batches[i]]
        b_test.drop('date', axis = 1, inplace = True)
        y_test = b_test['label']
        X_test = b_test.drop('label', axis = 1)

        auc = get_results(model, X_test, y_test)


        pred = model.predict(X_test)


        fair = all_fairness(X_test, y_test, pred)

        aucs.append(auc)
        fairness.append(fair)

    results = pd.DataFrame(fairness, columns=['S.P', 'TPR', 'FPR', 'GAP'])
    results['auc'] = aucs
    results = results[['auc', 'S.P', 'TPR', 'FPR', 'GAP']]

    return results.mean(axis=0)






def ours_cumulative(name, df, alpha):
    '''
        Use anticipated ratios from the time series forecaster
    '''
    aucs = []
    fairness = []

    predictions = time_series_forecast(name, df)

    batches = sorted(df.date.unique())


    if name == 'loan':
        batches = batches[2:]


    elif name == 'funding':
        batches = batches[2:]


    for i in range(1, len(batches)):
        
        ## Training 
        print(batches[0:i])
        
        b_train = df[df['date'].isin(batches[0:i])]
        b_train.drop('date', axis = 1, inplace = True)
        
        print(f"Use estimation from {batches[i]}")
        prediction = predictions[predictions['date'] == batches[i]]
        
        anticipated_weights = reweighing_anticipated(prediction)

        b1_new_weight = complex_reweighing(b_train, 'sensitive', 'label', anticipated_weights, alpha)

        b1_weighted_anticipated = reweighing(b_train, 'sensitive','label', b1_new_weight)

        weights = np.array(b1_weighted_anticipated['weight'])
        y_b1_weighted_anticipated = b1_weighted_anticipated['label']
        X_b1_weighted_anticipated = b1_weighted_anticipated.drop(['weight','label'], axis = 1)

        model = LogisticRegression(class_weight='balanced').fit(X_b1_weighted_anticipated, y_b1_weighted_anticipated, sample_weight = weights)


        ## Testing 
        print(batches[i])
        b_test = df[df['date'] == batches[i]]
        b_test.drop('date', axis = 1, inplace = True)
        y_test = b_test['label']
        X_test = b_test.drop('label', axis = 1)
        
        auc = get_results(model, X_test, y_test)
        
        
        pred = model.predict(X_test)
        
        
        fair = all_fairness(X_test, y_test, pred)
        
        aucs.append(auc)
        fairness.append(fair)
        
    results = pd.DataFrame(fairness, columns=['S.P', 'TPR', 'FPR', 'GAP'])
    results['auc'] = aucs

    results = results[['auc', 'S.P', 'TPR', 'FPR', 'GAP']]
    return results.mean(axis=0).to_numpy(), results, batches


def ours_cumulative_actual(name, df, alpha):
    '''
        Use the actual ratios for next time step (Cheating)
    '''
    aucs = []
    fairness = []

    batches = sorted(df.date.unique())

    if name == 'loan':
        batches = batches[2:]


    elif name == 'funding':
        batches = batches[2:]


    for i in range(1, len(batches)):
        
        ## Training 
        print(batches[0:i])
        
        b_train = df[df['date'].isin(batches[0:i])]
        b_train.drop('date', axis = 1, inplace = True)
        
        print(f"Use actual ratios from {batches[i]}")
        b_temp = df[df['date'] == batches[i]]

        
        
        actual_weights = weight_estimation(b_temp, 'sensitive', 'label')

        b1_new_weight = complex_reweighing(b_train, 'sensitive', 'label', actual_weights, alpha)

        b1_weighted_anticipated = reweighing(b_train, 'sensitive','label', b1_new_weight)

        weights = np.array(b1_weighted_anticipated['weight'])
        y_b1_weighted_anticipated = b1_weighted_anticipated['label']
        X_b1_weighted_anticipated = b1_weighted_anticipated.drop(['weight','label'], axis = 1)

        model = LogisticRegression(class_weight = 'balanced').fit(X_b1_weighted_anticipated, y_b1_weighted_anticipated, sample_weight = weights)


        ## Testing 
        print(batches[i])
        b_test = df[df['date'] == batches[i]]
        b_test.drop('date', axis = 1, inplace = True)
        y_test = b_test['label']
        X_test = b_test.drop('label', axis = 1)
        
        auc = get_results(model, X_test, y_test)
        
        
        pred = model.predict(X_test)
        
        
        fair = all_fairness(X_test, y_test, pred)
        
        aucs.append(auc)
        fairness.append(fair)
        
    results = pd.DataFrame(fairness, columns=['S.P', 'TPR', 'FPR'])
    results['auc'] = aucs

    return results.mean(axis=0)

def ours_cumulative_overfitting(name, df):
    '''
        append train and test and fix (Overfitting)
    '''
    aucs = []
    fairness = []

    batches = sorted(df.date.unique())

    if name == 'loan':
        batches = batches[2:]


    elif name == 'funding':
        batches = batches[2:]


    for i in range(1, len(batches)):
        
        ## Training (train + test)
        print(batches[0:i+1])
        
        b_train = df[df['date'].isin(batches[0:i+1])]
        b_train.drop('date', axis = 1, inplace = True)
        
        W = weight_estimation(b_train, 'sensitive', 'label') 
        b_train_reweighted = reweighing(b_train, 'sensitive', 'label', W)
        
        weights = np.array(b_train_reweighted['weight'])
        y_b_train = b_train_reweighted['label']
        X_b_train = b_train_reweighted.drop(['weight','label'], axis = 1)
        
        model = LogisticRegression(class_weight = 'balanced').fit(X_b_train, y_b_train, sample_weight = weights)
        

        ## Testing
        print('Test on') 
        print(batches[0:i+1])
        b_test = df[df['date'].isin(batches[0:i+1])]
        b_test.drop('date', axis = 1, inplace = True)
        y_test = b_test['label']
        X_test = b_test.drop('label', axis = 1)
        
        auc = get_results(model, X_test, y_test)
        
        
        pred = model.predict(X_test)
        
        
        fair = all_fairness(X_test, y_test, pred)
        
        aucs.append(auc)
        fairness.append(fair)
        
    results = pd.DataFrame(fairness, columns=['S.P', 'TPR', 'FPR'])
    results['auc'] = aucs

    return results.mean(axis=0)

def get_all_aggregated_values(df, sensitive, label):
  
        
    temp = df[(df['sensitive'] == sensitive) & (df['label'] == label )]
    temp = temp.groupby(['date']).size().to_frame(name = 'count').reset_index()
    temp.set_index('date', inplace=True)
    
    return temp


def get_time_series_predictions(name, df, sensitive, label):
    if name == 'jigsaw':
        window = 5

    elif name == 'loan':
        window = 3

    elif name == 'funding':
        window = 3

    elif name == 'adult':
        window = 2

    temp = get_all_aggregated_values(df, sensitive, label)
    sma_whole = temp[['count']]
    sma_whole[f'{sensitive,label}'] = sma_whole.rolling(window).mean()
    sma_whole.dropna(inplace = True)
    
    # plt.plot(sma_whole['count'], label = 'Real')
    # plt.plot(sma_whole[f'{sensitive,label}'], label = 'Pred')
    # plt.legend()
    # plt.xlabel('Time')
    # plt.ylabel('#')
    # plt.title(f'P{sensitive, label}')

    return sma_whole


def time_series_forecast(name, df):
    if name == 'jigsaw':
        df = pd.read_csv('./data/jigsaw_cleaned.csv')
        p_1_1 = get_time_series_predictions(name, df, 1, 1)['(1, 1)']
        p_1_0 = get_time_series_predictions(name, df, 1, 0)['(1, 0)']
        p_0_1 = get_time_series_predictions(name, df, 0, 1)['(0, 1)']
        p_0_0 = get_time_series_predictions(name, df, 0, 0)['(0, 0)']
        predictions = pd.concat([p_1_1, p_1_0, p_0_1, p_0_0], axis=1)
        predictions.reset_index(inplace = True)
        predictions['length'] = predictions.iloc[:,1:].sum(axis=1)
        predictions['S0'] = predictions['(0, 0)'] + predictions['(0, 1)']
        predictions['S1'] = predictions['(1, 0)'] + predictions['(1, 1)']
        predictions['Y0'] = predictions['(0, 0)'] + predictions['(1, 0)']
        predictions['Y1'] = predictions['(0, 1)'] + predictions['(1, 1)']

        return predictions

    elif name == 'loan':
        p_1_1 = get_time_series_predictions(name, df, 1, 1)['(1, 1)']
        p_1_0 = get_time_series_predictions(name, df, 1, 0)['(1, 0)']
        p_0_1 = get_time_series_predictions(name, df, 0, 1)['(0, 1)']
        p_0_0 = get_time_series_predictions(name, df, 0, 0)['(0, 0)']
        predictions = pd.concat([p_1_1, p_1_0, p_0_1, p_0_0], axis=1)
        predictions.reset_index(inplace = True)
        predictions['length'] = predictions.iloc[:,1:].sum(axis=1)
        predictions['S0'] = predictions['(0, 0)'] + predictions['(0, 1)']
        predictions['S1'] = predictions['(1, 0)'] + predictions['(1, 1)']
        predictions['Y0'] = predictions['(0, 0)'] + predictions['(1, 0)']
        predictions['Y1'] = predictions['(0, 1)'] + predictions['(1, 1)']

        return predictions


    elif name == 'funding':
        p_1_1 = get_time_series_predictions(name, df, 1, 1)['(1, 1)']
        p_1_0 = get_time_series_predictions(name, df, 1, 0)['(1, 0)']
        p_0_1 = get_time_series_predictions(name, df, 0, 1)['(0, 1)']
        p_0_0 = get_time_series_predictions(name, df, 0, 0)['(0, 0)']
        predictions = pd.concat([p_1_1, p_1_0, p_0_1, p_0_0], axis=1)
        predictions.reset_index(inplace = True)
        predictions['length'] = predictions.iloc[:,1:].sum(axis=1)
        predictions['S0'] = predictions['(0, 0)'] + predictions['(0, 1)']
        predictions['S1'] = predictions['(1, 0)'] + predictions['(1, 1)']
        predictions['Y0'] = predictions['(0, 0)'] + predictions['(1, 0)']
        predictions['Y1'] = predictions['(0, 1)'] + predictions['(1, 1)']

        return predictions


    elif name == 'adult':
        df = pd.read_csv('./data/adult_original.csv')
        p_1_1 = get_time_series_predictions(name, df, 1, 1)['(1, 1)']
        p_1_0 = get_time_series_predictions(name, df, 1, 0)['(1, 0)']
        p_0_1 = get_time_series_predictions(name, df, 0, 1)['(0, 1)']
        p_0_0 = get_time_series_predictions(name, df, 0, 0)['(0, 0)']
        predictions = pd.concat([p_1_1, p_1_0, p_0_1, p_0_0], axis=1)
        predictions.reset_index(inplace = True)
        predictions['length'] = predictions.iloc[:,1:].sum(axis=1)
        predictions['S0'] = predictions['(0, 0)'] + predictions['(0, 1)']
        predictions['S1'] = predictions['(1, 0)'] + predictions['(1, 1)']
        predictions['Y0'] = predictions['(0, 0)'] + predictions['(1, 0)']
        predictions['Y1'] = predictions['(0, 1)'] + predictions['(1, 1)']

        return predictions



def reweighing_anticipated(prediction):
    deprived = prediction['S0'].to_numpy()[0]  # P(S=0)
    favored = prediction['S1'].to_numpy()[0]   #P(S=1)
    DP = prediction['(0, 1)'].to_numpy()[0]  #P(S=0, Y=1)
    DN = prediction['(0, 0)'].to_numpy()[0]   #P(S=0, Y=0)
    FP = prediction['(1, 1)'].to_numpy()[0]   #P(S=1, Y=1)
    FN = prediction['(1, 0)'].to_numpy()[0]  #P(S=1, Y=0)

    positve = prediction['Y1'].to_numpy()[0]  #P(Y=1)
    negative = prediction['Y0'].to_numpy()[0] #P(Y=0)
    n = prediction['length'].to_numpy()[0]

    ## Get the weights 

    W_DP = (deprived * positve) / (n * DP)
    W_DN = (deprived * negative) / (n * DN)
    W_FP = (favored * positve) / (n * FP)
    W_FN = (favored * negative) / (n * FN)

    return np.array([W_DP, W_DN, W_FP, W_FN])


def inverse_reweighing_anticipated(prediction):
    deprived = prediction['S0'].to_numpy()[0]  # P(S=0)
    favored = prediction['S1'].to_numpy()[0]   #P(S=1)
    DP = prediction['(0, 1)'].to_numpy()[0]  #P(S=0, Y=1)
    DN = prediction['(0, 0)'].to_numpy()[0]   #P(S=0, Y=0)
    FP = prediction['(1, 1)'].to_numpy()[0]   #P(S=1, Y=1)
    FN = prediction['(1, 0)'].to_numpy()[0]  #P(S=1, Y=0)

    positve = prediction['Y1'].to_numpy()[0]  #P(Y=1)
    negative = prediction['Y0'].to_numpy()[0] #P(Y=0)
    n = prediction['length'].to_numpy()[0]

    ## Get the weights 

    W_DP = n / DP
    W_DN = n / DN
    W_FP = n / FP
    W_FN = n / FN

    return np.array([W_DP, W_DN, W_FP, W_FN])


def complex_reweighing(df, sensitive, label, anticipated_weights, alpha):
    
    current_weight = weight_estimation(df, sensitive, label)

    # Weighted average weight
    w_new = alpha * current_weight + (1-alpha) * anticipated_weights

    return w_new