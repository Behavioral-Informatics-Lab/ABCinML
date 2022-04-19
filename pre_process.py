
import pandas as pd
import numpy as np
from sklearn import preprocessing



def filter_rows(df_):
    '''
        Return only the loans which are done 
    '''
    return df_[df_["Status"] != "Current"]


def filter_columns(df_):
    """
        Filter out unwanted columns 
    """
    selected_columns = ['LoanNumber', 'ListedOnUTC', 'UserName', 'NewCreditCustomer',
                        'LoanDate', 'MaturityDate_Original', 'MaturityDate_Last',
                        'Age', 'DateOfBirth', 'Gender', 'Country', 'AppliedAmount',
                        'Amount', 'Interest', 'LoanDuration', 'MonthlyPayment',
                        'UseOfLoan', 'Education', 'MaritalStatus',
                        'NrOfDependants', 'EmploymentStatus', 'EmploymentDurationCurrentEmployer',
                        'WorkExperience', 'OccupationArea', 'HomeOwnershipType',
                        'IncomeFromPrincipalEmployer', 'IncomeFromPension', 'IncomeFromFamilyAllowance',
                        'IncomeFromSocialWelfare',
                        'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther',
                        'IncomeTotal', 'ExistingLiabilities', 'RefinanceLiabilities', 'DebtToIncome',
                        'FreeCash', 'DefaultDate', 'Status',
                        'CreditScoreEeMini',
                        'NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan',
                        'PreviousRepaymentsBeforeLoan', 'PreviousEarlyRepaymentsBefoleLoan',
                        'PreviousEarlyRepaymentsCountBeforeLoan'
                        ]

    return df_[selected_columns]


def rename_columns(df_):
    """
        Rename columns and add the null values
    """
    df_ = df_.replace(-1, np.nan)

    zero_replacements = ['Age', 'Education', 'MaritalStatus', 'EmploymentStatus', 'OccupationArea', 'CreditScoreEeMini']
    df_[zero_replacements] = df_[zero_replacements].replace(0.0, np.nan)
    df_['NewCreditCustomer'].replace({False : 0, True : 1}, inplace = True)
    value_replacement = {'EmploymentDurationCurrentEmployer': {
    'UpTo1Year' : 1,
    'UpTo2Years' : 2,
    'UpTo3Years' : 3,
    'UpTo4Years' : 4,
    'UpTo5Years' : 5,
    'MoreThan5Years': 6,
    'TrialPeriod' : 7,
    'Retiree' : 8,
    'Other': 9
    
    }}
    df_ = df_.replace(value_replacement)

    return df_


def add_new_columns(df_):
    """
    Create the required columns for modeling i.e., label
    """
    df_["Defaulted"] = df_['DefaultDate'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df_["PaidLoan"] = df_["Defaulted"].replace({0: 1, 1: 0})
    

    return df_


def filter_columns_for_training(df_):
    training_columns = ['LoanDate','PaidLoan', 'Gender','NewCreditCustomer', 'Amount',
                        'Interest', 'LoanDuration', 'Education',
                        'NrOfDependants', 'EmploymentDurationCurrentEmployer',
                        'IncomeFromPrincipalEmployer', 'IncomeFromPension',
                        'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
                        'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther',
                        'ExistingLiabilities', 'RefinanceLiabilities',
                        'DebtToIncome', 'FreeCash',
                        'CreditScoreEeMini', 'NoOfPreviousLoansBeforeLoan',
                        'AmountOfPreviousLoansBeforeLoan', 'PreviousRepaymentsBeforeLoan',
                        'PreviousEarlyRepaymentsBefoleLoan',
                        'PreviousEarlyRepaymentsCountBeforeLoan','HomeOwnershipType', 'EmploymentStatus']

    return df_[training_columns]

def fill_null_values(df):
    
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    
    df['Education'] = df['Education'].fillna(df['Education'].mode()[0])
    
    df['NrOfDependants'].replace({'10Plus':11}, inplace = True)
    df['NrOfDependants'] = pd.to_numeric(df['NrOfDependants'])
    df['NrOfDependants'] = df['NrOfDependants'].fillna(df['NrOfDependants'].mode()[0])
    
    df['EmploymentDurationCurrentEmployer'] = df['EmploymentDurationCurrentEmployer'].fillna(df['EmploymentDurationCurrentEmployer'].mode()[0])
    
    df['DebtToIncome'] = df['DebtToIncome'].fillna(df['DebtToIncome'].mean())
    
    df['FreeCash'] = df['FreeCash'].fillna(df['FreeCash'].mean())
    
    df['CreditScoreEeMini'] = df['CreditScoreEeMini'].fillna(df['CreditScoreEeMini'].mean())
    
    df['NoOfPreviousLoansBeforeLoan'] = df['NoOfPreviousLoansBeforeLoan'].fillna(df['NoOfPreviousLoansBeforeLoan'].mode()[0])
    
    df['AmountOfPreviousLoansBeforeLoan'] = df['AmountOfPreviousLoansBeforeLoan'].fillna(df['AmountOfPreviousLoansBeforeLoan'].mean())
    
    df['PreviousRepaymentsBeforeLoan'] = df['PreviousRepaymentsBeforeLoan'].fillna(df['PreviousRepaymentsBeforeLoan'].mean())
    
    df['PreviousEarlyRepaymentsBefoleLoan'] = df['PreviousEarlyRepaymentsBefoleLoan'].fillna(df['PreviousEarlyRepaymentsBefoleLoan'].mean())
    
    df['PreviousEarlyRepaymentsCountBeforeLoan'] = df['PreviousEarlyRepaymentsCountBeforeLoan'].fillna(df['PreviousEarlyRepaymentsCountBeforeLoan'].mode()[0])
    
    df['HomeOwnershipType'] = df['HomeOwnershipType'].fillna(df['HomeOwnershipType'].mode()[0])
    
    df['EmploymentStatus'] = df['EmploymentStatus'].fillna(df['EmploymentStatus'].mode()[0])
    
    return df


def change_dtypes(df_):
    df_['NewCreditCustomer'] = df_['NewCreditCustomer'].astype(object)
    df_['Gender'] = df_['Gender'].astype(object)
    df_['Education'] = df_['Education'].astype(object)
    df_['EmploymentStatus'] = df_['EmploymentStatus'].astype(object)
    df_['EmploymentDurationCurrentEmployer'] = df_['EmploymentDurationCurrentEmployer'].astype(object)
    df_['HomeOwnershipType'] = df_['HomeOwnershipType'].astype(object)
    
    return df_

def change_names(df_):

    df_ = df_.rename({'Gender': 'sensitive', 'PaidLoan' : 'label'}, axis=1)
    return df_


def filter_years(df_):

    df_['LoanDate'] = pd.to_datetime(df_['LoanDate'])
    df_['year'] = df_['LoanDate'].dt.year
    df_['month'] = df_['LoanDate'].dt.month 

    df_ = df_[df_['year'] != 2009]
    df_ = df_[~((df_['year'] == 2021) & (df_['month'] == 8))]

    return df_


def remove_non_binary_gender(df_):

    df_ = df_[df_['sensitive'] != 2]
    df_['sensitive'].replace({0:1, 1:0}, inplace = True)


    return df_

def pre_process_raw_data(df_):
    """
    Selects the required data and formats it as needed for the projects.
    :param df_: raw dataframe with loan details
    :return: processed dataframe with loans
    """
    df_ = filter_columns(df_)
    df_ = filter_rows(df_)
    df_ = rename_columns(df_)
    df_ = add_new_columns(df_)
    df_ = filter_columns_for_training(df_)
    df_ = fill_null_values(df_)
    df_ = change_dtypes(df_)
    df_ = change_names(df_)
    df_ = filter_years(df_)
    df_ = remove_non_binary_gender(df_)
    return df_


def preparing_dataframe(df):
    dataframe = df.drop('date', axis = 1)
    
    ## Standardizing 
    continous_variables = ['Amount', 'Interest', 'IncomeFromPrincipalEmployer','IncomeFromPension', 'IncomeFromFamilyAllowance',
                      'IncomeFromSocialWelfare','IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther', 
                       'DebtToIncome', 'FreeCash','CreditScoreEeMini','AmountOfPreviousLoansBeforeLoan','PreviousRepaymentsBeforeLoan',
                      'PreviousEarlyRepaymentsBefoleLoan']
    
    categorical_variables = ['NewCreditCustomer', 'LoanDuration','Education', 'NrOfDependants', 'EmploymentDurationCurrentEmployer',
              'ExistingLiabilities', 'RefinanceLiabilities', 'NoOfPreviousLoansBeforeLoan', 'PreviousEarlyRepaymentsCountBeforeLoan',
              'HomeOwnershipType', 'EmploymentStatus']
    
    dataframe[continous_variables] = preprocessing.StandardScaler().fit_transform(dataframe[continous_variables])
    
    dataframe = pd.get_dummies(dataframe, columns=categorical_variables)
    
    
    ## Feature importance 
    
    # train = dataframe.sample(10000)
    
    # y = train['label']
    # X = train.drop('label', axis = 1)
    
    # model_RF = SelectFromModel(estimator=LogisticRegression())
    # model_RF.fit(X, y)
    
    # good_features = X.columns[(model_RF.get_support())]
    
    # df_final = dataframe[good_features]
    # df_final['label'] = dataframe['label'] 
    # df_final['date'] = df['date']
    # df_final['sensitive'] = df['sensitive']

    df_final = dataframe.copy()
    df_final['date'] = df['date']
    
    return df_final