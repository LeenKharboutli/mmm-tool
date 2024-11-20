import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def preprocess_df(df):
    # Data Preprocessing
    df['date'] = pd.to_datetime(df['calendar_week'])
    df['day'] = [x.day for x in df['date']]
    df['week'] = df['date'].dt.isocalendar().week.astype('int')
    df['month'] = [x.month for x in df['date']]
    df['year'] = [x.year for x in df['date']]
    return df

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns_to_scale = ['paid_views', 'organic_views', 'google_impressions',
       'email_impressions', 'facebook_impressions', 'affiliate_impressions',
       'overall_views', 'day', 'week', 'month', 'year',
       'paid_views_0.1', 'paid_views_0.3', 'organic_views_0.1',
       'organic_views_0.3', 'google_impressions_0.1', 'google_impressions_0.3',
       'email_impressions_0.1', 'email_impressions_0.3',
       'facebook_impressions_0.1', 'facebook_impressions_0.3',
       'affiliate_impressions_0.1', 'affiliate_impressions_0.3',
       'overall_views_0.1', 'overall_views_0.3', 'month_1', 'month_10',
       'month_11', 'month_12', 'month_2', 'month_3', 'month_4', 'month_5',
       'month_6', 'month_7', 'month_8', 'month_9', 'division_A', 'division_B',
       'division_C', 'division_D', 'division_E', 'division_F', 'division_G',
       'division_H', 'division_I', 'division_J', 'division_K', 'division_L',
       'division_M', 'division_N', 'division_O', 'division_P', 'division_Q',
       'division_R', 'division_S', 'division_T', 'division_U', 'division_V',
       'division_W', 'division_X', 'division_Y', 'division_Z']

def encoding_categorical_features(df):
    # Encoding Categorical Features
    dummy_df = df[['month','division']].copy()
    dummy_df['month']= dummy_df['month'].astype('str')
    dummy_df = pd.get_dummies(dummy_df).astype('float')
    dummy_df = pd.concat([df, dummy_df], axis = 1)    
    mmxdf= dummy_df.drop(['date','division','calendar_week'], axis=1)
    missing_columns = [col for col in columns_to_scale if col not in mmxdf.columns]
    mmxdf[missing_columns] = 0
    mmxdf[columns_to_scale] = scaler.fit_transform(mmxdf[columns_to_scale])
    return mmxdf