import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# NOTE: do we need just one fitted scaler? How does this work...? 
scaler = MinMaxScaler()

def preprocess_df(df):
    """Performs feature engineering on input df.

    Args:
        df (pandas.core.frame.DataFrame): Unprocessed sample_media_spend_data dataframe

    Returns:
        pandas.core.frame.DataFrame: Modified dataframe
    """
 
    df['date'] = pd.to_datetime(df['calendar_week'])
    df['day'] = [x.day for x in df['date']]
    df['week'] = df['date'].dt.isocalendar().week.astype('int')
    df['month'] = [x.month for x in df['date']]
    df['year'] = [x.year for x in df['date']]
    return df

def encoding_categorical_features(df, columns_to_scale):
    """Encodes categorical features, then scales all target columns.

    Args:
        df (pandas.core.frame.DataFrame): The input dataframe
        columns_to_scale (list): Columns from the dataframe we need to scale

    Returns:
        pandas.core.frame.DataFrame: Encoded and rescaled df
    """

    dummy_df = df[['month','division']].copy()
    dummy_df['month']= dummy_df['month'].astype('str')
    dummy_df = pd.get_dummies(dummy_df).astype('float')
    dummy_df = pd.concat([df, dummy_df], axis = 1)    
    mmxdf= dummy_df.drop(['date','division','calendar_week'], axis=1)
    missing_columns = [col for col in columns_to_scale if col not in mmxdf.columns]
    mmxdf[missing_columns] = 0
    mmxdf[columns_to_scale] = scaler.fit_transform(mmxdf[columns_to_scale])
    return mmxdf