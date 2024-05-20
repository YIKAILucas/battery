import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder


class FeatureBuilder:
    @staticmethod
    def new_date_feature(df: pd.DataFrame):
        df['SN_Generated_Time_Y'] = df['SN_Generated_Time'].dt.year
        df['SN_Generated_Time_M'] = df['SN_Generated_Time'].dt.month
        df['SN_Generated_Time_D'] = df['SN_Generated_Time'].dt.day

        df['Packaging_Time_Y'] = df['Packaging_Time'].dt.year
        df['Packaging_Time_M'] = df['Packaging_Time'].dt.month
        df['Packaging_Time_D'] = df['Packaging_Time'].dt.day

        df['Stocking_Time_Y'] = df['Stocking_Time'].dt.year
        df['Stocking_Time_M'] = df['Stocking_Time'].dt.month
        df['Stocking_Time_D'] = df['Stocking_Time'].dt.day

        return df


class FeatureFilter:
    pass


class FeatureFinder:
    pass


class FeatureEncoder:
    @staticmethod
    def label_encoder(df: pd.DataFrame):
        labeled_df = df.copy()
        # 对类别型的特征进行编码
        categorical_cols = labeled_df.select_dtypes(include=['object', 'category']).columns
        le = LabelEncoder()
        try:
            for col in categorical_cols:
                labeled_df[col] = le.fit_transform(labeled_df[col])
        except Exception as e:
            print(e)
            print(col)
        return labeled_df
