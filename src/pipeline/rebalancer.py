import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class ReBalancer:
    """数据重采样类"""

    def __init__(self, data, label_col):
        self.data = data
        self.label_col = label_col

    def down_sample(
        self,
        df,
    ):
        """下采样

        Args:
            df (_type_): _description_
        """
        # 分离出多数类和少数类
        df_major = df[df.label_col == 0]
        df_minor = df[df.label_col == 1]

        # 下采样多数类
        df_major_downsampled = resample(
            df_major,
            replace=False,
            n_samples=len(df_minor),
        )

        # 合并少数类和下采样后的多数类
        df_balanced = pd.concat([df_major_downsampled, df_minor])

        # 显示新的类别分布
        print(df_balanced.is_damaged.value_counts())

        return df_balanced

    def smote_sample(self, X, y):
        """SMOTE采样

        Args:
            X (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        sm = SMOTE(random_state=42)

        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(X_train.shape, y_train.shape)

        return X_train, y_train
