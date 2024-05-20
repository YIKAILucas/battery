import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score


class ModelTrainer:
    model = None

    def get_param(self):
        return self.model.get_params()

    def xgb_clf(self):
        # 创建XGBoost分类器
        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            # scale_pos_weight=39,
            eval_metric="auc",
            n_estimators=500,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            seed=7,
            nthread=-1,
            use_label_encoder=False,
        )
        self.model = clf
        return clf

    def predict_with_threshold(self):
        # 训练分类器
        self.clf.fit(X_train, y_train.to_numpy().flatten())

        # 通过改变阈值来提高召回率
        threshold = 0.1  # 可以尝试不同的阈值以满足你的需求
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob > threshold).astype("int")

    @staticmethod
    def adjust_threshold(probs, threshold):
        """
        根据指定的阈值调整基于预测概率的分类结果，并计算召回率和精确率。

        参数:
        - probs (np.array): 从 clf.predict_proba() 得到的预测概率数组。
        - y_true (list or np.array): 真实的标签。
        - threshold (float): 用于分类的阈值。

        返回:
        - recall (float): 计算得到的召回率。
        - precision (float): 计算得到的精确率。
        """
        # 提取预测为正类的概率（对于二分类问题，通常是第二列）
        positive_probs = probs[:, 1]

        # 根据阈值生成新的预测结果
        y_pred = [1 if prob >= threshold else 0 for prob in positive_probs]

        return y_pred

    def set_model(self, model):
        self.model = model
