import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import plotly.figure_factory as ff
import plotly.express as px


class ModelEvaluator:
    """模型评估类

    Returns:
        _type_: _description_
    """

    @staticmethod
    def evaluate(y_test, y_pred) -> dict:
        """模型评估指标

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            dict: _description_
        """
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        tp / (tp + fp)

        print("Accuracy: %f" % accuracy)
        print("Precision: %f" % precision)
        print("Recall: %f" % recall)
        print(f"Specificity: {specificity}")
        # 显示混淆矩阵
        print("Confusion Matrix: ")
        # TODO 新增返回混淆矩阵
        matrix = confusion_matrix(y_test, y_pred)
        print(matrix)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
        }

    @staticmethod
    def show_importance(model, X):
        """显示特征重要性

        Args:
            model (_type_): _description_
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        importance = model.feature_importances_

        df_importance = pd.DataFrame(
            importance, index=X.columns, columns=["importance"]
        )
        df_sorted = df_importance.sort_values(by=["importance"], ascending=False)
        return df_sorted

    @staticmethod
    def plot_importance(model, X, threshold=0):
        """显示特征重要性

        Args:
            model (_type_): _description_
            X (_type_): _description_
            threshold (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        importance = model.feature_importances_

        df_importance = pd.DataFrame(
            importance, index=X.columns, columns=["importance"]
        )
        df_sorted = df_importance.sort_values(by="importance", ascending=False)

        if threshold >= 1:
            # threshold is a count
            df_sorted = df_sorted.iloc[: int(threshold)]
        elif threshold != 0 and threshold < 1:
            # threshold is a cutoff for feature importance
            df_sorted = df_sorted[df_sorted["importance"] > float(threshold)]
        else:
            pass
        df_sorted.rename(columns={"importance": "重要性"}, inplace=True)
        fig = px.bar(
            df_sorted,
            y="重要性",
            x=df_sorted.index,
            labels={"index": "特征"},
            height=800,
            width=1200,
        )
        fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=18))
        fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=18))
        fig.update_traces(textfont_size=20)

        fig.update_layout(title_text="特征重要性图")

        return fig

    @staticmethod
    def print_evaluation(y_true, y_pred, target_names=None, average="micro"):
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        report = sklearn.metrics.classification_report(y_true, y_pred)

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            np.array(y_true).ravel(), y_pred.ravel()
        )
        sklearn.metrics.auc(fpr, tpr)
        # svc_disp = plot_roc_curve(rfc, Xvalidation, y_pred)
        # plt.show()
        sklearn.metrics.multilabel_confusion_matrix(y_true, y_pred)
        # ConfusionMatrixDisplay.from_predictions(
        # y_true, predicted_labels, labels=lp_model.classes_
        # )
        # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, robust=True)
        # 单独 precision+recall+f1
        # precision = metrics.precision_score(y_true, y_pred, average=average)
        # metrics.recall_score(y_true, y_pred, average=average)
        # metrics.f1_score(y_true, y_pred, average=average)

        # 统一 precision+recall+f1
        prf = sklearn.metrics.precision_recall_fscore_support(
            y_true, y_pred, average=average
        )
        hamming = sklearn.metrics.hamming_loss(y_true, y_pred)
        # kappa不支持多标签
        # kappa = metrics.cohen_kappa_score(y_true, y_pred)

        print()
        print(f"acc \n{acc}")
        print(f"report \n{report}")
        print(f"precision_recall_fscore_support\n{prf}\n")
        # print(f'multilabel_confusion_matrix\n{confusion_mat}')
        # print(f'auc\n{auc}')
        print(f"hamming\n{hamming}\n")
        # print(f'kappa\n{kappa}\n')
        print("")

    @staticmethod
    def draw_confusion_matrix(c_matrix, classes):
        fig = ff.create_annotated_heatmap(
            c_matrix, x=classes, y=classes, colorscale="Greys"
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis=dict(title="预测值"),
            yaxis=dict(title="实际值"),
        )

        return fig
