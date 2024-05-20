from deprecated import deprecated
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from ydata_profiling import ProfileReport
import plotly.graph_objs as go


class StatisticalAnalyzer:
    def __init__(self, data):
        self.data = data

    def test_normality(self, column):
        """shapiro-wilk检验

        Args:
            column (_type_): _description_

        Returns:
            _type_: _description_
        """
        statistic, p_value = stats.shapiro(self.data[column])
        return statistic, p_value

    def conduct_hypothesis_testing(self, column1, column2):
        """student t检验

        Args:
            column1 (_type_): _description_
            column2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        statistic, p_value = stats.ttest_ind(self.data[column1], self.data[column2])
        return statistic, p_value

    def visualize_distribution(self, column):
        """展示数据分布

        Args:
            column (_type_): _description_
        """
        sns.histplot(self.data[column], kde=True)
        plt.show()

    def cal_pearson(self):
        """计算Pearson相关系数矩阵"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        corr = numeric_data.corr(method="pearson")

        return corr

    def plot_pearson(self, half=False, show_text=True, **kwargs):
        """绘制Pearson相关系数矩阵的热力图
        Args:
            half (bool, optional): 是否只显示半边. Defaults to False.
            show_text (bool, optional): 是否显示注释. Defaults to True.
            size (int, optional): 图像的大小. Defaults to 1000.
        """
        corr = self.cal_pearson()
        mask = (
            np.triu(np.ones_like(corr, dtype=bool), k=1)
            if half
            else np.ones_like(corr, dtype=bool)
        )
        z = np.where(mask, corr.values, None)
        annotation_text = np.where(mask, corr.round(2).values.astype(str), None)

        fig = go.Figure()

        # 添加一个白色的背景矩阵
        fig.add_trace(
            go.Heatmap(
                z=np.ones_like(corr),
                x=list(corr.columns),
                y=list(corr.index),
                colorscale=[[0, "white"], [1, "white"]],
                showscale=False,
                text=None,
            )
        )

        # 在背景矩阵上添加热图
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=list(corr.columns),
                y=list(corr.index),
                text=annotation_text if show_text else None,
                hoverinfo="x+y+z",
                hovertext=annotation_text,
                colorscale="Viridis",
                showscale=True,
            )
        )

        # 添加注释
        if show_text:
            annotations = [
                go.layout.Annotation(
                    text=annotation_text[i, j],
                    x=col,
                    y=row,
                    showarrow=False,
                    font=dict(size=12),
                )
                for i, row in enumerate(corr.index)
                for j, col in enumerate(corr.columns)
                if (half and j >= i or not half) and i != j
            ]
            fig.update_layout(annotations=annotations)

        size = kwargs.get("size", 1000)
        fig.update_layout(
            height=size,
            width=size,
            xaxis=dict(tickangle=-45),
        )
        fig.show()

    @deprecated
    def plot_pearson_old(self, half=False, show_text=True, **kwargs):
        """绘制Pearson相关系数矩阵的热力图
        Args:
            corr (_type_): _description_
            half (bool, optional): _description_. Defaults to False.
            show_text (bool, optional): _description_. Defaults to True.
            size (int, optional): _description_. Defaults to 1000.
        """
        corr = self.cal_pearson()
        if half:
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            z = np.where(mask, corr.values, None)
            annotation_text = np.where(mask, corr.round(2).values.astype(str), None)
        else:
            z = corr.values
            annotation_text = corr.round(2).values.astype(str)

        fig = go.Figure()

        # 添加一个白色的背景矩阵
        fig.add_trace(
            go.Heatmap(
                z=np.ones_like(corr),
                x=list(corr.columns),
                y=list(corr.index),
                colorscale=[[0, "white"], [1, "white"]],
                showscale=False,
                text=None,
            )
        )

        # 在背景矩阵上添加热图
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=list(corr.columns),
                y=list(corr.index),
                text=annotation_text if show_text else None,
                hoverinfo="x+y+z",
                hovertext=annotation_text,
                # hoverinfo='none',
                colorscale="Viridis",
                showscale=True,
            )
        )

        # 添加注释
        annotations = []
        if show_text:
            for i, row in enumerate(corr.index):
                for j, col in enumerate(corr.columns):
                    if half and j >= i:
                        if i != j:  # 不在对角线上
                            annotations.append(
                                go.layout.Annotation(
                                    text=annotation_text[i, j],
                                    x=col,
                                    y=row,
                                    showarrow=False,
                                    font=dict(size=12),
                                )
                            )
                    elif not half:
                        if i != j:
                            annotations.append(
                                go.layout.Annotation(
                                    text=annotation_text[i, j],
                                    x=col,
                                    y=row,
                                    showarrow=False,
                                    font=dict(size=12),
                                )
                            )

        size = kwargs.get("size", 1000)
        fig.update_layout(
            height=size,
            width=size,
            xaxis=dict(tickangle=-45),
            annotations=annotations if show_text else None,
        )
        fig.show()

    def _encode_dataframe(self):
        le = LabelEncoder()
        df_encoded = self.data.copy()
        for column in self.data.columns:
            if (
                self.data[column].dtype == "object"
                or self.data[column].dtype == "category"
            ):  # 如果列值的类型是对象类型（例如，字符串）
                df_encoded[column] = le.fit_transform(
                    self.data[column]
                )  # 适配并转换这一列
        return df_encoded

    def calculate_rank_correlations(self):
        """计算秩次相关"""
        df_encoded = self._encode_dataframe()

        # 使用 pandas 的 `corr` 函数计算 spearman rank correlation
        correlations = df_encoded.corr(method="spearman")
        return correlations

    def plot_spearman(self):
        """绘制Spearman相关系数矩阵的热力图"""
        corr = self.calculate_rank_correlations()
        # 对角线上的值填充为 1.0
        np.fill_diagonal(corr.values, 1.0)

        fig = px.imshow(
            corr,
            labels={"color": "Rank Correlation"},
            title="Rank Correlations Heatmap",
        )
        fig.show()

        return fig

    def cal_y_dist(self, label="is_damaged"):
        """计算某列正例和反例的占比"""
        positive_count = self.data[label].sum()  # Assuming 1s as the positive examples
        total_count = len(self.data[label])

        # Now, we can calculate the ratio
        ratio = positive_count / total_count
        print(f"正例占比: {round(ratio, 3)}")

        # Similarly for negative examples
        negative_count = total_count - positive_count
        negative_ratio = negative_count / total_count
        print(f"反例占比: {round(negative_ratio, 3)}")

    @staticmethod
    def show_report(df: pd.DataFrame, outpath="./your_report.html"):
        """输出 ydata profiling 报告"""
        # first_column = df.columns[0]
        # tmp = df.copy()
        # 检查第一个列名是否是元组
        # if isinstance(first_column, tuple):
        #     tmp.columns = [col[0] for col in tmp.columns]
        profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        profile.to_file(outpath)

    def cal_chi2(self, df: pd.DataFrame):
        """计算卡方"""
        # 存储每对列的卡方检验的P值
        chi2_values = pd.DataFrame(columns=df.columns, index=df.columns)

        # 对所有列进行卡方检验
        for column1 in df.columns:
            for column2 in df.columns:
                chi2, p, _, _ = stats.chi2_contingency(
                    pd.crosstab(df[column1], df[column2])
                )
                chi2_values.loc[column1, column2] = p

        # 输出卡方检验的P值结果
        print(chi2_values)
        return chi2_values

    @staticmethod
    def cal_col_correlation(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
        """计算列之间的相关系数"""
        correlation_matrix = df[cat_col].corr()

        return correlation_matrix

    def cal_spearman_col(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """计算Spearman相关系数"""
        correlation_matrix = df.corr(method="spearman")
        specific_column_correlation = correlation_matrix[column_name]
        return specific_column_correlation
