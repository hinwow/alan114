# ==================== 数据处理与可视化 ====================
import pandas as pd                  # 数据框操作（核心）
import numpy as np                   # 数值计算（核心）
import matplotlib.pyplot as plt      # 基础可视化（核心）
import seaborn as sns                # 高级统计可视化（核心）
import missingno as msno             # 缺失值可视化（需安装：pip install missingno）
from pandas_profiling import ProfileReport  # 自动化数据报告（需安装：pip install pandas-profiling）

# ==================== 特征工程（预处理/转换） ====================
# 标准化/归一化
from sklearn.preprocessing import StandardScaler    # Z-score标准化（核心）
from sklearn.preprocessing import MinMaxScaler      # 0-1归一化
from sklearn.preprocessing import RobustScaler      # 鲁棒缩放（抗异常值）

# 类别特征编码
from sklearn.preprocessing import OneHotEncoder     # 独热编码（核心）
from sklearn.preprocessing import OrdinalEncoder    # 序数编码（核心）
from category_encoders import TargetEncoder         # 目标编码（需安装：pip install category_encoders）

# 特征生成与选择
from sklearn.preprocessing import PolynomialFeatures  # 多项式特征生成（核心）
from sklearn.feature_selection import SelectKBest     # 基于统计的特征选择（核心）
from sklearn.feature_selection import f_classif       # 分类任务ANOVA检验（配合SelectKBest）
from sklearn.feature_selection import RFE             # 递归特征消除（核心）
from sklearn.decomposition import PCA                # 主成分分析（降维，核心）

# 文本特征提取
from sklearn.feature_extraction.text import CountVectorizer  # 词频向量化（核心）
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量化（核心）

# ==================== 模型与训练（分类/回归） ====================
# 线性模型
from sklearn.linear_model import LogisticRegression  # 逻辑回归（分类，核心）
from sklearn.linear_model import LinearRegression    # 线性回归（回归，核心）
from sklearn.linear_model import Ridge               # 岭回归（L2正则化，核心）
from sklearn.linear_model import Lasso               # 套索回归（L1正则化，核心）
from sklearn.linear_model import ElasticNet          # 弹性网络（L1+L2正则化）

# 树模型与集成学习
from sklearn.tree import DecisionTreeClassifier      # 决策树（分类，核心）
from sklearn.tree import DecisionTreeRegressor       # 决策树（回归，核心）
from sklearn.ensemble import RandomForestClassifier   # 随机森林（分类，核心）
from sklearn.ensemble import RandomForestRegressor    # 随机森林（回归，核心）
from sklearn.ensemble import GradientBoostingClassifier  # GBDT（分类，核心）
from sklearn.ensemble import AdaBoostClassifier        # AdaBoost（分类）
from xgboost import XGBClassifier, XGBRegressor       # XGBoost（需安装，核心）
from lightgbm import LGBMClassifier, LGBMRegressor    # LightGBM（需安装，核心）
from catboost import CatBoostClassifier, CatBoostRegressor  # CatBoost（需安装）

# 支持向量机（SVM）
from sklearn.svm import SVC                           # SVM（分类，核心）
from sklearn.svm import SVR                           # SVM（回归）

# 最近邻算法
from sklearn.neighbors import KNeighborsClassifier     # KNN（分类，核心）
from sklearn.neighbors import KNeighborsRegressor      # KNN（回归）

# 贝叶斯与概率模型
from sklearn.naive_bayes import GaussianNB            # 高斯朴素贝叶斯（分类，核心）
from sklearn.naive_bayes import MultinomialNB         # 多项式朴素贝叶斯（文本分类）

# 神经网络（sklearn内置）
from sklearn.neural_network import MLPClassifier      # 多层感知机（分类）
from sklearn.neural_network import MLPRegressor       # 多层感知机（回归）

# ==================== 模型选择与评估 ====================
# 数据集划分与交叉验证
from sklearn.model_selection import train_test_split   # 训练集-测试集划分（核心）
from sklearn.model_selection import KFold              # K折交叉验证（核心）
from sklearn.model_selection import StratifiedKFold    # 分层K折（保持类别分布，核心）
from sklearn.model_selection import RepeatedKFold      # 重复K折（增加稳定性）

# 超参数调优
from sklearn.model_selection import GridSearchCV       # 网格搜索（核心）
from sklearn.model_selection import RandomizedSearchCV  # 随机搜索（核心）
from sklearn.model_selection import cross_val_score     # 交叉验证评分（核心）

# 分类评估指标（核心）
from sklearn.metrics import accuracy_score            # 准确率
from sklearn.metrics import precision_score           # 精确率
from sklearn.metrics import recall_score              # 召回率
from sklearn.metrics import f1_score                   # F1分数
from sklearn.metrics import roc_auc_score             # ROC-AUC
from sklearn.metrics import confusion_matrix          # 混淆矩阵
from sklearn.metrics import classification_report      # 分类报告

# 回归评估指标（核心）
from sklearn.metrics import mean_squared_error         # 均方误差（MSE）
from sklearn.metrics import mean_absolute_error        # 平均绝对误差（MAE）
from sklearn.metrics import r2_score                   # R²分数
from sklearn.metrics import mean_squared_log_error     # 均方对数误差（MSLE）

# 评估可视化
from sklearn.metrics import RocCurveDisplay            # ROC曲线绘制（核心）
from sklearn.metrics import PrecisionRecallDisplay     # 精确率-召回率曲线绘制

# ==================== 不平衡数据处理 ====================
from imblearn.over_sampling import SMOTE               # 过采样（需安装：pip install imbalanced-learn，核心）
from imblearn.under_sampling import RandomUnderSampler  # 欠采样
from imblearn.pipeline import Pipeline as ImbPipeline   # 支持SMOTE的流水线（与sklearn兼容）

# ==================== 实用工具 ====================
from sklearn.pipeline import Pipeline                  # 流水线（整合预处理+模型，核心）
from sklearn.compose import ColumnTransformer          # 多列不同预处理组合（核心）
from sklearn.base import BaseEstimator, TransformerMixin  # 自定义转换器基类
from joblib import dump, load                           # 模型保存与加载（比pickle高效，核心）
    