# ==================== 数据处理与可视化 ====================
import pandas as pd                  # 数据框操作（核心）
import numpy as np                   # 数值计算（核心）
import matplotlib.pyplot as plt      # 基础可视化（核心）
import seaborn as sns                # 高级统计可视化（核心）
import missingno as msno             # 缺失值可视化（需安装：pip install missingno）
from pandas_profiling import ProfileReport  # 自动化数据报告（需安装：pip install pandas-profiling）

# ==================== 特征工程（预处理/转换） ====================
# 标准化/归一化（合并sklearn.preprocessing）
from sklearn.preprocessing import (
    StandardScaler,    # Z-score标准化（核心）
    MinMaxScaler,      # 0-1归一化
    RobustScaler,      # 鲁棒缩放（抗异常值）
    OneHotEncoder,     # 独热编码（核心）
    OrdinalEncoder,    # 序数编码（核心）
    PolynomialFeatures # 多项式特征生成（核心）
)
from category_encoders import TargetEncoder  # 目标编码（需安装：pip install category_encoders）

# 特征选择与降维（合并sklearn.feature_selection和sklearn.decomposition）
from sklearn.feature_selection import (
    SelectKBest,  # 基于统计的特征选择（核心）
    f_classif,    # 分类任务ANOVA检验（配合SelectKBest）
    RFE           # 递归特征消除（核心）
)
from sklearn.decomposition import PCA  # 主成分分析（降维，核心）

# 文本特征提取（合并sklearn.feature_extraction.text）
from sklearn.feature_extraction.text import (
    CountVectorizer,  # 词频向量化（核心）
    TfidfVectorizer   # TF-IDF向量化（核心）
)

# ==================== 模型与训练（分类/回归） ====================
# 线性模型（合并sklearn.linear_model）
from sklearn.linear_model import (
    LogisticRegression,  # 逻辑回归（分类，核心）
    LinearRegression,    # 线性回归（回归，核心）
    Ridge,               # 岭回归（L2正则化，核心）
    Lasso,               # 套索回归（L1正则化，核心）
    ElasticNet           # 弹性网络（L1+L2正则化）
)

# 树模型与集成学习（合并sklearn.tree和sklearn.ensemble）
from sklearn.tree import (
    DecisionTreeClassifier,  # 决策树（分类，核心）
    DecisionTreeRegressor    # 决策树（回归，核心）
)
from sklearn.ensemble import (
    RandomForestClassifier,    # 随机森林（分类，核心）
    RandomForestRegressor,     # 随机森林（回归，核心）
    GradientBoostingClassifier,  # GBDT（分类，核心）
    AdaBoostClassifier         # AdaBoost（分类）
)
# 第三方集成库（单独导入，因来自不同包）
from xgboost import XGBClassifier, XGBRegressor       # XGBoost（需安装，核心）
from lightgbm import LGBMClassifier, LGBMRegressor    # LightGBM（需安装，核心）
from catboost import CatBoostClassifier, CatBoostRegressor  # CatBoost（需安装）

# 支持向量机（合并sklearn.svm）
from sklearn.svm import (
    SVC,  # SVM（分类，核心）
    SVR   # SVM（回归）
)

# 最近邻算法（合并sklearn.neighbors）
from sklearn.neighbors import (
    KNeighborsClassifier,  # KNN（分类，核心）
    KNeighborsRegressor    # KNN（回归）
)

# 贝叶斯与概率模型（合并sklearn.naive_bayes）
from sklearn.naive_bayes import (
    GaussianNB,    # 高斯朴素贝叶斯（分类，核心）
    MultinomialNB  # 多项式朴素贝叶斯（文本分类）
)

# 神经网络（合并sklearn.neural_network）
from sklearn.neural_network import (
    MLPClassifier,  # 多层感知机（分类）
    MLPRegressor    # 多层感知机（回归）
)

# ==================== 模型选择与评估 ====================
# 数据集划分与交叉验证（合并sklearn.model_selection）
from sklearn.model_selection import (
    train_test_split,   # 训练集-测试集划分（核心）
    KFold,              # K折交叉验证（核心）
    StratifiedKFold,    # 分层K折（保持类别分布，核心）
    RepeatedKFold,      # 重复K折（增加稳定性）
    GridSearchCV,       # 网格搜索（核心）
    RandomizedSearchCV, # 随机搜索（核心）
    cross_val_score     # 交叉验证评分（核心）
)

# 评估指标（合并sklearn.metrics）
from sklearn.metrics import (
    accuracy_score,          # 准确率（分类）
    precision_score,         # 精确率（分类）
    recall_score,            # 召回率（分类）
    f1_score,                # F1分数（分类）
    roc_auc_score,           # ROC-AUC（分类）
    confusion_matrix,        # 混淆矩阵（分类）
    classification_report,   # 分类报告（分类）
    mean_squared_error,      # 均方误差（MSE，回归）
    mean_absolute_error,     # 平均绝对误差（MAE，回归）
    r2_score,                # R²分数（回归）
    mean_squared_log_error   # 均方对数误差（MSLE，回归）
)

# 评估可视化（合并sklearn.metrics）
from sklearn.metrics import (
    RocCurveDisplay,        # ROC曲线绘制（核心）
    PrecisionRecallDisplay  # 精确率-召回率曲线绘制
)

# ==================== 不平衡数据处理 ====================
from imblearn.over_sampling import SMOTE               # 过采样（需安装：pip install imbalanced-learn，核心）
from imblearn.under_sampling import RandomUnderSampler  # 欠采样
from imblearn.pipeline import Pipeline as ImbPipeline   # 支持SMOTE的流水线（与sklearn兼容）

# ==================== 实用工具 ====================
from sklearn.pipeline import Pipeline                  # 流水线（整合预处理+模型，核心）
from sklearn.compose import ColumnTransformer          # 多列不同预处理组合（核心）
from sklearn.base import (
    BaseEstimator,        # 自定义估计器基类
    TransformerMixin      # 自定义转换器基类
)
from joblib import dump, load                           # 模型保存与加载（比pickle高效，核心）
    