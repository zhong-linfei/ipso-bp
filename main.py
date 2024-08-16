import pandas as pd
from sklearn.preprocessing import StandardScaler  
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 读取 CSV 文件
file_train_path = './data/train.csv'
file_xtest_path = './data/test.csv'
file_ytest_path = './data/sample_submission.csv'
#训练集
df_train = pd.read_csv(file_train_path)
# 先执行correlations.py选取相关系数大于0.5的参数作为输入
X_train = df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']]
y_train = df_train['SalePrice']

#测试集输入
df_xtest = pd.read_csv(file_xtest_path)
X_test = df_xtest[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']]

#测试集输出
y_test = pd.read_csv(file_ytest_path)['SalePrice']

# 标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
#修改12345
model = Sequential()
model.add(Dense(54, input_dim=X_train_scaled.shape[1], activation='relu'))  # 输入层和第一个隐藏层
model.add(Dense(1, activation='relu'))  # 第二个隐藏层
model.add(Dense(1, activation='linear'))  # 输出层
