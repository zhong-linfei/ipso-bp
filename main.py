
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
# 1. 加载数据
# 读取 CSV 文件
base_path = r'C:\Users\17785\Desktop\test1\ipso-bp\data'
train_data = pd.read_csv(base_path + '\\train.csv')
test_data = pd.read_csv(base_path + '\\test.csv')
sample_submission = pd.read_csv(base_path + '\\sample_submission.csv')

# 2. 选择特征和目标
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 
            'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 
            'YearBuilt', 'YearRemodAdd']
target = 'SalePrice'

X = train_data[features]
y = train_data[target]

# 3. 数据预处理
# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_data[features])
y_test = sample_submission[target]
# 4. 构建神经网络模型
model = Sequential([
    Dense(120, input_dim=len(features), activation='relu'),
    Dense(60, activation='relu'),
    Dense(1)  # 输出层，线性激活函数
])

# 5. 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 6. 训练模型
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# 7. 评估模型
val_loss, val_mae = model.evaluate(X_val, y_val)
print(f'Validation MAE: {val_mae}')

# 8. 使用测试数据进行预测
predictions = model.predict(X_test)




# 9. 绘制实际值和预测值的折线图
plt.figure(figsize=(14, 8))
plt.plot(y_test.values, label='Actual SalePrice', color='blue')
plt.plot(predictions, label='Predicted SalePrice', color='red')
plt.xlabel('Samples')
plt.ylabel('SalePrice')
plt.title('Actual vs Predicted SalePrice - Line Chart')
plt.legend()
plt.show()


