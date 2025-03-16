import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# อ่านข้อมูล
data = pd.read_csv('data/car.csv')

# ทำการ log transform ราคา
data['selling_price'] = np.log1p(data['selling_price'])

# เลือกคุณสมบัติที่ใช้
X = data[['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']]
y = data['selling_price']

# แบ่งข้อมูลเป็น training และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize ข้อมูล
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล
model = Sequential([
    Dense(512, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),  # Dropout Layer เพื่อป้องกัน overfitting
    Dense(256, activation='relu'),
    Dropout(0.3),  # เพิ่ม Dropout ที่ layer ถัดไป
    Dense(1)  # Output Layer
])

# คอมไพล์โมเดล
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

# กำหนด EarlyStopping เพื่อตรวจสอบการเบรค
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# ฝึกโมเดล
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# ประเมินผลโมเดล
loss, mae = model.evaluate(X_test, y_test)

# แสดงผล
print(f'Mean Squared Error on Test Data: {loss}')
print(f'Mean Absolute Error on Test Data: {mae}')

# บันทึกโมเดล
model.save('car_model.h5')

# บันทึก scaler
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
