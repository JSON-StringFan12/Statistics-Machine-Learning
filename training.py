from tensorflow import keras
import numpy as np
import pandas as pd
from indexes import ColumnKey

division = [
    4, 1, 1, 12, 26, 53
]

filePath = './Statistics.xlsx'

ColumnKey.insert(3, [])

excel = pd.read_excel('./Statistics.xlsx')
excel = np.asarray(excel)
excel = np.delete(excel, [15, 16, 158], axis=0)
for v in excel:
    for i in range(0, len(v)):
        if i != 3:
            v[i] = ColumnKey[i].index(str(v[i]))
        v[i] = float(v[i])
        v[i] /= division[i]

excel = np.asarray(excel)

train_y = np.delete(excel, [0, 1, 2, 4, 5], axis=1)
train_x = np.delete(excel, 3, axis=1)

train_x = np.array(train_x, dtype=np.float)
train_y = np.array(train_y, dtype=np.float)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(5,)),
    keras.layers.Dense(80, activation='relu'),
    keras.layers.Dense(40, activation='relu'),
    keras.layers.Dense(1, activation='relu')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(train_y.shape)

model.fit(x=train_x, y=train_y, epochs=16, batch_size=1, shuffle=True)
model.save('model')
