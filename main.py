from tensorflow import keras
import numpy as np
import math
from indexes import ColumnKey

model = keras.models.load_model('model')

division = [
    4, 1, 1, 26, 53
]

questions = [
    'Learning Type (DL & hybrid, DL & virtual, DL only, virtual only or At School):',
    'Whether or Not Student is New (NEW or nan):',
    'Gender (M or F):',
    'City:',
    'Zip Code:',
]


def parseInput():
    s = [[54., 54., 54., 54., 54.]]
    for i in range(0, 5):
        print('Enter ' + questions[i])
        inp = input()
        for j in range(0, len(ColumnKey[i])):
            if ColumnKey[i][j] == inp:
                print('equal')
                s[0][i] = j
                s[0][i] /= division[i]
        if s[0][i] == 54:
            print('Answer not valid.')
            return parseInput()
    s = np.array(s, dtype=np.float)

    print(s)

    prediction = model.predict(s)
    if prediction > 1:
        prediction = 1
    print('Prediction:')
    return print(str(math.floor((prediction[0][0] * 12) + 0.5)) + 'th grade')


parseInput()

