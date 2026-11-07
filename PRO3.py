import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import utils

transactions1 = pd.read_csv('C:/Users/Donete/Documents/Projets/обнаружение-мошенничества/creditcard.csv', delimiter=',')
transactions = transactions1.head(1000)  

X = transactions.iloc[:, 0:30].values
Y = transactions[['Class']].values

X_t, X_test, Y_t, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

Sc = StandardScaler()
X_t = Sc.fit_transform(X_t)
X_test = Sc.transform(X_test)

Y_t_cat = utils.to_categorical(Y_t, 2)
Y_test_cat = utils.to_categorical(Y_test, 2)

model = Sequential()
model.add(Dense(30, input_dim=X_t.shape[1], activation='relu', name='input'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax', name='Output'))

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

results = model.fit(X_t, Y_t_cat, verbose=1, epochs=30, validation_data=(X_test, Y_test_cat))

model.save("модель_мошенничество.keras")

plt.figure(0)
plt.plot(results.history['accuracy'], label='Train Accuracy')
plt.plot(results.history['val_accuracy'], label='Test Accuracy')
plt.legend()
plt.title("Accuracy")
plt.show()

plt.figure(1)
plt.plot(results.history['loss'], label='Train Loss')
plt.plot(results.history['val_loss'], label='Test Loss')
plt.legend()
plt.title("Loss")
plt.show()

Y_res = model.predict(X_test)
acc = 0

for i in range(len(Y_res)):
    if np.argmax(Y_res[i]) == Y_test[i]:
        acc += 1

acc /= float(len(Y_res))
acc *= 100
print('accuracy = ', acc, '%')

count = 0
accFail = 0

for i in range(len(Y_res)):
    if Y_test[i] == 1:  
        count += 1
        if np.argmax(Y_res[i]) == Y_test[i]:
            accFail += 1

if count > 0:
    accFail /= count
    accFail *= 100
    print('accuracy (fraudulent transactions only) =', accFail, '%')
else:
    print('Aucune transaction frauduleuse dans l’échantillon sélectionné.')
