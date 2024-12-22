from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import os
import numpy as np

# Creating label map for actions
label_map = {label: num for num, label in enumerate(actions)}

# Initialize sequences and labels
sequences, labels = [], []

# Loading sequences and labels efficiently
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(action_path, str(sequence), f"{frame_num}.npy")
            if os.path.exists(file_path):
                window.append(np.load(file_path))
            else:
                print(f"File not found: {file_path}")
                window.append(np.zeros((63,)))  # Default value for missing frames, adjust as needed
        sequences.append(window)
        labels.append(label_map[action])

# Converting sequences and labels to numpy arrays and one-hot encoding labels
X = np.array(sequences)
y = to_categorical(labels)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# TensorBoard callback
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Building the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))  # Adjusted input shape based on sequence_length
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # Using len(actions) instead of actions.shape[0]

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[tb_callback])

# Saving the model
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
