import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Read data in from file
# authentic or counterfeit bill detection
# # categorization whether that bank note is considered to be authentic or a counterfeit note

data = pd.read_csv("banknotes.csv")
print(data)
print(data.head(10))

with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    # labeling
    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            # counterfeit bill
            "label": 1 if row[4] == "0" else 0
        })



# Separate data into training and testing groups
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]




X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)

X_training = np.array(X_training)
y_training = np.array(y_training)
X_testing = np.array(X_testing)
y_testing = np.array(y_testing)

# Create a neural network
model = tf.keras.models.Sequential()

# Add a hidden layer with 8 units, with ReLU activation
model.add(tf.keras.layers.Dense(8, input_shape=(4,), activation="relu"))

# Add output layer with 1 unit, with sigmoid activation
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Train neural network
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_training, y_training, epochs=20)

# Evaluate how well model performs
# the verbose parameter controls the amount of information displayed in the console during execution
model.evaluate(X_testing, y_testing, verbose=2)

model.save('banknote_model.keras')

# Make the prediction# Make the prediction

#example_data = np.array([[3.910200, 6.06500, -2.453400, -0.68234]])
#example_data2 = np.array([[2.37180, 7.49080, 0.015989, -1.74140]])
example_data3 = np.array([[0.60731, 3.95440, -4.772000, -4.48530]])

prediction = model.predict(example_data3)

# Convert the probability to binary output (0 or 1)
prediction_class = (prediction > 0.5).astype(int)

print(f"Prediction (probability): {prediction}")
print(f"Prediction (class): {prediction_class}")
# zero is authentic one is counterfeit

if prediction_class == 0:
    print("The bill is authentic.")
else:
    print("The bill is counterfeit.")




