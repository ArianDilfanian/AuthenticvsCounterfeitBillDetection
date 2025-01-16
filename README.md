# 💵 Authentic vs Counterfeit Bill Detection Using Machine Learning 🏦

## Overview 🚀
This project is designed to identify whether a given banknote is authentic or counterfeit using machine learning. The model leverages **neural networks** to classify banknotes based on four features extracted from a dataset. Using TensorFlow and a simple neural network, we can predict whether a bill is counterfeit or genuine.

## 🔧 Dependencies
Before running the code, make sure you have the following Python libraries installed:
- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `csv`

Install the dependencies using pip:
```bash
pip install tensorflow numpy pandas scikit-learn
```

## 📂 Dataset
The dataset used in this project is stored in `banknotes.csv`. The dataset contains 5 columns:
1. **V1**: Feature 1 (evidence of authenticity)
2. **V2**: Feature 2
3. **V3**: Feature 3
4. **V4**: Feature 4
5. **Class**: 0 for authentic bills, 1 for counterfeit bills

You can find the dataset [here](link-to-dataset) (replace with your actual link if needed).

## 🧠 Neural Network Model
The neural network model is built using TensorFlow and consists of:
- An **input layer** with 4 features.
- A **hidden layer** with 8 neurons and ReLU activation function.
- An **output layer** with 1 neuron using the sigmoid activation function, producing a probability that the banknote is counterfeit.

The model is trained using the **binary cross-entropy** loss function and **Adam** optimizer. The training accuracy and testing accuracy are evaluated to check the model's performance.

## 📊 Training and Testing
The dataset is split into training and testing sets using **train_test_split** from Scikit-learn. The model is trained for 20 epochs, and its performance is evaluated on the test set.

## 🖍️ Code Walkthrough
1. **Data Preprocessing**:
   - The CSV file is loaded and processed to separate the features (evidence) and labels (authentic/counterfeit).
   
2. **Model Creation**:
   - A simple neural network is created using TensorFlow, with one hidden layer and an output layer.

3. **Training**:
   - The model is trained using the training dataset for 20 epochs.
   
4. **Evaluation**:
   - The model is evaluated using the test set and the accuracy and loss are displayed.

5. **Prediction**:
   - A sample banknote is passed into the model to predict whether it is authentic or counterfeit.

### Example Usage:
```python
# Sample data input
example_data = np.array([[3.910200, 6.06500, -2.453400, -0.68234]])
prediction = model.predict(example_data)

# Convert the probability to binary output (0 or 1)
prediction_class = (prediction > 0.5).astype(int)

if prediction_class == 0:
    print("The bill is authentic.")
else:
    print("The bill is counterfeit.")
```

## 🚀 Model Performance:
After training for 20 epochs, the model achieved:
- **Training Accuracy**: 1.00
- **Testing Accuracy**: 1.00

### Evaluation:
```bash
Model Evaluation on Test Set:
- Accuracy: 100%
- Loss: 0.0607
```

## 🏆 Conclusion:
This machine learning model accurately detects counterfeit banknotes based on the provided features.



