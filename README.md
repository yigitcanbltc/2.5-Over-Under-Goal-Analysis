# Match Outcome Predictor

This project uses a neural network model to predict whether there will be more than 3 goals in a match based on various match statistics. The model processes input features, applies preprocessing steps, and trains a neural network to make accurate predictions.

## Features in the Dataset

The following features are used as input (`X`) for the prediction model:

- **1.5A**: Odds for under 1.5 goals in the match.
- **1.5Ü**: Odds for over 1.5 goals in the match.
- **2.5A**: Odds for under 2.5 goals in the match.
- **2.5Ü**: Odds for over 2.5 goals in the match.
- **3.5A**: Odds for under 3.5 goals in the match.
- **3.5Ü**: Odds for over 3.5 goals in the match.
- **KG Var**: Odds for both teams to score (Yes).
- **KG Yok**: Odds for both teams to score (No).
- **TG 0-1**: Odds for the total goals in the match to be between 0 and 1.
- **TG 6+**: Odds for the total goals in the match to exceed 6.
- **İY 1.5A**: Odds for under 1.5 goals in the first half.
- **İY 1.5Ü**: Odds for over 1.5 goals in the first half.

The target feature (`y`) is **Alt/Üst**, which indicates whether the total goals in the match are categorized as "over" or "under" based on specific thresholds.

## How It Works

1. **Data Preprocessing**:

   - Missing values are handled by filling them with zeros.
   - The target column (`Alt/Üst`) is encoded into numeric labels for model training.

2. **Model Architecture**:

   - The model is a feedforward neural network built using TensorFlow/Keras.
   - It has:
     - Input layer based on the number of features.
     - Two hidden layers with ReLU activation and dropout for regularization.
     - Output layer with softmax activation for multi-class classification.

3. **Training**:

   - The model uses categorical crossentropy as the loss function and Adam optimizer.
   - Validation and training performance are monitored to save the best-performing model.

4. **Evaluation**:
   - Predictions are made on the test set, and the accuracy of the model is reported.

## Requirements

- Python 3.7+
- TensorFlow
- pandas
- scikit-learn
- numpy

## Usage

1. Clone this repository and place your dataset (`matches.xlsx`) in the correct directory.
2. Install the required dependencies:
   ```bash
   pip install tensorflow pandas scikit-learn numpy
   3.Run the script:
    python match_predictor.py
   ```
