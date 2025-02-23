# Freight Cost Prediction Model Analysis

## Overview
This project implements and compares multiple machine learning approaches for freight cost prediction, including various neural network architectures and XGBoost. The models were evaluated on their ability to classify freight costs into different categories.

## Model Implementations

### Neural Network Architectures

| Training Instance | Optimizer Used | Regularizer Used | Early Stopping | Accuracy | Loss | F1 Score | Precision | Recall |
|------------------|----------------|------------------|----------------|----------|------|-----------|-----------|---------|
| Adam_Dropout_BatchNorm | Adam | Dropout + BatchNorm | Yes | 0.7158 | 0.6583 | 0.7090 | 0.7084 | 0.7158 |
| RMSprop_L2_Deep | RMSprop | L2 | Yes | 0.6479 | 0.8250 | 0.6103 | 0.6342 | 0.6479 |
| SGD_Momentum_BatchNorm | SGD | BatchNorm | Yes | 0.6647 | 0.7683 | 0.6522 | 0.6551 | 0.6647 |
| Adam_ElasticNet_Dropout | Adam | L1 + L2 | Yes | 0.3353 | 1.0992 | 0.1684 | 0.1124 | 0.3353 |
| RMSprop_L1_BatchNorm | RMSprop | L1 + BatchNorm | Yes | 0.7267 | 0.7044 | 0.7246 | 0.7237 | 0.7267 |


https://github.com/user-attachments/assets/35ae6145-6b03-43cf-866f-6e79d3cb633a


![image](https://github.com/user-attachments/assets/ab9592c8-1b07-450c-afec-e8c253b67394)
![image](https://github.com/user-attachments/assets/963c6bd5-48cc-4797-bb0d-bdea7c02a911)

You can see more plot images in the prediction plots directory

### XGBoost Implementation
Best Parameters:
- colsample_bytree: 1.0
- gamma: 0
- learning_rate: 0.1
- max_depth: 5
- min_child_weight: 3
- n_estimators: [value]

Performance Metrics:
- Best Cross-validation Score: 0.7487
- Class-wise Performance:
  - Class 0: Precision=0.80, Recall=0.79, F1=0.80
  - Class 1: Precision=0.85, Recall=0.81, F1=0.83
  - Class 2: Precision=0.65, Recall=0.69, F1=0.67
- Overall Accuracy: 0.76
- Macro Average: Precision=0.77, Recall=0.76, F1=0.77

## Performance Analysis

### Neural Network Models

1. **Best Performing Configurations**
   - RMSprop_L1_BatchNorm (72.67% accuracy)
   - Adam_Dropout_BatchNorm (71.58% accuracy)
   
2. **Optimization Techniques Impact**
   - RMSprop optimizer shows consistent performance across implementations
   - BatchNorm proves effective when combined with appropriate regularization
   - L1 regularization with BatchNorm achieves the best balance
   
3. **Challenges Identified**
   - ElasticNet regularization (L1 + L2) severely impacted model performance (33.53% accuracy)
   - Higher loss values don't necessarily indicate worse performance

### XGBoost Performance

1. **Advantages**
   - Best overall accuracy (74.87% cross-validation)
   - More consistent performance across classes
   - Better handling of class imbalance

2. **Class-wise Analysis**
   - Strong performance on Class 1 (0.83 F1-score)
   - Balanced precision and recall for Classes 0 and 1
   - Slightly lower performance on Class 2

## Comparative Analysis

1. **Model Comparison**
   - XGBoost achieves the highest accuracy (74.87%)
   - Best Neural Network close behind (72.67%)
   - Both approaches show viable performance

2. **Trade-offs**
   - Neural Networks:
     - More hyperparameters to tune
     - Greater variance in performance
     - Higher potential for improvement with more data
   - XGBoost:
     - More stable performance
     - Better out-of-the-box performance
     - Easier to tune

## Conclusions and Recommendations

### Key Findings
1. XGBoost provides the most reliable performance
2. Neural networks can achieve comparable results with proper optimization
3. Regularization strategy significantly impacts neural network performance

### Best Practices Identified
1. BatchNorm improves stability in neural networks
2. RMSprop and Adam optimizers outperform SGD
3. L1 regularization more effective than L2 for this dataset

### Recommendations
1. **For Production Deployment**
   - Consider XGBoost as primary model
   - Use RMSprop_L1_BatchNorm neural network as complementary model
   - Implement ensemble approach for robust predictions

2. **For Further Improvement**
   - Experiment with deeper architectures in neural networks
   - Fine-tune XGBoost hyperparameters further
   - Collect more training data for underperforming classes

## Future Work
1. Implement ensemble methods combining both approaches
2. Explore advanced architectures for neural networks
3. Investigate feature engineering opportunities
4. Consider adding more evaluation metrics



## Setup and Usage Instructions

### Environment Setup
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install required packages
pip install tensorflow
pip install xgboost
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
```

### Project Structure
```
freight-cost-prediction/
│
├── data/
│   └── freight_data.csv
│
├── models/
│   ├── best_model_RMSprop_L1_BatchNorm.h5
│   └── best_xgboost_model.json
│
├── notebooks/
│   └── freight_cost_prediction.ipynb
│
└── README.md
```

### Running the Notebook
1. Open the Jupyter notebook:
```bash
jupyter notebook notebooks/freight_cost_prediction.ipynb
```

2. Ensure data file is in the correct location:
```python
data_path = '../data/freight_data.csv'
df = pd.read_csv(data_path)
```

3. Run all cells or step through the notebook sections:
   - Data Preprocessing
   - Model Training
   - Results Analysis
   - Visualization

### Loading and Using Saved Models

#### Neural Network Model
```python
import tensorflow as tf

# Load the best neural network model
model = tf.keras.models.load_model('models/best_model_RMSprop_L1_BatchNorm.h5')

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
```

#### XGBoost Model
```python
import xgboost as xgb

# Load the best XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('models/best_xgboost_model.json')

# Make predictions
xgb_predictions = xgb_model.predict(X_test)
```

### Making New Predictions
```python
def prepare_data(data):
    """
    Prepare new data for prediction
    """
    # Apply the same preprocessing steps used during training
    numerical_features = ['Weight (Kilograms)', 'Line Item Quantity', 
                         'Line Item Value', 'Unit of Measure (Per Pack)']
    
    # Scale numerical features
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    return data

def predict_freight_cost(data, model_type='neural_network'):
    """
    Make predictions using the best saved model
    
    Args:
        data: Preprocessed input data
        model_type: 'neural_network' or 'xgboost'
    
    Returns:
        Predictions array
    """
    if model_type == 'neural_network':
        model = tf.keras.models.load_model('models/best_model_RMSprop_L1_BatchNorm.h5')
        predictions = model.predict(data)
        return np.argmax(predictions, axis=1)
    else:
        model = xgb.XGBClassifier()
        model.load_model('models/best_xgboost_model.json')
        return model.predict(data)

# Example usage
new_data = pd.read_csv('new_freight_data.csv')
prepared_data = prepare_data(new_data)
predictions = predict_freight_cost(prepared_data, model_type='neural_network')
```

### Model Retraining
To retrain models with new data:

1. Update the data file
2. Run the notebook sections for model training
3. Models will be automatically saved with updated weights

### Troubleshooting
Common issues and solutions:

1. **Missing dependencies**:
   - Run `pip install -r requirements.txt`
   - Ensure all required packages are installed

2. **CUDA errors**:
   - Check TensorFlow GPU compatibility
   - Verify CUDA toolkit installation

3. **Memory issues**:
   - Reduce batch size in model training
   - Use data generators for large datasets

4. **Model loading errors**:
   - Verify file paths
   - Check model file integrity
   - Ensure compatible TensorFlow versions

## Support
For issues and questions:
1. Check the troubleshooting section
2. Raise an issue in the repository
3. Contact the development team

[Previous conclusions and recommendations remain the same]
