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
