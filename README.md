# Neural Network Model Optimization for Freight Cost Prediction in Shipments

## Model Training Instances and Configuration Details

| Training Instance | Optimizer Used | Regularizer Used | Epochs | Early Stopping | Number of Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|------------------|----------------|------------------|---------|----------------|-----------------|---------------|-----------|-----------|---------|------------|
| Instance 1 (Adam_Dropout_BatchNorm) | Adam | Dropout + BatchNorm | 50 | Yes | 3 (64, 32, 3) | 0.001 | 0.71584 | 0.70899 | 0.71584 | 0.70845 |
| Instance 2 (RMSprop_L2_Deep) | RMSprop | L2 | 50 | Yes | 4 (128, 64, 32, 3) | 0.005 | 0.64795 | 0.61032 | 0.64795 | 0.63419 |
| Instance 3 (SGD_Momentum_BatchNorm) | SGD | BatchNorm | 50 | Yes | 5 (96, 64, 48, 32, 16, 3) | 0.01 | 0.66471 | 0.65222 | 0.66471 | 0.65507 |
| Instance 4 (Adam_ElasticNet_Dropout) | Adam | L1 + L2 | 50 | Yes | 4 (128, 96, 64, 32, 3) | 0.002 | 0.33529 | 0.16838 | 0.33529 | 0.11242 |
| Instance 5 (RMSprop_L1_BatchNorm) | RMSprop | L1 + BatchNorm | 50 | Yes | 4 (64, 48, 32, 3) | 0.001 | 0.72674 | 0.72462 | 0.72674 | 0.72372 |

## Analysis of Optimization Techniques

### Best Performing Model: Instance 5 (RMSprop_L1_BatchNorm)
![RMSprop_L1_BatchNorm_confusion_matrix](https://github.com/user-attachments/assets/7cdd94ee-7e43-49b0-b338-39ad87f93f4c)
- Highest accuracy: 0.72674
- Best F1 Score: 0.72462
- Most balanced precision-recall trade-off
- Optimal combination of RMSprop optimizer with L1 regularization and BatchNorm

### Performance Analysis

1. **Optimizer Effectiveness**
   - RMSprop showed strong performance in Instance 5 (72.67%) and decent performance in Instance 2 (64.79%)
   - Adam performed well in Instance 1 (71.58%) but poorly in Instance 4 (33.53%)
   - SGD with momentum achieved moderate performance (66.47%)

2. **Regularization Impact**
   - BatchNorm + L1 (Instance 5) provided the best regularization strategy
   - Dropout + BatchNorm (Instance 1) showed strong performance
   - ElasticNet (L1 + L2) in Instance 4 proved too aggressive, leading to underfitting
   - Single L2 regularization (Instance 2) showed moderate effectiveness

3. **Architecture Considerations**
   - 4-layer architecture proved optimal (Instance 5)
   - Deeper networks (5 layers in Instance 3) didn't necessarily improve performance
   - Layer size progression showed importance of balanced architecture

### Key Findings

1. **Optimal Configuration**
   - RMSprop optimizer with 0.001 learning rate
   - L1 regularization combined with BatchNorm
   - 4-layer architecture with moderate layer sizes
   - Early stopping implementation

2. **Notable Observations**
   - Combination of BatchNorm with L1 regularization provides superior results
   - Too aggressive regularization (Instance 4) can severely impact performance
   - Moderate-sized architectures outperform deeper networks

## Conclusion
The analysis reveals that the combination of RMSprop optimizer with L1 regularization and BatchNorm (Instance 5) provides the most robust performance for freight cost prediction. This configuration achieves the best balance between model complexity and regularization, resulting in superior accuracy (72.67%) and consistent performance across all metrics (F1: 0.72462, Precision: 0.72372, Recall: 0.72674).

## Future Recommendations

1. **Fine-tuning Opportunities**
   - Experiment with learning rate schedules for RMSprop
   - Investigate different BatchNorm configurations
   - Test variations of L1 regularization strength

2. **Architecture Optimization**
   - Focus on 4-layer architectures with varied neuron configurations
   - Explore residual connections for better gradient flow
   - Consider layer size optimization

3. **Regularization Strategy**
   - Further explore L1 + BatchNorm combinations
   - Test graduated regularization strategies
   - Investigate adaptive regularization techniques
