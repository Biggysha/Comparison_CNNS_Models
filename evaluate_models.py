import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_test = to_categorical(y_test, 10)

# Load models and evaluate
results = []
for name in ['lenet', 'alexnet', 'vggnet', 'googlenet', 'resnet']:
    model = tf.keras.models.load_model(f'models/{name}_mnist.h5')
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    results.append({'Model': name, 'Test Loss': loss, 'Test Accuracy': accuracy})

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results/model_results.csv', index=False)
print(results_df)


data = {
    'Model': ['lenet', 'alexnet', 'vggnet', 'googlenet', 'resnet'],
    'Test Loss': [0.033918, 0.040433, 0.029672, 0.030137, 0.045531],
    'Test Accuracy': [0.9902, 0.9903, 0.9925, 0.9914, 0.9869]
}

# Create a DataFrame
results_df = pd.DataFrame(data)

# Set the style for the plots
sns.set(style="whitegrid")

# Plot 1: Test Accuracy Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Test Accuracy', data=results_df, palette='viridis')
plt.title('Test Accuracy of Different Models on MNIST', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Test Accuracy', fontsize=14)
plt.ylim(0.98, 1.0)  # Set y-axis limits for better visualization
plt.savefig('results/accuracy_comparison.png')  # Save the plot
plt.show()

# Plot 2: Test Loss Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Test Loss', data=results_df, palette='magma')
plt.title('Test Loss of Different Models on MNIST', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Test Loss', fontsize=14)
plt.ylim(0.02, 0.05)  # Set y-axis limits for better visualization
plt.savefig('results/loss_comparison.png')  # Save the plot
plt.show()

# Plot 3: Combined Bar Plot (Accuracy and Loss)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Test Accuracy', data=results_df, palette='viridis')
plt.title('Test Accuracy', fontsize=14)
plt.ylim(0.98, 1.0)

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Test Loss', data=results_df, palette='magma')
plt.title('Test Loss', fontsize=14)
plt.ylim(0.02, 0.05)

plt.suptitle('Model Performance Comparison on MNIST', fontsize=16)
plt.tight_layout()
plt.savefig('results/combined_comparison.png')  # Save the plot
plt.show()