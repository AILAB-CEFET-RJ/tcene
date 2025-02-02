import pickle
import matplotlib.pyplot as plt

# Load the pickle file
file_path = 'c0.pkl'  # Replace {iteration} with the appropriate iteration number
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Access the 2D embeddings and cluster centers
z_2d = data['z_2d']
clust_2d = data['clust_2d']

# Plot the 2D embeddings and cluster centers
plt.figure(figsize=(8, 6))

# Plot the 2D embeddings
plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10, c='blue', label='Data Points', alpha=0.5)

# Plot the 2D cluster centers
plt.scatter(clust_2d[:, 0], clust_2d[:, 1], s=200, c='red', label='Cluster Centers', marker='X')

# Adding labels and title
plt.title('2D Visualization of Data and Cluster Centers')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

# Show the plot
plt.show()
