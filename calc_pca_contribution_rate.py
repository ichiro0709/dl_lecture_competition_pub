import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_embeddings(embedding_path):
    return np.load(embedding_path, allow_pickle=True)

def plot_pca_variance(embeddings):
    # データの標準化
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    # 主成分の最大数をデータの次元数に設定
    n_components = scaled_embeddings.shape[1]
    
    pca = PCA(n_components=n_components)
    pca.fit(scaled_embeddings)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_components + 1), explained_variance_ratio, marker='o', label='Individual')
    plt.plot(range(1, n_components + 1), cumulative_variance_ratio, marker='s', label='Cumulative')
    
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    
    plt.title('Explained Variance by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 閾値を超えるために必要な主成分の数を表示
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"Number of components for 95% variance: {n_components_95}")

if __name__ == "__main__":
    train_embeddings = load_embeddings("./data/train_embeddings.npy")
    plot_pca_variance(train_embeddings)
