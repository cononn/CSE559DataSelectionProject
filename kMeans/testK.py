# test different k values

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
import shutil
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# tried joblib, ended up being slower during testing


# function to process a file through batching
# batch size is set to 6 as each file has 6 sequences so they get processed at once in a batch
def processFile(filePath, tokenizer, model):
    # a single file will have multiple sequences
    with open(filePath, 'r') as f:
        data = f.readlines()

    sequences = [line.strip() for line in data]
    embeddings = []

    # process all sequences in file
    embeddings.extend(getEmbeddingsInBatch(sequences, tokenizer, model))

    # flatten to one embedding through averaging (for kMeans)
    fileEmbedding = np.mean(embeddings, axis=0)
    return fileEmbedding

# function to get embeddings from a batch (used in processFile)
def getEmbeddingsInBatch(sequences, tokenizer, model):
    # get tokens of all sequences in one file at a time
    tokens = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
        # normal embeddings are for individual tokens, cls embeddings represent entire sequence (we want this)
        clsEmbeddings = outputs.last_hidden_state[:, 0, :]
        clsEmbeddingsConverted = clsEmbeddings.squeeze(1).numpy()
    return clsEmbeddingsConverted

# need a separate function to call processFileBatch or else it errors (due to parallel process most likely)
# want to fork calling a function, not within the function
def processFileEmbedding(filePath, tokenizer, model):
    return processFile(filePath, tokenizer, model)

# process the files in parallel to save computation time (and ensure multiple CPUs are being utilized)
def processFilesParallel(filePaths, tokenizer, model):
    with multiprocessing.Pool() as Pool:
        # applies vars to processFileEmbedding to call function in parallel (based on how many files in filePaths)
        embeddings = Pool.starmap(processFileEmbedding, [(filePath, tokenizer, model) for filePath in filePaths])
    return embeddings

def plotClusters(X, labels, k):
    # PCA applied to reduce data to two dimensions
    pca = PCA(n_components=2)
    reducedData = pca.fit_transform(X)

    # scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reducedData[:, 0], reducedData[:, 1], c=labels, cmap='viridis', alpha=0.5)
    
    # color bar to distinguish clusters
    plt.colorbar(label='Cluster Label')

    # title
    plt.title("Cluster Visualization")

    # save to png
    outpath = f"clustersGraph_{k}.png"
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {outpath}")

    plt.close()

def main():
    # initialize parameters
    # use tcr BERT to gain numerical representations for data (to be able to use kMeans)
    # initially used hot one encoding => less representative of relationships within sequence
    modelName = "wukevin/tcr-bert"
    trainFolder = 'train_prefix'

    # initialize model
    # to turn files into numerical representation, use TCR-Bert (pre-trained)
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModel.from_pretrained(modelName)

    # load files within train folder (train_prefix)
    filePaths = [os.path.join(trainFolder, filename) for filename in os.listdir(trainFolder) if filename.endswith('.txt')]

    # process files to get embeddings (parallel computing: multiple files being processed at once)
    fileEmbeddings = processFilesParallel(filePaths, tokenizer, model)

    # k values being tested
    kValues = [3, 5, 7, 10, 15]

    # from embeddings, apply KMeans (batch size can be changed depending on number of data points and storage)
    batchSize = 5000
    X = np.array(fileEmbeddings)

    # test different cluster values to see which k value will be the best
    for k in kValues:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batchSize, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_
        # plot clusters
        plotClusters(X, labels, k)


if __name__ == "__main__":
    main()