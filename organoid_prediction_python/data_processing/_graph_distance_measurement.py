# code taken from this file: 
# https://github.com/jnowak90/GraVisGUI/blob/89a8f0945c27596cde13f0aad81fe17ed9d9b6a2/SourceCode/ShapeGUI.py#L1834
import networkx as nx
import numpy as np
import scipy as sp

def calculate_distance_matrix(visibilityGraphs:list)-> np.ndarray:
    """
    calculate the distance matrix of the input visibility graphs
    """
    distanceMatrix = np.zeros((len(visibilityGraphs), len(visibilityGraphs)))
    for index, graph1 in enumerate(visibilityGraphs):
        for pair, graph2 in enumerate(visibilityGraphs):
            distance = _calculate_Laplacian(graph1, graph2)
            distanceMatrix[index, pair] = distance
    return(distanceMatrix)

def _calculate_Laplacian(graph1, graph2):
    """
    calculate the distance between two graphs using the Kolmogorov-Smirnov statistic of the eigenvalue 
    distributions of the Laplacian matrices
    """
    laplacianGraph1 = nx.laplacian_matrix(graph1).toarray()
    laplacianGraph2 = nx.laplacian_matrix(graph2).toarray()
    eigenvaluesGraph1 = np.linalg.eig(laplacianGraph1)[0]
    normalizedEigenvaluesGraph1 = eigenvaluesGraph1 / np.max(eigenvaluesGraph1)
    eigenvaluesGraph2 = np.linalg.eig(laplacianGraph2)[0]
    normalizedEigenvaluesGraph2 = eigenvaluesGraph2 / np.max(eigenvaluesGraph2)
    distance = sp.stats.ks_2samp(normalizedEigenvaluesGraph1, normalizedEigenvaluesGraph2)[0]
    return(distance)

