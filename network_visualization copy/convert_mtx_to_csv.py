from scipy.io import mmread
import pandas as pd

matrix = mmread('network//steam_friends_network.mtx').tocoo()

edges = pd.DataFrame({
    'Source': matrix.row,
    'Target': matrix.col
})

edges = edges[edges['Source'] <= edges['Target']]
edges.to_csv('network//edges.csv', index=False)
