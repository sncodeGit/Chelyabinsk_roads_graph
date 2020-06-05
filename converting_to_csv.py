import pandas as pd
import networkx as nx
import config
import numpy as np
import pickle
from _collections import defaultdict

def convert_to_csv(graph, c_obj, c_nodes, output_path = config.path, adj_mat = False):
    if adj_mat:
        df_matrix = nx.to_pandas_adjacency(graph, dtype=np.uint8)
        df_matrix.to_csv(output_path + 'Adj_matrix_bbike_map.csv', index=False)

    adj_list = list(nx.generate_adjlist(graph))
    df_list = pd.DataFrame(adj_list, columns=['row'])
    df_list = pd.DataFrame(df_list.row.str.split(' ').tolist())
    df_list = df_list.rename(columns={0: 'Source'})
    df_list.to_csv(output_path + 'Adj_list_bbike_map.csv', index=False)
    df_nodes = pd.DataFrame(dict(graph.nodes(data=True))).T
    df_nodes.to_csv(output_path + 'All.csv', index=False)
    to_dict_nodes = df_nodes.loc[df_nodes['osmid'].isin(c_nodes)]
    to_dict_nodes.to_csv(output_path + 'Nodes_bbike_map.csv', index=False)
    to_dict_objects = df_nodes.loc[df_nodes['osmid'].isin(c_obj)]
    to_dict_objects.to_csv(output_path + 'Objects_bbike_map.csv', index=False)
    adj_list = pd.read_csv(output_path + 'Adj_list_bbike_map.csv')
    df_graph = df_nodes.merge(adj_list, left_on='osmid', right_on='Source')
    adj = df_graph.iloc[:, 6:].to_numpy()
    adj = [node[~pd.isnull(node)] for node in adj]
    df_graph['adj'] = adj
    df_graph = df_graph[['osmid', 'x', 'y', 'weight', 'adj']].set_index('osmid')

    with open(output_path + 'Chel_simp.pickle', 'wb') as f:
        pickle.dump(graph, f)

    return df_graph


def gaversin_distance(x1, y1, x2, y2):
    R = 6371
    x1 = x1 * np.pi / 180
    x2 = x2 * np.pi / 180
    y1 = y1 * np.pi / 180
    y2 = y2 * np.pi / 180
    sin1 = np.sin((y2 - y1) / 2)
    sin2 = np.sin((x2 - x1) / 2)
    dist = 2 * R * np.arcsin(np.sqrt(sin1 ** 2 + (sin2 ** 2) * np.cos(y1) * np.cos(y2)))

    return dist

def df_to_edges(graph: pd.DataFrame):
    graph_dict = graph.T.to_dict()
    edges = defaultdict(list)

    for src_id, row_dict in graph_dict.items():
        src_x = row_dict['x']
        src_y = row_dict['y']

        for dst_id in row_dict['adj']:
            dst_x = graph_dict[dst_id]['x']
            dst_y = graph_dict[dst_id]['y']

            weight = graph_dict[dst_id]['weight']
            if not np.isnan(weight):
                cost = weight * gaversin_distance(src_x, src_y, dst_x, dst_y)
            else:
                cost = gaversin_distance(src_x, src_y, dst_x, dst_y)

            edges[src_id].append((cost, dst_id))

    return edges