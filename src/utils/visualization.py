import plotly.graph_objects as go
import umap

def create_3d_visualization(graph_data, node_embeddings, labels):
    """Create 3D visualization of the patient graph"""
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding_3d = reducer.fit_transform(node_embeddings)
    
    edge_x, edge_y, edge_z = [], [], []
    for edge in graph_data.edge_index.t():
        start = embedding_3d[edge[0]]
        end = embedding_3d[edge[1]]
        edge_x.extend([start[0], end[0], None])
        edge_y.extend([start[1], end[1], None])
        edge_z.extend([start[2], end[2], None])
    
    fig = go.Figure(data=[
        go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='lightgray', width=1),
            hoverinfo='none'
        ),
        go.Scatter3d(
            x=embedding_3d[:, 0],
            y=embedding_3d[:, 1],
            z=embedding_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=labels,
                colorscale='Viridis',
                opacity=0.8
            )
        )
    ])
    
    fig.update_layout(
        title='3D Patient Graph Visualization',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        showlegend=False
    )
    
    return fig