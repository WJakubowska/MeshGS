import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

pio.renderers.default = "browser"

def save_plot_as_html(name, fig):
    dir = 'output_images'
    Path(dir).mkdir(parents=True, exist_ok=True)

    fig.write_html(
        f'{dir}/{name}', auto_open=True
    )

def plot_sphere(vertices, triangles, show_vertices = False):
    fig = go.Figure()

    for triangle in triangles:
        x = [triangle.v0.x, triangle.v1.x, triangle.v2.x, triangle.v0.x]
        y = [triangle.v0.y, triangle.v1.y, triangle.v2.y, triangle.v0.y]
        z = [triangle.v0.z, triangle.v1.z, triangle.v2.z, triangle.v0.z]

        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue'), showlegend=False))
        if show_vertices:
          fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='red'), showlegend=False))

    fig.update_layout(scene=dict(aspectmode="data"))
    save_plot_as_html('mesh.html', fig)


def plot_filled_triangle_sphere(vertices, triangles, fill_triangle_index=None, show_vertices=False):
    fig = go.Figure()

    for i, triangle in enumerate(triangles):
        x = [triangle.v0.x, triangle.v1.x, triangle.v2.x, triangle.v0.x]
        y = [triangle.v0.y, triangle.v1.y, triangle.v2.y, triangle.v0.y]
        z = [triangle.v0.z, triangle.v1.z, triangle.v2.z, triangle.v0.z]


        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue')))

        if fill_triangle_index is None or i == fill_triangle_index:
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, opacity=0.8, color='red'))

        if show_vertices:
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='red')))

    fig.update_layout(scene=dict(aspectmode="data"))
    save_plot_as_html('mesh_with_filled_face.html', fig)