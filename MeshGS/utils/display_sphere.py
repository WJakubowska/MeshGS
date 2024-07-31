import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import torch

pio.renderers.default = "browser"

def save_plot_as_html(name, fig):
    dir = 'output_images'
    Path(dir).mkdir(parents=True, exist_ok=True)

    fig.write_html(
        f'{dir}/{name}', auto_open=True
    )

def plot_sphere(vertices, triangles, show_vertices = False, name='mesh.html'):
    fig = go.Figure()

    for triangle in triangles:
        x = [triangle.v0.x, triangle.v1.x, triangle.v2.x, triangle.v0.x]
        y = [triangle.v0.y, triangle.v1.y, triangle.v2.y, triangle.v0.y]
        z = [triangle.v0.z, triangle.v1.z, triangle.v2.z, triangle.v0.z]

        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue'), showlegend=False))
        if show_vertices:
          fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='red'), showlegend=False))

    fig.update_layout(scene=dict(aspectmode="data"))
    save_plot_as_html(name, fig)


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


def plot_sphere_from_tensor_with_index(vertices, triangles, show_vertices=False, name='mesh.html'):
    fig = go.Figure()

    for triangle in triangles:
        v0, v1, v2 = triangle
        x = [vertices[v0][0], vertices[v1][0], vertices[v2][0], vertices[v0][0]]
        y = [vertices[v0][1], vertices[v1][1], vertices[v2][1], vertices[v0][1]]
        z = [vertices[v0][2], vertices[v1][2], vertices[v2][2], vertices[v0][2]]

        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue')))
        if show_vertices:
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='red')))

    fig.update_layout(scene=dict(aspectmode="data"))
    save_plot_as_html(name, fig)


def plot_rays_mesh_and_points(
    rays_origins: torch.Tensor,
    rays_directions: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    points: torch.Tensor,
    ray_length_scale: float = 1.0,
):
    vertices = vertices.detach().cpu().numpy()

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        )
    )
    data = []

    lines = []
    for i in range(len(rays_origins)):
        ray_origin, ray_direction = rays_origins[i], rays_directions[i]
        ray_end = ray_origin + ray_length_scale * ray_direction 
        lines.append(go.Scatter3d(
            x=[ray_origin[0], ray_end[0]],
            y=[ray_origin[1], ray_end[1]],
            z=[ray_origin[2], ray_end[2]],
            mode='lines',
            line=dict(width=0.5, color='blue'),
            name=f'Ray_{i}'
        ))

    data = data + lines

    triangles = []
    for face in faces:
        f = list(face)
        triangles.append(
            go.Scatter3d(
                x=vertices[:, 0][f + [f[0]]],
                y=vertices[:, 1][f + [f[0]]],
                z=vertices[:, 2][f + [f[0]]],
                mode='lines',
                marker=dict(size=5, color='black'),
                showlegend=False
            )
        )

    data = data + triangles
    fig = go.Figure(
        data=data,
        layout=layout
    )

    save_plot_as_html('RAY.html', fig)

def plot_selected_points_on_sphere(selected_pts, vertices, triangles, name='selected_points.html'):
    fig = go.Figure()

    vertices = vertices.cpu().detach().numpy()
    triangles = triangles.cpu().detach().numpy().astype(int)
    for triangle in triangles:
        triangle_vertices = vertices[triangle]
        x, y, z = zip(*triangle_vertices)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue'), showlegend=False))



    for i in range(selected_pts.shape[0]):
        x, y, z = zip(*selected_pts[i].cpu().detach().numpy())
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', name=f'Selected Points {i}', marker=dict(size=5, color='red'), showlegend=False))

    fig.update_layout(scene=dict(aspectmode="data"))
    save_plot_as_html(name, fig)


