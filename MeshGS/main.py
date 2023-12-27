from icosphere import Icosphere
from utils.display_sphere import plot_sphere, plot_filled_triangle_sphere
from utils.triangles_utils import get_unique_triangles


center_point = (400, 400, 0)
radius = [285, 140]
n_subdivisions = 2
icosphere_instance = Icosphere(n_subdivisions, center_point, radius)
vertices, triangles = icosphere_instance.vertices, icosphere_instance.triangles
plot_sphere(vertices, triangles)
unique_triangles = get_unique_triangles(triangles)
# plot_filled_triangle_sphere(vertices, triangles, fill_triangle_index=5, show_vertices=False)
print(len(unique_triangles))