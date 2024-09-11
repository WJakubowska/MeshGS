from meshes.mesh_structures import Vertex, Icosphere, UVSphere
from meshes.mesh_utils import get_triangles_as_indices
import torch.nn as nn
import trimesh
import torch 


class MeshGS(torch.nn.Module):
    def __init__(self, mesh_path=None, n_subdivisions=0, train_vertices=True, texture_size=8):
        super(MeshGS, self).__init__()

        self.opacity = None
        self.vertices = None
        self.faces = None
        self.texture = None
        self.offsets = None
        self.background = None
        self.mesh_path = mesh_path
        self.train_vertices = train_vertices
        self.texture_size = texture_size

        if mesh_path is not None:
            mesh = trimesh.load(mesh_path, force="mesh")

            centroid = mesh.centroid
            mesh.vertices -= centroid

            if n_subdivisions != 0:
                mesh = self.densify_mesh(mesh, n_subdivisions)

            vertices = []
            for vertex in mesh.vertices:
                vertex = Vertex(vertex[0], vertex[1], vertex[2])
                vertices.append(vertex)

            self.mesh_vertices = vertices
            self.mesh_faces = mesh.faces

        else:
            raise ValueError("Path to mesh file is not provided")

        self.setup_training_input(self.mesh_vertices, self.mesh_faces)

    def create_mesh_icosphere(self, center_point=(0, 0, 0), radius=[1], n_subdivisions=2):
        mesh = Icosphere(n_subdivisions, center_point, radius)
        vertices, triangles = mesh.vertices, mesh.triangles
        unique_vertices = mesh.get_all_vertices()
        triangles = get_triangles_as_indices(unique_vertices, triangles)
        return unique_vertices, triangles

    def create_mesh_sphere(self, n_slices=18, n_stacks=18):
        mesh = UVSphere(n_slices, n_stacks)
        unique_vertices, triangles = mesh.vertices, mesh.triangles
        triangles = get_triangles_as_indices(unique_vertices, triangles)
        return unique_vertices, triangles

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces

    def get_opacity(self):
        return self.opacity

    def get_texture(self):
        return self.texture

    def get_offsets(self):
        return self.offsets

    def get_background(self):
        return self.background

    def inverse_sigmoid(self, x):
        return torch.log(x / (1 - x))

    def densify_mesh(self, mesh, subdivisions=1):
        for _ in range(subdivisions):
            mesh = mesh.subdivide()
        return mesh

    def setup_training_input(self, mesh_vertices, mesh_faces):

        vertices_list = [(vertex.x, vertex.y, vertex.z) for vertex in mesh_vertices]
        vertices = torch.tensor(vertices_list, dtype=torch.float64, requires_grad=False)
        faces = torch.tensor(mesh_faces, dtype=torch.float64, requires_grad=False)
        num_faces = faces.shape[0]
        num_vertices = vertices.shape[0]

        self.faces = torch.nn.Parameter(faces, requires_grad=False)
        self.vertices = torch.nn.Parameter(vertices, requires_grad=False)

        if self.train_vertices == True:
            self.offsets = torch.nn.Parameter(torch.zeros(num_vertices, 3), requires_grad=True)
        else:
            self.offsets = torch.nn.Parameter(torch.zeros(num_vertices, 3), requires_grad=False)

        self.background = torch.nn.Parameter(torch.zeros(num_vertices), requires_grad=False)

        self.opacity = torch.nn.Parameter(self.inverse_sigmoid(0.1 * torch.ones(num_faces)),requires_grad=True)
        self.texture = torch.nn.Parameter(torch.randn(num_faces, self.texture_size, self.texture_size, 3) * 0.01, requires_grad=True)

        print("Mesh path: ", self.mesh_path)
        print("Vertices shape: ", self.vertices.shape)
        print("Offset shape: ", self.offsets.shape)
        print("Triangles shape: ", self.faces.shape)
        print("Opacity shape: ", self.opacity.shape)
        print("Texture shape: ", self.texture.shape)




class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # self.fc4 = nn.Linear(hidden_dim2, hidden_dim2//2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) # [n_rays, max_points, 3]
        
        return x