from utils.triangle import Triangle


def make_permutations(triangle):
    v0 = triangle.v0
    v1 = triangle.v1
    v2 = triangle.v2
    return [
        Triangle(v0, v1, v2),
        Triangle(v0, v2, v1),
        Triangle(v1, v2, v0),
        Triangle(v1, v0, v2),
        Triangle(v2, v1, v0),
        Triangle(v2, v0, v1),
    ]


def check_if_triangles_are_the_same(triangle1, triangle2):
    permutations = make_permutations(triangle2)
    if any(obj == triangle1 for obj in permutations):
        return True
    else:
        return False


def get_unique_triangles(triangles):
    unique_triangles = []

    for i, triangle in enumerate(triangles):
        is_unique = True
        for unique_triangle in unique_triangles:
            if check_if_triangles_are_the_same(triangle, unique_triangle):
                is_unique = False
                break
        if is_unique:
            unique_triangles.append(triangle)
    return unique_triangles


def get_triangles_as_indices(unique_vertices, triangles):
    for i, vertex in enumerate(unique_vertices):
        vertex.index = i

    triangles_with_shared_vertices = [
        Triangle(
            unique_vertices[unique_vertices.index(triangle.v0)],
            unique_vertices[unique_vertices.index(triangle.v1)],
            unique_vertices[unique_vertices.index(triangle.v2)],
        )
        for triangle in triangles
    ]

    return [
        [vertex.index for vertex in triangle.get_vertices()]
        for triangle in triangles_with_shared_vertices
    ]
