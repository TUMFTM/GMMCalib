import open3d as o3d
import numpy as np

def create_init_pc(box_size, num_points):
    # Create a box mesh
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=box_size[0], height=box_size[1], depth=box_size[2])

    # Sample points on each face of the box
    sampled_points = []

    for face in box_mesh.triangles:
        # Get the vertices of the face
        vertices = np.asarray(box_mesh.vertices)[face]

        # Compute the normal of the face
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        normal /= np.linalg.norm(normal)
        np.random.seed(3)
        # Sample points on the face using the barycentric coordinates method
        u = np.random.rand(num_points//6)
        v = np.random.rand(num_points//6)
        mask = u + v < 1
        u = u[mask]
        v = v[mask]
        points_on_face = vertices[0] + u[:, None] * (vertices[1] - vertices[0]) + v[:, None] * (vertices[2] - vertices[0])

        # Project points onto the plane of the face
        points_on_plane = points_on_face - np.dot(points_on_face - vertices[0], normal)[:, None] * normal

        # Add the sampled points to the list
        sampled_points.extend(points_on_plane)

    return np.array(sampled_points)

