import numpy as np
import trimesh
import pyrender
from PIL import Image
import os
import random
import concurrent.futures
import threading

def get_rand_dir():
    while True:
        x, y, z = np.random.uniform(-1, 1, 3)
        if x**2 + y**2 + z**2 <= 1:
            norm = np.sqrt(x**2 + y**2 + z**2)
            return np.array([x, y, z]) / norm
        
def get_orthonormal_basis(forward, world_up=np.array([0.0, 1.0, 0.0])):
    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)
    return right, up, forward

def generate_image(meshes_path, background_path, output_path):
    files = os.listdir(meshes_path)
    meshes = [f for f in files if f.endswith('.obj') or f.endswith('.fbx')]
    mesh_path = os.path.join(meshes_path, random.choice(meshes))

    mesh = trimesh.load(mesh_path, force="mesh", process=True)
    mesh = pyrender.Mesh.from_trimesh(mesh)

    background = Image.open(background_path).convert("RGBA")
    
    while True:
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1., 1., 1.])
        scene.add(mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        backward = get_rand_dir()
        x, y, z = get_orthonormal_basis(-backward)
        distance = np.random.uniform(20, 350)
        x_offset, y_offset = np.random.uniform(-3, 3), np.random.uniform(-3, 3)
        camera_pose = np.array([
            [x[0], y[0], z[0], -backward[0] * distance + x_offset],
            [x[1], y[1], z[1], -backward[1] * distance + y_offset],
            [x[2], y[2], z[2], -backward[2] * distance],
            [0.0, 0.0, 0.0, 1.0]
        ])

        scene.add(camera, pose=camera_pose)

        intensity = np.random.uniform(0, 45)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        backward = get_rand_dir()
        x, y, z = get_orthonormal_basis(-backward)
        light_pose = np.array([
            [x[0], y[0], z[0], 0.0],
            [x[1], y[1], z[1], 0.0],
            [x[2], y[2], z[2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(light, pose=light_pose)

        r = pyrender.OffscreenRenderer(viewport_width=background.size[0], viewport_height=background.size[1])
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        r.delete()

        total_pixels = color.shape[0] * color.shape[1]
        non_zero_count = np.count_nonzero(color)
        non_zero_percentage = (non_zero_count / total_pixels) * 100
        if non_zero_percentage > 2.0:
            render_image = Image.fromarray(color, 'RGBA')
            blended_image = Image.alpha_composite(background, render_image).convert("RGB")
            blended_image.save(output_path)
            return