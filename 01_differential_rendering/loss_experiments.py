import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)


def load_mesh(device):
    # Load the obj and ignore the textures and materials.
    verts, faces_idx, _ = load_obj("./data/teapot.obj")
    faces = faces_idx.verts_idx

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    teapot_mesh = Meshes(
        verts=[verts.to(device)],   
        faces=[faces.to(device)], 
        textures=textures
    )
    return teapot_mesh  

def create_renderers(device):

    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
    # edges. Refer to blending.py for more details. 
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=256, 
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=100, 
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )


    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=256, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    # We can add a point light in front of the object. 
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )

    return silhouette_renderer, phong_renderer


if __name__ == "__main__":

    # Set the cuda device 
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(f"using device {device}")
    else:
        device = torch.device("cpu")


    teapot_mesh = load_mesh(device)
    silhouette_renderer, phong_renderer = create_renderers(device)
    
    ## Generate ref image
    #####################

    distance = 3   # distance from camera to the object
    elevation = 50.0   # angle of elevation in degrees
    azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 
    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    # Render the teapot providing the values of R and T. 
    silhouette = silhouette_renderer(meshes_world=teapot_mesh, R=R, T=T)
    image_ref = phong_renderer(meshes_world=teapot_mesh, R=R, T=T)
    
    silhouette = silhouette.cpu().numpy()
    image_ref = image_ref.cpu().numpy()

    image_ref_proc = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
    image_ref_proc = image_ref_proc.cuda()

    ## Play with loss
    #################

    camera_position = nn.Parameter(
        torch.from_numpy(np.array([3.0,  6.9, +2.5], dtype=np.float32)).to(teapot_mesh.device))

    R = look_at_rotation(camera_position[None, :], device=device)  # (1, 3, 3)
    T = -torch.bmm(R.transpose(1, 2), camera_position[None, :, None])[:, :, 0]   # (1, 3)


    image = silhouette_renderer(meshes_world=teapot_mesh.clone(), R=R, T=T)

    # Calculate the silhouette loss
    loss = torch.sum((image[..., 3] - image_ref_proc) ** 2)
    
    print(loss)

    ## Save images
    ##############

    from PIL import Image
    silhouette_255 = (silhouette*255).astype(np.uint8).squeeze()
    im = Image.fromarray(silhouette_255)
    im.save("data/silhouette.png")

