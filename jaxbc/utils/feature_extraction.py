from jax_resnet import ResNet18
from jax_resnet import pretrained_resnet

import flax.linen as nn
import jax.numpy as jnp

from PIL import Image
import os

import numpy as np
import jax

def get_numpy_images(episode_path):
        # count number of pictures
    image_count = 0
    for _ in os.listdir(episode_path):
        image_count += 1
    pil_images = []
    for img_num in range(image_count):
        image_path = os.path.join(episode_path,f"{img_num}.png")
        pil_img = Image.open(image_path)
        pil_images.append(pil_img)
    images = np.concatenate([np.array(pil_image)[np.newaxis,:] for pil_image in pil_images],dtype=np.float64)
    images /= 255.0      
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # images = (images - mean) / std
    return images

def image_to_embedding(model,dir_path):

    episode_count = 0 
    for _ in os.listdir(dir_path):
        episode_count += 1

    for epi_num in range(episode_count):
        print("episode: ",epi_num)
        view = "front_rgb"
        episode_path = os.path.join(dir_path,f"episode{epi_num}")
        np_images = get_numpy_images(os.path.join(episode_path,view))

        epi_embedding = model.apply(variables,
                    np_images,
                    mutable=False)
        print(epi_embedding.shape)
        jax.numpy.save(episode_path+"/front_rgb_embeddings",epi_embedding)


if __name__ == "__main__":

    dir_path = "/home/andykim0723/jax_bc/data/pick_and_lift_simple/variation0/episodes"

    resnet18, variables = pretrained_resnet(18)
    model = resnet18()
    model = nn.Sequential(model.layers[0:-1])

    image_to_embedding(model,dir_path)