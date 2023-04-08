import bpy
import math
import random
import os
import yaml
import glob
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw  
import bpy_extras.object_utils




def get_images_in_directory(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and any(f.lower().endswith(ext) for ext in image_extensions)]
    return images


def compute_2d_bounding_box(obj):
    camera = bpy.context.scene.camera
    render = bpy.context.scene.render

    # Get the object's world matrix
    world_matrix = obj.matrix_world
    
    # Get the object's bounding box vertices
    vertices = [world_matrix @ v.co for v in obj.data.vertices]

    # Calculate 2D coordinates of the vertices in camera view
    vertices_2d = [bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera, v) for v in vertices]

    # Calculate the 2D bounding box
    min_x = min(v.x for v in vertices_2d)
    min_y = min(v.y for v in vertices_2d)
    max_x = max(v.x for v in vertices_2d)
    max_y = max(v.y for v in vertices_2d)

    # Convert normalized coordinates to pixel coordinates
    min_x_pixel = min_x * render.resolution_x
    min_y_pixel = (1 - min_y) * render.resolution_y
    max_x_pixel = max_x * render.resolution_x
    max_y_pixel = (1 - max_y) * render.resolution_y

    # Reorder the coordinates to ensure y1 >= y0
    y0, y1 = sorted([min_y_pixel, max_y_pixel])

    return [(min_x_pixel, y0), (max_x_pixel, y1)]


def set_random_camera_pose(config):
    # Set up the camera
    camera_config = config['camera']
    min_location = camera_config['location_min']
    max_location = camera_config['location_max']
    min_rotation = camera_config['rotation_min']
    max_rotation = camera_config['rotation_max']
    
    random_location = [
        random.uniform(min_location[0], max_location[0]),
        random.uniform(min_location[1], max_location[1]),
        random.uniform(min_location[2], max_location[2])
    ]
    
    random_rotation = [
        random.uniform(min_rotation[0], max_rotation[0]),
        random.uniform(min_rotation[1], max_rotation[1]),
        random.uniform(min_rotation[2], max_rotation[2])
    ]

    bpy.ops.object.camera_add(location=random_location)
    cam = bpy.context.active_object
    cam.rotation_euler = tuple(random_rotation)
    cam.data.type = 'PERSP'
    bpy.context.scene.camera = cam

def set_random_lighting(config):
    # Set up the lighting
    lighting_config = config['lighting']
    for _ in range(lighting_config['num_lights']):
        bpy.ops.object.light_add(type=lighting_config['light_type'], align='WORLD', location=(0, 0, lighting_config['light_distance']))
        light = bpy.context.active_object
        light.data.color = lighting_config['light_color']
        
        if lighting_config['light_type'] == "AREA":
            light.data.size = lighting_config['light_size']

        if lighting_config['type'] == 'random':
            light.data.energy = random.uniform(lighting_config['light_energy_min'], lighting_config['light_energy_max'])
        else:
            light.data.energy = lighting_config['light_energy_min']

def setup_scene(config):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Enable GPU rendering
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_x = config['resolution_x']
    bpy.context.scene.render.resolution_y = config['resolution_y']
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    set_random_lighting(config)
    set_random_camera_pose(config)


def import_dae_object(file_path):
    bpy.ops.wm.collada_import(filepath=file_path)
    obj = bpy.context.selected_objects[0]
    return obj


def create_background_plane():
    bpy.ops.mesh.primitive_plane_add(size=10, 
                enter_editmode=False, 
                align='WORLD', 
                location=(0, 0, -config['background_distance']), 
                scale=(1, 1, 1))
    plane = bpy.context.active_object
    plane.name = 'Background Plane'
    plane.rotation_euler = (0, 0, 0)
    return plane


def set_random_background(plane, config):
    background_images = glob.glob(os.path.join(config['backgrounds_dir'], '*.jpg'))
    background_image = random.choice(background_images)

    img = bpy.data.images.load(background_image)
    mat = bpy.data.materials.new(name="Background_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    diffuse_node = nodes.get("Principled BSDF")
    texture_node = nodes.new("ShaderNodeTexImage")
    texture_node.image = img

    links.new(texture_node.outputs["Color"], diffuse_node.inputs["Base Color"])

    if len(plane.data.materials) == 0:
        plane.data.materials.append(mat)
    else:
        plane.data.materials[0] = mat


def apply_texture_to_object(obj, image_path):
    # Set the object as active and selected
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    # Switch to edit mode and unwrap UVs using Smart UV Project with Scale to Bounds
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(scale_to_bounds=True)
    bpy.ops.object.mode_set(mode='OBJECT')



    # Load the image texture
    image_name = os.path.basename(image_path)
    image = bpy.data.images.load(image_path)

    # Create a new material
    material = bpy.data.materials.new(f"TextureMaterial_{image_name}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create shader nodes
    texture_node = nodes.new("ShaderNodeTexImage")
    texture_node.image = image
    texture_node.location = (-300, 300)

    uv_map_node = nodes.new("ShaderNodeUVMap")
    uv_map_node.location = (-500, 300)

    principled_node = nodes.new("ShaderNodeBsdfPrincipled")
    principled_node.location = (100, 300)

    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (300, 300)

    # Connect the nodes
    links.new(uv_map_node.outputs["UV"], texture_node.inputs["Vector"])
    links.new(texture_node.outputs["Color"], principled_node.inputs["Base Color"])
    links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Remove existing materials
    obj.active_material_index = 0
    while len(obj.material_slots) > 0:
        bpy.ops.object.material_slot_remove()

    # Assign the new material to the object
    obj.data.materials.append(material)

def apply_random_texture(obj, config):

    # Select a random texture image from the directory
    texture_image = os.path.join(config['texture_dir'], random.choice(os.listdir(config['texture_dir'])))
    apply_texture_to_object(obj, texture_image)



def render_image(output_path):
    print(f"Rendering image: {output_path}")  # Add this line
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

def set_random_pose(obj, location_range, rotation_range):
    location = (
        random.uniform(location_range['min'][0], location_range['max'][0]),
        random.uniform(location_range['min'][1], location_range['max'][1]),
        random.uniform(location_range['min'][2], location_range['max'][2]),
    )

    rotation = (
        math.radians(random.uniform(rotation_range['min'][0], rotation_range['max'][0])),
        math.radians(random.uniform(rotation_range['min'][1], rotation_range['max'][1])),
        math.radians(random.uniform(rotation_range['min'][2], rotation_range['max'][2])),
    )

    obj.location = location
    obj.rotation_euler = rotation


def generate_dataset(config):
    setup_scene(config)
    background_plane = create_background_plane()

    # Create output subdirectories
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, 'images')
    label_dir = os.path.join(output_dir, 'labels')
    labeled_image_dir = os.path.join(output_dir, 'labeled_images')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(labeled_image_dir, exist_ok=True)

    # Iterate through all .dae files in the model_dir
    for model_path in glob.glob(os.path.join(config['model_dir'], '*.dae')):
        obj = import_dae_object(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        for i in range(config['num_samples']):
            set_random_camera_pose(config)
            set_random_lighting(config)
            set_random_background(background_plane, config)
            apply_random_texture(obj, config)
            set_random_pose(obj, config['pose_randomization']['location'], config['pose_randomization']['rotation'])

            output_prefix = f"{model_name}_{i:04}"
            render_image(os.path.join(image_dir, f"{output_prefix}.png"))

            bounding_box_2d = compute_2d_bounding_box(obj)
            pose = np.array([obj.location, obj.rotation_euler], dtype=np.float32)
            label_data = {
                '2d_bounding_box': bounding_box_2d,
                'pose': pose
            }
            np.save(os.path.join(label_dir, f"{output_prefix}.npy"), label_data)

            img = Image.open(os.path.join(image_dir, f"{output_prefix}.png"))
            draw = ImageDraw.Draw(img)
            bbox_color = (255, 0, 0)
            draw.rectangle(bounding_box_2d, outline=bbox_color, width=2)
            img.save(os.path.join(labeled_image_dir, f"{output_prefix}.png"))

        # Delete the current object before importing the next one
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.delete()




if __name__ == '__main__':
    # print cwd
    print(os.getcwd())
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    generate_dataset(config)
    print("Done!")