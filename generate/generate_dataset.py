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
import cv2
from pathlib import Path
import shutil




def get_images_in_directory(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and any(f.lower().endswith(ext) for ext in image_extensions)]
    return images

def clip_coordinates(x1, y1, x2, y2, img_width, img_height):
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    return x1, y1, x2, y2


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

    # return [(min_x_pixel, y0), (max_x_pixel, y1)]
    return min_x_pixel, y0, max_x_pixel, y1



def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = clip_coordinates(x1, y1, x2, y2, img_width, img_height)
    x_center = (x1 + x2) / (2 * img_width)
    y_center = (y1 + y2) / (2 * img_height)
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return [x_center, y_center, width, height]


def world_to_camera_view(scene, world_coordinates):
    """
    Convert world coordinates to camera view (2D screen-space) coordinates.

    :param scene: The scene object
    :param world_coordinates: The world coordinates of the point (x, y, z)
    :return: 2D coordinates of the point in camera view
    """
    camera = scene.camera
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, world_coordinates)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )

    return co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]



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
    
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Select all light objects in the current scene
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)

    # Delete selected light objects
    bpy.ops.object.delete()

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


def create_background_plane(config):
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
    print("ranom background image: ", background_image)

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
        (random.uniform(rotation_range['min'][0], rotation_range['max'][0])),
        (random.uniform(rotation_range['min'][1], rotation_range['max'][1])),
        (random.uniform(rotation_range['min'][2], rotation_range['max'][2])),
    )

    obj.location = location
    obj.rotation_euler = rotation


def draw_3d_axes(image_path, obj, output_folder, scene):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    origin = np.array(obj.location)
    axes_length = 1.0

    # Convert Blender Matrix to NumPy array
    matrix_world_np = np.array(obj.matrix_world.to_3x3())

    x_axis = origin + matrix_world_np @ np.array([axes_length, 0, 0])
    y_axis = origin + matrix_world_np @ np.array([0, axes_length, 0])
    z_axis = origin + matrix_world_np @ np.array([0, 0, axes_length])

    origin_2d = world_to_camera_view(scene, Vector(origin))
    x_axis_2d = world_to_camera_view(scene, Vector(x_axis))
    y_axis_2d = world_to_camera_view(scene, Vector(y_axis))
    z_axis_2d = world_to_camera_view(scene, Vector(z_axis))

    draw.line((*origin_2d, *x_axis_2d), fill=(255, 0, 0), width=2)
    draw.line((*origin_2d, *y_axis_2d), fill=(0, 255, 0), width=2)
    draw.line((*origin_2d, *z_axis_2d), fill=(0, 0, 255), width=2)

    output_path = os.path.join(output_folder, os.path.basename(image_path))
    img.save(output_path)


def process_cad_obj(config, class_id, model_path, image_path, label_path, image_labeled_path, image_with_axes_path, poses_path):
    cam = setup_scene(config)
    background_plane = create_background_plane(config)
    obj = import_dae_object(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]


    for i in range(config['num_samples']):
        
        set_random_camera_pose(config)
        set_random_lighting(config)
        set_random_background(background_plane, config)
        apply_random_texture(obj, config)
        set_random_pose(obj, config['pose_randomization']['location'], config['pose_randomization']['rotation'])

        # Save the rendered image
        output_prefix = f"{model_name}_{i:04}"
        output_image_path = os.path.join(config['output_dir'], image_path, f"{output_prefix}.png")
        render_image(output_image_path)

        # Save the object's 2D bounding box and 3D pose information as a label in YOLO format
        bounding_box_2d = compute_2d_bounding_box(obj)
        yolo_bbox = convert_bbox_to_yolo_format(bounding_box_2d, config['resolution_x'], config['resolution_y'])
        output_label_path = os.path.join(config['output_dir'], label_path, f"{output_prefix}.txt")
        with open(output_label_path, 'w') as label_file:
            label_file.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

        # Save the rendered image with the bounding box
        img = Image.open(output_image_path)
        draw = ImageDraw.Draw(img)
        bbox_color = (255, 0, 0)  # Red color for the bounding box
        draw.rectangle(bounding_box_2d, outline=bbox_color, width=2)  # Change width as needed
        
        # Save the image with the bounding box
        output_image_with_bbox_path = os.path.join(config['output_dir'], image_labeled_path, f"{output_prefix}_with_bbox.png")
        img.save(output_image_with_bbox_path)

        # Draw 3D axes on the image with the bounding box and save it in a separate folder
        draw_3d_axes(output_image_with_bbox_path, obj, os.path.join(config['output_dir'], image_with_axes_path), bpy.context.scene)

        # Save the object's pose information
        output_pose_path = os.path.join(config['output_dir'], poses_path, f'{model_name}_pose_{i:04d}.txt')
        with open(output_pose_path, 'w') as pose_file:
            pose_file.write(f"location: {obj.location}\n")
            pose_file.write(f"rotation_euler: {obj.rotation_euler}\n")


    # Delete the current object before importing the next one
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.delete()

    return model_name, class_id


import concurrent.futures

def generate_dataset(config):
    # Clear the dataset folder
    output_dir = config['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    

    # Create the output subdirectories
    image_path = "images/train"
    label_path = "labels/train"
    image_path_val = "images/val"
    label_path_val = "labels/val"
    image_labeled_path = "images_labeled/train"
    image_with_axes_path = "images_with_axes/train"
    poses_path = "poses/train"
    os.makedirs(os.path.join(config['output_dir'], image_path), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], label_path), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], image_labeled_path), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], image_with_axes_path), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], poses_path), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], image_path_val), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], label_path_val), exist_ok=True)

    class_id = 0
    class_id_mapping = {}

    

    # # Iterate through all .dae files in the model_dir
    # for model_path in glob.glob(os.path.join(config['model_dir'], '*.dae')):
        
    #     model_name, class_id = process_cad_obj(config, class_id, model_path, image_path, label_path, image_labeled_path, image_with_axes_path, poses_path)
    #     class_id_mapping[class_id] = model_name

    #     # Increment the class ID for the next object
    #     class_id += 1


    # Get the list of DAE files
    dae_files = glob.glob(os.path.join(config['model_dir'], '*.dae'))
    # Create a ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_cad_obj, config, class_id, model_path, image_path, label_path, image_labeled_path, image_with_axes_path, poses_path)
            for class_id, model_path in enumerate(dae_files)
        ]

        for future in concurrent.futures.as_completed(futures):
            mapping = future.result()
            class_id = mapping[1]
            model_name = mapping[0]
            class_id_mapping[class_id] = model_name
            print(future.result())
            # future.result()
        


    # Splitting the dataset into training and validation sets
    split_ratio = config['validation_split']
    img_dir = os.path.join(config['output_dir'], image_path)
    val_dir = os.path.join(config['output_dir'], image_path_val)
    os.makedirs(val_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    random.shuffle(img_files)
    num_val_images = int(len(img_files) * split_ratio)

    for i in range(num_val_images):
        img_file = img_files[i]
        label_file = img_file.replace('.png', '.txt')
        shutil.move(os.path.join(img_dir, img_file), os.path.join(val_dir, img_file))
        shutil.move(os.path.join(config['output_dir'], label_path, label_file), os.path.join(config['output_dir'], label_path_val, label_file))

    # Save class and path information to a YAML file
    dataset_info = {
        # 'path': os.path.abspath(config['output_dir']),
        'path': config['output_dir'],
        'train': image_path,
        'val': image_path_val,
        'test': None,
        'names': class_id_mapping
    }

    # for i, name in enumerate(class_id_mapping):
    #     print(f"Class ID {i}: {name}")
    
    # print(class_id_mapping)

    with open(os.path.join(config['output_dir'], 'classes.yaml'), 'w') as f:
        f.write("# My Dataset YOLO 🚀, GPL-3.0 license\n")
        f.write("# Example usage: yolo train data=classes.yaml\n")
        # f.write("# parent\n")
        # f.write("# ├── generate (path of generate_dataset function)\n")
        # f.write("# └── datasets\n")
        # f.write("#     └── my_dataset\n\n")
        yaml.dump(dataset_info, f, default_flow_style=False)



if __name__ == '__main__':
    # print cwd
    print(os.getcwd())
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    generate_dataset(config)
    print("Done!")