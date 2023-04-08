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



def setup_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Enable GPU rendering
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_x = config['resolution_x']
    bpy.context.scene.render.resolution_y = config['resolution_y']
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Set up the camera
    bpy.ops.object.camera_add(location=(0, 0, config['camera_distance']))
    cam = bpy.context.active_object
    cam.rotation_euler = (0, 0, 0)
    cam.data.type = 'PERSP'
    bpy.context.scene.camera = cam

    # Set up the lighting
    bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, 0, 10))
    light = bpy.context.active_object
    light.data.energy = 3



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
    setup_scene()

    background_plane = create_background_plane()

    # Iterate through all .dae files in the model_dir
    for model_path in glob.glob(os.path.join(config['model_dir'], '*.dae')):
        obj = import_dae_object(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        for i in range(config['num_samples']):
            set_random_background(background_plane, config)
    
            apply_random_texture(obj, config)
            set_random_pose(obj, config['pose_randomization']['location'], config['pose_randomization']['rotation'])

            # Save the rendered image with the model name
            # output_image_path = os.path.join(config['output_dir'], f'{model_name}_image_{i:04d}.png')
            # render_image(output_image_path)

            output_prefix = f"{os.path.splitext(model_name)[0]}_{i:04}"
            output_dir = config['output_dir']
            output_path = os.path.join(output_dir, f"{output_prefix}.png")  # Define output_path here
            render_image(output_path)

            # Save the object's 2D bounding box and 3D pose information as a label
            output_label_path = os.path.join(config['output_dir'], f'{model_name}_label_{i:04d}.txt')
            with open(output_label_path, 'w') as label_file:
                bounding_box_2d = compute_2d_bounding_box(obj)
                label_file.write(f"2d_bounding_box: {bounding_box_2d}\n")
                label_file.write(f"location: {obj.location}\n")
                label_file.write(f"rotation_euler: {obj.rotation_euler}\n")

            output_label_path = os.path.join(config['output_dir'], f'{model_name}_label_{i:04d}.npy')
            bounding_box_2d = compute_2d_bounding_box(obj)
            pose = np.array([obj.location, obj.rotation_euler], dtype=np.float32)
            label_data = {
                '2d_bounding_box': bounding_box_2d,
                'pose': pose
            }
            np.save(output_label_path, label_data)


            
        

            # Save the rendered image with the bounding box
            # Draw bounding boxes on the rendered image
            img = Image.open(output_path)
            draw = ImageDraw.Draw(img)
            bbox_color = (255, 0, 0)  # Red color for the bounding box
            draw.rectangle(bounding_box_2d, outline=bbox_color, width=2)  # Change width as needed
            
            # Save the image with the bounding box
            img.save(os.path.join(output_dir, f"{output_prefix}_with_bbox.png"))

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