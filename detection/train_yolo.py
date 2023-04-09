import os
from ultralytics import YOLO
import yaml
import random
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EPOCHS = 100

def train(model, dataset_config):
    model.train(data= dataset_config, epochs=EPOCHS)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    success = model.export(format="onnx")  # export the model to ONNX format
    return metrics, success


def validate(model, dataset_config, dataset_path):
    with open(dataset_config, 'r') as f:
        config = yaml.safe_load(f)
        validate_path = os.path.join(dataset_path, config['val'])
        # list images in val_path
        files = os.listdir(validate_path)

        for image in files:    
            #image = random.choice(files)
            filepath = os.path.join(validate_path, image)
            print(filepath)
            results = model(filepath)  # predict on an image
            # print(results)
            return filepath, results



if __name__=="__main__":
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # pretrained_model = "/home/gebmer/repos/assembly_vision/runs/detect/train4/weights/best.pt"
    pretrained_model = "/home/geraldebmer/repos/assembly_vision/runs/detect/train6/weights/best.pt"
    model = YOLO(pretrained_model)

    # Use the model
    # dataset_path = "/home/gebmer/repos/assembly_vision/dataset/"
    dataset_path = "/home/geraldebmer/repos/assembly_vision/dataset/"
    dataset_config = dataset_path + "classes.yaml"

    # metrics, success = train(model, dataset_config)
    # print(metrics)
    # print(success)
    
    # validate 
    with open(dataset_config, 'r') as f:
        config = yaml.safe_load(f)
        validate_path = os.path.join(dataset_path, config['val'])
        # list images in val_path
        files = os.listdir(validate_path)

        # for image in files:    
        image = random.choice(files)
        filepath = os.path.join(validate_path, image)
        print(filepath)
        results = model(filepath)  # predict on an image

        img = None
        for result in results[0]:
            # print(result)
            img = result.orig_img
            # break
            for box in result.boxes:
                cls = box.cls
                conf = box.conf
                # print class_id: confidence
                # print(f"{result.names[class_id]}: {confidence:.2f}")
                label = result.names[cls.item()]
                color = (255, 128, 0)
                thickness = 2
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                
                # Draw the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                
                # Draw the label text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text = f"{label} {conf.item():.2f}"
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_w, text_h = text_size
                cv2.rectangle(img, (x1, y1 - text_h - 3), (x1 + text_w, y1), color, -1)
                cv2.putText(img, text, (x1, y1 - 3), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Display the image with bounding boxes and annotations
        if img.any() != None:
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No results")