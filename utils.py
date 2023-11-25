import cv2
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import json
from PIL import ImageStat, Image
import numpy as np
import os
import shutil
from metrics import BinaryMetrics
import torch
import torchvision.transforms as transforms
from scipy.stats import entropy

import plotly.express as px


def print_map(df):
    """
    input: data frame with relevant information
    output: map showing the location where images were taken
    """
    df_temp = df[["longitude", "latitude", "tag"]].drop_duplicates()
    df_temp["size"] = 6

    fig = px.scatter_mapbox(
        df_temp,
        lat="latitude",
        lon="longitude",
        #    color="tag",
        #    color_continuous_scale=color_scale,
        size="size",
        zoom=9,
        height=900,
        width=900,
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def print_image(
    df, path, file, include_polyline=False, include_bounding_box=False, target=True
):
    """
    function to show images
    polylines and bounding boxes can be plotted optionally
    """
    print(file)
    image = cv2.imread(path + file)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    window_name = "Image"
    if target:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)
    if not include_polyline:
        plt.imshow(image)
        plt.show()
        return
    for idx, row in df[df["path"] == file].iterrows():
        thickness = row.width // 100
        isClosed = row["closed"]
        pts = row["poly2d"].reshape((-1, 1, 2))
        image = cv2.polylines(image, [pts], isClosed, color, thickness)
        if include_bounding_box:
            x = min([i[0] for i in row.poly2d])
            y = min([i[1] for i in row.poly2d])
            w = max([i[0] for i in row.poly2d]) - min([i[0] for i in row.poly2d])
            h = max([i[1] for i in row.poly2d]) - min([i[1] for i in row.poly2d])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness // 2)

    plt.imshow(image)
    plt.show()


def parse_input():
    """
    function reads json data for each dataset and extracts relevant information for
    sensors
    images
    labels
    """
    data = []

    # iterate over all direcotries and open json file
    for d in os.listdir("./orig_data/DB/"):
        file_name = glob(f"./orig_data/DB/{d}/*.json")[0]
        print(file_name)
        with open(file_name) as json_file:
            raw = json.load(json_file)
        raw = raw["openlabel"]

        # extract sensor data
        map_sensor_to_wh = {}
        for k, v in raw["streams"].items():
            if "stream_properties" in v.keys():
                map_sensor_to_wh[k] = {
                    "width": v["stream_properties"]["intrinsics_pinhole"]["width_px"],
                    "height": v["stream_properties"]["intrinsics_pinhole"]["height_px"],
                }
            else:
                map_sensor_to_wh[k] = {
                    "width": np.nan,
                    "height": np.nan,
                }

        # extract labels
        for k, v in raw["frames"].items():
            frame = k.rjust(3, "0")
            csv_name = glob(f"./orig_data/DB/{d}/novatel_oem7_inspva/{frame}_*.csv")[0]
            df_longlat = pd.read_csv(csv_name)
            long = df_longlat["longitude"].values[0]
            lat = df_longlat["latitude"].values[0]

            for k2, v2 in v["objects"].items():
                for k3, v3 in v2["object_data"].items():
                    for b in v3:
                        temp = {}
                        temp["longitude"] = long
                        temp["latitude"] = lat
                        temp["tag"] = raw["metadata"]["tagged_file"]
                        temp["type"] = raw["objects"][k2]["type"]
                        temp["type"] = raw["objects"][k2]["type"]
                        coordinate_system = b["coordinate_system"]
                        label_uid = b["uid"]
                        temp["name"] = b["name"]
                        temp["label_uid"] = label_uid
                        temp["object_uid"] = k2
                        temp["sensor"] = coordinate_system

                        temp["height"] = map_sensor_to_wh[coordinate_system]["height"]
                        temp["width"] = map_sensor_to_wh[coordinate_system]["width"]
                        temp[
                            "path"
                        ] = f"{d}{v['frame_properties']['streams'][coordinate_system]['uri']}"
                        temp["dataset"] = d
                        temp["timestamp"] = v["frame_properties"]["timestamp"]
                        temp["label_type"] = k3
                        temp["tag"] = raw["metadata"]["tagged_file"]

                        # get details if label is a track
                        if temp["type"] == "track" and k3 == "poly2d":
                            temp["closed"] = b["closed"]
                            temp["val"] = b["val"]
                            temp["name"] = b["name"]
                            for t in b["attributes"]["text"]:
                                temp[t["name"]] = t["val"]
                        data.append(temp)
    return pd.DataFrame(data)


def parse_rail_sem_input(path):
    """
    function reads json data for railsem dataset and extracts relevant
    information for images and labels
    """
    data = []
    uid = 0
    for file_name in glob(f"{path}*.json"):
        with open(file_name) as json_file:
            raw = json.load(json_file)
            for o in raw["objects"]:
                if o["label"] == "rail":
                    if "polyline-pair" in o.keys():
                        assert len(o["polyline-pair"]) == 2
                        for p in o["polyline-pair"]:
                            temp = {}
                            temp["path"] = raw["frame"] + ".jpg"
                            temp["width"] = raw["imgWidth"]
                            temp["height"] = raw["imgHeight"]
                            temp["type"] = "rail"
                            temp["label_uid"] = uid
                            uid += 1
                            temp["val"] = p
                            temp["closed"] = False
                            data.append(temp)
                    else:
                        temp = {}
                        temp["path"] = raw["frame"] + ".jpg"
                        temp["width"] = raw["imgWidth"]
                        temp["height"] = raw["imgHeight"]
                        temp["type"] = "rail"
                        temp["label_uid"] = uid
                        uid += 1
                        temp["val"] = o["polyline"]
                        temp["closed"] = False
                        data.append(temp)
    return pd.DataFrame(data)


def calculate_brightness(im_file):
    """
    calculate average brightness of image
    """
    im = Image.open(im_file).convert("L")
    stat = ImageStat.Stat(im)
    return stat.rms[0]


def calculate_entropy(im_file):
    """
    calculate entropy of image
    """
    image = cv2.imread(im_file)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _bins = 128
    hist, _ = np.histogram(gray_image.ravel(), bins=_bins, range=(0, _bins))
    prob_dist = hist / hist.sum()
    image_entropy = entropy(prob_dist, base=2)
    return image_entropy


def generate_train_val_test_split(df, seed=1, val_ratio=0.15, test_ratio=0.15):
    """
    split dataset into validation, train, and test split
    """
    df_temp = df[["tag"]].drop_duplicates().sample(frac=1, random_state=seed)
    # define number of tags in each set
    test_num = int(len(df_temp) * test_ratio)
    val_num = int(len(df_temp) * val_ratio)
    train_num = len(df_temp) - test_num - val_num
    # generate set
    df_train = df_temp.iloc[:train_num]
    df_train["set"] = "train"
    df_val = df_temp.iloc[train_num : train_num + val_num]
    df_val["set"] = "val"
    df_test = df_temp.iloc[train_num + val_num :]
    df_test["set"] = "test"

    # the sum of the sizes of all sub sets must match the size of the original set
    assert len(df_temp) == len(df_val) + len(df_train) + len(df_test)
    df = pd.merge(
        how="left",
        left=df,
        right=pd.concat([df_test, df_train, df_val]),
        left_on=["tag"],
        right_on=["tag"],
    )
    return df


def generate_mask_df(df: pd.DataFrame, path: str, file: str):
    """
    takes an image and the repective labels and generates mask
    """
    print(file)
    df = df[df["path"] == file]
    img = cv2.imread(path + file)
    thickness = int(img.shape[1] / 100)
    mask = np.zeros_like(img)
    for idx, row in df.iterrows():
        cv2.polylines(
            mask,
            np.int32([row["poly2d"]]),
            isClosed=False,
            color=(255, 255, 255),
            thickness=thickness,
        )
    return mask


def generate_mask_path(path_image: str, path_label: str, print=False):
    """
    input: path of image and label
    output: mask
    """
    with open(path_label, "r") as f:
        labels = f.read().splitlines()
    img = cv2.imread(path_image)
    h, w = img.shape[:2]
    mask = np.zeros_like(img).astype(np.float32)
    for label in labels:
        class_id, *poly = label.split(" ")
        poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)  # Read poly, reshape
        poly *= [w, h]  # Unscale
        cv2.polylines(
            img, [poly.astype("int")], True, (0, 255, 0), 4
        )  # Draw Poly Lines

        cv2.fillPoly(
            mask, [poly.astype("int")], (255, 255, 255), cv2.LINE_AA
        )  # Draw area
    if print:
        fig, ax = plt.subplots(2)
        fig.suptitle("Original image and the respective mask")
        ax[0].imshow(img)
        ax[1].imshow(mask)
    return mask / 255


def find_contours(mask, epsilon=0.8):
    """
    given a mask this function generates the countour around the
    """
    mask_temp = mask.copy()
    imgray = cv2.cvtColor(mask_temp, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    labels = []
    coordinates = []
    for idx, contour in enumerate(contours):
        con_short = cv2.approxPolyDP(contour, epsilon=0.8, closed=True)
        coord = [point[0] for point in con_short]
        coord += [coord[0]]  # close polygon
        label_line = "0 " + " ".join(
            [f"{cord[0]/mask.shape[1]} {cord[1]/mask.shape[0]}" for cord in coord]
        )

        coordinates.append(coord)
        labels.append(label_line)
    return labels, coordinates


def make_directory_structure(path, dataset):
    """
    make directory structure
    TODO: put everything in loop
    """
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(f"{path}{dataset}"):
        os.makedirs(f"{path}{dataset}")

    if not os.path.exists(f"{path}{dataset}/train"):
        os.makedirs(f"{path}{dataset}/train")

    if not os.path.exists(f"{path}{dataset}/train/labels"):
        os.makedirs(f"{path}{dataset}/train/labels")

    if not os.path.exists(f"{path}{dataset}/train/images"):
        os.makedirs(f"{path}{dataset}/train/images")

    if not os.path.exists(f"{path}{dataset}/val"):
        os.makedirs(f"{path}{dataset}/val")

    if not os.path.exists(f"{path}{dataset}/val/images"):
        os.makedirs(f"{path}{dataset}/val/images")

    if not os.path.exists(f"{path}{dataset}/val/labels"):
        os.makedirs(f"{path}{dataset}/val/labels")

    if not os.path.exists(f"{path}{dataset}/test"):
        os.makedirs(f"{path}{dataset}/test")

    if not os.path.exists(f"{path}{dataset}/test/images"):
        os.makedirs(f"{path}{dataset}/test/images")

    if not os.path.exists(f"{path}{dataset}/test/labels"):
        os.makedirs(f"{path}{dataset}/test/labels")


def create_yml_file(path, dataset):
    yaml_content = f"""
    path: ./{dataset}
    train: train/images
    val: val/images
    test: test/images

    nc: 1
    names: ['track']
        """
    with open(f"{path}{dataset}.yml", "w") as f:
        f.write(yaml_content)


def write_dataset_to_directory(
    df, path_yml, path_data, dataset, only_label=False, path_orig_data="./orig_data/DB/"
):
    make_directory_structure(dataset=dataset, path=path_data)
    create_yml_file(path=path_yml, dataset=dataset)
    for idx, row in df[["path", "set"]].drop_duplicates().iterrows():
        source = f"{path_orig_data}{row['path']}"
        filename = row["path"].replace("/", "_")
        destination = f"{path_data}{dataset}/{row['set']}/images/{filename}"
        shutil.copy(source, destination)

        mask = generate_mask_df(df, path=path_orig_data, file=row["path"])
        labels, coordinates = find_contours(mask)
        with open(
            f"{path_data}{dataset}/{row['set']}/labels/{filename.replace('png','txt').replace('jpg', 'txt')}",
            "w",
        ) as f:
            f.write("\n".join(labels))


def apply_fast_line_detection(
    path,
    kernel_size=0,
    length_treshold=10,
    min_aspect_ration=0,
    max_aspect_ration=float("inf"),
    canny_th1=50,
    canny_th2=50,
    canny_aperture_size=3,
):
    image_orig = cv2.imread(path)
    image = cv2.imread(path, cv2.CV_8UC1)
    if kernel_size > 0:
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    fld = cv2.ximgproc.createFastLineDetector(
        length_threshold=length_treshold,
        do_merge=True,
        canny_th1=canny_th1,
        canny_th2=canny_th2,
        canny_aperture_size=canny_aperture_size,
    )
    lines = fld.detect(image)
    line_image = np.copy(image_orig)
    mask = np.zeros_like(image)

    if lines is not None:
        for line in lines.astype(int):
            for x1, y1, x2, y2 in line:
                aspect = abs(x1 - x2) / max(abs(y1 - y2), 1)
                if aspect > min_aspect_ration and aspect < max_aspect_ration:
                    cv2.line(
                        line_image,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        image.shape[1] // 100,
                    )
                    cv2.line(
                        mask,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        mask.shape[1] // 100,
                    )
    plt.imshow(line_image)
    plt.show()
    return mask


def compare_masks(target_mask, prediction_mask, verbose=False):
    assert target_mask.max() <= 1 and prediction_mask.max()
    transform = transforms.Compose([transforms.ToTensor()])
    target_tensor = transform(target_mask)
    target_tensor = target_tensor[None, :]

    pred_tensor = transform(prediction_mask)
    pred_tensor = pred_tensor[None, :]

    bm = BinaryMetrics()
    pixel_acc, dice, precision, specificity, recall, f1_score, iou = bm(
        y_pred=pred_tensor, y_true=target_tensor
    )
    if verbose:
        print(
            f"""
            pixel_acc: {pixel_acc:.4f}
            dice: {dice:.4f}
            precision: {precision:.4f}
            specificity: {specificity:.4f}
            recall: {recall:.4f}
            f1_score: {f1_score:.4f}
            iou: {iou:.4f}
            ------------------------------------
        """
        )
    return pixel_acc, dice, precision, specificity, recall, f1_score, iou


def q10(x):
    return x.quantile(0.1)


def q90(x):
    return x.quantile(0.9)


def q5(x):
    return x.quantile(0.05)


def q95(x):
    return x.quantile(0.95)


def q1(x):
    return x.quantile(0.01)


def q99(x):
    return x.quantile(0.99)
