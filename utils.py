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


def print_image(df, path, include_polyline=False, include_bounding_box=False):
    """
    function to show images
    polylines and bounding boxes can be plotted optionally
    """
    print(path)
    image = cv2.imread(f"./orig_data/DB/{path}")
    window_name = "Image"
    color = (255, 0, 0)
    if not include_polyline:
        plt.imshow(image)
        plt.show()
        return
    for idx, row in df[df["path"] == path].iterrows():
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
                        temp["object_uid"] = k2
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


def calculate_brightness(im_file):
    """
    calculate average brightness of image
    """
    im = Image.open(im_file).convert("L")
    stat = ImageStat.Stat(im)
    return stat.rms[0]


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


def generate_mask(df, path):
    """
    takes an image and the repective labels and generates mask
    """
    df = df[df["path"] == path]
    img = cv2.imread(f"./orig_data/DB/{path}")
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


def write_dataset_to_directory(df, path_yml, path_data, dataset, only_label=False):
    make_directory_structure(dataset=dataset, path=path_data)
    create_yml_file(path=path_yml, dataset=dataset)
    for idx, row in df[["path", "set"]].drop_duplicates().iterrows():
        source = f"./orig_data/DB/{row['path']}"
        filename = row["path"].replace("/", "_")
        destination = f"{path_data}{dataset}/{row['set']}/images/{filename}"
        shutil.copy(source, destination)

        mask = generate_mask(df, row["path"])
        labels, coordinates = find_contours(mask)
        with open(
            f"{path_data}{dataset}/{row['set']}/labels/{filename.replace('png','txt')}",
            "w",
        ) as f:
            f.write("\n".join(labels))


def apply_fast_line_detection(
    path,
    kernel_size=0,
    length_treshold=10,
    min_aspect_ration=0,
    max_aspect_ration=float("inf"),
):
    image_orig = cv2.imread(path)
    image = cv2.imread(path, cv2.CV_8UC1)
    if kernel_size > 0:
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    fld = cv2.ximgproc.createFastLineDetector(
        length_threshold=length_treshold, do_merge=True
    )
    lines = fld.detect(image)
    line_image = np.copy(image_orig)

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
    plt.imshow(line_image)
    plt.show()
