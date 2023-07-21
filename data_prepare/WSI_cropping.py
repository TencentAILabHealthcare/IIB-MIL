###
import argparse
import glob
import multiprocessing
import os
import pickle
import random
import threading
import time
import warnings
from os import makedirs, path
from os.path import join
from xml.etree.ElementTree import parse

import numpy as np
import openslide as slide
import pandas as pd
import shapely.geometry as shgeo
from skimage import img_as_ubyte, io, transform
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from tqdm import tqdm

warnings.simplefilter("ignore")


# %%
def parse_filename_from_directory(input_file_list):
    output_file_list = [
        os.path.basename(os.path.splitext(item)[0]) for item in input_file_list
    ]
    return output_file_list


def thres_saturation(img, t=15):
    """
    Color saturation is the intensity and purity of a color as displayed in an image.
    The higher the saturation of a color, the more vivid and intense it is.
    The lower a color's saturation, the closer it is to pure gray on the grayscale.
    :param img:
    :param t:
    :return: boolean
    """
    img = rgb2hsv(img)
    h, w, c = img.shape
    sat_img = img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return ave_sat >= t


def crop_slide(
    img,
    save_slide_path,
    position=(0, 0),
    step=(0, 0),
    patch_size=224,
    scale=10,
    down_scale=1,
):  # position given as (x, y) at nx scale
    patch_name = "{}_{}".format(step[0], step[1])

    img_nx_path = join(
        save_slide_path,
        f"{patch_name}-tile-r{position[1] * down_scale}-c{position[0] * down_scale}-{patch_size}x{patch_size}.png",
    )
    if path.exists(img_nx_path):
        return 1

    img_x = img.read_region(
        (position[0] * down_scale, position[1] * down_scale),
        0,
        (patch_size * down_scale, patch_size * down_scale),
    )
    img_x = np.array(img_x)[..., :3]
    # if down_scale!=1:
    img = transform.resize(
        img_x,
        (img_x.shape[0] // down_scale, img_x.shape[0] // down_scale),
        order=1,
        anti_aliasing=False,
    )
    # if thres_saturation(img, 30): # -1 for all
    try:
        io.imsave(img_nx_path, img_as_ubyte(img))
    except Exception as e:
        print(e)


def slide_to_patch(out_base, img_slides, patch_size, step_size, scale, down_scale=1):
    makedirs(out_base, exist_ok=True)
    for s in tqdm(range(len(img_slides))):
        img_slide = img_slides[s]
        img_name = img_slide.split(path.sep)[-1].split(".")[0]
        bag_path = join(out_base, img_name)

        makedirs(bag_path, exist_ok=True)
        img = slide.OpenSlide(img_slide)

        try:
            if int(np.floor(float(img.properties["openslide.mpp-x"]) * 10)) == 2:
                down_scale = 40 // scale
            else:
                down_scale = 20 // scale
        except Exception as e:
            print("tiff --> No properties 'openslide.mpp-x'")

        dimension = img.level_dimensions[0]
        # dimension and step at given scale
        # print(dimension,down_scale)
        step_y_max = int(np.floor(dimension[1] / (step_size * down_scale)))  # rows
        step_x_max = int(np.floor(dimension[0] / (step_size * down_scale)))  # columns
        print("number :", step_x_max, step_y_max, step_x_max * step_y_max)
        num = step_x_max * step_y_max
        count = 0
        for j in range(step_y_max):
            for i in range(step_x_max):
                start_time = time.time()
                crop_slide(
                    img,
                    bag_path,
                    (i * step_size, j * step_size),
                    step=(j, i),
                    patch_size=patch_size,
                    scale=scale,
                    down_scale=down_scale,
                )
                end_time = time.time()
                count += 1
                print(f"{count}/{num}", (end_time - start_time) / 60)


def extract_coord_from_xml(xml_path: str) -> list:
    """
    extract annotation coordinates from xml file

    Args:
        xml_path (str): xml file path

    Returns:
        list: a list contains annotation group information, location convert to row & col
    """
    try:
        with open(xml_path, "rt", encoding="utf-8") as infile:
            xml_doc = parse(infile)
    except:
        print(f"=>>>>>>>> cant load file {xml_path}")
        return []

    coord_group = []
    for annot in xml_doc.iterfind("Annotations/Annotation"):
        annot_name = annot.get("Name")
        annot_color = annot.get("Color")
        # print(f"{annot_name} with {annot_color}")
        coords = []
        for coord in annot.iterfind("Coordinates/Coordinate"):
            x = float(coord.get("X"))
            y = float(coord.get("Y"))
            row_idx, col_idx = y, x
            coords.append((row_idx, col_idx))
        if len(coords) < 3:
            print(f"{infile} annotation error coordinate less than 3")
            continue
        polygon = shgeo.Polygon(coords)
        polygon = polygon.buffer(0.01)
        coord_group.append(
            {
                "Name": annot_name,
                "Color": annot_color,
                "Coords": coords,
                "Polygon": polygon,
            }
        )

    return coord_group


def convert_diag_to_cycle_coords(pst_lt: tuple, pst_rb: tuple):
    """
    Args:
        pst_lt (tuple): [description]
        pst_rb (tuple): [description]

    Returns:
        [type]: [description]
    """
    row_start, col_start = pst_lt
    row_end, col_end = pst_rb

    right_top_pst = (row_start, col_end)
    left_bottom_pst = (row_end, col_start)

    return [pst_lt, right_top_pst, pst_rb, left_bottom_pst]


def get_patch_size(pst_lt, pst_rb):
    row_start, col_start = pst_lt
    row_end, col_end = pst_rb
    return (col_end - col_start) * (row_end - row_start)


def is_patch_in_annotation(
    pst_lt: tuple, pst_rb: tuple, annot_group: list, thres=0.1
) -> bool:
    # print(pst_lt)
    # print(pst_rb)
    patch_coords = convert_diag_to_cycle_coords(pst_lt, pst_rb)
    patch_polygon = shgeo.Polygon(patch_coords)
    patch_polygon = patch_polygon.buffer(0.01)  # added by zhenyulin
    patch_szie = get_patch_size(pst_lt, pst_rb)
    for each in annot_group:
        insert_area = patch_polygon.intersection(each["Polygon"]).area
        if insert_area / patch_szie > thres:
            return True
    return False


def slide_to_patch_in_annotation(out_base, img_slide, patch_size, step_size, scale):
    CROP_ALL = False
    patch_in_annotation_list = list()
    makedirs(out_base, exist_ok=True)

    img_name = img_slide.split(path.sep)[-1].split(".")[0]
    bag_path = join(out_base, img_name)
    xml_path = path.join(path.dirname(img_slide), img_name + ".xml")
    if not path.exists(xml_path):
        CROP_ALL = True
        print(".xml not found, crop all patches..", img_name)
    else:
        annot_group = extract_coord_from_xml(xml_path)
    makedirs(bag_path, exist_ok=True)
    try:
        img = slide.OpenSlide(img_slide)
    except Exception as e:
        print(e)
        return img_name, []
    down_scale = 1
    try:
        if int(np.floor(float(img.properties["openslide.mpp-x"]) * 10)) == 2:
            down_scale = 40 // scale
        else:
            down_scale = 20 // scale
    except Exception as e:
        print(e)

    dimension = img.level_dimensions[0]
    # dimension and step at given scale
    step_y_max = int(np.floor(dimension[1] / (step_size * down_scale)))  # rows
    step_x_max = int(np.floor(dimension[0] / (step_size * down_scale)))  # columns
    for j in range(step_y_max):
        for i in range(step_x_max):
            patch_name = "{}_{}".format(j, i)
            o_r_s, o_c_s = j * step_size * down_scale, i * step_size * down_scale
            o_r_e, o_c_e = o_r_s + patch_size, o_c_s + patch_size
            if CROP_ALL or is_patch_in_annotation(
                (o_r_s, o_c_s), (o_r_e, o_c_e), annot_group, thres=0.01
            ):
                mark = 1
                img_nx_path = join(
                    bag_path,
                    f"{patch_name}-tile-r{j * step_size * down_scale}-c{i * step_size * down_scale}-{patch_size}x{patch_size}.png",
                )
                # enable to crop at the same time
                crop_slide(
                    img,
                    bag_path,
                    mark,
                    (i * step_size, j * step_size),
                    step=(j, i),
                    patch_size=patch_size,
                    scale=scale,
                    down_scale=down_scale,
                )
                if path.exists(img_nx_path):
                    print("in annotation", img_nx_path)
                patch_in_annotation_list.append(img_nx_path)

    return img_name, patch_in_annotation_list


def slide_to_patch_display(out_base, img_slide, patch_size, step_size, scale):
    img_name = img_slide.split(path.sep)[-1].split(".")[0]
    bag_path = join(out_base, img_name)
    makedirs(bag_path, exist_ok=True)
    img = slide.OpenSlide(img_slide)
    down_scale = 1
    try:
        if int(np.floor(float(img.properties["openslide.mpp-x"]) * 10)) == 2:
            down_scale = 40 // scale
        else:
            down_scale = 20 // scale
    except Exception as e:
        print("tiff --> No properties 'openslide.mpp-x'")
    dimension = img.level_dimensions[0]
    # dimension and step at given scale
    step_y_max = int(np.floor(dimension[1] / (step_size * down_scale)))  # rows
    step_x_max = int(np.floor(dimension[0] / (step_size * down_scale)))  # columns
    return step_x_max * step_y_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop the WSIs into patches")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of threads for parallel processing, too large may result in errors",
    )
    parser.add_argument(
        "--overlap", type=int, default=0, help="Overlap pixels between adjacent patches"
    )
    parser.add_argument("--patch_size", type=int, default=1120, help="Patch size")
    parser.add_argument("--scale", type=int, default=20, help="20x 10x 5x")
    parser.add_argument(
        "--dataset", type=str, default="./dataset", help="Dataset folder name"
    )
    parser.add_argument(
        "--output", type=str, default="./result/tiled_patchs", help="Output folder name"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display patch numbers under this setting",
    )
    parser.add_argument(
        "--annotation", action="store_true", help="Obtain patches in annotation region"
    )
    args = parser.parse_args()

    print(
        "Cropping patches, this could take a while for big dataset, please be patient"
    )
    step = args.patch_size - args.overlap

    # obtain dataset paths
    path_base = args.dataset
    out_base = args.output
    if path.isdir(path_base):
        all_slides = (
            glob.glob(f"{path_base}/*.svs")
            + glob.glob(f"{path_base}/*.tif")
            + glob.glob(f"{path_base}/*.tiff")
            + glob.glob(f"{path_base}/*.mrxs")
            + glob.glob(f"{path_base}/*.ndpi")
        )
    elif path.isfile(path_base):
        df = pd.read_csv(path_base)
        all_slides = df.Slide_Path.values.tolist()
    else:
        raise ValueError(f"Please check dataset folder {path_base}")

    print("Number of .svs .mrxs .ndpi .tif/f", len(all_slides))

    # display the overview
    if args.display:
        patch_numbers = list()
        random.shuffle(all_slides)
        for s in tqdm(all_slides[:100]):
            patch_numbers.append(
                slide_to_patch_display(out_base, s, args.patch_size, step, args.scale)
            )
        print(
            f"patch numbers: mean-{np.mean(patch_numbers)}, min-{np.min(patch_numbers)}, max={np.max(patch_numbers)}"
        )

    # obtain patches in or not in annotation area
    elif args.annotation:
        pool = multiprocessing.Pool(args.num_threads)
        dict_name2img = dict()
        tasks = []
        results = []
        for s in all_slides:
            tasks.append((out_base, s, args.patch_size, step, args.scale))
        pbar = tqdm(total=len(tasks))

        def update(*a):
            pbar.update()

        for t in tasks:
            results.append(
                pool.apply_async(slide_to_patch_in_annotation, t, callback=update)
            )
        for result in results:
            name, patch_list = result.get()
            dict_name2img[name] = patch_list
        for s in tqdm(all_slides):
            name, patch_list = slide_to_patch_in_annotation(
                out_base, s, args.patch_size, step, args.scale
            )
            dict_name2img[name] = patch_list
        pickle.dump(dict_name2img, open(f"{out_base}/dict_name2imgs.pkl", "wb"))
        print(f"saving dict_name2imgs.pkl to {out_base}..")

    # crop all patches
    else:
        each_thread = int(np.floor(len(all_slides) / args.num_threads))
        threads = []
        for i in range(args.num_threads):
            if i < (args.num_threads - 1):
                t = threading.Thread(
                    target=slide_to_patch,
                    args=(
                        out_base,
                        all_slides[each_thread * i : each_thread * (i + 1)],
                        args.patch_size,
                        step,
                        args.scale,
                    ),
                )
            else:
                t = threading.Thread(
                    target=slide_to_patch,
                    args=(
                        out_base,
                        all_slides[each_thread * i :],
                        args.patch_size,
                        step,
                        args.scale,
                    ),
                )
            threads.append(t)

        for thread in threads:
            thread.start()

        dict_name2img = dict()
        ID_list = parse_filename_from_directory(all_slides)
        print(ID_list)
        for case_ID in ID_list:
            patch_list = glob.glob(f"{out_base}/{case_ID}/*.png")
            dict_name2img[case_ID] = patch_list
        pickle.dump(dict_name2img, open(f"{out_base}/dict_name2imgs.pkl", "wb"))
        print(f"saving dict_name2imgs.pkl to {out_base}..")
