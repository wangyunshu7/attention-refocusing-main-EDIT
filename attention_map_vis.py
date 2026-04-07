import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch
import os
import imageio
import abc
import copy
from IPython.display import display
from typing import Union, Tuple, List, Optional


def get_all_attention(attn_maps_mid, attn_maps_up, attn_maps_down, res=16):
    result = []

    for attn_map_integrated in attn_maps_up:
        if attn_map_integrated == []: continue
        attn_map = attn_map_integrated[0][0]
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        # print(H)
        if H == res:
            result.append(attn_map.reshape(-1, res, res, attn_map.shape[-1]))
    for attn_map_integrated in attn_maps_mid:

        # for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated[0]
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        # print(H)
        if (H == res):
            result.append(attn_map.reshape(-1, res, res, attn_map.shape[-1]))
    # import pdb; pdb.set_trace()
    for attn_map_integrated in attn_maps_down:
        if attn_map_integrated == []: continue
        attn_map = attn_map_integrated[0][0]
        if attn_map == []: continue
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        # print(H)
        if (H == res):
            result.append(attn_map.reshape(-1, res, res, attn_map.shape[-1]))

    result = torch.cat(result, dim=0)
    result = result.sum(0) / result.shape[0]
    return result

def get_all_self_att(self_first, self_second, self_third,res=16):
    result = {256: [], 1024: [], 4096: [], 64: [], 94: [], 1054: [], 286: [], 4126: []}
    # import pdb; pdb.set_trace()
    all_att = [self_first, self_second, self_third]
    for self_att in all_att:
        for att in self_att:
            if att != []:
                temp = att[0]
                for attn_map in temp:
                    current_res = attn_map.shape[1]
                    # print(current_res)
                    result[current_res].append(attn_map)

    result = result[256]
    result = torch.cat(result, dim=0)
    result = result.sum(0) / result.shape[0]
    return result.reshape(res,res,result.shape[-1])

def show_image_relevance(image_relevance, image: Image.Image,
                         relevnace_res=16, is_self=False, x=0,
                         y=0,radius=20):  # Retro: image_relevance{Tensor:(16,16)} rensor([[0.0006,....],..])
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    # Retro add:
    image_relevance_to_show = image_relevance.detach().cpu().numpy()
    image_relevance_to_show = (image_relevance_to_show - image_relevance_to_show.min()) / (
            image_relevance_to_show.max() - image_relevance_to_show.min())
    image_relevance_to_show = np.uint8(255 * image_relevance_to_show)
    # image_relevance_to_show = Image.fromarray(image_relevance_to_show)
    # image_relevance_to_show.save("./visualization_outputs/heatmap16.jpg")
    image_relevance_to_show = image_relevance_to_show.reshape(image_relevance_to_show.shape[0], -1, 1)

    # image = image[0].resize((relevnace_res ** 2, relevnace_res ** 2))  # {Image} 256*256
    image = np.array(image)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)# Retro {ndarray:(256,256,3)} [[[16,21,25],[18,22,26],...],...]

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1],
                                              image_relevance.shape[-1])  # Retro: {Tensor:(1,1,16,16)}
    image_relevance = image_relevance.cuda(0)  # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2,
                                                      mode='bilinear')  # Retro: {Tensor:(1,1,256,256)}
    image_relevance = image_relevance.cpu()  # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (
            image_relevance.max() - image_relevance.min())  # Retro: 归一化->[0, 1]
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)  # Retro: {Tensor:(256,256)}
    image = (image - image.min()) / (image.max() - image.min())  # Retro: 归一化->[0,1] {ndarray:(256,256,3)}

    vis = show_cam_on_image(image, image_relevance.detach().cpu())
    vis = np.uint8(255 * vis)
    if is_self:
        mapped_x = int((x / 15) * 255)
        mapped_y = int((y / 15) * 255)
        radius = radius  # 圆点半径
        color = (0, 0, 255)  # 红色 (BGR 格式)
        cv2.circle(vis, (mapped_y, mapped_x), radius, color, -1)  # -1 表示填充整个圆
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    return vis, image_relevance_to_show


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img

def attention_map_vis_show(latent,input,att_first, att_second, att_third,self_first, self_second, self_third,timestep,res=16):
    #self_first[1][0][0].shape:8,4096,4096
    #att_first[3][0][0].shape:8,256,77
    autoencode = input['encoder']
    tokenizer = input['tokenizer']
    prompt = input['prompt']
    indices_to_alter = input['token_indice']
    iter_id = input['iter_id']

    ori_image = autoencode.decode(latent)
    ori_image = torch.clamp(ori_image[0], min=-1, max=1) * 0.5 + 0.5
    ori_image = ori_image.detach().cpu().numpy().transpose(1, 2, 0) * 255
    ori_image = ori_image.astype(np.uint8)

    ori_image_save_path = './visual/atten_vis/ori_image/' + prompt + str(iter_id)
    os.makedirs(ori_image_save_path, exist_ok=True)
    ori_save_image = Image.fromarray(ori_image)
    ori_save_image.save(ori_image_save_path + f"/iter_{timestep:02d}.jpg")

    tokens = tokenizer.encode(prompt)
    # decoder = tokenizer.decode
    decoder_prompt = prompt.lstrip().split(" ")

    cross_attn = get_all_attention(att_second, att_first, att_third)

    images=[]
    max_position_per_index = []
    for i in indices_to_alter:
        image = cross_attn[:, :, i]
        ii = torch.argmax(image)
        max_position_per_index.append(ii)
        image, image_16 = show_image_relevance(image,
                                               ori_image, is_self=False, x=ii // 16,
                                               y=ii % 16,
                                               radius=20)  # Retro: the key step, process attention map and original image
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
        image = text_under_image(image, decoder_prompt[i-1] + f" {ii // 16 + 1},{ii % 16 + 1}")
        images.append(image)

    pil_img = view_images(np.stack(images, axis=0), display_image=False)
    cross_save_path = './visual/atten_vis/cross/' + prompt + str(iter_id)
    os.makedirs(cross_save_path, exist_ok=True)
    pil_img.save(cross_save_path + f"/iter_{timestep:02d}.jpg")

    ####TODO:self-attention map save
    self_attn = get_all_self_att(self_first, self_second, self_third)
    boxes = input['boxes']
    self_save_path = './visual/atten_vis/self/' + prompt + str(iter_id)
    os.makedirs(self_save_path, exist_ok=True)
    for word_id in range(len(indices_to_alter)):
        images = []
        max_position_per_index = []
        x1, y1, x2, y2 = [ int(item * res) for item in boxes[word_id][0]]
        for i in range(x1,x2):
            for j in range(y1,y2):
                index = res * i + j
                image = self_attn[ : ,: ,index]
                ii = torch.argmax(image)
                max_position_per_index.append(ii)
                image, image_16 = show_image_relevance(image,
                                                       ori_image, is_self=True, x=ii // 16,
                                                       y=ii % 16,
                                                       radius=20)  # Retro: the key step, process attention map and original image
                image = image.astype(np.uint8)
                image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
                image = text_under_image(image,  f"({i},{j});({ii // 16 + 1},{ii % 16 + 1})")
                images.append(image)
        pil_img = view_images(np.stack(images, axis=0), display_image=False)
        pil_img.save(self_save_path + f"/{decoder_prompt[indices_to_alter[word_id] - 1]}_iter_{timestep:02d}.jpg")


    return


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img
