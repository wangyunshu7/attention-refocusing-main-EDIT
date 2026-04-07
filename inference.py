import argparse
from typing import List, Dict

from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os 
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from ldm.util import instantiate_from_config
from trainer import batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from scipy.io import loadmat
from functools import partial
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from chatGPT import read_txt_hrs, load_gt, load_box, save_img, read_csv, generate_box_gpt4, Pharse2idx_2, process_box_phrase, format_box, draw_box_2
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image, ImageDraw, ImageFont
from urllib.request import urlopen
import re
import spacy
import inflect
import random
from ldm.models.diffusion.loss import get_attention_map_index_to_wordpiece, get_indices


nlp = spacy.load("en_core_web_trf")
p = inflect.engine()



def attention_map_vis(autoencoder,attn_vis_dict,prompt,token_indice):
    save_path = "./visual/attention_map_vis/"+prompt
    os.makedirs(save_path, exist_ok=True)
    #将attention map放缩到同一个维度上



def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas



def load_ckpt(ckpt_path,device):
    
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device)
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config




def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask



@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None]*len(phrases) if images==None else images 
    phrases = [None]*len(images) if phrases==None else phrases 

    # version = "openai/clip-vit-large-patch14"
    version = "/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(  get_clip_feature(model, processor, phrase, is_image=False) )
        image_features.append( get_clip_feature(model, processor, image,  is_image=True) )

    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 

    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }

    return batch_to_device(out, device) 


def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize( (512, 512) )
    return image



@torch.no_grad()
def prepare_batch_kp(meta, batch=1, max_persons_per_image=8):
    
    points = torch.zeros(max_persons_per_image*17,2)
    idx = 0 
    for this_person_kp in meta["locations"]:
        for kp in this_person_kp:
            points[idx,0] = kp[0]
            points[idx,1] = kp[1]
            idx += 1
    
    # derive masks from points
    masks = (points.mean(dim=1)!=0) * 1 
    masks = masks.float()

    out = {
        "points" : points.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
    }

    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_hed(meta, batch=1):
    
    pil_to_tensor = transforms.PILToTensor()

    hed_edge = Image.open(meta['hed_image']).convert("RGB")
    hed_edge = crop_and_resize(hed_edge)
    hed_edge = ( pil_to_tensor(hed_edge).float()/255 - 0.5 ) / 0.5

    out = {
        "hed_edge" : hed_edge.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_canny(meta, batch=1):
    """ 
    The canny edge is very sensitive since I set a fixed canny hyperparamters; 
    Try to use the same setting to get edge 

    img = cv.imread(args.image_path, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img,100,200)
    edges = PIL.Image.fromarray(edges)

    """
    
    pil_to_tensor = transforms.PILToTensor()

    canny_edge = Image.open(meta['canny_image']).convert("RGB")
    canny_edge = crop_and_resize(canny_edge)

    canny_edge = ( pil_to_tensor(canny_edge).float()/255 - 0.5 ) / 0.5

    out = {
        "canny_edge" : canny_edge.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_depth(meta, batch=1):
    
    pil_to_tensor = transforms.PILToTensor()

    depth = Image.open(meta['depth']).convert("RGB")
    depth = crop_and_resize(depth)
    depth = ( pil_to_tensor(depth).float()/255 - 0.5 ) / 0.5

    out = {
        "depth" : depth.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 



@torch.no_grad()
def prepare_batch_normal(meta, batch=1):
    """
    We only train normal model on the DIODE dataset which only has a few scene.

    """
    
    pil_to_tensor = transforms.PILToTensor()

    normal = Image.open(meta['normal']).convert("RGB")
    normal = crop_and_resize(normal)
    normal = ( pil_to_tensor(normal).float()/255 - 0.5 ) / 0.5

    out = {
        "normal" : normal.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 





def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    return labelmap_rgb

@torch.no_grad()
def prepare_batch_sem(meta, batch=1):

    pil_to_tensor = transforms.PILToTensor()

    sem = Image.open( meta['sem']  ).convert("L") # semantic class index 0,1,2,3,4 in uint8 representation 
    sem = TF.center_crop(sem, min(sem.size))
    sem = sem.resize( (512, 512), Image.NEAREST ) # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly
    try:
        sem_color = colorEncode(np.array(sem), loadmat('color150.mat')['colors'])
        Image.fromarray(sem_color).save("sem_vis.png")
    except:
        pass 
    sem = pil_to_tensor(sem)[0,:,:]
    input_label = torch.zeros(152, 512, 512)
    sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

    out = {
        "sem" : sem.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 




# def run(meta, config, starting_noise=None):

    # - - - - - prepare models - - - - - # 
# @torch.no_grad()
def run(meta,models,info_files, p, starting_noise=None,iter_id=0, img_id=0, save=True, stage="img"):
    model, autoencoder, text_encoder, diffusion, config = models

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    
    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])



    # - - - - - update config from args - - - - - # 
    config.update( vars(args) )
    config = OmegaConf.create(config)


    # - - - - - prepare batch - - - - - #
    if "keypoint" in meta["ckpt"]:
        batch = prepare_batch_kp(meta, config.batch_size)
    elif "hed" in meta["ckpt"]:
        batch = prepare_batch_hed(meta, config.batch_size)
    elif "canny" in meta["ckpt"]:
        batch = prepare_batch_canny(meta, config.batch_size)
    elif "depth" in meta["ckpt"]:
        batch = prepare_batch_depth(meta, config.batch_size)
    elif "normal" in meta["ckpt"]:
        batch = prepare_batch_normal(meta, config.batch_size)
    elif "sem" in meta["ckpt"]:
        batch = prepare_batch_sem(meta, config.batch_size)
    else:
        batch = prepare_batch(meta, config.batch_size)
    context = text_encoder.encode(  [meta["prompt"]]*config.batch_size  )
    uc = text_encoder.encode( config.batch_size*[""] )
    with torch.no_grad():
        if args.negative_prompt is not None:
            uc = text_encoder.encode( config.batch_size*[args.negative_prompt] )


    # - - - - - sampler - - - - - # 
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250 
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50 


    # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process
    inpainting_extra_input = None # used as model input 
    if "input_image" in meta:
        # inpaint mode 
        assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'
        
        inpainting_mask = draw_masks_from_boxes( batch['boxes'], model.image_size  ).cuda()
        
        input_image = F.pil_to_tensor( Image.open(meta["input_image"]).convert("RGB").resize((512,512)) ) 
        input_image = ( input_image.float().unsqueeze(0).cuda() / 255 - 0.5 ) / 0.5
        z0 = autoencoder.encode( input_image )
        
        masked_z = z0*inpainting_mask
        inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
    

    # - - - - - input for gligen - - - - - #

    grounding_input = grounding_tokenizer_input.prepare(batch)
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)

    input = dict(
                x = starting_noise, 
                timesteps = None, 
                context = context, 
                grounding_input = grounding_input,
                inpainting_extra_input = inpainting_extra_input,
                grounding_extra_input = grounding_extra_input,
                boxes=meta['ll'],
                object_position = meta['position'],
                token_indice = meta['token_indice'],
                part_of_speech = meta['part_of_speech'],
                include_entities = meta['include_entities'],
                bbox = meta['bbox'],
                prompt = meta['prompt'],
                attn_map_idx_to_wp = meta['attn_map_idx_to_wp'],
                syngen_indice = meta['syngen_indice'],
                recitify = meta['recitify'],
                stage = stage,
                froze_area = meta['froze_area'],
                encoder = autoencoder,
                tokenizer = text_encoder,
                iter_id = iter_id,
                save_attn_vis = meta['save_attn_vis'],
                is_edited = meta['is_edited']
            )


    # - - - - - start sampling - - - - - #
    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

    # samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0, loss_type=args.loss_type)
    samples_fake, noise_list = sampler.sample(S=steps, shape=shape, input=input, uc=uc,
                                              guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0,
                                              loss_type=args.loss_type)


    ##TODO:添加编辑过程
    if stage == 'fg':
        clip_model, preprocess = clip.load("/root/.cache/clip/ViT-B-32.pt", device=device)
        samples_img = autoencoder.decode(samples_fake)
        # check/find inpaint box
        all_boxes = meta['locations']
        all_phrases = meta['phrases']
        worse_boxes = []
        better_boxes = []
        worse_id = []
        for i, box in enumerate(all_boxes):
            img_box = [round(value * 512) for value in box]
            left, top, right, bottom = img_box
            left = max(0, left - 20)
            top = max(0, top - 20)
            right = min(512, right + 20)
            bottom = min(512, bottom + 20)
            cropped_img = samples_img[:, :, top:bottom, left:right]
            cropped_img = torch.clamp(cropped_img[0], min=-1, max=1) * 0.5 + 0.5
            cropped_img = cropped_img.detach().cpu().numpy().transpose(1, 2, 0) * 255
            try:
                cropped_img = Image.fromarray(cropped_img.astype(np.uint8))
            except:
                continue

            image = preprocess(cropped_img).unsqueeze(0).to(device)
            texts = ["a photo of " + all_phrases[i]]
            text = clip.tokenize(texts).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image)
                text_features = clip_model.encode_text(text)
                cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
                print("clip_sim:", cosine_similarity)
            if cosine_similarity < 0.3:
                box = [left / 512, top / 512, right / 512, bottom / 512]
                worse_boxes.append(box)
                worse_id.append(i)
            else:
                better_boxes.append(box)
        print("worse boxes: ", worse_id)
        # mask noise
        if len(worse_boxes) > 0 and len(better_boxes) > 0:
            save_before_edited = meta["save_folder_name"]+"_before_edited"
            ####TODO:保存编辑前的图像
            output_folder1 = os.path.join(args.folder, save_before_edited + '/img')
            os.makedirs(output_folder1, exist_ok=True)
            output_folder2 = os.path.join(args.folder, save_before_edited + '/box')
            os.makedirs(output_folder2, exist_ok=True)
            start = len(os.listdir(output_folder2))
            image_ids = list(range(start, start + config.batch_size))
            print(image_ids)
            font = ImageFont.truetype("Roboto-LightItalic.ttf", size=20)
            for image_id, sample in zip(image_ids, samples_img):
                img_name = meta['prompt'].replace(' ', '_') + str(int(image_id)) + '.png'
                sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
                # sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
                sample = sample.detach().cpu().numpy().transpose(1, 2, 0) * 255
                sample = Image.fromarray(sample.astype(np.uint8))
                img2 = sample.copy()
                draw = ImageDraw.Draw(sample)
                boxes = meta['location_draw']
                text = meta["phrases"]
                info_files.update({img_name: (text, boxes)})
                for i, box in enumerate(boxes):
                    t = text[i]

                    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=128, width=2)
                    draw.text((box[0] + 5, box[1] + 5), t, fill=200, font=font)
                save_img(output_folder2, sample, meta['prompt'], iter_id, img_id)
                save_img(output_folder1, img2, meta['prompt'], iter_id, img_id)
            #####
            masked_noises = []
            for noise in noise_list:
                mask = torch.zeros_like(noise).to(device)
                rdm_noise = torch.randn_like(noise).to(device)

                for box in better_boxes:
                    latent_box = [round(value * 64) for value in box]
                    left, top, right, bottom = latent_box
                    left = max(0, left - 5)
                    top = max(0, top - 5)
                    right = min(512, right + 5)
                    bottom = min(512, bottom + 5)
                    mask[:, :, (top):(bottom), (left):(right)] = 1
                    blurred_mask = F.gaussian_blur(mask, kernel_size=3, sigma=2.0)
                    blurred_mask = blurred_mask / blurred_mask.max()
                for box in worse_boxes:
                    latent_box = [round(value * 64) for value in box]
                    left, top, right, bottom = latent_box
                    left = max(0, left - 5)
                    top = max(0, top - 5)
                    right = min(512, right + 5)
                    bottom = min(512, bottom + 5)
                    blurred_mask[:, :, (top):(bottom), (left):(right)] = 0

                noise = noise * blurred_mask
                masked_noises.append(noise)

            return noise_list, worse_id, blurred_mask
    ####end

    with torch.no_grad():
        samples_fake = autoencoder.decode(samples_fake)


    # save images
    if save :
        path = meta["save_folder_name"]
        output_folder1 = os.path.join(args.folder, meta["save_folder_name"] + '/img')
        os.makedirs(output_folder1, exist_ok=True)
        output_folder2 = os.path.join(args.folder, meta["save_folder_name"] + '/box')
        os.makedirs(output_folder2, exist_ok=True)
        start = len( os.listdir(output_folder2) )
        image_ids = list(range(start,start+config.batch_size))
        print(image_ids)
        font = ImageFont.truetype("Roboto-LightItalic.ttf", size=20)
        for image_id, sample in zip(image_ids, samples_fake):
            img_name = meta['prompt'].replace(' ', '_') + str(int(image_id))+'.png'
            sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
            sample = sample.cpu().numpy().transpose(1,2,0) * 255 
            sample = Image.fromarray(sample.astype(np.uint8))
            img2 = sample.copy()
            draw = ImageDraw.Draw(sample)
            boxes = meta['location_draw']
            text = meta["phrases"]
            info_files.update({img_name: (text, boxes)})
            for i, box in enumerate(boxes):
                t = text[i]

                draw.rectangle([(box[0], box[1]),(box[2], box[3])], outline=128, width=2)
                draw.text((box[0]+5, box[1]+5), t, fill=200,font=font )
            save_img(output_folder2, sample,meta['prompt'],iter_id,img_id)
            save_img(output_folder1,img2,meta['prompt'],iter_id ,img_id )
    return samples_fake,None,None


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable[2].tokenizer.decode(t)
                         for idx, t in enumerate(stable[2].tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable[2].tokenizer(prompt)['input_ids']) - 1}

    return token_idx_to_word

def get_token_pof(token,doc):
    get_part_of_speech = False
    pof_index = None
    for doc_token in doc:
        if token == doc_token.text:
            if doc_token.pos_ in ["NOUN", "PROPN"]:
                pof_index=1
                get_part_of_speech = True
            elif doc_token.pos_ in ["ADJ","amod", "nmod", "compound",
                                    "npadvmod", "advmod", "acomp",'relcl']:
                pof_index=0
                get_part_of_speech = True
        if get_part_of_speech:
            break

    return  pof_index, get_part_of_speech

def get_token_indice(token, token_idx_to_word,token_indice):
    token = token.split("'")[0]
    # token_plural = p.plural(token)
    find_indx = False
    for key, value in token_idx_to_word.items():
        if (token in value) and int(key) not in token_indice:
            token_indice.append(int(key))
            find_indx = True
            break
    if not find_indx:
        try:
            for key, value in token_idx_to_word.items():
                if value in token and int(key) not in token_indice:
                    token_indice.append(int(key))
                    find_indx = True
                    break
        except:
            pass
    return find_indx, token_indice

def get_token_bbox(token, layout, new_size=1, old_size = 256):
    multi_token_box = []
    for key, value in layout.items():
        if token in key:
            x1,y1,x2,y2 = value
            value = [ max(min(item*new_size/old_size,new_size),0) for item in value]
            # value = [value[0], value[1], min(value[2]+1, 64) if value[2]==value[0] else value[2],
            #          min(value[3]+1, 64) if value[3]==value[1] else value[3]]
            multi_token_box.append(value)
    if len(multi_token_box) == 1:
        multi_token_box = multi_token_box[0]

    return multi_token_box


def Pharse2idx_2(prompt, name_box):
    prompt = prompt.replace('.', '')
    prompt = prompt.replace(',', '')
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    bbox_to_self_att = []
    for obj in name_box.keys():
        obj_position = []
        in_prompt = False
        for word in obj.split(' '):
            if word in prompt_list:
                obj_first_index = prompt_list.index(word) + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word + 's' in prompt_list:
                obj_first_index = prompt_list.index(word + 's') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word + 'es' in prompt_list:
                obj_first_index = prompt_list.index(word + 'es') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
        if in_prompt:
            bbox_to_self_att.append(np.array(name_box[obj]))

            object_positions.append(obj_position)

    return object_positions, bbox_to_self_att

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,  default="visual", help="root folder for output")
    parser.add_argument('--ckpt', type=str, default='/root/autodl-tmp/gligen_model/diffusion_pytorch_model.bin', help='path to the checkpoint')

    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    # parser.add_argument("--negative_prompt", type=str,  default="cropped images", help="")
    
    parser.add_argument("--file_save",default='output', type=str)
    parser.add_argument("--layout",default='layout', type=str)
    parser.add_argument("--loss_type", choices=['standard','SAR','CAR','SAR_CAR','Syngen_PCA_RCA','PCA_RCA_SAR','Syngen_RCA_SAR','Syngen_PCA_SAR','Syngen_PCA_RCA_SAR'],default='Syngen_PCA_RCA', help='Choose one option among the four options for what types of losses ')
    parser.add_argument("--prompt_path",default='./T2I-CompBench_dataset/color_val.txt',type=str)
    parser.add_argument("--p_start", default=0, type=int)
    parser.add_argument("--recitify", default=False, type=bool)
    parser.add_argument("--save_attn_vis", default=False, type=bool)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()
    


    meta_list = [ 

        # - - - - - - - - GLIGEN on text grounding for generation - - - - - - - - # 
        dict(
            ckpt = args.ckpt,
            prompt =None,
            phrases = None,
            locations = None,
            alpha_type = [0.3, 0.0, 0.7],
            save_folder_name=args.file_save,
            ll = None,
            token_indice = None,
            part_of_speech = None,
            only_noun = False,
            bbox = None,
            froze_area=None
        )
    ]

    device = args.device

    info_files = {}
    models = load_ckpt(meta_list[0]["ckpt"],device)
    i=0
    prompt_layout_path = "./layout_generation"
    prompts = []
    with open(args.prompt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.replace("\n", "").replace(".", "").strip())

    sub_file_name = args.prompt_path.split("/")[-1]
    all_caption_layout = {}
    with open(prompt_layout_path+f"/{sub_file_name}", "r") as f:
        lines = f.readlines()
        for line_index, line in enumerate(lines):
            caption_key = line.split("::::")[0]
            layout_dict = line.split("::::")[-1].replace("\n", "")
            pattern = r"'(.*?)': \[(.*?)\]" #键值对
            matches = re.findall(pattern, layout_dict)
            # 创建一个字典来存储拆分后的键值对
            layout_dict = {key: list(map(int, value.split(', '))) for key, value in matches} #把prompt和layout封装成字典
            all_caption_layout[caption_key] = layout_dict#字典嵌套caption_key作为外层key， layout_dict作为value

    sub_file_name = sub_file_name.split(".")[0]
    for p_id, user_input in enumerate(prompts):
        if p_id >= args.p_start:
            for meta in meta_list:
                pp = user_input
                meta["prompt"] = user_input
                text = user_input
                if user_input in all_caption_layout.keys():
                    o_names = []
                    o_boxes = []
                    for key, value in all_caption_layout[user_input].items():
                        o_names.append(key.split("-")[0])
                        o_boxes.append(tuple([x * 2 for x in value]))

                    ###增加part_of_speech等参数
                    doc = nlp(user_input)
                    token_idx_to_word = get_indices_to_alter(models, user_input)
                    bbox, token_indice, part_of_speech, layout_key = [], [], [], []
                    layout = all_caption_layout[user_input]
                    for key, value in layout.items():
                        layout_key.append(key.split("-")[0])
                    layout_key = list(set(layout_key))
                    for tokens in layout_key:
                        tokens = tokens.split(" ")
                        for token in tokens:#把 red cat 遍历 分为 red 和 cat
                            token_pof, get_pof = get_token_pof(token, doc)
                            if get_pof:
                                part_of_speech.append(token_pof)
                                find_indx, token_find_index = get_token_indice(token, token_idx_to_word, token_indice)
                                if find_indx:
                                    token_indice = token_find_index
                                    bbox.append(get_token_bbox(token, layout))

                    #syngen_loss 需要的
                    attn_map_idx_to_wp = get_attention_map_index_to_wordpiece(models[2].tokenizer, user_input)
                    syngen_indice = get_indices(models[2].tokenizer, user_input)
                    #yellow monkey---->yellow 和 monkey 都对应那个monkey框
                    #part_of_speech 1代表名词，0代表修饰词 [[0,1,1] 对应 ‘yellow monkey’ ‘dog’
                    #token_indice [2,5] 对应 ‘monkey’ ‘dog’在句子的位置在第2和第5个，前提是这些词在词典里存在
                    if len(token_indice) == len(bbox) == len(part_of_speech) and len(token_indice) != 0:
                        sorted_tokens = sorted(zip(token_indice, part_of_speech, bbox))
                        token_indice, part_of_speech, bbox = zip(*sorted_tokens)
                        include_entities = True if all(x == 1 for x in part_of_speech) else False
                        fixed_seed = random.randint(0, 10**6)
                        #number of generated images for one prompt
                        for k in range(10):
                            torch.manual_seed(fixed_seed + k)
                            starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)
                            p, ll  = format_box(o_names, o_boxes)#pharse=="a yellow monkey"加 ”a“
                            l = np.array(o_boxes)
                            name_box = process_box_phrase(o_names, o_boxes)
                            #generate format box and positions for losses
                            position, box_att = Pharse2idx_2(pp, name_box)
                            #save layout
                            layout_folder = args.layout
                            os.makedirs( layout_folder, exist_ok=True)
                            draw_box_2(o_names, box_att ,layout_folder,str(i) + '_' +meta["prompt"].replace(' ',"_") + '.jpg' )
                            print('position', position )
                            # phrase
                            meta["phrases"] = p
                            # location integer to visual box
                            meta['location_draw'] = l
                            #location scale, the input GLIGEN
                            meta["locations"] = l/512
                            # the box format using for CAR and SAR loss
                            meta['ll'] = box_att
                            # the locations of words which out of GPT4, label of boxes
                            meta['position'] = position
                            # 迭代优化的过程中需要的新的参数
                            meta['prompt'] = user_input
                            meta['token_indice'] = token_indice
                            meta['part_of_speech'] = part_of_speech
                            meta['include_entities'] = include_entities  #是否全是名词
                            meta['bbox'] = bbox
                            meta['attn_map_idx_to_wp'] = attn_map_idx_to_wp  #不带特殊词【cls】的完整句子和字符位置索引
                            meta['syngen_indice'] = syngen_indice #带特殊词【cls】的完整句子和字符位置索引
                            meta['save_folder_name'] = args.file_save +"/"+ sub_file_name
                            meta['recitify'] = args.recitify
                            meta['save_attn_vis'] = args.save_attn_vis
                            meta['is_edited'] = False

                            if args.recitify:
                                fg_noise, worse_id, froze_area = run(meta, models, info_files, args, starting_noise, k,p_id, stage='fg')

                                if worse_id == None: continue
                                meta['froze_area'] = froze_area
                                name_worse_box = process_box_phrase([o_names[id] for id in worse_id],
                                                                    [o_boxes[id] for id in worse_id])
                                worse_l = np.array([l[id] for id in worse_id])
                                meta['locations'] = worse_l / 512
                                worse_position, worse_box_att = Pharse2idx_2(pp, name_worse_box)
                                meta['ll'] = worse_box_att
                                meta['position'] = worse_position
                                meta['is_edited'] = True
                                print(f"Starting second phase.")
                                run(meta, models, info_files, args, fg_noise, iter_id=k, img_id=p_id)

                            else:
                                run(meta, models, info_files, args, starting_noise, iter_id=k, img_id=p_id)








