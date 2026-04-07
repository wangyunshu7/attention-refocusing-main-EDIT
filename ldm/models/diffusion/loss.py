import itertools
from typing import List, Union, Dict
import torch.distributions as dist
import math
import torch
from ldm.models.diffusion.gaussian_smoothing import GaussianSmoothing
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import spacy
from collections import defaultdict
parser = spacy.load("en_core_web_trf")
start_token = "<|startoftext|>"
end_token = "<|endoftext|>"

def loss_one_att_outside(attn_map, bboxes, object_positions, t):
    loss = 0
    object_number = len(bboxes)
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))

    for obj_idx in range(object_number):

        for obj_box in bboxes[obj_idx]:
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                                         int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
            mask[y_min: y_max, x_min: x_max] = 1.
            mask_out = 1. - mask
            index = (mask == 1.).nonzero(as_tuple=False)
            index_in_key = index[:, 0] * H + index[:, 1]
            att_box = torch.zeros_like(attn_map)
            att_box[:, index_in_key, :] = attn_map[:, index_in_key, :]

            att_box = att_box.sum(axis=1) / index_in_key.shape[0]
            att_box = att_box.reshape(-1, H, H)
            activation_value = (att_box * mask_out).reshape(b, -1).sum(dim=-1)  # / att_box.reshape(b, -1).sum(dim=-1)
            loss += torch.mean(activation_value)

    return loss / object_number


def caculate_loss_self_att(self_first, self_second, self_third, bboxes, object_positions, t, list_res=[256],
                           smooth_att=True, sigma=0.5, kernel_size=3):
    all_attn = get_all_self_att(self_first, self_second, self_third)
    cnt = 0
    total_loss = 0
    for res in list_res:
        attn_maps = all_attn[res]
        for attn in attn_maps:
            total_loss += loss_one_att_outside(attn, bboxes, object_positions, t)
            cnt += 1

    return total_loss / cnt


def get_all_self_att(self_first, self_second, self_third):
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
    return result


def get_all_attention(attn_maps_mid, attn_maps_up, attn_maps_down, res):
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


def _get_data(box, image):
    xmin, ymin, xmax, ymax = [int(val*image.shape[0]) for val in box]
    if xmin != xmax and ymin != ymax:
        max_data = image[xmin:xmax, ymin:ymax].max()
        mean_data = image[xmin:xmax, ymin:ymax].mean()
    elif xmin == xmax and ymin == ymax:
        max_data = image[xmin, ymin]
        mean_data = image[xmin, ymin]
    elif xmin == xmax:
        max_data = image[xmin, ymin:ymax].max()
        mean_data = image[xmin, ymin:ymax].mean()
    elif ymin == ymax:
        max_data = image[xmin:xmax, ymin].max()
        mean_data = image[xmin:xmax, ymin].mean()

    return max_data, mean_data

def caculate_loss_PCA_RCA_loss(attn_maps_mid, attn_maps_up, attn_maps_down, bboxes, token_indice, t, res=16,
                                smooth_att=True, sigma=0.5, kernel_size=3):

    attn = get_all_attention(attn_maps_mid, attn_maps_up, attn_maps_down, res)

    attn_text = attn[:, :, 1:-1]
    attn_text *= 100
    attn_text = torch.nn.functional.softmax(attn_text, dim=-1)

    max_indices_list, mean_indices_list = [], []
    token_indice = [index - 1 for index in token_indice]
    for index,i in enumerate(token_indice):
        image = attn_text[:, :, i]
        if smooth_att:
            smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to("cuda")
            input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            image = smoothing(input).squeeze(0).squeeze(0)
        box = bboxes[index]
        if type(box[0]) is not list:
            max_data, mean_data = _get_data(box, image)
        else:
            max_data, mean_data = zip(*[_get_data(sub_box, image) for sub_box in box])
            max_data = min(max_data)
            mean_data = min(mean_data)
        max_indices_list.append(max_data)
        mean_indices_list.append(mean_data)

    all_indice_list = [max_indices_list, mean_indices_list]
    return all_indice_list


def get_indices(tokenizer, prompt: str) -> Dict[str, int]:
    """Utility function to list the indices of the tokens you wish to alter"""
    ids = tokenizer(prompt).input_ids
    indices = {
        i: tok
        for tok, i in zip(
            tokenizer.convert_ids_to_tokens(ids), range(len(ids))
        )
    }
    return indices

def get_attention_map_index_to_wordpiece(tokenizer, prompt):
    attn_map_idx_to_wp = {}

    wordpieces2indices = get_indices(tokenizer, prompt)

    # Ignore `start_token` and `end_token`
    for i in list(wordpieces2indices.keys())[1:-1]:
        wordpiece = wordpieces2indices[i]
        wordpiece = wordpiece.replace("</w>", "")
        attn_map_idx_to_wp[i] = wordpiece

    return attn_map_idx_to_wp


def extract_attribution_indices(doc):
    # doc = parser(prompt)
    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees

def _align_indices(prompt, spacy_pairs,syngen_indice):
    wordpieces2indices = syngen_indice
    paired_indices = []
    collected_spacy_indices = (
        set()
    )  # helps track recurring nouns across different relations (i.e., cases where there is more than one instance of the same word)

    for pair in spacy_pairs:
        curr_collected_wp_indices = (
            []
        )  # helps track which nouns and amods were added to the current pair (this is useful in sentences with repeating amod on the same relation (e.g., "a red red red bear"))
        for member in pair:
            for idx, wp in wordpieces2indices.items():
                if wp in [start_token, end_token]:
                    continue

                wp = wp.replace("</w>", "")
                if member.text.lower() == wp.lower():
                    if idx not in curr_collected_wp_indices and idx not in collected_spacy_indices:
                        curr_collected_wp_indices.append(idx)
                        break
                # take care of wordpieces that are split up
                elif member.text.lower().startswith(
                        wp.lower()) and wp.lower() != member.text.lower():  # can maybe be while loop
                    wp_indices = align_wordpieces_indices(
                        wordpieces2indices, idx, member.text
                    )
                    # check if all wp_indices are not already in collected_spacy_indices
                    if wp_indices and (wp_indices not in curr_collected_wp_indices) and all(
                            [wp_idx not in collected_spacy_indices for wp_idx in wp_indices]):
                        curr_collected_wp_indices.append(wp_indices)
                        break

        for collected_idx in curr_collected_wp_indices:
            if isinstance(collected_idx, list):
                for idx in collected_idx:
                    collected_spacy_indices.add(idx)
            else:
                collected_spacy_indices.add(collected_idx)

        if curr_collected_wp_indices:
            paired_indices.append(curr_collected_wp_indices)
        else:
            print(f"No wordpieces were aligned for {pair} in _align_indices")

    return paired_indices


def align_wordpieces_indices(
        wordpieces2indices, start_idx, target_word
):
    """
    Aligns a `target_word` that contains more than one wordpiece (the first wordpiece is `start_idx`)
    """

    wp_indices = [start_idx]
    wp = wordpieces2indices[start_idx].replace("</w>", "")

    # Run over the next wordpieces in the sequence (which is why we use +1)
    for wp_idx in range(start_idx + 1, len(wordpieces2indices)):
        if wp.lower() == target_word.lower():
            break

        wp2 = wordpieces2indices[wp_idx].replace("</w>", "")
        if target_word.lower().startswith(wp.lower() + wp2.lower()) and wp2.lower() != target_word.lower():
            wp += wordpieces2indices[wp_idx].replace("</w>", "")
            wp_indices.append(wp_idx)
        else:
            wp_indices = (
                []
            )  # if there's no match, you want to clear the list and finish
            break

    return wp_indices


def extract_attribution_indices_with_verb_root(doc):
    '''This function specifically addresses cases where a verb is between
       a noun and its modifier. For instance: "a dog that is red"
       here, the aux is between 'dog' and 'red'. '''

    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        subtree = []
        stack = []

        # if w is a verb/aux and has a noun child and a modifier child, add them to the stack
        if w.pos_ != 'AUX' or w.dep_ in modifiers:
            continue

        for child in w.children:
            if child.dep_ in modifiers or child.pos_ in ['NOUN', 'PROPN']:
                if child.pos_ not in ['AUX', 'VERB']:
                    subtree.append(child)
                stack.extend(child.children)
        # did not find a pair of noun and modifier
        if len(subtree) < 2:
            continue

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                # we don't want to add 'is' or other verbs to the loss, we want their children
                if node.pos_ not in ['AUX']:
                    subtree.append(node)
                stack.extend(node.children)

        if subtree:
            if w.pos_ not in ['AUX']:
                subtree.append(w)
            subtrees.append(subtree)
    return subtrees

def extract_attribution_indices_with_verbs(doc):
    '''This function specifically addresses cases where a verb is between
       a noun and its modifier. For instance: "a dog that is red"
       here, the aux is between 'dog' and 'red'. '''

    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp",
                 'relcl']
    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                if child.pos_ not in ['AUX', 'VERB']:
                    subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                # we don't want to add 'is' or other verbs to the loss, we want their children
                if node.pos_ not in ['AUX', 'VERB']:
                    subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees

def extract_entities_only(doc):
    entities = []
    for w in doc:
        if w.pos_ in ['NOUN', 'PROPN']:
            entities.append([w])
    return entities

def unify_lists(list_of_lists):
    def flatten(lst):
        for elem in lst:
            if isinstance(elem, list):
                yield from flatten(elem)
            else:
                yield elem

    def have_common_element(lst1, lst2):
        flat_list1 = set(flatten(lst1))
        flat_list2 = set(flatten(lst2))
        return not flat_list1.isdisjoint(flat_list2)

    lst = []
    for l in list_of_lists:
        lst += l
    changed = True
    while changed:
        changed = False
        merged_list = []
        while lst:
            first = lst.pop(0)
            was_merged = False
            for index, other in enumerate(lst):
                if have_common_element(first, other):
                    # If we merge, we should flatten the other list but not first
                    new_merged = first + [item for item in other if item not in first]
                    lst[index] = new_merged
                    changed = True
                    was_merged = True
                    break
            if not was_merged:
                merged_list.append(first)
        lst = merged_list

    return lst

def _extract_attribution_indices(prompt,syngen_indice,include_entities):
    modifier_indices = []
    doc = parser(prompt)
    # extract standard attribution indices
    modifier_sets_1 = extract_attribution_indices(doc)
    modifier_indices_1 = _align_indices(prompt, modifier_sets_1,syngen_indice)
    if modifier_indices_1:
        modifier_indices.append(modifier_indices_1)

    # extract attribution indices with verbs in between
    modifier_sets_2 = extract_attribution_indices_with_verb_root(doc)
    modifier_indices_2 = _align_indices(prompt, modifier_sets_2,syngen_indice)
    if modifier_indices_2:
        modifier_indices.append(modifier_indices_2)

    modifier_sets_3 = extract_attribution_indices_with_verbs(doc)
    modifier_indices_3 = _align_indices(prompt, modifier_sets_3,syngen_indice)
    if modifier_indices_3:
        modifier_indices.append(modifier_indices_3)

    # entities only
    if include_entities:
        modifier_sets_4 = extract_entities_only(doc)
        modifier_indices_4 = _align_indices(prompt, modifier_sets_4,syngen_indice)
        modifier_indices.append(modifier_indices_4)

    # make sure there are no duplicates
    modifier_indices = unify_lists(modifier_indices)
    # print(f"Final modifier indices collected:{modifier_indices}")

    return modifier_indices

def split_indices(related_indices: List[int]):
    noun = [related_indices[-1]]  # assumes noun is always last in the list
    modifier = related_indices[:-1]
    if isinstance(modifier, int):
        modifier = [modifier]
    return noun, modifier

def _flatten_indices(related_indices):
    flattened_related_indices = []
    for item in related_indices:
        if isinstance(item, list):
            flattened_related_indices.extend(item)
        else:
            flattened_related_indices.append(item)
    return flattened_related_indices

def _get_outside_indices(subtree_indices, attn_map_idx_to_wp):
    flattened_subtree_indices = _flatten_indices(subtree_indices)
    outside_indices = [
        map_idx
        for map_idx in attn_map_idx_to_wp.keys() if (map_idx not in flattened_subtree_indices)
    ]
    return outside_indices

def _symmetric_kl(attention_map1, attention_map2):
    # Convert map into a single distribution: 16x16 -> 256
    if len(attention_map1.shape) > 1:
        attention_map1 = attention_map1.reshape(-1)
    if len(attention_map2.shape) > 1:
        attention_map2 = attention_map2.reshape(-1)

    p = dist.Categorical(probs=attention_map1.float())
    q = dist.Categorical(probs=attention_map2.float())

    kl_divergence_pq = dist.kl_divergence(p, q)
    kl_divergence_qp = dist.kl_divergence(q, p)

    avg_kl_divergence = (kl_divergence_pq + kl_divergence_qp) / 2
    return avg_kl_divergence

def _calculate_outside_loss(attention_maps, src_indices, outside_loss):
    negative_loss = []
    computed_pairs = set()
    pair_counter = 0

    for outside_idx in outside_loss:
        if isinstance(src_indices, list):
            wp_neg_loss = []
            for t in src_indices:
                pair_key = (t, outside_idx)
                if pair_key not in computed_pairs:
                    wp_neg_loss.append(
                        _symmetric_kl(
                            attention_maps[:,:,t], attention_maps[:,:,outside_idx]
                        )
                    )
                    computed_pairs.add(pair_key)
            negative_loss.append(max(wp_neg_loss) if wp_neg_loss else 0)
            pair_counter += 1

        else:
            pair_key = (src_indices, outside_idx)
            if pair_key not in computed_pairs:
                negative_loss.append(
                    _symmetric_kl(
                        attention_maps[:,:,src_indices], attention_maps[:,:,outside_idx]
                    )
                )
                computed_pairs.add(pair_key)
                pair_counter += 1

    return negative_loss, pair_counter

def calculate_negative_loss(
        attention_maps, modifier, noun, subtree_indices, attn_map_idx_to_wp
):
    outside_indices = _get_outside_indices(subtree_indices, attn_map_idx_to_wp)

    negative_noun_loss, num_noun_pairs = _calculate_outside_loss(
        attention_maps, noun, outside_indices
    )
    if outside_indices:
      negative_noun_loss = -sum(negative_noun_loss) / len(outside_indices)
    else:
      negative_noun_loss = 0

    if modifier:
        negative_modifier_loss, num_modifier_pairs = _calculate_outside_loss(
            attention_maps, modifier, outside_indices
        )
        if outside_indices:
          negative_modifier_loss = -sum(negative_modifier_loss) / len(outside_indices)
        else:
          negative_modifier_loss = 0

        negative_loss = (negative_modifier_loss + negative_noun_loss) / 2
    else:
        negative_loss = negative_noun_loss

    return negative_loss

def calculate_positive_loss(attention_maps, modifier, noun):
    src_indices = modifier
    dest_indices = noun

    if isinstance(src_indices, list) and isinstance(dest_indices, list):
        wp_pos_loss = [
            _symmetric_kl(attention_maps[:,:,s], attention_maps[:,:,d])
            for (s, d) in itertools.product(src_indices, dest_indices)
        ]
        positive_loss = max(wp_pos_loss)
    elif isinstance(dest_indices, list):
        wp_pos_loss = [
            _symmetric_kl(attention_maps[:,:,src_indices], attention_maps[:,:,d])
            for d in dest_indices
        ]
        positive_loss = max(wp_pos_loss)
    elif isinstance(src_indices, list):
        wp_pos_loss = [
            _symmetric_kl(attention_maps[:,:,s], attention_maps[:,:,dest_indices])
            for s in src_indices
        ]
        positive_loss = max(wp_pos_loss)
    else:
        positive_loss = _symmetric_kl(
            attention_maps[:,:,src_indices], attention_maps[:,:,dest_indices]
        )

    return positive_loss

def _calculate_losses(
        attention_maps,
        all_subtree_pairs,
        subtree_indices,
        attn_map_idx_to_wp,
):
    positive_loss = []
    negative_loss = []
    for pair in all_subtree_pairs:
        noun, modifier = pair
        positive_loss.append(
            calculate_positive_loss(attention_maps, modifier, noun)
        )
        negative_loss.append(
            calculate_negative_loss(
                attention_maps, modifier, noun, subtree_indices, attn_map_idx_to_wp
            )
        )

    positive_loss = sum(positive_loss)
    negative_loss = sum(negative_loss)

    return positive_loss, negative_loss

def _attribution_loss(
        attention_maps: List[torch.Tensor],
        prompt: Union[str, List[str]],
        attn_map_idx_to_wp,
        syngen_indice,
        include_entities
) -> torch.Tensor:
    subtrees_indices = None
    if not subtrees_indices:
        subtrees_indices = _extract_attribution_indices(prompt,syngen_indice,include_entities)#根据不同的需求选择不同的解析方式，为每种解释方式的token编码索引
    subtrees_indices = subtrees_indices

    loss = 0

    for subtree_indices in subtrees_indices:
        noun, modifier = split_indices(subtree_indices)
        all_subtree_pairs = list(itertools.product(noun, modifier))
        if noun and not modifier:
            if isinstance(noun, list) and len(noun) == 1:
                processed_noun = noun[0]
            else:
                processed_noun = noun
            loss += calculate_negative_loss(
                attention_maps, modifier, processed_noun, subtree_indices, attn_map_idx_to_wp
            )
        else:
            positive_loss, negative_loss = _calculate_losses(
                attention_maps,
                all_subtree_pairs,
                subtree_indices,
                attn_map_idx_to_wp,
            )

            loss += positive_loss
            loss += negative_loss

    return loss

def caculate_loss_syngen_loss(attn_maps_mid, attn_maps_up, attn_maps_down,t,prompt,include_entities,attn_map_idx_to_wp,syngen_indice,res=16):
    attention_maps = get_all_attention(attn_maps_mid, attn_maps_up, attn_maps_down, res)
    loss = _attribution_loss(attention_maps, prompt, attn_map_idx_to_wp,syngen_indice,include_entities)

    return loss

def caculate_loss_att_fixed_cnt(attn_maps_mid, attn_maps_up, attn_maps_down, bboxes, object_positions, t, res=16,
                                smooth_att=True, sigma=0.5, kernel_size=3):
    attn = get_all_attention(attn_maps_mid, attn_maps_up, attn_maps_down, res)

    obj_number = len(bboxes)
    total_loss = 0

    attn_text = attn[:, :, 1:-1]
    attn_text *= 100
    attn_text = torch.nn.functional.softmax(attn_text, dim=-1)
    current_res = attn.shape[0]
    H = W = current_res

    min_all_inside = 1000
    max_outside = 0

    for obj_idx in range(obj_number):
        num_boxes = 0

        for obj_position in object_positions[obj_idx]:
            true_obj_position = obj_position - 1
            att_map_obj = attn_text[:, :, true_obj_position]
            if smooth_att:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(att_map_obj.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                att_map_obj = smoothing(input).squeeze(0).squeeze(0)
            other_att_map_obj = att_map_obj.clone()
            att_copy = att_map_obj.clone()

            for obj_box in bboxes[obj_idx]:
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                                             int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)

                if att_map_obj[y_min: y_max, x_min: x_max].numel() == 0:
                    max_inside = 1.

                else:
                    max_inside = att_map_obj[y_min: y_max, x_min: x_max].max()
                if max_inside < 0.1:
                    total_loss += 6 * (1. - max_inside)
                elif max_inside < 0.2:
                    total_loss += 1. - max_inside
                elif t < 15:
                    total_loss += 1. - max_inside
                if max_inside < min_all_inside:
                    min_all_inside = max_inside

                # find max outside the box, find in the other boxes

                att_copy[y_min: y_max, x_min: x_max] = 0.
                other_att_map_obj[y_min: y_max, x_min: x_max] = 0.

            for obj_outside in range(obj_number):
                if obj_outside != obj_idx:
                    for obj_out_box in bboxes[obj_outside]:
                        x_min_out, y_min_out, x_max_out, y_max_out = int(obj_out_box[0] * W), \
                                                                     int(obj_out_box[1] * H), int(
                            obj_out_box[2] * W), int(obj_out_box[3] * H)

                        if other_att_map_obj[y_min_out: y_max_out, x_min_out: x_max_out].numel() == 0:
                            max_outside_one = 0
                        else:
                            max_outside_one = other_att_map_obj[y_min_out: y_max_out, x_min_out: x_max_out].max()

                        att_copy[y_min_out: y_max_out, x_min_out: x_max_out] = 0.

                        if max_outside_one > 0.15:
                            total_loss += 4 * max_outside_one
                        elif max_outside_one > 0.1:
                            total_loss += max_outside_one
                        elif t < 15:
                            total_loss += max_outside_one
                        if max_outside_one > max_outside:
                            max_outside = max_outside_one

            max_background = att_copy.max()
            total_loss += len(bboxes[obj_idx]) * max_background / 2.

    return total_loss / obj_number, min_all_inside, max_outside


def caculate_ground(ground_first, ground_second, ground_third, bboxes,
                    object_positions, t):
    attn_ground = get_all_self_att(ground_first, ground_second, ground_third)
    attn_maps = attn_ground[286]
    loss = 0
    loss_self = 0
    object_number = len(bboxes)

    # import pdb; pdb.set_trace()
    attn_map = torch.mean(torch.stack(attn_maps), dim=0)
    # for attn_map in attn_maps:
    if t < 15:
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        for obj_idx in range(object_number):

            for obj_box in bboxes[obj_idx]:
                mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                                             int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1.
                mask_out = 1. - mask
                index = (mask == 1.).nonzero(as_tuple=False)

                index_in_key = index[:, 0] * H + index[:, 1]
                att_box = torch.zeros_like(attn_map)
                att_box[:, index_in_key, :] = attn_map[:, index_in_key, :]
                box_ids = np.arange(0, object_number, 1)

                att_box = att_box.sum(axis=1) / index_in_key.shape[0]
                att_box_square = att_box[:, :256].reshape(-1, H, H)

                activation_value = (att_box_square * mask_out).reshape(b, -1).sum(
                    dim=-1)  # / att_box.reshape(b, -1).sum(dim=-1)
                cp_att_box = torch.zeros(size=(att_box.shape[0], 30))
                cp_att_box[:, :] = att_box[:, 256:]
                cp_att_box[:, obj_idx] = 0
                loss_self += torch.mean(activation_value)
                loss += cp_att_box.amax(dim=1).mean()

    return loss / object_number, loss_self / object_number

def recitify_attention_map_right(attn, box, token_indice, use_type="recitify"):
    #
    return attn


def recitify_attention_map(input,save_attn,use_type='check',check_threshold=0.3):
    attention_map = save_attn
    bbox = input["bbox"]
    token_indice = input["token_indice"]

    del attention_map["mid_cross"]
    #首先将attention_map全部resize到最大维度
    all_attention_map = []
    for key,value in attention_map.items():
        for e_value in value:
            if len(e_value) > 0:
                e_value = e_value[0][0]
                res = int(math.sqrt(e_value.shape[1]))
                e_value = e_value.reshape(1, -1, res, res, e_value.shape[-1])[0]
                all_attention_map.append(e_value)

    tensor_dict = defaultdict(list)
    for tensor in all_attention_map:
        tensor_dict[tensor.shape].append(tensor)

    all_attention_map = []
    max_res = 0
    for key, value in tensor_dict.items():
        value = torch.cat(value, dim=0)
        all_attention_map.append(value.sum(0) / value.shape[0])
        max_res = key[2] if key[2] > max_res else max_res
    for idx,item in enumerate(all_attention_map):
        all_attention_map[idx] = torch.nn.functional.interpolate(item.permute(2, 0, 1).unsqueeze(0), size=(max_res, max_res), mode='bilinear',
                                                        align_corners=False)
    stacked_tensor = torch.stack(all_attention_map)
    mean_attention_map = (stacked_tensor.sum(0) / stacked_tensor.shape[0])[0]
    if use_type == "check":
        for idx, p_index in enumerate(token_indice):
            box = bbox[idx]
            is_right = check_right(mean_attention_map[p_index,:,:],box,check_threshold)
            if not is_right:
                return False
        return True

    token_max_attn_box = []
    for idx, p_index in enumerate(token_indice):
        box = bbox[idx]
        token_max_attn_box.append(find_max_attn_box(box , mean_attention_map[p_index,:,:]))

    if use_type == "locate":
        return token_max_attn_box

    #根据最大找到的最大的attn_sum box替换到原始的box位置
    for key, value in attention_map.items():
        if len(value) > 1:
            for v_idx,e_value in enumerate(value):
                if len(e_value) > 0:
                    for idx, p_index in enumerate(token_indice):
                        attn_max_sum_box = token_max_attn_box[idx]
                        ori_box = bbox[idx]
                        attention_map[key][v_idx][0][0][:,:,p_index] = recitify_token_attention_map(e_value[0][0][:,:, p_index],ori_box,attn_max_sum_box)

    return attention_map, token_max_attn_box

def check_right(attention_map,box,check_threshold):
    attention_map_size = attention_map.size()[0]
    if not isinstance(box[0], List):
        x1, y1, x2, y2 = [int(x * attention_map_size) for x in box]
        mean_value = torch.stack(attention_map[x1:x2,y1:y2]).mean(dim=0)
        if mean_value <= check_threshold:
            return False
    else:
        for eve_box in box:
            x1, y1, x2, y2 = [int(x * attention_map_size) for x in eve_box]
            mean_value = torch.stack(attention_map[x1:x2, y1:y2]).mean(dim=0)
            if mean_value <= check_threshold:
                return False
    return True



def find_max_attn_box(box,attention_map):
    attention_map_size = attention_map.size()[0]
    max_attn_box = []
    if not isinstance(box[0],List):
        max_attn_box.append(iteration_detection(box,attention_map,attention_map_size))
    else:
        for eve_box in box:
            max_attn_box.append(iteration_detection(eve_box,attention_map,attention_map_size))

    return max_attn_box

def iteration_detection(box,attention_map,attention_map_size):
    x1, y1, x2, y2 = [int(x * attention_map_size) for x in box]
    w, h = x2 - x1, y2 - y1
    max_value = -1
    max_position = (0, 0)
    if (attention_map_size - w) != 0 and (attention_map_size - h) != 0:
        for i in range(attention_map_size - w):
            for j in range(attention_map_size - h):
                window = attention_map[i:i + w, j:j + h]
                window_sum = window.sum()
                if window_sum > max_value:
                    max_value = window_sum
                    max_position = (i,j)
    elif (attention_map_size - w) != 0:
        for i in range(attention_map_size - w):
            window = attention_map[i:i + w, :]
            window_sum = window.sum()
            if window_sum > max_value:
                max_value = window_sum
                max_position = (i,0)
    elif (attention_map_size - h) != 0:
        for j in range(attention_map_size - h):
            window = attention_map[:, j:j + h]
            window_sum = window.sum()
            if window_sum > max_value:
                max_value = window_sum
                max_position = (0, j)
    else:
        max_position = (0, 0)

    max_attn_box = [max_position[0]/attention_map_size, max_position[1]/attention_map_size,
                    (max_position[0]+w)/attention_map_size, (max_position[1]+h)/attention_map_size]
    return max_attn_box

def recitify_token_attention_map(attention_map, ori_box, attn_max_box, alpha = 10):
    attention_map_size = int(math.sqrt(attention_map.shape[-1]))
    attention_map = attention_map.reshape(attention_map.size()[0], attention_map_size, attention_map_size)
    min_value = attention_map.min()
    if not isinstance(ori_box[0],List):
        ori_box = [ori_box]
    for idx,e_attn_max_box in enumerate(attn_max_box):
        o_x1, o_y1, o_x2, o_y2 = [int(x * attention_map_size) for x in ori_box[idx]]
        t_x1, t_y1, t_x2, t_y2 = [int(x*attention_map_size) for x in e_attn_max_box]
        if o_x2 - o_x1 != t_x2 - t_x1:
            o_x2 = o_x1 + (t_x2 - t_x1)
        if o_y2 - o_y1 != t_y2 - t_y1:
            o_y2 = o_y1 + (t_y2 - t_y1)

        c_tensor = attention_map[:,t_x1:t_x2,t_y1:t_y2] * alpha
        attention_map[:, t_x1:t_x2, t_y1:t_y2] = min_value
        attention_map[:, o_x1:o_x2, o_y1:o_y2] = c_tensor


    attention_map = attention_map.reshape(attention_map.size()[0], attention_map_size**2)

    return attention_map