from typing import List

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import math
from ldm.models.diffusion.loss import caculate_loss_att_fixed_cnt, caculate_loss_self_att, caculate_ground, caculate_loss_PCA_RCA_loss, caculate_loss_syngen_loss,recitify_attention_map
from attention_map_vis import attention_map_vis_show

class PLMSSampler(object):
    def __init__(self, diffusion, model, schedule="linear", alpha_generator_func=None, set_alpha_scale=None):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.device = diffusion.betas.device
        self.ddpm_num_timesteps = diffusion.num_timesteps
        self.schedule = schedule
        self.alpha_generator_func = alpha_generator_func
        self.set_alpha_scale = set_alpha_scale

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=False):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.diffusion.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.diffusion.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.diffusion.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # @torch.no_grad()
    def sample(self, S, shape, input, uc=None, guidance_scale=1, mask=None, x0=None, loss_type=None):
        self.make_schedule(ddim_num_steps=S)
        # import pdb; pdb.set_trace()
        return self.plms_sampling(shape, input, uc, guidance_scale, mask=mask, x0=x0, loss_type=loss_type)

    # @torch.no_grad()
    def plms_sampling(self, shape, input, uc=None, guidance_scale=1, mask=None, x0=None, loss_type=None):

        b = shape[0]
        img = input["x"]

    ###TODO
        stage = input["stage"]
        if stage == 'fg':
            img = input["x"]
        elif loss_type == 'SAR_CAR':
            img = input["x"]
        else:
            img = input["x"][0]
            noise_series = input["x"]
            input["x"] = img
            #set froze area mask
            froze_mask = input["froze_area"]
    ###

        if img == None:
            img = torch.randn(shape, device=self.device)
            input["x"] = img

        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]

        old_eps = []

        if self.alpha_generator_func != None:
            alphas = self.alpha_generator_func(len(time_range))

        #scale_range:
        scale_range = (1., 0.5)
        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.ddim_timesteps))
        save_attn = {'up_cross': [], 'mid_cross': [], 'down_cross': []}
        is_right = True

        noise_list = []

        for i, step in enumerate(time_range):
            print('step ', i)

            # set alpha and restore first conv layer
            if self.alpha_generator_func != None:
                self.set_alpha_scale(self.model, alphas[i])
                if alphas[i] == 0:
                    self.model.restore_first_conv_from_SD()

            # run
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=self.device,
                                 dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.diffusion.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img
                input["x"] = img

            if loss_type != None and loss_type != 'standard':
                if input['object_position'] != [] and i <= 30:
                    if loss_type == 'SAR_CAR':
                        x = self.update_loss_self_cross(input, i, index, ts)
                    elif loss_type == 'SAR':
                        x = self.update_only_self(input, i, index, ts)
                    elif loss_type == 'CAR':
                        x = self.update_loss_only_cross(input, i, index, ts)
                    elif loss_type == "Syngen_PCA_RCA":
                        x = self.update_loss_Syngen_RCA_PCA(input, i, index, ts, scale_range)
                    elif loss_type == "PCA_RCA_SAR":
                        x = self.update_loss_PCA_RCA_SAR(input, i, index, ts, scale_range)
                    elif loss_type == "Syngen_PCA_SAR":
                        x = self.update_loss_Syngen_PCA_SAR(input, i, index, ts, scale_range)
                    elif loss_type == "Syngen_RCA_SAR":
                        x = self.update_loss_Syngen_RCA_SAR(input, i, index, ts, scale_range)
                    elif loss_type == "Syngen_PCA_RCA_SAR":
                        x = self.update_loss_Syngen_PCA_RCA_SAR(input, i, index, ts, scale_range)
                    torch.cuda.empty_cache()
                    input["x"] = x

            attn_save_timestep = [0, 5, 10, 15, 20, 25, 30, 40, 49]
            if i in attn_save_timestep and input['save_attn_vis'] and not input['is_edited']:
                e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
                    input)
                attention_map_vis_show(x, input, att_first, att_second, att_third, self_first, self_second, self_third,
                                       i)
                del e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3
                torch.cuda.empty_cache()

            img, pred_x0, e_t = self.p_sample_plms(input, ts, index=index, uc=uc, guidance_scale=guidance_scale,
                                                  old_eps=old_eps, t_next=ts_next)

            ###TODO
            if stage == 'fg':
                noise_list.append(img)
            elif loss_type == 'SAR_CAR':
                input["x"] = img
            elif step > 150:
                img = img * (1. - froze_mask) + noise_series[i] * froze_mask

            #END

            input["x"] = img
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)

        if stage == 'fg':
            return img, noise_list

        return img,None

    def _compute_loss(self, max_attention_per_index: List[torch.Tensor], part_of_speech: List[List[int]],
                      return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses = [max(0, 1. - curr_max) for sublist in max_attention_per_index for curr_max in sublist]
        sliced_losses = [losses[i:i + len(max_attention_per_index[0])] for i in
                         range(0, len(losses), len(max_attention_per_index[0]))]

        adj_loss = [sum(value for value, pos in zip(curr_max, part_of_speech) if pos == 0) / part_of_speech.count(0) for
                    curr_max in
                    sliced_losses] if part_of_speech.count(0) != 0 else [0, 0]
        noun_loss = [sum(value for value, pos in zip(curr_max, part_of_speech) if pos == 1) / part_of_speech.count(1)
                     for curr_max in
                     sliced_losses] if part_of_speech.count(1) != 0 else [0, 0]

        #pca_n_loss  rca_n_loss pca_m_loss rca_m_loss
        loss = [noun_loss[0], adj_loss[0],noun_loss[1],  adj_loss[1]]

        if return_losses:
            return loss, sliced_losses
        else:
            return loss


    def update_loss_Syngen_RCA_PCA(self,input, index1, index, ts, scale_range, type_loss='Syngen_PCA_RCA',scale_factor =20 ):
        interation_timestep = [ 5, 10, 15, 20, 30]###TODO:要记得加0回去
        thresholds = {0: [[0.15], [0.1]], 5: [[0.15], [0.4]], 10: [[0.6], [0.6]], 20: [[0.9], [0.7]],
                      30: [[0.95], [0.8]]}
        step_size = scale_factor * np.sqrt(scale_range[index1])

        loss = torch.tensor(10000)
        loss_list = torch.stack([loss] * 4)

        x = deepcopy(input["x"])
        input["timesteps"] = ts

        token_indice = input['token_indice']
        token_box = input['bbox']
        prompt = input['prompt']
        include_entities = input['include_entities']
        attn_map_idx_to_wp = input['attn_map_idx_to_wp']
        syngen_indice = input['syngen_indice']
        part_of_speech = input['part_of_speech']


        x = x.requires_grad_(True)
        input['x'] = x
        e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
            input)
        losses = caculate_loss_PCA_RCA_loss(att_second, att_first, att_third,
                                            bboxes=token_box,
                                            token_indice=token_indice,
                                            t=index1)
        loss_list = self._compute_loss(max_attention_per_index=losses, part_of_speech=part_of_speech)
        syngen_loss = caculate_loss_syngen_loss(att_second, att_first, att_third,
                                                t=index1, prompt=prompt,
                                                include_entities=include_entities,
                                                attn_map_idx_to_wp=attn_map_idx_to_wp,
                                                syngen_indice=syngen_indice)
        if index1 not in thresholds.keys():
            loss = syngen_loss + loss_list[0] + loss_list[2]
            print(
                f"PCA loss: {loss_list[0]:0.4f} | RCA loss: {loss_list[2]:0.4f} | Syngen loss: {syngen_loss:0.4f}")
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]
            x = x - grad_cond
            x = x.detach()
            del loss_list, syngen_loss, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3

        else:
            iteration = 0
            max_iter = 25

            threshold = thresholds[index1]
            if isinstance(threshold[0], list):
                compare_threshold = [threshold[0][0], threshold[0][0] * 0.5, threshold[1][0], threshold[1][0] * 0.5]
            else:
                compare_threshold = [threshold[0], threshold[0] * 0.5, threshold[1], threshold[1] * 0.5]


            target_loss = [max(0, 1 - t) for t in compare_threshold]
            while (any([x > y for x, y in zip(loss_list, target_loss)]) and (
                    index1 == 0 or index1 == 5 or (index1 == 10 and iteration <= 10))) or (
                    any([x > y for x, y in zip(loss_list[0:2], target_loss[0:2])]) and (
                    index1 != 5 and (index1 == 10 and iteration > 10))):
                iteration += 1

                if iteration > max_iter:
                    break

                x = x.requires_grad_(True)
                input['x'] = x

                e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
                    input)
                losses = caculate_loss_PCA_RCA_loss(att_second, att_first, att_third,
                                                    bboxes=token_box,
                                                    token_indice=token_indice,
                                                    t=index1)
                syngen_loss = caculate_loss_syngen_loss(att_second, att_first, att_third,
                                                        t=index1, prompt=prompt,
                                                        include_entities=include_entities,
                                                        attn_map_idx_to_wp=attn_map_idx_to_wp,
                                                        syngen_indice=syngen_indice)
                loss_list = self._compute_loss(max_attention_per_index=losses, part_of_speech=part_of_speech)


                if all(l != 0 for l in loss_list):  # loss:n.max adj.max n.mean adj.mean
                    # 修改成只是用non的损失
                    if loss_list[2] > target_loss[2] or loss_list[0] > target_loss[0]:
                        loss = loss_list[0] + loss_list[2] + syngen_loss
                        print(
                            f"PCA loss: {loss_list[0]:0.4f} | RCA loss: {loss_list[2]:0.4f} | Syngen loss: {syngen_loss:0.4f}")
                    elif loss_list[1] > target_loss[1] or loss_list[3] > target_loss[3]:
                        loss = loss_list[1] + loss_list[3] + syngen_loss
                        print(
                            f"PCA loss: {loss_list[1]:0.4f} | RCA loss: {loss_list[3]:0.4f} | Syngen loss: {syngen_loss:0.4f}")
                    else:
                        loss = syngen_loss
                        print(f"Syngen loss: {syngen_loss:0.4f}")

                    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]
                    x = x - grad_cond * step_size
                    x = x.detach()
            del losses, syngen_loss, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3
        torch.cuda.empty_cache()
        return x

    def update_loss_Syngen_PCA_RCA_SAR(self,input, index1, index, ts, scale_range, type_loss='Syngen_PCA_RCA_SAR',scale_factor =20 ):
        interation_timestep = [0, 5, 10, 15, 20, 30]###TODO:要记得加0回去
        # thresholds = {0: [[0.15], [0.1]], 5: [[0.15], [0.4]], 10: [[0.6], [0.6]], 20: [[0.9], [0.7]],
        #               30: [[0.95], [0.8]]}

        thresholds = {0: [[0.5], [0.4]], 5: [[0.55], [0.5]], 10: [[0.55], [0.6]], 20: [[0.6], [0.7]],
                      30: [[0.6], [0.8]]}
        step_size = scale_factor * np.sqrt(scale_range[index1])
        if index1 < 2:
            loss_scale = 1
        elif index1 < 5:
            loss_scale = 1
        elif index1 < 10:
            loss_scale = 0.5
        elif index1 < 20:
            loss_scale = 0.5
        else:
            loss_scale = 0.1


        loss = torch.tensor(10000)
        loss_list = torch.stack([loss] * 4)

        x = deepcopy(input["x"])
        input["timesteps"] = ts

        token_indice = input['token_indice']
        token_box = input['bbox']
        prompt = input['prompt']
        include_entities = input['include_entities']
        attn_map_idx_to_wp = input['attn_map_idx_to_wp']
        syngen_indice = input['syngen_indice']
        part_of_speech = input['part_of_speech']
        bboxes = input['boxes']
        object_positions = input['object_position']


        x = x.requires_grad_(True)
        input['x'] = x
        e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
            input)
        losses = caculate_loss_PCA_RCA_loss(att_second, att_first, att_third,
                                            bboxes=token_box,
                                            token_indice=token_indice,
                                            t=index1)
        loss_list = self._compute_loss(max_attention_per_index=losses, part_of_speech=part_of_speech)
        syngen_loss = caculate_loss_syngen_loss(att_second, att_first, att_third,
                                                t=index1, prompt=prompt,
                                                include_entities=include_entities,
                                                attn_map_idx_to_wp=attn_map_idx_to_wp,
                                                syngen_indice=syngen_indice)
        self_attn_loss = caculate_loss_self_att(self_first, self_second, self_third, bboxes=bboxes,
                                       object_positions=object_positions, t=index1) * loss_scale
        if index1 not in thresholds.keys():
            loss = syngen_loss + loss_list[0] + loss_list[2] + self_attn_loss
            print(
                f"PCA loss: {loss_list[0]:0.4f} | RCA loss: {loss_list[2]:0.4f} | Syngen loss: {syngen_loss:0.4f} | SAR loss:{self_attn_loss:0.4f}")
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]
            x = x - grad_cond
            x = x.detach()
        else:
            iteration = 0
            iter = {0: 20, 5: 10, 10: 5, 20: 3, 30: 2}
            max_iter = iter[index1]

            threshold = thresholds[index1]
            if isinstance(threshold[0], list):
                compare_threshold = [threshold[0][0], threshold[0][0] * 0.75, threshold[1][0], threshold[1][0] * 0.75]
            else:
                compare_threshold = [threshold[0], threshold[0] * 0.5, threshold[1], threshold[1] * 0.5]


            target_loss = [max(0, 1 - t) for t in compare_threshold]
            while (any([x > y for x, y in zip(loss_list, target_loss)]) and (
                    index1 == 0 or index1 == 5 )) or (
                    any([x > y for x, y in zip([loss_list[1], loss_list[3]], [target_loss[1], target_loss[3]])]) and (
                    index1 > 5 )):
                iteration += 1

                if iteration > max_iter:
                    break

                x = x.requires_grad_(True)
                input['x'] = x

                e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
                    input)
                losses = caculate_loss_PCA_RCA_loss(att_second, att_first, att_third,
                                                    bboxes=token_box,
                                                    token_indice=token_indice,
                                                    t=index1)
                syngen_loss = caculate_loss_syngen_loss(att_second, att_first, att_third,
                                                        t=index1, prompt=prompt,
                                                        include_entities=include_entities,
                                                        attn_map_idx_to_wp=attn_map_idx_to_wp,
                                                        syngen_indice=syngen_indice)
                loss_list = self._compute_loss(max_attention_per_index=losses, part_of_speech=part_of_speech)
                self_attn_loss = caculate_loss_self_att(self_first, self_second, self_third, bboxes=bboxes,
                                                        object_positions=object_positions, t=index1) * loss_scale

                if all(l != 0 for l in loss_list):  # loss:n.max adj.max n.mean adj.mean
                    # 修改成只是用non的损失
                    if loss_list[2] > target_loss[2] or loss_list[0] > target_loss[0]:
                        loss = loss_list[0] + loss_list[2] + syngen_loss + self_attn_loss
                        print(
                            f"PCA loss: {loss_list[0]:0.4f} | RCA loss: {loss_list[2]:0.4f} | Syngen loss: {syngen_loss:0.4f} | SAR loss : {self_attn_loss:0.4f}")
                    elif loss_list[1] > target_loss[1] or loss_list[3] > target_loss[3]:
                        loss = loss_list[1] + loss_list[3] + syngen_loss + self_attn_loss
                        print(
                            f"PCA loss: {loss_list[1]:0.4f} | RCA loss: {loss_list[3]:0.4f} | Syngen loss: {syngen_loss:0.4f} | SAR loss : {self_attn_loss:0.4f}")
                    else:
                        loss = syngen_loss + self_attn_loss
                        print(f"Syngen loss: {syngen_loss:0.4f} | SAR loss : {self_attn_loss:0.4f}")

                    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]
                    x = x - grad_cond * step_size
                    x = x.detach()
        del loss_list,losses, syngen_loss,self_attn_loss ,att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3
        torch.cuda.empty_cache()
        return x

    def update_loss_PCA_RCA_SAR(self,input, index1, index, ts, scale_range, type_loss='PCA_RCA_SAR',scale_factor =20 ):
        interation_timestep = [0, 5, 10, 15, 20, 30]###TODO:要记得加0回去
        thresholds = {0: [[0.15], [0.1]], 5: [[0.15], [0.4]], 10: [[0.6], [0.6]], 20: [[0.9], [0.7]],
                      30: [[0.95], [0.8]]}
        step_size = scale_factor * np.sqrt(scale_range[index1])
        if index1 < 2:
            loss_scale = 4
        elif index1 < 5:
            loss_scale = 4
        elif index1 < 10:
            loss_scale = 3
        elif index1 < 20:
            loss_scale = 3
        else:
            loss_scale = 1

        x = deepcopy(input["x"])
        input["timesteps"] = ts

        token_indice = input['token_indice']
        token_box = input['bbox']
        prompt = input['prompt']
        include_entities = input['include_entities']
        attn_map_idx_to_wp = input['attn_map_idx_to_wp']
        syngen_indice = input['syngen_indice']
        part_of_speech = input['part_of_speech']
        bboxes = input['boxes']
        object_positions = input['object_position']


        x = x.requires_grad_(True)
        input['x'] = x
        e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
            input)
        losses = caculate_loss_PCA_RCA_loss(att_second, att_first, att_third,
                                            bboxes=token_box,
                                            token_indice=token_indice,
                                            t=index1)
        loss_list = self._compute_loss(max_attention_per_index=losses, part_of_speech=part_of_speech)
        self_attn_loss = caculate_loss_self_att(self_first, self_second, self_third, bboxes=bboxes,
                                       object_positions=object_positions, t=index1) * loss_scale

        if index1 not in thresholds.keys():
            loss =  loss_list[0] + loss_list[2] + self_attn_loss
            print(
                f"PCA loss: {loss_list[0]:0.4f} | RCA loss: {loss_list[2]:0.4f} | SAR loss:{self_attn_loss:0.4f}")
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]
            x = x - grad_cond
            x = x.detach()
            del loss_list,losses, self_attn_loss, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3

        else:
            iteration = 0
            max_iter = 25

            threshold = thresholds[index1]
            if isinstance(threshold[0], list):
                compare_threshold = [threshold[0][0], threshold[0][0] * 0.5, threshold[1][0], threshold[1][0] * 0.5]
            else:
                compare_threshold = [threshold[0], threshold[0] * 0.5, threshold[1], threshold[1] * 0.5]


            target_loss = [max(0, 1 - t) for t in compare_threshold]
            while (any([x > y for x, y in zip(loss_list, target_loss)]) and (
                    index1 == 0 or index1 == 5 or (index1 == 10 and iteration <= 10))) or (
                    any([x > y for x, y in zip(loss_list[0:2], target_loss[0:2])]) and (
                    index1 != 5 and (index1 == 10 and iteration > 10))):
                iteration += 1

                if iteration > max_iter:
                    break

                x = x.requires_grad_(True)
                input['x'] = x

                e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
                    input)
                losses = caculate_loss_PCA_RCA_loss(att_second, att_first, att_third,
                                                    bboxes=token_box,
                                                    token_indice=token_indice,
                                                    t=index1)
                loss_list = self._compute_loss(max_attention_per_index=losses, part_of_speech=part_of_speech)
                self_attn_loss = caculate_loss_self_att(self_first, self_second, self_third, bboxes=bboxes,
                                                        object_positions=object_positions, t=index1) * loss_scale

                if all(l != 0 for l in loss_list):  # loss:n.max adj.max n.mean adj.mean
                    # 修改成只是用non的损失
                    if loss_list[2] > target_loss[2] or loss_list[0] > target_loss[0]:
                        loss = loss_list[0] + loss_list[2] + self_attn_loss
                        print(
                            f"PCA loss: {loss_list[0]:0.4f} | RCA loss: {loss_list[2]:0.4f}  | SAR loss : {self_attn_loss:0.4f}")
                    elif loss_list[1] > target_loss[1] or loss_list[3] > target_loss[3]:
                        loss = loss_list[1] + loss_list[3] + self_attn_loss
                        print(
                            f"PCA loss: {loss_list[1]:0.4f} | RCA loss: {loss_list[3]:0.4f} | SAR loss : {self_attn_loss:0.4f}")
                    else:
                        loss = self_attn_loss
                        print(f" SAR loss : {self_attn_loss:0.4f}")

                    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]
                    x = x - grad_cond * step_size
                    x = x.detach()
            del loss_list, losses, self_attn_loss, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3
        torch.cuda.empty_cache()
        return x

    def update_loss_Syngen_PCA_SAR(self, input, index1, index, ts, scale_range, type_loss='Syngen_PCA_SAR', scale_factor=20):
        interation_timestep = [0, 5, 10, 15, 20, 30]  ###TODO:要记得加0回去
        thresholds = {0: [[0.15], [0.1]], 5: [[0.15], [0.4]], 10: [[0.6], [0.6]], 20: [[0.9], [0.7]],
                      30: [[0.95], [0.8]]}
        step_size = scale_factor * np.sqrt(scale_range[index1])
        if index1 < 2:
            loss_scale = 4
        elif index1 < 5:
            loss_scale = 4
        elif index1 < 10:
            loss_scale = 3
        elif index1 < 20:
            loss_scale = 3
        else:
            loss_scale = 1

        loss = torch.tensor(10000)
        loss_list = torch.stack([loss] * 4)

        x = deepcopy(input["x"])
        input["timesteps"] = ts

        token_indice = input['token_indice']
        token_box = input['bbox']
        prompt = input['prompt']
        include_entities = input['include_entities']
        attn_map_idx_to_wp = input['attn_map_idx_to_wp']
        syngen_indice = input['syngen_indice']
        part_of_speech = input['part_of_speech']
        bboxes = input['boxes']
        object_positions = input['object_position']


        x = x.requires_grad_(True)
        input['x'] = x
        e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
            input)
        losses = caculate_loss_PCA_RCA_loss(att_second, att_first, att_third,
                                            bboxes=token_box,
                                            token_indice=token_indice,
                                            t=index1)
        loss_list = self._compute_loss(max_attention_per_index=losses, part_of_speech=part_of_speech)
        syngen_loss = caculate_loss_syngen_loss(att_second, att_first, att_third,
                                                t=index1, prompt=prompt,
                                                include_entities=include_entities,
                                                attn_map_idx_to_wp=attn_map_idx_to_wp,
                                                syngen_indice=syngen_indice)
        self_attn_loss = caculate_loss_self_att(self_first, self_second, self_third, bboxes=bboxes,
                                                object_positions=object_positions, t=index1) * loss_scale
        if index1 not in thresholds.keys():
            loss = syngen_loss + loss_list[0] + self_attn_loss
            print(
                f"PCA loss: {loss_list[0]:0.4f} | Syngen loss: {syngen_loss:0.4f} | SAR loss:{self_attn_loss:0.4f}")
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]
            x = x - grad_cond
            x = x.detach()
            del loss_list, loss, syngen_loss, self_attn_loss, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3

        else:
            iteration = 0
            max_iter = 25

            threshold = thresholds[index1]
            if isinstance(threshold[0], list):
                compare_threshold = [threshold[0][0], threshold[0][0] * 0.5, threshold[1][0], threshold[1][0] * 0.5]
            else:
                compare_threshold = [threshold[0], threshold[0] * 0.5, threshold[1], threshold[1] * 0.5]

            target_loss = [max(0, 1 - t) for t in compare_threshold]
            while (any([x > y for x, y in zip(loss_list, target_loss)]) and (
                    index1 == 0 or index1 == 5 or (index1 == 10 and iteration <= 10))) or (
                    any([x > y for x, y in zip(loss_list[0:2], target_loss[0:2])]) and (
                    index1 != 5 and (index1 == 10 and iteration > 10))):
                iteration += 1

                if iteration > max_iter:
                    break

                x = x.requires_grad_(True)
                input['x'] = x

                e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
                    input)
                losses = caculate_loss_PCA_RCA_loss(att_second, att_first, att_third,
                                                    bboxes=token_box,
                                                    token_indice=token_indice,
                                                    t=index1)
                syngen_loss = caculate_loss_syngen_loss(att_second, att_first, att_third,
                                                        t=index1, prompt=prompt,
                                                        include_entities=include_entities,
                                                        attn_map_idx_to_wp=attn_map_idx_to_wp,
                                                        syngen_indice=syngen_indice)
                loss_list = self._compute_loss(max_attention_per_index=losses, part_of_speech=part_of_speech)
                self_attn_loss = caculate_loss_self_att(self_first, self_second, self_third, bboxes=bboxes,
                                                        object_positions=object_positions, t=index1) * loss_scale

                if all(l != 0 for l in [loss_list[0],loss_list[1]]):  # loss:n.max adj.max n.mean adj.mean
                    # 修改成只是用non的损失
                    if loss_list[0] > target_loss[0]:
                        loss = loss_list[0] + syngen_loss + self_attn_loss
                        print(
                            f"PCA loss: {loss_list[0]:0.4f} | Syngen loss: {syngen_loss:0.4f} | SAR loss : {self_attn_loss:0.4f}")
                    elif loss_list[1] > target_loss[1]:
                        loss = loss_list[1] + syngen_loss + self_attn_loss
                        print(
                            f"PCA loss: {loss_list[1]:0.4f} | Syngen loss: {syngen_loss:0.4f} | SAR loss : {self_attn_loss:0.4f}")
                    else:
                        loss = syngen_loss + self_attn_loss
                        print(f"Syngen loss: {syngen_loss:0.4f} | SAR loss : {self_attn_loss:0.4f}")

                    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]
                    x = x - grad_cond * step_size
                    x = x.detach()
            del loss_list, losses, syngen_loss, self_attn_loss, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3
        torch.cuda.empty_cache()
        return x

    def update_loss_Syngen_RCA_SAR(self,input, index1, index, ts, scale_range, type_loss='Syngen_RCA_SAR',scale_factor =20 ):
        interation_timestep = [0, 5, 10, 15, 20, 30]###TODO:要记得加0回去
        thresholds = {0: [[0.15], [0.1]], 5: [[0.15], [0.4]], 10: [[0.6], [0.6]], 20: [[0.9], [0.7]],
                      30: [[0.95], [0.8]]}
        step_size = scale_factor * np.sqrt(scale_range[index1])
        if index1 < 2:
            loss_scale = 4
        elif index1 < 5:
            loss_scale = 4
        elif index1 < 10:
            loss_scale = 3
        elif index1 < 20:
            loss_scale = 3
        else:
            loss_scale = 1


        loss = torch.tensor(10000)
        loss_list = torch.stack([loss] * 4)

        x = deepcopy(input["x"])
        input["timesteps"] = ts

        token_indice = input['token_indice']
        token_box = input['bbox']
        prompt = input['prompt']
        include_entities = input['include_entities']
        attn_map_idx_to_wp = input['attn_map_idx_to_wp']
        syngen_indice = input['syngen_indice']
        part_of_speech = input['part_of_speech']
        bboxes = input['boxes']
        object_positions = input['object_position']


        x = x.requires_grad_(True)
        input['x'] = x
        e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
            input)
        losses = caculate_loss_PCA_RCA_loss(att_second, att_first, att_third,
                                            bboxes=token_box,
                                            token_indice=token_indice,
                                            t=index1)
        loss_list = self._compute_loss(max_attention_per_index=losses, part_of_speech=part_of_speech)
        syngen_loss = caculate_loss_syngen_loss(att_second, att_first, att_third,
                                                t=index1, prompt=prompt,
                                                include_entities=include_entities,
                                                attn_map_idx_to_wp=attn_map_idx_to_wp,
                                                syngen_indice=syngen_indice)
        self_attn_loss = caculate_loss_self_att(self_first, self_second, self_third, bboxes=bboxes,
                                       object_positions=object_positions, t=index1) * loss_scale
        if index1 not in thresholds.keys():
            loss = syngen_loss + loss_list[2] + self_attn_loss
            print(
                f" RCA loss: {loss_list[2]:0.4f} | Syngen loss: {syngen_loss:0.4f} | SAR loss:{self_attn_loss:0.4f}")
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]
            x = x - grad_cond
            x = x.detach()
            del loss_list,loss, syngen_loss,self_attn_loss, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3

        else:
            iteration = 0
            max_iter = 25

            threshold = thresholds[index1]
            if isinstance(threshold[0], list):
                compare_threshold = [threshold[0][0], threshold[0][0] * 0.5, threshold[1][0], threshold[1][0] * 0.5]
            else:
                compare_threshold = [threshold[0], threshold[0] * 0.5, threshold[1], threshold[1] * 0.5]


            target_loss = [max(0, 1 - t) for t in compare_threshold]
            while (any([x > y for x, y in zip(loss_list, target_loss)]) and (
                    index1 == 0 or index1 == 5 or (index1 == 10 and iteration <= 10))) or (
                    any([x > y for x, y in zip(loss_list[0:2], target_loss[0:2])]) and (
                    index1 != 5 and (index1 == 10 and iteration > 10))):
                iteration += 1

                if iteration > max_iter:
                    break

                x = x.requires_grad_(True)
                input['x'] = x

                e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
                    input)
                losses = caculate_loss_PCA_RCA_loss(att_second, att_first, att_third,
                                                    bboxes=token_box,
                                                    token_indice=token_indice,
                                                    t=index1)
                syngen_loss = caculate_loss_syngen_loss(att_second, att_first, att_third,
                                                        t=index1, prompt=prompt,
                                                        include_entities=include_entities,
                                                        attn_map_idx_to_wp=attn_map_idx_to_wp,
                                                        syngen_indice=syngen_indice)
                loss_list = self._compute_loss(max_attention_per_index=losses, part_of_speech=part_of_speech)
                self_attn_loss = caculate_loss_self_att(self_first, self_second, self_third, bboxes=bboxes,
                                                        object_positions=object_positions, t=index1) * loss_scale

                if all(l != 0 for l in loss_list):  # loss:n.max adj.max n.mean adj.mean
                    # 修改成只是用non的损失
                    if loss_list[2] > target_loss[2] :
                        loss = loss_list[2] + syngen_loss + self_attn_loss
                        print(
                            f" RCA loss: {loss_list[2]:0.4f} | Syngen loss: {syngen_loss:0.4f} | SAR loss : {self_attn_loss:0.4f}")
                    elif loss_list[3] > target_loss[3]:
                        loss = loss_list[3] + syngen_loss + self_attn_loss
                        print(
                            f" RCA loss: {loss_list[3]:0.4f} | Syngen loss: {syngen_loss:0.4f} | SAR loss : {self_attn_loss:0.4f}")
                    else:
                        loss = syngen_loss + self_attn_loss
                        print(f"Syngen loss: {syngen_loss:0.4f} | SAR loss : {self_attn_loss:0.4f}")

                    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]
                    x = x - grad_cond * step_size
                    x = x.detach()
            del loss_list,losses, syngen_loss,self_attn_loss, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3
        torch.cuda.empty_cache()
        return x


    def update_loss_self_cross(self, input, index1, index, ts, type_loss='self_accross'):

        if index1 < 2:
            loss_scale = 4
            max_iter = 2
        elif index1 < 5:
            loss_scale = 4
            max_iter = 6
        elif index1 < 10:
            loss_scale = 3
            max_iter = 3
        elif index1 < 20:
            loss_scale = 3
            max_iter = 2
        else:
            loss_scale = 1
            max_iter = 2

        loss_threshold = 0.1
        max_index = 10
        x = deepcopy(input["x"])
        iteration = 0
        loss = torch.tensor(10000)
        input["timesteps"] = ts

        print("optimize", index1)
        min_inside = 0
        # import pdb; pdb.set_trace()
        max_outside = 1
        if (index1 < max_index):
            while (loss.item() > loss_threshold and iteration < max_iter and (
                    index1 < max_index and (min_inside < 0.2))):  # or max_outside>0.15
                x = x.requires_grad_(True)
                input['x'] = x
                e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
                    input)
                bboxes = input['boxes']
                object_positions = input['object_position']

                # self att losss
                loss1 = caculate_loss_self_att(self_first, self_second, self_third, bboxes=bboxes,
                                               object_positions=object_positions, t=index1) * loss_scale
                # cross attention-loss
                loss2, min_inside, max_outside = caculate_loss_att_fixed_cnt(att_second, att_first, att_third,
                                                                             bboxes=bboxes,
                                                                             object_positions=object_positions,
                                                                             t=index1)

                print('min, max', min_inside, max_outside)
                loss2 *= loss_scale
                # self attention loss in gate-self attention
                loss3, loss_self = caculate_ground(ground1, ground2, ground3, bboxes=bboxes,
                                                   object_positions=object_positions, t=index1)

                loss = loss2 + loss1 + loss3 * loss_scale * 3

                print('loss', loss, loss1, loss2, loss3 * loss_scale * 3, loss_self * loss_scale / 2)
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]

                x = x - grad_cond
                x = x.detach()
                iteration += 1
                del loss1, loss2, loss3, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3

        if (index1 >= 10):

            while ((index1 % 5 == 0 and index1 <= 35) and (iteration < max_iter and (
                    min_inside < 0.2))):  # or (min_inside > 0.2 and max_outside< 0.1)  or max_outside>0.15
                x = x.requires_grad_(True)
                input['x'] = x
                e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model(
                    input)
                bboxes = input['boxes']
                object_positions = input['object_position']
                loss1 = caculate_loss_self_att(self_first, self_second, self_third, bboxes=bboxes,
                                               object_positions=object_positions, t=index1) * loss_scale
                loss2, min_inside, max_outside = caculate_loss_att_fixed_cnt(att_second, att_first, att_third,
                                                                             bboxes=bboxes,
                                                                             object_positions=object_positions,
                                                                             t=index1)
                print('min, max', min_inside, max_outside)
                loss2 *= loss_scale
                loss3, loss_self = caculate_ground(ground1, ground2, ground3, bboxes=bboxes,
                                                   object_positions=object_positions, t=index1)
                loss = loss1 + loss2 + loss3 * loss_scale * 3
                print('loss', loss, loss1, loss2, loss3 * loss_scale * 3, loss_self * loss_scale / 2)
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x])[0]
                x = x - grad_cond
                x = x.detach()
                iteration += 1
                del loss1, loss2, loss3, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3
        return x

    def update_loss_only_cross(self, input, index1, index, ts, type_loss='self_accross'):

        if index1 < 10:
            loss_scale = 3
            max_iter = 5
        elif index1 < 20:
            loss_scale = 2
            max_iter = 5
        else:
            loss_scale = 1
            max_iter = 1
        loss_threshold = 0.1

        max_index = 30
        x = deepcopy(input["x"])
        iteration = 0
        loss = torch.tensor(10000)
        input["timesteps"] = ts

        print("optimize", index1)
        while loss.item() > loss_threshold and iteration < max_iter and (index1 < max_index):
            print('iter', iteration)
            x = x.requires_grad_(True)
            input['x'] = x
            e_t, att_first, att_second, att_third, self_first, self_second, self_third = self.model(input)

            bboxes = input['boxes']
            object_positions = input['object_position']
            loss2 = caculate_loss_att_fixed_cnt(att_second, att_first, att_third, bboxes=bboxes,
                                                object_positions=object_positions, t=index1) * loss_scale
            loss = loss2
            print('loss', loss)
            hh = torch.autograd.backward(loss)
            grad_cond = x.grad
            x = x - grad_cond
            x = x.detach()
            iteration += 1
            torch.cuda.empty_cache()
        return x

    def update_only_self(self, input, index1, index, ts, type_loss='self_accross'):
        if index1 < 10:
            loss_scale = 4
            max_iter = 5
        elif index1 < 20:
            loss_scale = 3
            max_iter = 5
        else:
            loss_scale = 1
            max_iter = 1
        loss_threshold = 0.1

        max_index = 30
        x = deepcopy(input["x"])
        iteration = 0
        loss = torch.tensor(10000)
        input["timesteps"] = ts

        print("optimize", index1)
        while loss.item() > loss_threshold and iteration < max_iter and (index1 < max_index):
            print('iter', iteration)
            x = x.requires_grad_(True)
            input['x'] = x
            e_t, att_first, att_second, att_third, self_first, self_second, self_third = self.model(input)

            bboxes = input['boxes']
            object_positions = input['object_position']
            loss = caculate_loss_self_att(self_first, self_second, self_third, bboxes=bboxes,
                                          object_positions=object_positions, t=index1) * loss_scale
            print('loss', loss)
            hh = torch.autograd.backward(loss)
            grad_cond = x.grad

            x = x - grad_cond
            x = x.detach()
            iteration += 1
            torch.cuda.empty_cache()
        return x

    @torch.no_grad()
    def p_sample_plms(self, input, t, index, guidance_scale=1., uc=None, old_eps=None, t_next=None):
        x = deepcopy(input["x"])
        b = x.shape[0]

        def get_model_output(input):
            e_t, first, second, third, _, _, _, _, _, _ = self.model(input)
            if uc is not None and guidance_scale != 1:
                unconditional_input = dict(x=input["x"], timesteps=input["timesteps"], context=uc,
                                           inpainting_extra_input=input["inpainting_extra_input"],
                                           grounding_extra_input=input['grounding_extra_input'], save_attn = None)
                e_t_uncond, _, _, _, _, _, _, _, _, _ = self.model(unconditional_input)
                e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            return e_t

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=self.device)
            a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
            sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index], device=self.device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * torch.randn_like(x)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        input["timesteps"] = t
        e_t = get_model_output(input)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            input["x"] = x_prev
            input["timesteps"] = t_next
            e_t_next = get_model_output(input)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t


