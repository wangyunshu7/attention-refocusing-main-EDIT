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

    interation_timestep = [0, 5, 10, 15, 20, 30]
    thresholds = {0: [[0.15], [0.1]], 5: [[0.15], [0.4]], 10: [[0.6], [0.6]], 20: [[0.9], [0.7]],
                  30: [[0.95], [0.8]]}
    step_size = scale_factor * np.sqrt(scale_range[index1])

    loss_threshold = 0.1
    max_index = 10
    x = deepcopy(input["x"])
    iteration = 0
    loss = torch.tensor(10000)
    input["timesteps"] = ts

    print("optimize", index1)
    # import pdb; pdb.set_trace()
    if index1 in interation_timestep:
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