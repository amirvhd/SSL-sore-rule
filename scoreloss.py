# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


def gaussian_kernel(x1, x2, gamma=1):
    a = -F.mse_loss(x1, x2, reduction='sum')
    print(a.shape)
    return torch.exp(a)


class SCORELoss(nn.Module):
    def __init__(self, out_dim, ncrops,
                 center_momentum=0.9, l=2.4):
        super().__init__()
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.out = out_dim
        self.n_sample = 4
        self.dist = torch.distributions.Normal(loc=torch.zeros(self.out, device="cuda"),
                                               scale=torch.ones(self.out, device="cuda")
                                               )
        self.l = l

    def forward(self, mean_st, var_st, teacher_output):
        self.batch = mean_st.shape[0]

        samples = self.dist.rsample(sample_shape=torch.tensor([self.n_sample, self.batch]))
        student_sample = (
                (samples * var_st.unsqueeze(0)) + mean_st.unsqueeze(0))

        teacher_output = (teacher_output - self.center)
        teacher_out = teacher_output.detach().chunk(2)

        student_sample_chunk = student_sample.chunk(self.ncrops, dim=1)
        loss_second_part = 0
        loss_first_part = 0
        n_loss = 0

        for i in range(self.n_sample):
            for iq, q in enumerate(teacher_out):
                for iv in range(len(student_sample_chunk)):
                    loss_first_part += 2 * F.l1_loss(student_sample_chunk[iv][i, :, :], q)
                    n_loss += 1

            if i != self.n_sample - 1:
                for j in range(i + 1, self.n_sample):
                    loss_second_part += - (1 / (self.n_sample - 1)) * F.l1_loss(student_sample[i, :, :],
                                                                                student_sample[j, :, :])

        total_loss = 10 * (
                (loss_first_part + 0.000009) / n_loss + self.l * (loss_second_part + 0.000009) / self.n_sample)
        self.update_center(teacher_output)
        return total_loss, loss_first_part / n_loss, loss_second_part / self.n_sample

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
