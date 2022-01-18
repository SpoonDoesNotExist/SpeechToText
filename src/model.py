"""
Модель QuartzNet BxR
Классы B_block и R_block соответствуют частям модели, указанным в статье

R_block:    Separable convolution + Batch norm + ReLU
B_block:    Состоит из R подряд идущих R_block
"""

class Pointwise_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, 1)

    def forward(self, x):
        return self.conv(x)


class R_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, activate=True):
        super().__init__()
        self.activate = activate
        self.conv_depth = nn.Conv1d(
            in_c, in_c, kernel_size,
            padding='same', groups=in_c
        )
        self.conv_point = nn.Conv1d(in_c, out_c, 1)
        self.batch_norm = nn.BatchNorm1d(out_c)

    def forward(self, x):
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.batch_norm(x)
        if(self.activate):
            x = F.relu(x)

        return x


class B_block(nn.Module):
    def __init__(self, R_size, in_c, out_c, kernel_size):
        super().__init__()
        self.conv_pw = Pointwise_conv(in_c, out_c)
        self.batch_norm = nn.BatchNorm1d(out_c)
        self.R = nn.Sequential(*self.generate_R_part(R_size, in_c, out_c,kernel_size))

    def forward(self, x):
        x_add = self.conv_pw(x)
        x_add = self.batch_norm(x_add)
        x = self.R(x)
        x = x+x_add
        x = F.relu(x)

        return x

    def generate_R_part(self, R_size, in_c, out_c,kernel_size):
        r_list = []
        for i in range(R_size):
            if i < R_size-1:
                r_list.append(
                    R_block(
                        in_c,
                        in_c,
                        kernel_size
                    )
                )
            else:
                r_list.append(
                    R_block(
                        in_c,
                        out_c,
                        kernel_size,
                        activate=False
                    )
                )
        return r_list


class QuartzNet(nn.Module):
    def __init__(self, params):
        super().__init__()

        c_params=params['c_params']
        b_params=params['b_params']
        b_size=params['b_size']

        self.C_1 = R_block(
            c_params['c_1']['in'],
            c_params['c_1']['out'],
            c_params['c_1']['kernel']
        )

        self.B = nn.Sequential(
            *[B_block(b_size,p['in'], p['out'], p['kernel']) for p in b_params]
        )

        self.C_2 = R_block(
            c_params['c_2']['in'],
            c_params['c_2']['out'],
            c_params['c_2']['kernel']
        )
        self.C_3 = R_block(
            c_params['c_3']['in'],
            c_params['c_3']['out'],
            c_params['c_3']['kernel']
        )
        self.C_4 = Pointwise_conv(
            c_params['c_4']['in'],
            c_params['c_4']['out']
        )

    def forward(self, x):
        sizes = x.size()  # (batch,chanel,feature,time)
        x = x.view(sizes[0],sizes[1] * sizes[2],sizes[3])  # (batch,chanel,time)

        x = self.C_1(x)
        x = self.B(x)
        x = self.C_2(x)
        x = self.C_3(x)
        x = self.C_4(x)

        return x