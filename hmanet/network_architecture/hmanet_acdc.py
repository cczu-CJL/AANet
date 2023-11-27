import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import to_3tuple

from hmanet.network_architecture.neural_network import SegmentationNetwork


class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()
class Conv3dGeLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, use_batchnorm=True):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         bias=not (use_batchnorm))
        gelu = nn.GELU()
        IN = nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)
        super(Conv3dGeLU, self).__init__(conv, IN, gelu)


class DecoderConv_AttenBlock(nn.Module):
    def __init__(self,window_size, in_channels, out_channels, skip_channels=0, head=1, depths=1,spatial_kernel=3, topk=1, tag=0):
        super().__init__()
        self.conv1 = nn.Sequential(
                 nn.Conv3d(out_channels + skip_channels, out_channels, 3, 1, 1, bias=False),
                 nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True),
                 nn.LeakyReLU(inplace=True))


        self.conv_attn =nn.Sequential(
            *([Conv_Atten_inverse_Layer(out_channels, head, window_size, spatial_kernel, topk=topk) for _ in range(depths)]))

        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
            nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True))
        if tag==0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, [1,2,2],[1,2,2]),
                nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True))
        elif tag==1:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, [2,2,2],[2,2,2]),
                nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True))
        elif tag==2:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, [1, 2, 2], [1, 2, 2]),
                nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True))

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        out = self.conv_attn(x)
        return out

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = nn.LeakyReLU(inplace=True)

        self.norm1 = nn.InstanceNorm3d(out_dim, eps=1e-5, affine=True)
        self.last = last
        self.norm2 = nn.InstanceNorm3d(out_dim, eps=1e-5, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activate(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if not self.last:
            x = self.activate(x)
        return x
class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=32):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1=[1,patch_size[1]//2,patch_size[2]//2]
        stride2=[1,patch_size[1]//2,patch_size[2]//2]
        self.proj1 = project(in_chans,embed_dim//2,stride1,1,False)
        self.proj2 = project(embed_dim//2,embed_dim,stride2,1,True)

        self.norm = nn.InstanceNorm3d(embed_dim, eps=1e-5, affine=True)

    def forward(self, x):
        """Forward function."""

        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))
        x = self.proj1(x)
        x = self.proj2(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class Atten(nn.Module):
    def __init__(self, dims, head):
        super().__init__()
        self.head = head
        self.scale = (dims // head) ** -0.5
        self.qkv = nn.Linear(dims, dims * 3, bias=True)
        self.proj = nn.Linear(dims, dims)
    def forward(self, c1):
        B, N, C = c1.shape

        qkv = self.qkv(c1).reshape(B, -1, 3, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)

        out = self.proj(x_atten)
        return out

class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))

        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class QKVGather(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, r_idx, qkv):

        n, p3, w3, c_kv = qkv.size()
        topk = r_idx.size(-1)
        topk_Qkv = torch.gather(qkv.view(n, 1, p3, w3, c_kv).expand(-1, p3, -1, -1, -1),
                                dim=2,
                                index=r_idx.view(n, p3, topk, 1, 1).expand(-1, -1, -1, w3, c_kv)
                               )
        return topk_Qkv
class TopkRouting(nn.Module):
    def __init__(self, qk_dim, topk=4, qk_scale=None):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5

        self.emb_q = nn.Linear(qk_dim, qk_dim)
        self.emb_k = nn.Linear(qk_dim, qk_dim)

        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, g_win):

        query_hat, key_hat = self.emb_q(g_win), self.emb_k(g_win)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)

        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)

        return r_weight, topk_index
class DownConv(nn.Module):

    def __init__(self, cin, cout,kernel_size=[1,3,3],stride=[1,2,2],padding=[0,1,1]):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(cin, cout,kernel_size=kernel_size,stride=stride,padding=padding, bias=False),
            nn.InstanceNorm3d(cout, eps=1e-5, affine=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7, topk=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = self.dim ** -0.5
        self.topk = topk

        self.router = TopkRouting(qk_dim=dim,
                                  qk_scale=self.scale,
                                  topk=self.topk)
        self.qkv_gather = QKVGather()
        self.wo = nn.Linear(dim, dim)

        self.attn_act = nn.Softmax(dim=-1)

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Atten(dim, num_heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim),
                                 nn.GELU(),
                                 nn.Linear(4*dim, dim))
        self.g_qkv = nn.Linear(dim, 3*dim)

        self.mlp2 = nn.Sequential(nn.Linear(dim, 4*dim),
                                  nn.GELU(),
                                 nn.Linear(4*dim, dim))


    def forward(self, x_in, x_g_in):

        N, C, S_in, H_in, W_in = x_in.size()

        pad_r = (self.window_size[0] - S_in % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - H_in % self.window_size[1]) % self.window_size[1]
        pad_g = (self.window_size[2] - W_in % self.window_size[2]) % self.window_size[2]

        shortcut_x = F.pad(x_in, (0, pad_g, 0, pad_b, 0, pad_r))

        _, _, S, H, W = shortcut_x.size()  # padded size
        shortcut_x = shortcut_x.permute(0, 2, 3, 4, 1)


        shortcut_x = rearrange(shortcut_x, 'b (p1 Ms) (p2 Mh) (p3 Mw) c-> b (p1 p2 p3) Ms Mh Mw c', Ms=self.window_size[0], Mh=self.window_size[1], Mw=self.window_size[2], c=C)

        gN, gC, gS_in, gH_in, gW_in = x_g_in.size()

        pad_r_g = (S // self.window_size[0] - gS_in % (S // self.window_size[0])) % (S // self.window_size[0])
        pad_b_g = (H // self.window_size[1] - gH_in % (H // self.window_size[1])) % (H // self.window_size[1])
        pad_g_g = (W // self.window_size[2] - gW_in % (W // self.window_size[2])) % (W // self.window_size[2])
        x_g = F.pad(x_g_in, (0, pad_g_g, 0, pad_b_g, 0, pad_r_g))
        gN, gC, gS, gH, gW = x_g.size()

        x_g = rearrange(x_g, 'b c Ms Mh Mw  -> b (Ms Mh Mw) c')
        x_g = self.attn(self.norm1(x_g)) + x_g
        x_g = rearrange(x_g, 'b (Ms Mh Mw) c -> b Ms Mh Mw c', Ms=gS, Mh=gH, Mw=gW)
        x_g = x_g + self.mlp(self.norm1(x_g))

        shortcut_x_g = rearrange(x_g, 'b (p1 Ms) (p2 Mh) (p3 Mw) c -> b (p1 p2 p3) Ms Mh Mw c', p1=S // self.window_size[0], p2=H // self.window_size[1], p3=W // self.window_size[2])



        _, _, gMs, gMh, gMw, _ = shortcut_x_g.size()

        g_win = shortcut_x_g.mean([2, 3, 4])

        r_weight, r_idx = self.router(g_win)
        g_qkv_pix_sel = rearrange(shortcut_x_g, 'b nw Ms Mh Mw c -> b nw (Ms Mh Mw) c')
        g_qkv_pix_sel = self.qkv_gather(r_idx=r_idx, qkv=g_qkv_pix_sel)  # (n, p^3, topk, s_kv*h_kv*w_kv, c)


        shortcut_l = rearrange(shortcut_x, 'b nw Ms Mh Mw c -> b nw (Ms Mh Mw) c')
        shortcut_g = rearrange(g_qkv_pix_sel, 'b nw k W3 c ->b nw (k W3) c')
        shortcut = torch.cat((shortcut_l, shortcut_g), dim=2)


        qkv_pix = self.g_qkv(self.norm1(shortcut))
        q_pix_sel, k_pix_sel, v_pix_sel = qkv_pix.split([self.dim, self.dim, self.dim], dim=-1)

        q_pix_sel = rearrange(q_pix_sel, 'b nw w3 (m c) -> (b nw) m w3 c', m=self.num_heads)
        k_pix_sel = rearrange(k_pix_sel, 'b nw w3 (m c) -> (b nw) m c w3', m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'b nw w3 (m c) -> (b nw) m w3 c', m=self.num_heads)



        attn_weight = (q_pix_sel * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel


        out = rearrange(out, '(b nw) m w3 c -> b nw w3 (m c)', nw = (S*H*W) // (self.window_size[0]*self.window_size[1]*self.window_size[2]))

        l_out = out[:,:,:self.window_size[0] * self.window_size[1] * self.window_size[2],:]

        l_out = self.wo(l_out)
        l_out = l_out + shortcut_l

        l_out = rearrange(l_out, 'b (p1 p2 p3) (Ms Mh Mw) c -> b (p1 Ms) (p2 Mh) (p3 Mw) c',
                      p1=S // self.window_size[0], p2=H // self.window_size[1], p3=W // self.window_size[2],
                      Ms=self.window_size[0], Mh=self.window_size[1], Mw=self.window_size[2])
        l_out = l_out + self.mlp2(self.norm1(l_out))

        g_out = x_g.permute(0, 4, 1, 2, 3)
        l_out = l_out.permute(0, 4, 1, 2, 3)

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            l_out = l_out[:, :, :S_in, :H_in, :W_in].contiguous()
        if pad_r_g > 0 or pad_b_g > 0 or pad_g_g > 0:
            g_out = g_out[:, :, :gS_in, :gH_in, :gW_in].contiguous()
        return l_out, g_out

class Conv_Atten_inverse_Layer(nn.Module):
    def __init__(self, dim, head, window_size, spatial_kernel, topk):
        super().__init__()

        self.conv1 = BasicResBlock(dim, dim, 3, 1, 1, False)

        self.sa_conv = nn.Conv3d(2, 1, kernel_size=spatial_kernel,
                            padding=spatial_kernel // 2, bias=False)

        self.sigmoid = nn.Sigmoid()


        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
        self.conv_fc1 = nn.Sequential(
           nn.Conv3d(dim, dim//2, kernel_size=1, stride=1),
           nn.LeakyReLU(inplace=True)
        )
        self.conv_fc2 = nn.Conv3d(dim//2, dim, kernel_size=1, stride=1)

        self.attn = BiLevelRoutingAttention(dim=dim, num_heads=head, window_size=window_size, topk=topk)

        if head == 2:
            self.down = nn.Sequential(
                nn.Conv3d(dim, dim, (1, 3, 3), (1, 2, 2), (0, 1, 1), bias=False),
                nn.InstanceNorm3d(dim, eps=1e-5, affine=True),
            )
            self.up = nn.Sequential(
                nn.ConvTranspose3d(dim, dim, (1, 2, 2), (1, 2, 2)),
                nn.InstanceNorm3d(dim, eps=1e-5, affine=True))
        elif head == 4:
            self.down = nn.Sequential(
                nn.Conv3d(dim, dim, (3, 3, 3), (2, 2, 2), (1, 1, 1), bias=False),
                nn.InstanceNorm3d(dim, eps=1e-5, affine=True),
            )
            self.up = nn.Sequential(
                nn.ConvTranspose3d(dim, dim, (2, 2, 2), (2, 2, 2)),
                nn.InstanceNorm3d(dim, eps=1e-5, affine=True))
        elif head == 8:
            self.down = nn.Sequential(
                nn.Conv3d(dim, dim, (3, 3, 3), (2, 2, 2), (0, 1, 1), bias=False),
                nn.InstanceNorm3d(dim, eps=1e-5, affine=True),
            )
            self.up = nn.Sequential(
                nn.ConvTranspose3d(dim, dim, (2, 2, 2), (2, 2, 2),output_padding=(1,0,0)),
                nn.InstanceNorm3d(dim, eps=1e-5, affine=True))
        else:
            self.down = nn.Sequential(
                nn.Conv3d(dim, dim,  1, 1, bias=False),
                nn.InstanceNorm3d(dim, eps=1e-5, affine=True),
            )
            self.up = nn.Sequential(
                nn.ConvTranspose3d(dim, dim, 1, 1, bias=False),
                nn.InstanceNorm3d(dim, eps=1e-5, affine=True))

    def forward(self, inp):


        conv = self.conv1(inp)

        max_out, _ = torch.max(conv, dim=1, keepdim=True)
        avg_out = torch.mean(conv, dim=1, keepdim=True)
        spatial_vectors = self.sigmoid(self.sa_conv(torch.cat([max_out, avg_out], dim=1)))
        inverse_spatial_vectors = torch.ones(spatial_vectors.shape, device=inp.device) - spatial_vectors


        trans = inp * inverse_spatial_vectors + conv

        g_inp = self.down(inp)
        l_trans, g_trans = self.attn(trans, g_inp)
        l_trans = self.up(g_trans) + l_trans


        cha_avg_out  = self.gap(l_trans)
        cha_avg_out  = self.conv_fc1(cha_avg_out)
        cha_avg_out = self.conv_fc2(cha_avg_out)
        channel_vectors = self.sigmoid(cha_avg_out)
        inverse_channel_vectors = torch.ones(channel_vectors.shape, device=inp.device) - channel_vectors

        fuse1 = conv * inverse_channel_vectors + l_trans


        return fuse1



class hmanet(SegmentationNetwork):
    def __init__(self, crop_size=[14, 160, 160], embedding_dim=48, input_channels=1, num_classes=13, conv_op=nn.Conv3d,
                 depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16), patch_size=(1, 4, 4),
                 window_size=(7, 7, 7, 7),  # window_size=(4,4,8,4),
                 deep_supervision=True, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,  # drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,  # patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op

        embed_dim = embedding_dim
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=input_channels, embed_dim=embed_dim*2)

        self.encoder1 = nn.Sequential(
            *([Conv_Atten_inverse_Layer(embed_dim * 2, 2, window_size=[7, 5, 5], spatial_kernel=9, topk=6) for _ in range(depths[0])]))

        self.down2 = BasicResBlock(embed_dim * 2, embed_dim * 4, [1,3,3], [1,2,2], [0,1,1], True)
        self.encoder2 = nn.Sequential(
            *([Conv_Atten_inverse_Layer(embed_dim * 4, 4, window_size=[7, 5, 5], spatial_kernel=7, topk=3) for _ in range(depths[1])]))

        self.down3 = BasicResBlock(embed_dim * 4, embed_dim * 8, [3,3,3],[2,2,2],[1,1,1], True)
        self.encoder3 = nn.Sequential(
            *([Conv_Atten_inverse_Layer(embed_dim * 8, 8, window_size=[7, 10, 10], spatial_kernel=5, topk=1) for _ in range(depths[2])]))


        self.down4 = DownConv(embed_dim * 8, embed_dim * 16, kernel_size=[1,3,3],stride=[1,2,2],padding=[0,1,1])
        self.encoder4 = nn.Sequential(
            *([Conv_Atten_inverse_Layer(embed_dim * 16, 16, window_size=[7, 5, 5], spatial_kernel=3, topk=1) for _ in range(depths[3])]))


        self.ups0 = DecoderConv_AttenBlock([7, 10, 10], embed_dim * 16, embed_dim * 8, embed_dim * 8, head=8, depths=depths[2], spatial_kernel=5, topk=1, tag=2)
        self.ups1 = DecoderConv_AttenBlock([7, 5, 5], embed_dim * 8, embed_dim * 4, embed_dim * 4, head=4, depths=depths[1], spatial_kernel=7, topk=3, tag=1)
        self.ups2 = DecoderConv_AttenBlock([7, 5, 5], embed_dim * 4, embed_dim * 2, embed_dim * 2,head=2, depths=depths[0], spatial_kernel=9, topk=6, tag=0)

        self.ups16 = nn.ConvTranspose3d(embed_dim * 8, num_classes, (2, 4, 4), (2, 4, 4))
        self.ups8 = nn.ConvTranspose3d(embed_dim * 4, num_classes, (1, 4, 4), (1, 4, 4))
        self.ups4 = nn.ConvTranspose3d(embed_dim * 2, num_classes, (1, 4, 4), (1, 4, 4))
    def forward(self, input):

        x = self.patch_embed(input)
        bridge1 = self.encoder1(x)


        CT= self.down2(bridge1)
        bridge2 = self.encoder2(CT)


        CT = self.down3(bridge2)
        bridge3 = self.encoder3(CT)


        CT = self.down4(bridge3)
        bridge4 = self.encoder4(CT)


        bridge3 = self.ups0(bridge4, bridge3)

        out1 = self.ups1(bridge3, bridge2)

        out2 = self.ups2(out1, bridge1)

        seg_outputs = []
        seg_outputs.append(self.ups16(bridge3))
        seg_outputs.append(self.ups8(out1))
        seg_outputs.append(self.ups4(out2))
        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]
class makelayer(nn.Module):
    def __init__(self, dim, head, window_size, spatial_kernel, depth, topk):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv_Atten_inverse_Layer(dim, head, window_size=window_size, spatial_kernel=spatial_kernel, topk=topk)
            for _ in range(depth)
        ])

    def forward(self, x, g_x):
        for blk in self.blocks:
            x, g_x = blk(x, g_x)
        return x, g_x

if __name__ == '__main__':
    img = torch.randn([1, 1, 14, 160, 160])  # 14, 160, 160
    model = hmanet()
    out = model(img)
    print(type(out))
    print(len(out))
