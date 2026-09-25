// Compute Shader 2x Upscaler (MiniESPCN)
// Conv3x3(3->32)+ReLU -> Conv5x5(32->16)+ReLU -> Conv3x3(16->12) -> PixelShuffle(2)
//
// TG = 8x8 LR -> 16x16 HR; halo = 4; IN = 16x16 LR
// Shared: s_input 16x16x3 + s_l1 14x14x32 + s_l2 10x10x16 ~= 34.5 KB
// Dispatch: Dispatch(ceil(W_LR/8), ceil(H_LR/8), 1)

#include "weights.hlsl"

Texture2D<float4>   InputTexture  : register(t0);
RWTexture2D<float4> OutputTexture : register(u0);

#define TG_X 8
#define TG_Y 8
#define HALO 4

#define IN_W  (TG_X + 2 * HALO)   // 16
#define IN_H  (TG_Y + 2 * HALO)   // 16

#define L1_W  (IN_W - 2)          // 14
#define L1_H  (IN_H - 2)          // 14
#define L2_W  (L1_W - 4)          // 10
#define L2_H  (L1_H - 4)          // 10

groupshared float3 s_input[IN_H][IN_W];
groupshared float  s_l1[32][L1_H][L1_W];
groupshared float  s_l2[16][L2_H][L2_W];

[numthreads(TG_X, TG_Y, 1)]
void CSMain(uint3 gid  : SV_GroupID,
            uint3 gtid : SV_GroupThreadID)
{
    uint2 texSize;
    InputTexture.GetDimensions(texSize.x, texSize.y);

    int2 inBase = int2(gid.xy * uint2(TG_X, TG_Y)) - int2(HALO, HALO);

    // ---------- Загрузка входа с ZERO-padding ----------
    for (int y = gtid.y; y < IN_H; y += TG_Y) {
        for (int x = gtid.x; x < IN_W; x += TG_X) {
            int2 p = inBase + int2(x, y);
            if (p.x < 0 || p.y < 0 || p.x >= (int)texSize.x || p.y >= (int)texSize.y) {
                s_input[y][x] = float3(0.0f, 0.0f, 0.0f);
            } else {
                s_input[y][x] = InputTexture[p].rgb;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---------- Conv1 3x3 (3->32) + ReLU ----------
    for (int y = gtid.y; y < L1_H; y += TG_Y) {
        for (int x = gtid.x; x < L1_W; x += TG_X) {
            int iy = y + 1, ix = x + 1;
            [unroll]
            for (int c = 0; c < 32; c++) {
                float sum = conv1_bias[c];
                [unroll]
                for (int dy = -1; dy <= 1; dy++) {
                    [unroll]
                    for (int dx = -1; dx <= 1; dx++) {
                        float3 v = s_input[iy + dy][ix + dx];
                        sum += v.r * conv1_weight[c][0][dy+1][dx+1]
                             + v.g * conv1_weight[c][1][dy+1][dx+1]
                             + v.b * conv1_weight[c][2][dy+1][dx+1];
                    }
                }
                s_l1[c][y][x] = max(0.0f, sum);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---------- Conv2 5x5 (32->16) + ReLU ----------
    for (int y = gtid.y; y < L2_H; y += TG_Y) {
        for (int x = gtid.x; x < L2_W; x += TG_X) {
            int iy = y + 2, ix = x + 2;
            [unroll]
            for (int c = 0; c < 16; c++) {
                float sum = conv2_bias[c];
                [unroll]
                for (int in_c = 0; in_c < 32; in_c++) {
                    [unroll]
                    for (int dy = -2; dy <= 2; dy++) {
                        [unroll]
                        for (int dx = -2; dx <= 2; dx++) {
                            sum += s_l1[in_c][iy + dy][ix + dx]
                                 * conv2_weight[c][in_c][dy + 2][dx + 2];
                        }
                    }
                }
                s_l2[c][y][x] = max(0.0f, sum);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---------- Conv3 3x3 (16->12), register accumulation ----------
    int iy = gtid.y + 1, ix = gtid.x + 1;

    float l3_0  = conv3_bias[0],  l3_1  = conv3_bias[1],  l3_2  = conv3_bias[2];
    float l3_3  = conv3_bias[3],  l3_4  = conv3_bias[4],  l3_5  = conv3_bias[5];
    float l3_6  = conv3_bias[6],  l3_7  = conv3_bias[7],  l3_8  = conv3_bias[8];
    float l3_9  = conv3_bias[9],  l3_10 = conv3_bias[10], l3_11 = conv3_bias[11];

    [unroll]
    for (int in_c = 0; in_c < 16; in_c++) {
        [unroll]
        for (int dy = -1; dy <= 1; dy++) {
            [unroll]
            for (int dx = -1; dx <= 1; dx++) {
                float v = s_l2[in_c][iy + dy][ix + dx];
                l3_0  += v * conv3_weight[0][in_c][dy+1][dx+1];
                l3_1  += v * conv3_weight[1][in_c][dy+1][dx+1];
                l3_2  += v * conv3_weight[2][in_c][dy+1][dx+1];
                l3_3  += v * conv3_weight[3][in_c][dy+1][dx+1];
                l3_4  += v * conv3_weight[4][in_c][dy+1][dx+1];
                l3_5  += v * conv3_weight[5][in_c][dy+1][dx+1];
                l3_6  += v * conv3_weight[6][in_c][dy+1][dx+1];
                l3_7  += v * conv3_weight[7][in_c][dy+1][dx+1];
                l3_8  += v * conv3_weight[8][in_c][dy+1][dx+1];
                l3_9  += v * conv3_weight[9][in_c][dy+1][dx+1];
                l3_10 += v * conv3_weight[10][in_c][dy+1][dx+1];
                l3_11 += v * conv3_weight[11][in_c][dy+1][dx+1];
            }
        }
    }

    // ---------- PixelShuffle + BOUNDS CHECK ----------
    int2 outPixel = int2(gid.xy * uint2(TG_X, TG_Y)) + int2(gtid.xy);

    if (outPixel.x >= (int)texSize.x || outPixel.y >= (int)texSize.y)
        return;

    int2 outBase = outPixel * 2;

    OutputTexture[outBase + int2(0, 0)] = float4(l3_0,  l3_4,  l3_8,  1.0f);
    OutputTexture[outBase + int2(1, 0)] = float4(l3_1,  l3_5,  l3_9,  1.0f);
    OutputTexture[outBase + int2(0, 1)] = float4(l3_2,  l3_6,  l3_10, 1.0f);
    OutputTexture[outBase + int2(1, 1)] = float4(l3_3,  l3_7,  l3_11, 1.0f);
}
