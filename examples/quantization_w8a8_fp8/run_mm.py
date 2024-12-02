import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import triton
from mm_kernel import matmul_kernel, triton_scaled_mm

def custom_matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c

import time
torch.manual_seed(0)
a = torch.randn((3072, 3072), device="cuda", dtype=torch.float16)
b = torch.randn((3072, 3072), device="cuda", dtype=torch.float16)
a = a.to(torch.float8_e4m3fnuz)
b = b.T
b = b.to(torch.float8_e4m3fnuz)
scales = torch.ones((3072, 1), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fnuz)

all_t = []
triton_scaled_mm(a, b, scale_a=scales, scale_b=scales, out_dtype=torch.float16)
for _ in range(10):
    st = time.time()
    triton_output = triton_scaled_mm(a, b, scale_a=scales, scale_b=scales, out_dtype=torch.float16)
    all_t.append(time.time() - st)
print(f'Triton matmul running time: {sum(all_t)/len(all_t)}s')

all_t = []
torch.matmul(a.to(torch.float16), b.to(torch.float16))
for _ in range(10):
    st = time.time()
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    all_t.append(time.time() - st)
print(f'Torch matmul running time: {sum(all_t)/len(all_t)}s')

if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")