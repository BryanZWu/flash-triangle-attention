'''
Code adapted from the triton flash-attn-2.0 tutorial:
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

Modifications: 
- In AF2 triangular attention, an additional bias is applied to the 
  attention scores: "The decision whether edge ij will receive an
  update from edge ik is not only determined by their
  query-key similarity (as in standard attention [95]),
  but also modulated by the information bjk derived from
  the third edge jk of this triangle"
- Triangle attention input is pw for all QKV: (B, H, L, L, D)

Notes on nomenclature:
- Shapes:
    - B: batch
    - H: heads
    - L: context length
    - D: dimensions
Tensors:
    - Q: (B, H, QL1, QL2, D) Query
    - K: (B, H, KN1, KL2, D) Key
    - V: (B, H, KN1, KL2, D) Value
    - B_pw: (B, H, KN1, KL2) PW Bias
    - M: (B, H, QL1, QL2) Max values
    - O: (B, H, QL1, QL2, D) Output
'''

import pytest
import torch
# import triton.tools.experimental_descriptor

import triton
import triton.language as tl

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Helper function to check if we're using HIP (AMD) backend
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

# Helper function to check if we're using CUDA backend
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr, B_pw_block_ptr,
                    start_ql2, pre_bias_qk_scale, post_bias_qk_scale,
                    BLOCK_QL2: tl.constexpr, H: tl.constexpr, BLOCK_KL2: tl.constexpr,  #
                    offs_ql2: tl.constexpr, offs_kl2: tl.constexpr,
                    L: tl.constexpr, fp8_v: tl.constexpr):
    """
    Inner kernel for attention forward pass computation
    Computes attention scores and updates accumulator for a block of the attention matrix
    """
    lo, hi = 0, L

    # Advance block pointers to correct starting positions
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    B_pw_block_ptr = tl.advance(B_pw_block_ptr, (0, lo))

    # Loop over the entire sequence of KV, accumulating (Q @ K)V
    for start_n in range(lo, hi, BLOCK_KL2):
        start_n = tl.multiple_of(start_n, BLOCK_KL2)

        # Load and print K block shape
        k = tl.load(K_block_ptr)

        qk = tl.dot(q, k)

        # Apply attention bias, and scaling (pre-bias being 1/sqrt(head_dim) and post-bias being 1/log(2))
        b_pw = tl.load(B_pw_block_ptr)
        qk = (qk * pre_bias_qk_scale + b_pw) * post_bias_qk_scale

        # Update new max value if needed. Subtract for stability
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        # Attn scores
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # Scale down prev denominator and numerator by exp(m_i - m_ij),
        # if the max changed.
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load V and compute attention output
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)  # Convert to FP8 if needed
        else:
            p = p.to(tl.float16)  # Otherwise use FP16
        acc = tl.dot(p, v, acc)  # Update accumulator with attention values

        m_i = m_ij

        # Advance block pointers
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_KL2, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_KL2))
        B_pw_block_ptr = tl.advance(B_pw_block_ptr, (0, BLOCK_KL2))

    return acc, l_i, m_i

configs = [
    triton.Config({'BLOCK_QL2': B_QL2, 'BLOCK_KL2': B_KL2}, num_stages=s, num_warps=w) \
    for B_QL2 in [64, 128]\
    for B_KL2 in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_QL2 = conf.kwargs["BLOCK_QL2"]
    BLOCK_KL2 = conf.kwargs["BLOCK_KL2"]
    if BLOCK_QL2 * BLOCK_KL2 < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["L", "D"])
@triton.jit
def _attn_fwd(Q, K, V, B_pw, sm_scale, M, Out,  # sm_scale: scaling factor for softmax, M: max values, Out: output tensor
              stride_qb, stride_qh, stride_ql1, stride_ql2, stride_qd,  # Strides for Q tensor: batch, head, seq_len, seq_len, head_dim
              stride_kb, stride_kh, stride_kl1, stride_kl2, stride_kd,  # Strides for K tensor
              stride_vb, stride_vh, stride_vl1, stride_vl2, stride_vd,  # Strides for V tensor
              stride_ob, stride_oh, stride_ol1, stride_ol2, stride_od,  # Strides for Output tensor
              stride_bb, stride_bh, stride_bl1, stride_bl2,  # Strides for B tensor
              B, H, L,  # b=batch_size, h=num_heads, n=sequence_length
              D: tl.constexpr,  # Size of each attention head
              BLOCK_QL2: tl.constexpr,  # Block size for sequence dimension
              BLOCK_KL2: tl.constexpr,  # Block size for attention computation
              ):
    """
    Attention forward pass kernel: 
    Q: [B, H, L1, QL2, D]
    K: [B, H, L1, KL2, D]
    V: [B, H, L1, KL2, D]
    B: [B, H, QL2, KL2]
    (Note: in practice all L is equal since we work with a pw representation)


    Naive implementation:
    1. attn_logits = Q @ K.transpose(-2, -1) # [B, H, L, L, L]
    2. attn_logits = attn_logits / math.sqrt(head_dim)
    3. attn_logits = attn_logits + bjk # [B, H, L, L, L]
    4. attn_score = softmax(attn_logits, axis=-1) # [B, H, L, L, L]
    5. attn_out = attn_score @ V # [B, H, L, L, D]

    Of these, step 3 is not handled by out-of-the-box flash attention.
    """
    # Ensure block size doesn't exceed head dimension
    tl.static_assert(BLOCK_KL2 <= D, "Block size (BLOCK_KL2) must not exceed head dimension (D)")

    # Grid dimension explanation:
    # - program_id(0): Indexes along inner sequence length (L/BLOCK_QL2 blocks)
    #      the L along which we aggregate q and k
    # - program_id(1): Indexes along outer sequence length (L blocks)
    # - program_id(2): Indexes along batch*heads dimension (Z*H blocks)
    start_ql2 = tl.program_id(0) 
    off_l1 = tl.program_id(1)
    off_bh = tl.program_id(2)

    # Convert combined batch-head index into separate indices
    off_b = off_bh // H
    off_h = off_bh % H

    # Calculate base offset for current batch and head, and also first L dim
    # (which is effectively just another dim to batch along for now)
    # Note that we assume similar strides for Q, K, V, as the original
    # triton flash attention kernel does.
    qvk_offset = (
        off_b.to(tl.int64) * stride_qb +
        off_h.to(tl.int64) * stride_qh +
        off_l1.to(tl.int64) * stride_ql1
    )
    b_offset = (
        off_b.to(tl.int64) * stride_bb +
        off_h.to(tl.int64) * stride_bh
    )

    # Flash attention 2 slices Q instead of KV slicing, so that the output
    # is written by one warp/SM--do the same for triangular attention.

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(L, D),
        strides=(stride_ql2, stride_qd),
        offsets=(start_ql2 * BLOCK_QL2, 0), # Tile Q across blocks
        block_shape=(BLOCK_QL2, D),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(L, D),
        strides=(stride_vl2, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_KL2, D),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(D, L),
        strides=(stride_kd, stride_kl2),
        offsets=(0, 0),
        block_shape=(D, BLOCK_KL2),
        order=(0, 1),
    )
    B_pw_block_ptr = tl.make_block_ptr(
        base=B_pw + b_offset,
        shape=(L, L),
        strides=(stride_bl1, stride_bl2),
        offsets=(start_ql2 * BLOCK_QL2, 0),
        block_shape=(BLOCK_QL2, BLOCK_KL2),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(L, D),
        strides=(stride_ol2, stride_od),
        offsets=(start_ql2 * BLOCK_QL2, 0),
        block_shape=(BLOCK_QL2, D),
        order=(1, 0),
    )
    # Initialize per-block data structures:
    offs_ql2 = start_ql2 * BLOCK_QL2 + tl.arange(0, BLOCK_QL2)  
    offs_kl2 = tl.arange(0, BLOCK_KL2)

    # Initialize tracking variables for softmax computation
    m_i = tl.zeros([BLOCK_QL2], dtype=tl.float32) - float("inf")  # Max values for stability
    l_i = tl.zeros([BLOCK_QL2], dtype=tl.float32) + 1.0          # Sum for normalization
    acc = tl.zeros([BLOCK_QL2, D], dtype=tl.float32)      # Accumulated attention outputs

    pre_bias_qk_scale = sm_scale # 1/sqrt(head_dim)
    post_bias_qk_scale = 1.44269504 # 1/log(2), to account for log2 in softmax

    # Load query block - remains in SRAM throughout computation
    q = tl.load(Q_block_ptr)

    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q,
        K_block_ptr, V_block_ptr, B_pw_block_ptr,
        start_ql2, pre_bias_qk_scale, post_bias_qk_scale,
        BLOCK_QL2, D, BLOCK_KL2,
        offs_ql2, offs_kl2,
        L, V.dtype.element_ty == tl.float8e5
        )

    # Final processing:
    # 1. Compute logsumexp values for backward pass
    m_i += tl.math.log2(l_i)
    # 2. Normalize softmax by dividing by denominator
    acc = acc / l_i[:, None]
    # 3. Store outputs
    m_ptrs = M + off_bh * L + offs_ql2
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, b_pw, sm_scale):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        # Initialize output tensor
        o = torch.empty_like(q)

        # Initialize scaling factor storage
        # M is the max val for each query position--QL2.
        # Shape (B, H, L1, L2), equivalent to attn_scores.max(dim=-1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2], q.shape[3]), device=q.device, dtype=torch.float32)

        def grid(args):
            return (triton.cdiv(q.shape[-2], args["BLOCK_QL2"]), q.shape[-3], q.shape[0] * q.shape[1])

        ctx.grid = grid

        # Launch regular kernel
        _attn_fwd[grid](
            q, k, v, b_pw, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3), k.stride(4),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3), v.stride(4),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3), o.stride(4),
            b_pw.stride(0), b_pw.stride(1), b_pw.stride(2), b_pw.stride(3),
            B=q.shape[0], H=q.shape[1], L=q.shape[2],
            D=HEAD_DIM_K,
        )

        # Save tensors needed for backward pass
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o

attention = _attention.apply


@pytest.mark.parametrize("Z, H, L, HEAD_DIM", [(1, 2, 1024, 32)])
def test_op(Z, H, L, HEAD_DIM, dtype=torch.float16):
    """
    Test function for attention implementation
    Compares Triton implementation with PyTorch reference implementation
    """
    # Set random seed for reproducibility
    torch.manual_seed(20)

    # Initialize input tensors
    q = (torch.empty((Z, H, L, L, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, L, L, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, L, L, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    b = (torch.empty((Z, H, L, L), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    dout = torch.randn_like(q)

    
    ref_out = reference_tt_attn(q, k, v, b, sm_scale)

    # Compute reference gradients
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    ref_db, b.grad = b.grad.clone(), None

    # Triton implementation
    tri_out = attention(q, k, v, b, sm_scale).half()

    # Compare results--forward pass
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)

    backwards_pass_implemented = False
    if backwards_pass_implemented:
        tri_out.backward(dout)
        tri_dv, v.grad = v.grad.clone(), None
        tri_dk, k.grad = k.grad.clone(), None
        tri_dq, q.grad = q.grad.clone(), None

        # Verify gradients match
        assert torch.allclose(ref_dv, tri_dv, atol=1e-2)
        assert torch.allclose(ref_dk, tri_dk, atol=1e-2)
        assert torch.allclose(ref_dq, tri_dq, atol=1e-2)

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
# BATCH, N_HEADS, HEAD_DIM = 4, 32, 64 # To test on A100
B, H, D = 4, 8, 32 # To test on L4
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    if mode == "bwd":
        continue
    
    # Peak memory usage benchmark
    configs.append(
        triton.testing.Benchmark(
            x_names=["L"],
            x_vals=[2**i for i in range(4, 13)],
            line_arg="provider",
            line_vals=["triton", "vanilla-torch"], # "sdpa-kernel"],
            line_names=["Triton", "Vanilla PyTorch"], # "SDPA Kernel"],
            # line_vals=["triton-fp16", "vanilla-torch", "sdpa-kernel"],
            # line_names=["Triton [FP16]", "Vanilla PyTorch", "SDPA Kernel"],
            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
            ylabel="Memory (GB)",
            plot_name=f"fused-attention-batch{B}-head{H}-d{D}-{mode}-memory",
            args={
                "H": H,
                "B": B,
                "D": D,
                "mode": mode,
                "metric": "memory",
            },
        ))
    # Add latency benchmark
    configs.append(
        triton.testing.Benchmark(
            x_names=["L"],
            x_vals=[2**i for i in range(4, 13)],
            line_arg="provider",
            line_vals=["triton-fp16", "vanilla-torch"], # "sdpa-kernel"],
            line_names=["Triton [FP16]", "Vanilla PyTorch"], # "SDPA Kernel"],
            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
            ylabel="Time (ms)",
            plot_name=f"fused-attention-batch{B}-head{H}-d{D}-{mode}-latency",
            args={
                "H": H,
                "B": B,
                "D": D,
                "mode": mode,
                "metric": "time",
            },
        ))

@triton.testing.perf_report(configs)
def bench_attention(B, H, L, D, mode, provider, device=DEVICE, metric="tflops"):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16

    # Helper function to measure peak memory
    def get_peak_memory():
        if device.type == "cuda":
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB
        return 0

    # Reset memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        q = torch.randn((B, H, L, L, D), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((B, H, L, L, D), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((B, H, L, L, D), dtype=dtype, device=device, requires_grad=True)
        b = torch.randn((B, H, L, L), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3

        if "triton" in provider:
            # Handle FP8 case
            if mode == "fwd" and "fp8" in provider:
                q = q.to(torch.float8_e5m2)
                k = k.to(torch.float8_e5m2)
                # v should be contiguous as B, H, D, L instead of B, H, L, D
                v = v.permute(0, 1, 2, 4, 3).contiguous()
                v = v.permute(0, 1, 2, 4, 3)
                v = v.to(torch.float8_e5m2)
                b = b.to(torch.float8_e5m2)

            fn = lambda: attention(q, k, v, b, sm_scale)

            # Setup backward pass if needed
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)

            ms = triton.testing.do_bench(fn)
            peak_mem = get_peak_memory()

        elif provider == "vanilla-torch":
            fn = lambda: reference_tt_attn(q, k, v, b, sm_scale)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn)
            peak_mem = get_peak_memory()
        elif provider == "sdpa-kernel":
            # Disabled for now--kernel not available on the L4 I'm testing this on
            def fn():
                # Memory efficient SDPA kernel
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=b)
            ms = triton.testing.do_bench(fn)
            peak_mem = get_peak_memory()
        else:
            raise ValueError(f"Invalid provider: {provider}")

        if metric == "memory":
            return peak_mem
        elif metric == "time":
            return ms  # Return raw milliseconds
        else:  # tflops
            raise NotImplementedError("Haven't done the FLOPs metric for Triangular Attn yet")
            flops_per_matmul = 2.0 * B * H * L * L * D
            total_flops = 2 * flops_per_matmul
            if mode == "bwd":
                total_flops *= 2.5  # Additional computations for backward pass
            return total_flops * 1e-12 / (ms * 1e-3)

    except torch.cuda.OutOfMemoryError:
        # If we hit OOM, return None for vanilla PyTorch but continue for Triton
        if metric == "memory":
            return float('inf')  # Indicate OOM in memory benchmark
        else:
            return float('nan')  # Indicate OOM in FLOPS benchmark

def reference_tt_attn(q, k, v, b, sm_scale):
    """
    Reference implementation of attention for correctness.

    If a debugging_dict is provided, we will check intermediate values against the debugging_dict.
    """
    Z, H, L = q.shape[:3]
    # Step 1: QK^T. 
    qk = torch.matmul(q, k.transpose(-2, -1))

    # Step 2: Scale and add bias--attn scaling happens before bias addition
    qk_scaled = qk * sm_scale 
    qk_scaled_bias = qk_scaled + b.view(Z, H, 1, L, L)

    # Step 3a: Softmax (builtin)
    p = torch.softmax(qk_scaled_bias.float(), dim=-1).to(v.dtype)

    # Step 4: Output
    ref_out = torch.matmul(p, v)
    return ref_out

if __name__ == "__main__":
    # Run basic correctness test
    test_op(Z=2, H=4, L=128, HEAD_DIM=32, dtype=torch.float16)

    # Run benchmarks
    bench_attention.run(save_path="test_results/", print_data=True)