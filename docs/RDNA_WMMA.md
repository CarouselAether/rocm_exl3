# RDNA WMMA reference

The WMMA layout and fragment rules this port relies on. Everything here comes
from reading the code as of `perf/phase0` (HEAD 97063b3). No GPU was run to
write it.

Every claim has one of these tags:

- **[code `file:line`]**: read directly from the source.
- **[NOTES `line`]**: from `exllamav3/exllamav3_ext/rocm/RDNA_NOTES.md`.
- **[UNVERIFIED]**: an inference or an open question. It is not a rule.

Path abbreviations:

| short | path |
|---|---|
| `W` | `exllamav3/exllamav3_ext/rocm/rdna_wmma.hip.h` |
| `GI` | `exllamav3/exllamav3_ext/rocm/quant/exl3_gemm_inner_rdna.hip.h` |
| `GK` | `exllamav3/exllamav3_ext/rocm/quant/exl3_gemm_kernel_rdna.hip.h` |
| `KM` | `exllamav3/exllamav3_ext/rocm/quant/exl3_kernel_map_rdna.hip.h` |
| `KMC` | `exllamav3/exllamav3_ext/rocm/quant/exl3_kernel_map_rdna.hip` |
| `MK` | `exllamav3/exllamav3_ext/rocm/quant/exl3_moe_kernel_rdna.hip.h` |
| `MS` | `exllamav3/exllamav3_ext/rocm/quant/exl3_moe_shape_rdna.hip.h` |
| `RC` | `exllamav3/exllamav3_ext/rocm/quant/reconstruct_rdna.hip` |
| `NOTES` | `exllamav3/exllamav3_ext/rocm/RDNA_NOTES.md` |
| `WC` | `rocm_tools/wmma_check.hip` |
| `GC` | `rocm_tools/gemm_check.hip` |

Measurement provenance: all measured facts come from gfx1151 (Strix Halo,
RDNA 3.5, wave32) under ROCm 7.2.4 [NOTES 4-6]. The other gfx11 targets in
`setup.py:288` (gfx1100, gfx1101, gfx1102, gfx1150) and the gfx12 targets
(gfx1200, gfx1201) have never been measured. See section 6.

---

## 1. Intrinsics, operand types, arch selection

### gfx11 (gfx1100/1101/1151): wave32 builtins used

| wrapper | builtin | A/B operand type | C/D type | line |
|---|---|---|---|---|
| `rdna_wmma::mma_sync` | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(B, A, C)` | `half16_t` (16 x `_Float16`) | `float8_t` (8 x float) | [code W:285] |
| `rdna_wmma::mma_sync_bf16` | `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(B, A, C)` | `bf16x16_t` (16 x `__bf16`) | `float8_t` | [code W:494] |
| `rdna_wmma::mma_sync_f16<opsel>` | `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(B, A, C, opsel)` | `half16_t` | `half16_t` (16 halves, 8 written per op) | [code W:516] |
| `rdna_wmma::mma_sync_i8<sa,sb,clamp>` | `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(sb, B, sa, A, C, clamp)` | `int32x4_t` (16 x int8 packed) | `int32x8_t` | [code W:616-619] |

Vector typedefs: [code W:71-72, 77-78, 84]. Fragment structs: [code W:86-194].

**Rule 1.1: operand order is `(B, A, C)` for every variant.** [code W:202-203,
264, 285, 455-458]. [NOTES 11-13]: "Wrong operand order produces a transposed
result, not a crash." The bf16 order was checked against a CPU reference: the
correct order gave 0/256 mismatches, and the swapped order gave 240
transpose-matches [code W:455-458].

**Rule 1.2: int8 sign flags pair with the vector that follows them.** The
builtin signature is `(s0, v0, s1, v1, C, clamp)`. Under `(B, A, C)` order the
first flag describes **B**. `mma_sync_i8<signed_a, signed_b>` takes the natural
order and swaps internally [code W:571-580, 608-620], [NOTES 29-33]. Verified:
`flags(1,0)` computed signed-B times unsigned-A [code W:579]. Signedness is a
property of the multiply only, because the loaders copy bytes verbatim
[code W:582-583].

**Rule 1.3: `opsel` is a compile-time immediate**, so it is a template
parameter [code W:507-508], [NOTES 24-26].

**Rule 1.4: bf16 has no C fragment of its own.** It accumulates to fp32 with
the f16 form's C layout and reuses `WmmaFragC` plus every store and accumulate
helper [code W:160-163, 460-461], [NOTES 23-24].

Other types in the header that are **not** WMMA fragments: `FragA`/`FragB`/
`FragC`/`FragC_h` (`Vec<half2,4>`, `Vec<half2,2>`, `Vec<float,4>`,
`Vec<half2,2>`) are PTX-layout-compatible types. They exist so that upstream's
`exl3_dq.cuh` and `codebook.cuh` compile unchanged [code W:15-18, 50-65].
`dq_dispatch` writes its output in `FragB`, which uses the **PTX** mma B layout
[code GI:38-43, 379-380].

### Arch selection

- Only `rdna_wmma::mma_sync` is arch-guarded. Under `__gfx1200__ ||
  __gfx1201__` its body is `__builtin_trap()` [code W:270-286]. The reason
  given: gfx11 WMMA encodings do not exist on gfx12, and LLVM fails with
  "Cannot select: llvm.amdgcn.wmma.f32..." [code W:271-272], [NOTES
  1060-1063].
- `mma_sync_bf16`, `mma_sync_f16` and `mma_sync_i8` are deliberately left
  unguarded. If a gfx12 build instantiates them, it fails loudly at compile
  time [code W:279-281], [NOTES 1066-1069].
- There is no gfx12 WMMA implementation. The planned gfx12 port uses
  "half-size fragments, no operand duplication across wave halves" and needs
  RDNA4 hardware to validate [code W:276-278], [NOTES 1073-1075].
- The runtime steer is in `rocm_py/__init__.py:577-610`. On a device whose
  `gcnArchName` contains `gfx120`, it sets `BlockSparseMLP.fused_mode_buffers
  = None` after `load_local`. MoE then runs per expert, which keeps the trap
  unreachable. `EXL3_ROCM_RDNA4_FUSED_MOE=1` skips the steer
  [code rocm_py/__init__.py:590-608], [NOTES 1068-1070].
- There is no wave64 variant. All builtins are `_w32`, and lanes are
  `threadIdx.x & 31` [code W:238, 251, …].
- gfx11 is not split by chip. Every gfx11 target gets the same code, because
  the only guard is the gfx12 macro [code W:270].

---

## 2. A/B fragment layout (gfx11, all variants)

The layout table is stated at [code W:205-211], [code W:564-569] (int8) and
[NOTES 15-21]. It was validated by `tests/wmma_smoke2.cpp` "256/256 cells,
0 transposed, 0 swapped, re-proven on ROCm 7.2.4" [code W:205-206].

| fragment | lane `L` (0..31) holds | element index within the lane's vector | per-lane storage | line |
|---|---|---|---|---|
| A (f16/bf16) | row `L % 16` of the 16x16 A tile | element `k` = `A[L%16][k]`, k = 0..15 (16 consecutive halves) | 16 x 16-bit = 32 B | [code W:207, 238-241, 471-472] |
| B (f16/bf16) | column `L % 16` of the 16x16 B tile (`B` is `[K, N]`) | element `k` = `B[k][L%16]`, k = 0..15 | 16 x 16-bit = 32 B | [code W:208, 244-261, 475-486] |
| A (int8) | row `L % 16` | byte `k` = `A[L%16][k]` | 16 bytes = "4 VGPRs" | [code W:566, 585-593] |
| B (int8) | column `L % 16` | byte `k` = `B[k][L%16]` | 16 bytes = "4 VGPRs" | [code W:567, 595-606] |

**Rule 2.1: lanes 16-31 replicate lanes 0-15 for A and B.** Every loader
indexes by `lane % 16` alone: `load_matrix_a` [code W:239], `load_matrix_b`
[code W:252], `load_matrix_a_bf16` [code W:472], `load_matrix_b_bf16`
[code W:481], `load_matrix_a_i8` [code W:592], `load_matrix_b_i8`
[code W:602]. Lane `L` and lane `L+16` therefore load byte-identical
fragments. This is how the code satisfies the gfx11 duplication requirement.
The code never states the requirement as a rule. The only textual evidence is
the gfx12 note, which contrasts gfx12's "no operand duplication across wave
halves" with gfx11 [code W:277], [NOTES 1074]. What gfx11 does when the halves
differ has never been tested ([UNVERIFIED], see section 6).

**Rule 2.2: A load is one 32-byte vector dereference.** The caller bakes the
sub-K column offset into the pointer, so the load is always 16 consecutive
halves, "one GLOBAL_LOAD_DWORDX8" [code W:219-222, 241]. Alignment was
measured on gfx1151. The dynamic LDS base is 32-byte aligned. A deliberately
misaligned by 16, 8, 4 or 2 bytes does not fault: the backend splits the
access [code W:223-232], [NOTES 48-50].

**Rule 2.3: B load is 16 strided scalar reads**, `src[k*stride + col]`. It
cannot be vectorised because each lane owns a column [code W:245, 258-261].

Element-to-VGPR packing: the code only calls the data "16 halves" in an
`ext_vector_type(16)`. That the halves pack two per VGPR, with even `k` in the
low half, is [UNVERIFIED]; the code does not state it. For int8, the only
statement is "4 VGPRs" [code W:566-567].

---

## 3. C/D accumulator layout

### fp32 (f16 and bf16 inputs) and int32 (int8 inputs)

| lane `L` | vector element `i` (0..7) holds `C[row][col]` with | line |
|---|---|---|
| any | `row = L % 16` | [code W:209, 296, 423] |
| any | `col = i*2 + col_base`, `col_base = (L >= 16) ? 1 : 0` | [code W:209-210, 297, 302, 423, 428] |

Lanes 0-15 therefore hold the even columns of their row, and lanes 16-31 hold
the odd columns of the same row. Each lane stores 8 values [code W:210]. int32
uses the same mapping [code W:568-569, 627-634], [NOTES 20].

Helpers that encode this layout, all with the same `row` / `col_base`
arithmetic:

- Unchecked: `load_accumulate_c` [W:290-305], `load_accumulate_c_half`
  [W:308-323], `store_matrix_c` [W:416-431], `store_matrix_c_half`
  [W:434-449], `store_matrix_c_i32` [W:622-635].
- Checked variants test `row < valid_rows` and `col < valid_cols`:
  [W:326-344], [W:347-365], [W:368-389], [W:392-413], [W:637-656].

Stores go one scalar at a time, stride 2 elements per lane, into a row-major
`C` with a caller-supplied row stride [code W:425-430].

### fp16 accumulate (`v_wmma_f16_16x16x16_f16`)

- The (row, col) mapping is the same as fp32: `row = L % 16`, element `i` at
  column `i*2 + col_base` [code W:500-505].
- **Rule 3.1: element `i` lives in half-slot `i*2 + opsel`** of the 16-half
  vector. A single op writes 8 of the 16 slots. With `opsel=0` only slots
  0,2,..,14 are written, with `opsel=1` only 1,3,..,15, and the unwritten slots
  keep their prior contents. This was verified on gfx1151 by dumping the raw
  fragment [code W:126-130, 151, 156].
- Consequence: two independent accumulators can share one fragment (one at
  `opsel=0`, one at `opsel=1`) [code W:136-139], [NOTES 24-26]. `clear_half<opsel>`
  and `get<opsel>(i)` implement this [code W:147-157]. The store helpers are
  `store_matrix_c_f16<opsel>` and `store_matrix_c_f16_checked<opsel>`
  [code W:519-555].
- A single-accumulator fp16 fragment is the same 8 VGPRs as fp32, so fp32 is
  strictly better unless the opsel packing is used [code W:132-135], [NOTES
  27-29], [code GI:45-48].
- Nothing in production uses the fp16-accumulate form. `GI` uses only
  `WmmaFragC` / `mma_sync` [code GI:303, 451].

---

## 4. Production path: LDS tiles into fragments (`exl3_gemm_kernel_inner`)

### Callers

- The dense cooperative GEMM and mgemm: `exl3_gemm_kernel` [code GK:79-81]
  and `exl3_mgemm_kernel` [code GK:276-278]. These are instantiated by the 24
  `quant/comp_units_rdna/exl3_comp_unit_*_cb*.hip` units via `GK`.
- Fused MoE: `moe_gemm_tile` → `exl3_gemm_kernel_inner` [code MK:151-161].
  It runs with `MOE_TILESIZE_K = 32` [code MS:20-24], so `TILEBLOCKS_K = 2`.
- Standalone check: `GC:96`.

### Fixed geometry

| constant | value | line |
|---|---|---|
| `TILESIZE_M` | 16 (static_assert) | [code GI:148] |
| threads per sub_k group | 256 = 8 waves (`NUM_WARPS`) | [code GI:71, 102, 147] |
| `TILESIZE_K` | multiple of 16. The RDNA shape table uses 16; MoE uses 32 | [code GI:149], [NOTES 176-179], [code MS:20-24] |
| `TILESIZE_N` | multiple of 128, and `TILEBLOCKS_N % NUM_WARPS == 0` (192 rejected) | [code GI:150-152], [NOTES 146-149] |
| `FRAGS_N_PER_WARP` | `TILEBLOCKS_N / NUM_WARPS`: one 16-wide WMMA N-block per fragment | [code GI:104-106] |
| blockDim | `256 * TILESIZE_K / 16`; `sub_k = threadIdx.x / 256` | [code GK:50], [code GI:171-174] |

Each wave owns `FRAGS_N_PER_WARP` consecutive 16x16 output blocks. Block index
is `warp_id * FRAGS_N_PER_WARP + n` [code GI:371, 502]. Each `sub_k` group
handles one 16-wide K slice of the K tile [code GI:361, 377].

### A: global → LDS → fragment

- LDS A tile: `TILESIZE_M` rows with row stride **`SH_A_STRIDE = TILESIZE_K +
  8` halves** [code GI:113, 115]. The copy is `uint4` (8 halves). The
  destination index is `m * (SH_A_STRIDE/8) + k`, so the row index has to be
  recovered [code GI:209-226, 336-340].
- The fragment load is `load_matrix_a(frag, sh1_a_ptr + m*16*SH_A_STRIDE +
  sub_k*16, SH_A_STRIDE)` [code GI:361-362]. Lane L reads row L%16, 16
  consecutive halves starting at column `sub_k*16` of the K tile.
- The stated reason for the padding: rows are 16 halves (32 B), and with 32
  4-byte banks an unpadded stride puts rows 0, 4, 8 and 12 on the same bank, a
  4-way collision. Padding by one 8-half group breaks that. It "replaces
  upstream's XOR swizzle" [code GI:108-112, 31-36]. The host LDS accounting
  mirrors it [code KMC:94-103].
- Rows at or beyond `size_m` are never written (`pred_a_gl` false,
  [code GI:225]), so stale LDS feeds those A rows into the WMMA. This is
  harmless for the valid rows because every output path skips `row >= size_m`
  [code GI:511, 532, 554]. The guard relies on C row `i` depending only on A
  row `i`, which holds for any matmul.

### B: quantized → dequant (PTX layout) → shuffle transpose → LDS → fragment

1. All 32 lanes run `dq_dispatch<bits, cb>(b_quant, lane_id << 3, frag0,
   frag1)`. Each lane gets 8 halves of one 16x16 block in the PTX mma B
   layout [code GI:376-380].
2. The dq-native layout that `GI` relies on: lane L holds k-rows
   `(L%4)*2 + {0,1,8,9}` for n-column `(L/8)*2 + ((L>>2)&1)` and that column
   plus 8. That is the arithmetic of [code GI:400-427] and is stated at
   [NOTES 320-321] (where "rows" means k and "columns" means n).
3. `__shfl_down(x, 4, 32)` of all four `half2`s [code GI:385-388]. Lanes with
   bit 2 clear (0-3, 8-11, 16-19, 24-27) combine their own values (column
   `c0`) with those of lane+4 (column `c0+1`). They write all 256 elements of
   the 16x16 block into the warp-private LDS tile `B_lds[k][n]`, where
   `r0 = (L%4)*2`, `c0 = (L/8)*2`, and `c1 = c0 + 8`
   [code GI:390-427]. `RC:46-74` uses the same pattern, with `c0 = L/8` in
   half2 units.
4. `mem_fence()` (`s_waitcnt(0)`) → `load_matrix_b(frag_b, B_lds,
   SH_B_DQ_STRIDE)` → `mem_fence()` [code GI:438-440]. The justification:
   B_lds is warp-private and the wave32 lanes are converged here, so draining
   LDS is enough and a `__syncthreads` would be stronger than needed
   [code GI:432-437]. The `mem_fence` definition is at [code W:790-800].
5. **B staging stride `EXL3_GEMM_SH_B_DQ_STRIDE = 18` halves** [code KM:82-102],
   [code GI:117-120]. A stride of 17 put the active-lane groups 0-3 vs 16-19
   and 8-11 vs 24-27 on the same banks, measured at ~24% stall time by PMC.
   18 halves is 9 dwords, and 9 is coprime with 32 [code KM:84-88],
   [NOTES 139-143]. Four consumers must agree on the value: the kernel,
   `exl3_gemm_smem_bytes`, the launch, and `GC`. A mismatch once
   under-allocated LDS by 256 B [code KM:90-101], [NOTES 144-146].
6. Staging-tile sizing: `NUM_WARPS * TILEBLOCKS_K * 16 * SH_B_DQ_STRIDE`,
   indexed by the block-wide warp id `warp_id + sub_k*NUM_WARPS`
   [code GI:121-128, 372-375], [NOTES 150-155]. This was a real bug at
   `TILEBLOCKS_K = 2` [code MS:26-31].

### C: fragment → output

- `c_row = lane % 16` and `c_col_base = (lane >= 16) ? 1 : 0`
  [code GI:176-178].
- The output tile position of element `j` of fragment `n` is `row = m*16 +
  c_row` and `col = (warp_id*FRAGS_N_PER_WARP + n)*16 + j*2 + c_col_base`
  [code GI:499-503, 510, 517].
- Split-K partial sums: `read_sum_gl` / `write_sum_gl` do scalar global
  read-modify-write at `row * size_n_stride + col` in fp32 or fp16
  [code GI:505-545].
- The cross-sub_k reduction (`TILEBLOCKS_K > 1`) stages raw fragment elements
  per thread in `sh_c` at `8 * FRAGS_N_PER_WARP * t`. Both sides index by `t`
  [code GI:467-497], [NOTES 156-160]. It needs no layout knowledge because
  partner threads share a lane id and so share the fragment layout.
- Hadamard output (`shmem_out_had`, used only by `exl3_gemm_kernel`) writes
  fp32 row-major `sh_c[row * TILESIZE_N + col]` with no padding
  [code GI:547-561], and then `had_*_r_128_inner` runs per 128-column chunk
  [code GI:563-587].

### LDS budget

The static_assert checks `2*SH_STAGES*sh_a + 2*SH_STAGES*sh_b + 2*sh_b_dq +
4*sh_c <= SMEM_MAX` [code GI:154-161]. The 64 KB default for gfx1151 is at
[code KM:38-67]. The per-arch budgets in `setup.py:299-303` are 92160 for
gfx110x and gfx120x, and 65536 for gfx1150/1151.

---

## 5. Deliberate differences from the CUDA/NVIDIA mma logic

| # | NVIDIA / upstream | RDNA port | reason given | cite |
|---|---|---|---|---|
| D1 | `mma.sync.m16n8k16`: 8 halves/lane A, 4 floats/lane C | WMMA 16x16x16: 16 halves/lane A, 8 floats/lane C | The fragment layouts do not correspond, so every C-touching path was re-derived rather than shimmed | [code GI:14-24], [code W:212-214] |
| D2 | 2 fragments per 16-wide N block (N=8) | 1 fragment per 16-wide N block | WMMA N = 16 | [code GI:104-106] |
| D3 | `ldmatrix` + XOR swizzle of the A tile in smem | Plain A tile with padded stride `TILESIZE_K+8`, read by `load_matrix_a` | The swizzle is meaningless for 16-consecutive-half-per-lane reads, and padding fixes the 4-way bank collision | [code GI:16, 31-36, 108-113] |
| D4 | dq fragments fed straight into mma B | dq → `__shfl_down 4` transpose → warp-private LDS (stride 18) → `load_matrix_b` | The WMMA B fragment needs lane L to hold column L%16, but dq emits the PTX layout | [code GI:38-43, 365-440] |
| D5 | fp16-accumulate path on sm_86 | Always fp32 accumulate | RDNA has no fp32 rate penalty, and gfx11 fp16 C is the same 8 VGPRs | [code GI:45-48] |
| D6 | `cp.async` + wait/fence pipeline | Synchronous `uint4` copies; stage wait = `__syncthreads` | RDNA 3.5 has no async global→LDS at this width. This costs the load/compute overlap | [code GI:17-18, 50-54, 329, 621-623] |
| D7 | Operand order `(A, B)` | Builtin takes `(B, A, C)` | Empirical; the wrong order gives a transposed result | [code W:202-203], [NOTES 11-13] |
| D8 | `TILESIZE_M` 16/32/64 | 16 only (static_assert) | 32/64-row MoE tiles were not ported | [code GI:5-6, 148], [code MK:7-8] |
| D9 | `SMEM_MAX` ~90-100 KB | 64 KB default, clamped at runtime | gfx1151 LDS is 64 KB/workgroup | [code KM:38-67], [NOTES 48-50] |

---

## 6. Unverified rules and suspicious numbers (needs a test; nothing was changed)

1. **`SH_A_STRIDE = TILESIZE_K + 8` [code GI:113]: the "breaks that" claim is
   not measured.** Unlike stride 18, no PMC figure is cited. Arithmetic
   ([UNVERIFIED]): at TK=16 the stride is 24 halves = 12 dwords, so row start
   banks are `12r mod 32` = 0,12,24,4,16,28,8,20,0,… and rows r and r+8 still
   share a start bank (2-way, down from 4-way). At TK=32 (MoE) it is 20 dwords,
   `20r mod 32`, which also repeats at r+8. Whether the 32-byte read is
   serviced so that this matters needs a PMC (`SQ_LDS_BANK_CONFLICT`) run.
2. **Stale comment:** [code GI:382-384] says "stride 17 to avoid bank
   conflicts". The code uses `EXL3_GEMM_SH_B_DQ_STRIDE = 18` [code KM:102], and
   17 is the stride documented as wrong.
3. **"Fused-MoE is the only live WMMA user" [NOTES 1062-1065] disagrees with
   the code.** `exl3_gemm_kernel` and `exl3_mgemm_kernel` [code GK:79, 276]
   call `exl3_gemm_kernel_inner`, which calls `rdna_wmma::mma_sync`
   [code GI:451]. They are instantiated by the 24 GEMM comp units. On gfx120x
   those kernels compile to `__builtin_trap()`, and `rocm_py` steers only MoE
   [code rocm_py/__init__.py:590-608]. The code read for this document does
   not establish whether m > 1 dense GEMM dispatch can reach them on gfx12.
   Needs confirmation.
4. **Lane replication (Rule 2.1) is satisfied but never tested for
   necessity.** No test loads different data into lanes 16-31. Nobody has
   measured whether gfx11 hardware reads lanes 16-31 at all, requires them to
   match, or produces wrong results when they differ. A future "optimisation"
   that loads only 16 lanes, or splits K across wave halves, would not be
   caught.
5. **Why `(B, A, C)` + `row = L%16` works.** [UNVERIFIED hypothesis, not in
   code]: swapping the operands makes the hardware compute the transpose, so
   the ISA's column-per-lane D layout reads back as row-per-lane. The code
   treats the pair only as an empirical fact. If either the order or the C
   helper is ever changed alone, the result is transposed.
6. **VGPR packing** of the 16-half A/B vector (two per VGPR, low half = even
   k), and of the fp16-accumulate slots (slot `2i+opsel` = VGPR i, low/high
   half), is not stated in code. The code only speaks in vector-slot terms
   [code W:126-130].
7. **All gfx11 targets other than gfx1151 are assumed to have the same
   layout.** `setup.py:288` builds for gfx1100/1101/1102/1150 with no
   separate guard, and every measurement is gfx1151-only [NOTES 4-6].
8. **gfx12 layout is completely unknown.** "Half-size fragments, no operand
   duplication" is a comment-level expectation [code W:276-278], [NOTES
   1073-1075] and has never run on hardware.
9. **Warp-private B_lds ordering depends on `s_waitcnt(0)` alone**
   [code GI:432-440], with no `wave_barrier`. [NOTES 125-131] records that
   `wave_barrier` alone is insufficient for cross-lane LDS exchange. The inverse
   case (waitcnt without a scheduling barrier, relying on convergence plus
   pointer aliasing) is argued in the comment but not independently tested
   beyond `GC` passing.
10. **Magic number `0.088388347648f`** (1/sqrt(128)) in the hadamard output
    [code GI:579, 584]. This is correct by arithmetic, but a WMMA gate test
    does not cover it.
11. **Dead re-proof path:** the header says to re-prove with
    `../../rocm_exl3_legacy/tests/wmma_smoke2.cpp` [code W:27-30, 205]. That
    file is not in the repo; the live equivalent is `WC`. `ROCM_PORT_MAP.md`
    still calls the header `rdna_wmma.hip (368)` [rocm_tools/ROCM_PORT_MAP.md:105].
12. **int8 load alignment:** `load_matrix_a_i8` dereferences a 16-byte
    `int32x4_t` at `A + (L%16)*stride` bytes [code W:592]. The alignment
    measurement covers only the f16 A load [code W:225-232]. The int8 path has
    no production caller [code exl3_gemv_int8_rdna.hip:22-24], so this is
    latent.
13. **`clamp = true` for int8** is never exercised anywhere.

---

## 7. Test coverage

### What `rocm_tools/wmma_check.hip` tests today

It compiles the shipped header [code WC:3-4, 21]. It runs one wave (1 block x
32 threads), one 16x16x16 tile, with row stride 16 for A, B and C. Inputs are
non-symmetric so that operand-order and transpose bugs are distinguishable
[code WC:5-8, 137-143]. The checks are tolerance-based:

| check | wrappers | tolerance | line |
|---|---|---|---|
| f32 mma + store | `load_matrix_a/b`, `mma_sync`, `store_matrix_c` | abs 0.06 | [WC:27-35, 164-174] |
| orientation | same, compares against `ref^T` | abs 0.06 | [WC:172, 175] |
| fp16 store | `store_matrix_c_half` | abs 0.5 | [WC:37-45, 177-183] |
| accumulate | `load_accumulate_c` (fp32) | abs 0.06 | [WC:48-57, 185-191] |
| bounds-checked store | `store_matrix_c_checked` (vr=9, vc=5), guard region untouched (exact) | abs 0.06 / exact | [WC:60-68, 193-209] |
| fp16 accumulate opsel=0/1 | `mma_sync_f16`, `store_matrix_c_f16` | rel 0.004 + 0.02 | [WC:82-91, 211-227] |
| dual accumulator | opsel=0 holds A·B, opsel=1 holds B·A in one fragment | rel | [WC:93-108, 229-252] |
| bf16 → fp32 + orientation | `*_bf16`, `store_matrix_c` | rel 0.01 + 0.02 | [WC:72-80, 254-292] |
| int8, all 4 signedness combos, bytes > 127 | `*_i8`, `mma_sync_i8<sa,sb>`, `store_matrix_c_i32` | **exact** | [WC:114-123, 294-336] |

`GC` separately exercises the production `exl3_gemm_kernel_inner` against
upstream's reconstruct mapping. It uses random data with a relative tolerance
of `0.02*|ref| + 0.05`, and only these settings: `TK=16`
(`TILEBLOCKS_K = 1`), `m=16`, `c_fp32=true`, `shmem_out_had=false`. The
N/bits/cb/slices combinations are listed at [code GC:272-279]. Setup and
coverage are at [code GC:1-17, 102-198].

### What a gate test (known A/B → exact expected C through the production WMMA path) still needs

- **Exactness.** Every fp16/bf16 check above uses a tolerance, and only int8
  is exact. A gate should pick inputs whose products and partial sums are
  exactly representable in fp32, such as small integers or powers of two. It
  should then compare bit-for-bit, so that a single swapped element cannot
  hide inside a tolerance.
- **The production B path.** `WC` feeds B from a plain row-major array. The
  transpose chain (dq_dispatch → `__shfl_down 4` → B_lds stride 18 →
  `load_matrix_b`) is only covered by `GC`, and only with a tolerance. An exact
  gate through that path can use **one-hot A rows** (`A[i][k] = 1` iff `k =
  p(i)`), so that `C[i][:] == W[p(i)][:]` bit-exactly against
  `ref_reconstruct`. This also checks `out_col` and `c_row` mapping, and
  multi-k-tile accumulation (the partial sums stay exact).
- **Padded/offset A.** `SH_A_STRIDE = TK+8` and the `sub_k*16` column offset
  [code GI:361] are untested in `WC`, and appear in `GC` only at TK=16.
- **`TILEBLOCKS_K = 2`** (MoE's live geometry: split sub_k, `threadblock_reduce`,
  block-wide `sh_b_dq` indexing). No standalone test covers it, and both past
  defects lived there [code MS:26-31].
- **`size_m < 16`** (partial tiles with stale A rows; output rows must stay
  untouched), and `c_fp32 = false` (fp16 output, `read_sum_gl` in half).
- **Sliced mode** (`size_n_stride != size_n`) [code GI:92-97, 186].
- **`shmem_out_had = true`**, the path `exl3_gemm_kernel` always uses
  [code GK:79-81] [code KMC:105-107]. It is not exact because of the Hadamard,
  but it needs at least a tolerance check.
- **Multi-wave blocks.** `WC` runs 32 threads. Production runs 256 or 512, so
  `threadIdx.x & 31` and per-warp B_lds offsets need coverage at block scale.
- **Chained mma across K.** `WC` issues one `mma_sync` per fragment. Nothing
  checks accumulation over several K steps into the same `WmmaFragC` in
  isolation.
- **Untested helpers:** `load_accumulate_c_half`, `load_accumulate_c_checked`,
  `load_accumulate_c_half_checked`, `store_matrix_c_half_checked`,
  `store_matrix_c_f16_checked`, `store_matrix_c_i32_checked`, `clear_half`,
  and i8 `clamp=true`.
- **A negative test for lane replication** (section 6 item 4): load
  deliberately different data into lanes 16-31 and record what the hardware
  does, so that Rule 2.1 is known rather than assumed.
- **Per-arch runs:** gfx1100/1101 (gfx11) and gfx1200/1201 (gfx12, where
  `mma_sync` is expected to trap) are unverified until run on hardware.

---

## 8. WMMA fp32 accumulation is not IEEE-exact

Measured on gfx1151 on 2026-09-27 with the pip ROCm SDK (.venv10, AMD clang
23.0.0git), using `rocm_tools/wmma_gate.hip`. The shipped header was not changed.

**Finding.** `v_wmma_f32_16x16x16_f16`, `v_wmma_f32_16x16x16_bf16` and
`v_wmma_f16_16x16x16_f16` do not accumulate like a sequence of IEEE adds, even
when every product and the exact sum are representable in fp32.

- Each product of fp16 inputs is exact: 3*3 gives 9, and 16 x (1*1) gives 16.
- The multi-term sum is not exact.
- int8 (`v_wmma_i32_16x16x16_iu8`) is bit-exact against integer math.

**First gate run.** The inputs were small integers and dyadics, and the check
was `==`. Every case with f16 or bf16 inputs failed on a subset of cells, and
every int8 case passed. For example:

- got -4.25000048, want -4.25
- got 2.38e-06, want 0

No case matched the transposed expectation. The layouts are correct; the
arithmetic is not IEEE.

**Accumulator probe.** The gate's INFO probe runs one mma with K = 16. The exact
sum is representable in fp32 in every row. Emax is the exponent of the largest
|product|, and one unit below is 2^(Emax-23), which is ulp_f32 of the largest
product.

| dot product | exact | got | error, in 2^(Emax-23) |
|---|---|---|---|
| 9 - 9 | 0 | -2.38e-07 | -0.25 |
| 1 - 1 | 0 | -5.96e-08 | -0.5 |
| 1024 - 1024 | 0 | -6.10e-05 | -0.5 |
| 4 - 1 | 3 | 2.99999976 | -0.5 |
| 8 - 1 | 7 | 6.99999952 | -0.5 |
| 9 - 9 + 9 - 9 | 0 | 0 | 0 |
| -1 - 1 (one sign) | -2 | -2 | 0 |
| 1024 + 2^-13 (one term) | 1024.000122 | 1024.000122 | 0 |
| 1024 + 8 x 2^-16 | 1024.000122 | 1024 | -1 |
| 1024 + 14 x 2^-14 | 1024.000854 | 1024.000732 | -1 |
| 1024 + 4x2^-12 + 4x2^-13 + 6x2^-14 | 1024.001831 | 1024.001709 | -1 |
| 1024 + 15 x 2^-12 | 1024.003662 | 1024.003662 | 0 |
| -1024 + 8 x 2^-16 | -1023.999878 | -1023.999756 | +1 |
| 1 + 14 x 2^-24 | 1.000000834 | 1.000000715 | -1 |

**Reading of the probe.** This is a hypothesis consistent with the data, not a
confirmed model.

- **Alignment truncation.** Products far below the largest one are aligned to
  its exponent, and the bits below a window of roughly 2^(Emax-24..-23) are
  dropped. They are dropped even when their exact sum would have been
  representable: 8 x 2^-16 added to 1024 is lost completely. The coordinator's
  alignment-truncation hypothesis is therefore **confirmed** for small terms.
  A pure per-term truncation would lose all 14 x 2^-14; only 2^-13 of their
  sum was lost. So a few guard bits survive, or pairs are summed before
  truncation.
- **Mixed-sign bias.** A single subtraction such as 4 - 1 or 1 - 1 is off by
  -0.25 to -0.5 x 2^(Emax-23). This is not explained by truncating the terms,
  because the terms are exact integers. The pattern looks like a
  negation/rounding artefact in a wide fixed-point adder. Balanced pairs
  (9 - 9 + 9 - 9) cancel it.
- **Size of the error.** The error per mma is at most about one ulp_f32 of the
  largest intermediate magnitude. It is not proportional to the result, so a
  result that is exactly 0 can come back as ±2^-22.

**Consequences.**

- Bit-exact equality against IEEE math is not a valid oracle for any f16 or
  bf16 WMMA path. This includes the "exact in any order" claim for small
  integers.
- `rocm_tools/wmma_gate.hip` therefore checks f16 and bf16 cases in two ways:
  1. **Math check.** A per-cell bound against a double reference: 16 x
     ulp_f32(M_s) per 16-deep K step, where M_s is the largest of |C in|, the
     products, the exact partial sums and |C out|. fp16 accumulation adds
     ulp_f16(result) per step, and an fp16 store adds ulp_f16(want). The
     transpose diagnostic is kept.
  2. **Golden check.** A bit-for-bit comparison against
     `rocm_tools/wmma_gate.golden`, recorded with `--record` on gfx1151. This is
     the regression gate.
- Measured maximum error, in ulp_f32 of the largest intermediate magnitude:

  | case | max error |
  |---|---|
  | f32 K=64 cases, including LDS and 8-wave | 4.5 ulp |
  | K=16 f32 store / accumulate cases | 2.0 ulp |
  | bf16 K=32 | 2.0 ulp |
  | f16 accumulate cases | 0.5 ulp |
  | f16 store | 0.5 ulp |
  | C identity (A = 0) | 0, bit-exact |
  | int8 | 0, bit-exact |

  All 27 cases pass. Two default-mode runs were bit-identical to the golden file
  and to each other.
- Tolerances elsewhere, such as `wmma_check.hip` and `gemm_check.hip`, are
  justified by this measurement and not only by fp16 input rounding.
- Upstream CUDA mma may not share these semantics. Bitwise A/B comparisons
  between an NVIDIA reference and this port cannot be expected to match on WMMA
  paths.

**Related INFO result (section 6 item 4).** In this run, lanes 16-31 were given
A or B data that differed from lanes 0-15. The even C columns matched the
lanes 0-15 data in 128/128 cells. The odd C columns matched the lanes 16-31 data
in 126/128 cells for divergent A and 123/128 for divergent B. The probe tests
the lanes 0-15 product first, so the remaining cells were counted as matching it
within the bound; they are probably cells where the two products coincide.
No cell matched neither. The hardware does read both halves: lanes 0-15
feed the even output columns and lanes 16-31 feed the odd ones. The replication
in Rule 2.1 is therefore required for correctness, not merely harmless. This
was measured once, on gfx1151 only.

### 8.x Confirmation: the behaviour is inherent to the silicon (2026-09-27)

AMD's instruction-level emulator had the same mismatch, and fixed it by modelling the hardware exactly:
- ROCm/rocm-systems issue #12056: "`V_DOT2_F32_{F16,BF16}` and WMMA do not match gfx1151 silicon".
- PR #12120: "Match RDNA3 and RDNA4 DOT2 and WMMA arithmetic to hardware".

The PR's model:
- RDNA3 `v_wmma_f32_16x16x16_f16` runs as **eight sequential DOT2 steps** with architecture-specific integer (fixed-point) accumulation. RDNA4 runs as four DOT4 steps.
- Each architecture has its own subnormal handling and rounding boundaries.
- The model matches silicon bit for bit (1.2e8 F32 outputs checked).

No mode or flag changes it. That explains the truncation and the mixed-sign bias measured above.

**Decision: no mitigation.**
- An exact result would need split-precision emulation (hi/lo fp16 operand split, about 3 WMMAs per product), which triples matrix cost.
- The effect is 1–5 ulp_f32 of the largest intermediate. That is about 4000× smaller than the fp16 input quantization (2^-11 relative), and far below EXL3 weight quantization error.
- Correctness is guarded by the bounded check plus the golden-file regression (`rocm_tools/wmma_gate.hip`), not by IEEE bit-exactness.
- RDNA4 (gfx12) will round differently again, so it needs its own golden file if it is ever tested.
