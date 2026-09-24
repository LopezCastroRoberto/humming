#pragma once

#include <humming/utils/all.cuh>

// Persistent flat raster over grouped-contiguous expert tiles. The fields
// below preserve the ordinary Humming pipeline/epilogue scheduler interface.
// Grouped-contiguous A and activation-scale loaders use the absolute m_offset
// and current_shape_m fields, matching the generic scheduler interface.
template <class Ctx>
class RsW4a8GroupedScheduler {
  using SharedStorage = typename Ctx::SharedStorage;
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;
  static constexpr uint32_t kNumExperts = Ctx::kNumExperts;
  static constexpr uint32_t kNumThreads = Ctx::kNumThreads;
  static constexpr uint32_t kNBlocks = CEIL_DIV(ProblemShape::N, BlockShape::N);

  SharedStorage &smem;
  CUtensorMap *tensor_map_buffer;
  int32_t current_iter = -1;
  uint32_t total_mn_blocks;
  uint32_t num_m_blocks;
  uint32_t num_blocks_in_group = 0;

  CUDA_INLINE uint32_t find_expert(uint32_t global_m_block) const {
    uint32_t lower = 0;
    uint32_t upper = kNumExperts;
    // A power-of-two count is common for EP, but not a property of grouped
    // GEMM. ceil(log2(E)) iterations also cover other expert counts.
    PRAGMA_UNROLL
    for (uint32_t step = 0;
         step < constexpr_log2(2 * kNumExperts - 1); ++step) {
      const uint32_t middle = (lower + upper) >> 1;
      if (global_m_block >= smem.expert_m_block_offset[middle]) {
        lower = middle;
      } else {
        upper = middle;
      }
    }
    return lower;
  }

  // Mirror of Scheduler<Ctx>::update_tensor_map_c: retarget the per-CTA C
  // tensor map's row dim to this expert's current_shape_m so the ws.cuh
  // TMA-C epilogue stores into the correct grouped-contiguous region.
  CUDA_INLINE void update_tensor_map_c() {
    if constexpr (Ctx::kUseTmaC) {
      if (threadIdx.x < 32) {
        tma_wait_store_group<0>();
        __syncwarp();
        if (threadIdx.x == 0) {
          tensor_map_replace_global_dim<1>(smem.tensor_map_buffer, current_shape_m);
          tensor_map_buffer[blockIdx.x] = smem.tensor_map_buffer[0];
          tensor_map_release_cta();
          tensor_map_acquire_cta(tensor_map_buffer + blockIdx.x);
        }
        __syncwarp();
      }
    }
  }

 public:
  uint32_t m_block_id = 0;
  uint32_t n_block_id = 0;
  uint32_t expert_id = 0;
  uint32_t old_expert_id = (1u << 30);
  uint32_t current_shape_m = 0;
  uint32_t m_offset = 0;

  // Generic Scheduler<Ctx> interface expected by the humming_ws.cuh driver.
  // Non-StreamK grouped-contiguous => these are compile-time constants; the
  // driver only reads them (never StreamK-writes) on this path.
  uint32_t slice_iters = ProblemShape::K / BlockShape::K;
  uint32_t k_block_id = 0;
  uint32_t slice_count = 1;
  uint32_t slice_id = 0;
  uint32_t locks_offset = 0;

  CUDA_INLINE RsW4a8GroupedScheduler(
      SharedStorage &smem_, uint32_t shape_m,
      bool use_int64_expert_layout,
      const uint32_t *expert_layout_ptr, const void *c,
      CUtensorMap *tensor_map_buffer_)
      : smem(smem_), tensor_map_buffer(tensor_map_buffer_) {
    static_assert(Ctx::kIsGroupedContiguousGemm);
    static_assert(!Ctx::kUseStreamK);
    static_assert(Ctx::kMultiCastSizeA == 1);
    static_assert(Ctx::kMultiCastSizeB == 1);
    static_assert(Ctx::kRasterGroupM == 1);
    // Generic-driver parity: this kernel reuses the ws.cuh epilogue, which
    // reads the per-CTA C tensor map the scheduler must stage + retarget.
    // Copy the base C tensor map into smem (mirrors Scheduler<Ctx> ctor).
    if constexpr (Ctx::kUseTmaC) {
      if (threadIdx.x == 0)
        smem.tensor_map_buffer[0] = reinterpret_cast<const CUtensorMap *>(c)[0];
      __syncwarp();
    }

    if (use_int64_expert_layout)
      legacy_load_2d<Ctx::kUseCpAsync, kNumExperts + 1, kNumThreads, 2, 1>(
          expert_layout_ptr, smem.expert_offset);
    else
      legacy_load_2d<Ctx::kUseCpAsync, kNumExperts + 1, kNumThreads, 1, 1>(
          expert_layout_ptr, smem.expert_offset);
    if constexpr (Ctx::kUseCpAsync) cp_async_commit_group();
    if constexpr (Ctx::kUseCpAsync) cp_async_wait_group<0>();
    __syncthreads();

    // One warp scans tile counts in 32-expert chunks. Unlike the original
    // one-warp initialization, this covers every expert, including EP1's 256
    // local experts, without serializing a 256-entry prefix sum on lane 0.
    if (threadIdx.x < 32) {
      const uint32_t lane = threadIdx.x;
      uint32_t carry = 0;
      for (uint32_t base = 0; base < kNumExperts; base += 32) {
        const uint32_t expert = base + lane;
        uint32_t tokens = 0;
        if (expert < kNumExperts) {
          const uint32_t next_offset =
              expert + 1 < kNumExperts ? smem.expert_offset[expert + 1] : shape_m;
          tokens = next_offset - smem.expert_offset[expert];
          smem.expert_tokens[expert] = tokens;
        }
        uint32_t blocks = CEIL_DIV(tokens, BlockShape::M);
        PRAGMA_UNROLL
        for (uint32_t delta = 1; delta < 32; delta <<= 1) {
          const uint32_t previous = __shfl_up_sync(0xffffffff, blocks, delta);
          if (lane >= delta) blocks += previous;
        }
        if (expert < kNumExperts)
          smem.expert_m_block_offset[expert + 1] = carry + blocks;
        carry += __shfl_sync(0xffffffff, blocks, 31);
      }
      if (lane == 0) {
        smem.expert_m_block_offset[0] = 0;
        smem.total_m_blocks[0] = carry;
      }
    }
    __syncthreads();
    num_m_blocks = smem.total_m_blocks[0];
    total_mn_blocks = num_m_blocks * kNBlocks;
  }

  CUDA_INLINE bool get_next_block() {
    const uint32_t next_mn_index =
        static_cast<uint32_t>(++current_iter) * gridDim.x + blockIdx.x;
    if (next_mn_index >= total_mn_blocks) return false;

    // Wider M tiles benefit from the shorter L2 raster; narrow tiles retain
    // the original 16-block grouping.
    constexpr uint32_t kNum1DBlocksPerGroup = BlockShape::M >= 160 ? 8 : 16;
    const uint32_t num_blocks_per_group = kNBlocks * kNum1DBlocksPerGroup;
    const uint32_t raster_group = next_mn_index / num_blocks_per_group;
    uint32_t first_m_block = raster_group * kNum1DBlocksPerGroup;
    uint32_t in_group = next_mn_index % num_blocks_per_group;
    num_blocks_in_group = MIN(kNum1DBlocksPerGroup, num_m_blocks - first_m_block);

    const uint32_t global_m_block =
        first_m_block + in_group % num_blocks_in_group;
    n_block_id = in_group / num_blocks_in_group;
    expert_id = find_expert(global_m_block);
    const uint32_t expert_first_block = smem.expert_m_block_offset[expert_id];
    m_block_id = global_m_block - expert_first_block;
    m_offset = smem.expert_offset[expert_id] + m_block_id * BlockShape::M;
    current_shape_m =
        smem.expert_offset[expert_id] + smem.expert_tokens[expert_id];
    if (old_expert_id != expert_id) update_tensor_map_c();
    old_expert_id = expert_id;
    return true;
  }
};
