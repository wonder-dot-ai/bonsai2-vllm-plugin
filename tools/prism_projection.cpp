// Small instrumented graph harness; all numerical operations execute in the
// pinned Prism libraries. llama_mul_mat_hadamard is included from that source.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "llama-impl.h"
#include "ggml-quants.h"
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C" const char * bonsai_prism_commit() { return ggml_commit(); }

// Host arrays use [batch, width] and row-major packed [rows, width/128*34].
// perm_nk/perm_rep enable the ssm_out feature permutation from build_lora_mm.
extern "C" int bonsai_projection(
    const void * packed, const float * signs, const float * input,
    int width, int rows, int batch, int block, int perm_nk, int perm_rep,
    int cuda, float * rotated_out, float * packed_out, float * dense_out) {
    try {
        if (width <= 0 || rows <= 0 || batch <= 0 || block <= 0 ||
            (block & (block-1)) || width % block || width % 128 ||
            perm_nk <= 0 || perm_rep <= 0 || width % (perm_nk * perm_rep)) return 1;
        using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
        using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
        using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
        backend_ptr backend(cuda ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init(),
                            ggml_backend_free);
        if (!backend) return 2;
        if (!cuda) ggml_backend_cpu_set_n_threads(backend.get(), 4);
        ggml_init_params params = {ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(128, false)*2,
                                   nullptr, true};
        context_ptr ctx(ggml_init(params), ggml_free);
        if (!ctx) return 3;
        auto * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_PQ2_0, width, rows);
        auto * f = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, batch);
        auto * s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
        auto * h = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, block, block);
        ggml_set_input(x);
        auto * cur = x;
        if (perm_rep > 1) {
            cur = ggml_reshape_4d(ctx.get(), cur, width/(perm_nk*perm_rep), perm_nk, perm_rep, batch);
            cur = ggml_cont(ctx.get(), ggml_permute(ctx.get(), cur, 0, 2, 1, 3));
            cur = ggml_reshape_2d(ctx.get(), cur, width, batch);
        }
        cur = ggml_mul(ctx.get(), cur, s);
        auto * rotated = llama_mul_mat_hadamard(ctx.get(), cur, h);
        ggml_set_output(rotated);
        auto * y = ggml_mul_mat(ctx.get(), w, rotated);
        auto * yf = ggml_mul_mat(ctx.get(), f, rotated);
        ggml_mul_mat_set_prec(yf, GGML_PREC_F32);
        ggml_set_output(y);
        ggml_set_output(yf);
        auto * transform_graph = ggml_new_graph_custom(ctx.get(), 128, false);
        ggml_build_forward_expand(transform_graph, rotated);
        auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
        ggml_build_forward_expand(graph, y);
        ggml_build_forward_expand(graph, yf);
        buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()),
                          ggml_backend_buffer_free);
        if (!buffer) return 4;
        std::vector<float> dense(size_t(width)*rows);
        dequantize_row_pq2_0(static_cast<const block_pq2_0 *>(packed), dense.data(), dense.size());
        std::vector<float> had(size_t(block)*block);
        const float scale = 1.0f/std::sqrt(float(block));
        for (int i=0; i<block; ++i)
            for (int j=0; j<block; ++j)
                had[size_t(i)*block+j] = (__builtin_popcount(unsigned(i & j)) & 1) ? -scale : scale;
        ggml_backend_tensor_set(w, packed, 0, ggml_nbytes(w));
        ggml_backend_tensor_set(f, dense.data(), 0, ggml_nbytes(f));
        ggml_backend_tensor_set(x, input, 0, ggml_nbytes(x));
        ggml_backend_tensor_set(s, signs, 0, ggml_nbytes(s));
        ggml_backend_tensor_set(h, had.data(), 0, ggml_nbytes(h));
        if (ggml_backend_graph_compute(backend.get(), transform_graph) != GGML_STATUS_SUCCESS) return 5;
        ggml_backend_tensor_get(rotated, rotated_out, 0, ggml_nbytes(rotated));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 6;
        ggml_backend_tensor_get(y, packed_out, 0, ggml_nbytes(y));
        ggml_backend_tensor_get(yf, dense_out, 0, ggml_nbytes(yf));
        return 0;
    } catch (...) { return 7; }
}
