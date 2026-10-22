#include "llama-moe-slot.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-experts.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace {
// Parallel pread threads per layer wave.
//
// This path is bandwidth-bound (measured: on-device decode time tracks total bytes
// read, not critical-path bytes), so whether the storage can serve more concurrent
// reads decides whether raising this helps at all. It is a tunable rather than a
// constant so it can be swept on device without a rebuild -- set
// LLAMA_MOE_IO_THREADS. Defaults to 4, the value all earlier measurements used.
constexpr int kMoeSsdIoThreadsDefault = 4;

static int moe_ssd_io_threads() {
    static const int n = []() {
        const char * s = std::getenv("LLAMA_MOE_IO_THREADS");
        const int v = s ? std::atoi(s) : 0;
        return (v > 0 && v <= 64) ? v : kMoeSsdIoThreadsDefault;
    }();
    return n;
}

// Sentinel values for slot_to_expert.
//   kSlotFree    -1: empty slot, available for allocation
//   kSlotPending -2: claimed by a prefetch worker; contents are not written yet. The main path must skip it
//                    (neither overwrite it nor treat it as a valid cache entry).
constexpr int32_t kSlotFree    = -1;
constexpr int32_t kSlotPending = -2;

// One I/O wave per layer. A layer's unique set never exceeds n_slots (<= 256), so this is
// always a single wave: the wave loop only adds a join + thread respawn per boundary, and
// the disk idles across it. Measured 2026-09-12 (paired A/B, 8/8 reps positive): 13 waves ->
// 1 wave is worth +3.65% prefill (261.9 -> 271.2 t/s), at zero memory cost.
// Kept as a named constant (rather than deleting the loop) so the cap stays explicit.
constexpr int kMoeSsdIoChunk = 256;

static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Xcode can capture stderr (as with [MBMtmd]). llama-cli callbacks swallow plain LLAMA_LOG_INFO.
static void moe_log(const char * fmt, ...) {
    char buf[768];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
//    fprintf(stderr, "[moe-ssd] %s\n", buf);
//    fflush(stderr);
    LLAMA_LOG_INFO("[moe-ssd] %s\n", buf);
}

static bool park_window_enabled() {
    const char * e = std::getenv("LLAMA_MOE_PARK_WINDOW");
    if (!e || !*e) {
        return true;
    }
    return e[0] != '0';
}

using metal_buf_fn = bool (*)(ggml_backend_buffer_t);

static metal_buf_fn lookup_metal_buf_fn(ggml_backend_buffer_t buf, const char * name) {
    if (!buf) {
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
    if (!buft) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev) {
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) {
        return nullptr;
    }
    return (metal_buf_fn) ggml_backend_reg_get_proc_address(reg, name);
}
} // namespace

// Next-layer router × current hidden (attn_post_norm). Decode only. LLAMA_MOE_LOOKAHEAD_PREFETCH=1.
// Predicted ids are preadv'd into a ping-pong tail of the shared bank after install, overlapping
// the current layer's GPU work. Hits remap for free; misses still install on the critical path.
struct lookahead_expert_prefetch {
    static constexpr int32_t kMaxDepth = 16;

    struct hit {
        int32_t expert;
        int32_t slot;
    };

    bool enabled = false;
    int32_t depth = 8;
    /// Number of experts actually read from disk; 0 = predict only, without disk reads (use predictions to guide eviction).
    /// Controlled by LLAMA_MOE_PREFETCH_READ; defaults to depth (legacy behavior).
    int32_t read_n = 0;
    int32_t n_slots = 0;
    int32_t n_expert = 0;
    int32_t last_moe_layer = -1;

    int32_t slot_floor() const { return n_slots - 2 * depth; }
    int32_t bank_base(int k) const { return slot_floor() + depth * (k & 1); }

    std::thread th;
    std::mutex mtx;
    std::condition_variable cv_go;
    std::condition_variable cv_done;
    bool quit = false;
    bool job_pending = false;
    bool job_running = false;
    int32_t job_layer = -1;
    int32_t job_bank = -1;
    int32_t job_n = 0;
    int32_t job_expert[kMaxDepth] = {};
    int32_t arm_layer = -1;
    int32_t arm_bank = -1;
    int32_t arm_n = 0;
    int32_t arm_expert[kMaxDepth] = {};

    std::function<void(int, int, int)> read_fn;
    // In window mode, slots are chosen by the window LRU, not by the ping-pong region; pass to read_fn
    // slot = -1 as a sentinel (meaning "choose one yourself from the window"). See the lambda in impl.
    bool window_mode = false;
    std::atomic<uint64_t> n_hits{0};
    std::atomic<uint64_t> n_armed{0};
    // Diagnostics: is prefetch actually working? See arm()/take().
    std::atomic<uint64_t> n_arm_skipped{0};   // Number of jobs skipped because the previous job had not finished
    std::atomic<uint64_t> n_waits{0};         // Number of actual blocking waits in take()
    std::atomic<int64_t>  n_wait_ns{0};       // Total time spent blocked
    std::atomic<uint64_t> n_pred_read{0};     // Number of experts actually read by the worker
    std::atomic<uint64_t> n_pred_published{0};// Number of prefetch results published into the window (became cache entries)
    std::atomic<uint64_t> n_predict_only{0};  // Number of prediction-only operations with no disk reads (lightweight combined mode)
    std::atomic<uint64_t> n_evict_saved{0};   // Evictions skipped because an expert was predicted to be used
    std::atomic<uint64_t> n_evict_forced{0};  // Number of times pass 0 found no non-predicted candidate and had to enter pass 1
    std::atomic<uint64_t> n_pred_cached{0};   // Predictions filtered out by arm because they were already in the target layer
    std::atomic<uint64_t> n_used_from_pf{0};  // Number of main-path experts that came from the **prefetch region**
                                              // (not retained after use; overwritten when this layer finishes)
    std::atomic<uint64_t> n_evict_total{0};   // Total number of evictions
    std::atomic<uint64_t> n_pred_bytes{0};    // Number of bytes actually read by the worker
    bool ready = false;

    void init(int32_t slots, int32_t experts, int32_t last_moe, std::function<void(int, int, int)> fn) {
        n_slots = slots;
        n_expert = experts;
        last_moe_layer = last_moe;
        read_fn = std::move(fn);
        ready = true;

        int want = 0;
        const char * on = std::getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH");
        if (on && on[0] != '\0' && on[0] != '0') {
            want = 8;
            if (const char * d = std::getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH")) {
                const int v = atoi(d);
                if (v >= 0 && v <= kMaxDepth) {
                    want = v;
                }
            }
        } else {
            // the mechanism was called FATE before this rename; keep the old spellings loud
            if (std::getenv("LLAMA_MOE_FATE") || std::getenv("LLAMA_MOE_FATE_DEPTH")) {
                moe_log("lookahead-prefetch: off, and LLAMA_MOE_FATE* is no longer read; "
                        "use LLAMA_MOE_LOOKAHEAD_PREFETCH[=_DEPTH]");
            } else {
                moe_log("lookahead-prefetch: off (LLAMA_MOE_LOOKAHEAD_PREFETCH=%s)", on ? on : "unset");
            }
        }
        {
            const char * rs = std::getenv("LLAMA_MOE_PREFETCH_READ");
            read_n = rs ? std::atoi(rs) : -1;   // -1 = follow depth
        }
        set_depth(want);
        if (read_n < 0 || read_n > depth) {
            read_n = depth;                     // Default to depth (legacy behavior)
        }
    }

    void set_depth(int32_t want) {
        if (!ready) {
            return;
        }
        if (want < 0) {
            want = 0;
        }
        if (want > kMaxDepth) {
            want = kMaxDepth;
        }
        if (want > 0 && n_slots < 2 * want + 8) {
            moe_log("lookahead-prefetch: off, bank %d too small for depth %d", (int) n_slots, (int) want);
            want = 0;
        }
        const int32_t cur = enabled ? depth : 0;
        if (want == cur && (want == 0 || th.joinable())) {
            return;
        }

        {
            std::unique_lock<std::mutex> lk(mtx);
            if (job_running) {
                cv_done.wait(lk, [this]() { return !job_running; });
            }
            job_pending = false;
            arm_layer = -1;
            arm_n = 0;
            if (want > 0) {
                depth = want;
                enabled = true;
            } else {
                enabled = false;
            }
        }

        if (enabled && !th.joinable()) {
            quit = false;
            th = std::thread([this]() { worker(); });
        }

        if (enabled) {
            moe_log("lookahead-prefetch: on depth=%d reserved=[%d,%d)", (int) depth, (int) slot_floor(), (int) n_slots);
        } else {
            moe_log("lookahead-prefetch: off (depth=0)");
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            quit = true;
        }
        cv_go.notify_all();
        if (th.joinable()) {
            th.join();
        }
        enabled = false;
    }

    void drain() {
        if (!enabled) {
            return;
        }
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [this]() { return !job_running && !job_pending; });
        arm_layer = -1;
        arm_n = 0;
    }

    void report() const {
        if (n_armed.load(std::memory_order_relaxed) == 0) {
            return;
        }
        const uint64_t armed = n_armed.load(std::memory_order_relaxed);
        const uint64_t hits = n_hits.load(std::memory_order_relaxed);
        const uint64_t preds = armed * (uint64_t) depth;
        moe_log("lookahead-prefetch: armed=%llu hits=%llu (%.1f%% of predictions)",
            (unsigned long long) armed,
            (unsigned long long) hits,
            preds ? 100.0 * (double) hits / (double) preds : 0.0);
    }

    void reset() {
        std::lock_guard<std::mutex> lk(mtx);
        arm_layer = -1;
        arm_n = 0;
    }

    void take(int32_t layer, std::vector<hit> & out) {
        if (!enabled) {
            return;
        }
        std::unique_lock<std::mutex> lk(mtx);

        if (window_mode) {
            // In window mode, the worker and cache **share the same slots**, but **do not need to wait for each other**.
            //
            // Why this works (the key to combining cache + prefetch):
            //   1. When the worker claims a slot, it marks it `kSlotPending`; the main path skips it during allocation/eviction —
            //      so it neither overwrites a slot being written nor treats incomplete data as a valid entry;
            //   2. If the worker has already **published** an expert needed by the main path, `expert_to_slot` contains it,
            //      so it is a hit; otherwise the main path handles it as a miss and reads it itself (at most one duplicate read);
            //   3. The worker does **not hold a lock** while reading from disk (see install_prefetch_into_window),
            //      so it does not block the main thread on the mutex.
            //
            // Previously, this waited unconditionally for all K reads to finish, wasting 43 ms per token in measurements —
            // that was the real reason cache and prefetch interfered, not an inherent conflict between them.
            arm_layer = -1;
            arm_n = 0;
            return;
        }

        // Ping-pong mode: the worker writes to its dedicated tail region [slot_floor, n_slots), while the main path uses only
        // [0, slot_floor) (see alloc_hi = slot_floor in process()); the regions **do not overlap**,
        // so the main path need not wait for the worker.
        //
        // Previously, cv_done.wait() waited unconditionally for the **entire batch of K experts**, including experts the main path
        // did not need at all. The measured cost was high: 54.6 ms wasted per token at K=8 and 259 ms at K=16,
        // longer than a whole step, wiping out the benefit of prefetch.
        // Now, **accept only completed prefetch results for the current layer**; skip unfinished results,
        // and let the main path read misses normally (it would need to read them anyway).
        if (job_running) {
            // The result is not ready yet. arm_layer still holds the previous value, so the consistency check below will reject it.
            return;
        }
        if (arm_layer != layer || arm_n <= 0) {
            arm_layer = -1;
            arm_n = 0;
            return;
        }
        const int32_t base = bank_base(arm_bank);
        for (int k = 0; k < arm_n; ++k) {
            out.push_back({ arm_expert[k], base + k });
        }
        arm_layer = -1;
        arm_n = 0;
    }

    void arm(int32_t layer_next, const std::vector<int32_t> & pred) {
        if (!enabled || layer_next < 1 || layer_next > last_moe_layer || pred.empty()) {
            return;
        }

        // read_n == 0: predict only, without disk reads. This is a lightweight combination of caching and prediction.
        // The caller (impl::process) stores the predictions for use by eviction.
        if (read_n <= 0) {
            n_predict_only.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        std::lock_guard<std::mutex> lk(mtx);
        if (job_pending || job_running) {
            // The previous prefetch is still running, so skip this one. This happens often when reads are slower than compute,
            // and prefetch then provides almost no benefit. Measuring this reveals whether it is a bottleneck.
            n_arm_skipped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        job_layer = layer_next;
        job_bank = layer_next & 1;
        job_n = 0;
        std::vector<uint8_t> seen((size_t) n_expert, 0);
        for (int32_t e : pred) {
            if (job_n >= read_n) {   // Read only read_n experts (<= depth); read_n == 0 returned above
                break;
            }
            if (e < 0 || e >= n_expert || seen[(size_t) e]) {
                continue;
            }
            seen[(size_t) e] = 1;
            job_expert[job_n++] = e;
        }
        if (job_n <= 0) {
            return;
        }
        for (int k = job_n; k < kMaxDepth; ++k) {
            job_expert[k] = -1;
        }
        job_pending = true;
        n_armed.fetch_add(1, std::memory_order_relaxed);
        cv_go.notify_one();
    }

    void worker() {
        for (;;) {
            int32_t layer = -1;
            int32_t bank = -1;
            int32_t n = 0;
            int32_t ex[kMaxDepth];
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_go.wait(lk, [this]() { return quit || job_pending; });
                if (quit) {
                    return;
                }
                job_pending = false;
                job_running = true;
                layer = job_layer;
                bank = job_bank;
                n = job_n;
                memcpy(ex, job_expert, sizeof(ex));
            }
            const int32_t base = bank_base(bank);
            const int n_th = std::max(1, std::min(moe_ssd_io_threads(), n));
            std::atomic<int> next{0};
            auto read_one = [&]() {
                for (;;) {
                    const int k = next.fetch_add(1, std::memory_order_relaxed);
                    if (k >= n) {
                        break;
                    }
                    if (ex[k] < 0) {
                        continue;
                    }
                    try {
                        // In window mode, pass -1 so impl can choose a slot using the target layer window LRU.
                        // Prefetch writes into the **target layer's own window**, not the tail ping-pong region —
                        // so the result is immediately a cache hit for that layer, with no extra copy.
                        read_fn(layer, ex[k], window_mode ? -1 : (base + k));
                        n_pred_read.fetch_add(1, std::memory_order_relaxed);
                        n_pred_published.fetch_add(1, std::memory_order_relaxed);
                    } catch (...) {
                        ex[k] = -1;
                    }
                }
            };
            if (n_th == 1) {
                read_one();
            } else {
                std::vector<std::thread> helpers;
                helpers.reserve((size_t) n_th);
                for (int t = 0; t < n_th; ++t) {
                    helpers.emplace_back(read_one);
                }
                for (auto & h : helpers) {
                    h.join();
                }
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                arm_layer = layer;
                arm_bank = bank;
                arm_n = n;
                memcpy(arm_expert, ex, sizeof(arm_expert));
                job_running = false;
            }
            cv_done.notify_all();
        }
    }
};

struct llama_moe_slot_runtime::impl {
    struct binding {
        ggml_tensor * tensor = nullptr;
        const llama_moe_experts_entry * entry = nullptr;
    };

    // A "packed view" used for a read: prefill and decode each have their own tensors and file layout.
    // Factor it out so read_expert_directv can select by phase without duplicating the read logic.
    struct pack_view {
        const std::vector<binding> * bindings = nullptr;
        const std::vector<size_t>  * order    = nullptr;
        size_t base = 0;
        size_t stride = 0;
        std::string path;
        bool ok = false;

        bool valid() const { return ok && bindings && order; }
    };

    struct layer_state {
        bool enabled = false;
        bool pack_ok = false;
        int32_t window_start = 0;   // Starting slot of this layer's decode window (0 means the window is disabled)
        /// Total width of this layer's window (**cache region c_il + prefetch staging region K**). May vary by layer —
        /// see llama_model::moe_layer_windows(): larger for hot layers, smaller for cold ones.
        /// window_start is always 0 (each layer has its own decode tensor, so its slot space is [0, width)).
        int32_t window_width = 0;
        size_t pack_base = 0;
        size_t pack_bytes = 0;
        size_t pack_stride = 0;
        std::string pack_path;
        std::vector<binding> bindings;
        std::vector<size_t> pack_order; // bindings sorted by file offset, i.e. experts packed order

        // Decode-specific set (independent window per layer, ne[2] = c). Prefill uses the set above.
        bool dec_pack_ok = false;
        size_t dec_pack_base = 0;
        size_t dec_pack_stride = 0;
        std::string dec_pack_path;
        std::vector<binding> dec_bindings;
        std::vector<size_t> dec_pack_order;

        /// Number of successful prefetch results published into this window (diagnostic; n_hits counts main-path hits,
        /// whereas this counter measures worker output).
        uint64_t n_prefetch_published = 0;
        std::vector<int32_t> slot_to_expert;
        std::vector<int32_t> expert_to_slot;
        std::vector<uint64_t> slot_age;
        uint64_t age = 0;
        std::mutex mutex;
    };
    struct map_userdata {
        llama_moe_slot_runtime * runtime = nullptr;
        int layer = -1;
    };

    struct pending_install {
        int32_t expert = -1;
        int32_t slot = -1;
    };

    struct install_timing {
        int n = 0;
        int n_th = 0;
        bool pack = false;
        int64_t wall_ns = 0;
        int64_t pread_sum_ns = 0;
        int64_t pread_max_ns = 0;
        size_t bytes = 0;
    };

    // Feasibility statistics for merging reads of adjacent experts. See the comment in accumulate_runs().
    struct run_stats {
        int n          = 0;   // Total number of contiguous runs (grouped by consecutive expert IDs only)
        int experts    = 0;   // Number of experts in runs (should equal the total number of misses)
        int max        = 0;   // Longest run
        int ok_n       = 0;   // Runs whose slots are also consecutive (can be merged with zero copies)
        int ok_ex      = 0;   // Number of experts covered by those runs
        int scatter_n  = 0;   // Runs with non-consecutive slots that require a scratch buffer
        int scatter_ex = 0;
    };

    struct ubatch_acc {
        int n_tokens = 0;
        int n_used = 0;
        int n_layers = 0;
        int unique_sum = 0;
        int miss_sum = 0;
        int n_th = 0;
        int pack_layers = 0;
        int64_t wall_ns = 0;
        int64_t pread_sum_ns = 0;
        size_t bytes = 0;
        run_stats runs;
    };

    struct io_stats {
        std::atomic<uint64_t> n_miss{0};
        std::atomic<uint64_t> n_bytes{0};
        std::atomic<int64_t>  pread_ns{0};
    };

    /// Per-layer list of experts predicted to be used in the next step, written by the previous layer's arm() and consumed by this layer's process().
    /// Used to improve eviction at **zero I/O cost**: entries predicted to be used should not be evicted.
    std::vector<std::vector<int32_t>> predicted;
    std::mutex pred_mtx;

    // Complete process-wide I/O accounting (main path + prefetch); see readv_span.
    std::atomic<uint64_t> io_bytes_all{0};
    std::atomic<uint64_t> io_calls_all{0};
    uint64_t last_io_bytes = 0;
    uint64_t last_io_calls = 0;
    // Cumulative time spent managing experts during graph execution; see add_process_ns.
    std::atomic<int64_t> process_ns{0};
    // Diagnostics: split process() into three internal sections to locate the unexplained increase from 3.5 to 19.5 ms.
    // Their sum is a subset of process_ns; the difference is the remaining work (allocation, eviction, statistics, etc.).
    std::atomic<int64_t> readids_ns{0};
    std::atomic<int64_t> install_ns{0};
    std::atomic<int64_t> writeids_ns{0};
    // Diagnostics: sum of the sizes of the per-layer unions of experts used (= actual (row, expert) matvec count in the graph).
    // Used to tell whether ε grows with union size (kernel count) or with bytes.
    std::atomic<int64_t> unique_sum{0};
    // Diagnostics: per-layer cumulative misses (decode steps only). Distinguishes "about 1.x per layer" from
    // "many misses in only a few layers" — the former means 40 serial waits; the latter requires addressing only a few layers.
    std::vector<uint64_t> miss_per_layer;
    /// Histogram of the **positions** of misses in selected_experts (sorted by descending score).
    /// Used to determine whether sorting can be replaced by reserving a fixed tail segment:
    /// if misses cluster near the end, split off the last few positions as a second group while preserving summation order (and byte-for-byte output).
    /// A2: number of **trailing positions** among each MoE layer's n_expert_used that may have their reads deferred.
    ///
    /// Motivation: misses are concentrated near the end of score order (measured values: 0:0.03 ... 7:0.22). Defer misses at those positions
    /// until the earlier resident experts have finished computing, so the reads overlap GPU compute.
    /// Splitting at fixed positions (rather than sorting by residency) preserves summation order, keeping output byte-for-byte identical.
    int32_t matmul_split = 0;

    /// Per-layer list of deferred (expert, slot) pairs, written by phase A and consumed by phase B.
    std::vector<std::vector<pending_install>> deferred;

    uint64_t miss_pos[16] = {};
    uint64_t miss_pos_steps = 0;
    uint64_t miss_per_layer_steps = 0;
    int64_t last_process_ns = 0;
    uint64_t last_flush_idx = 0;
    // Baseline for deltas of prefetch diagnostic fields; see flush_ubatch.
    uint64_t last_arm_skipped = 0;
    uint64_t last_wait_count = 0;
    int64_t  last_wait_ns = 0;
    uint64_t last_pred_read = 0;
    uint64_t last_pf_hits = 0;
    uint64_t last_pf_pub = 0;

    explicit impl(const llama_model & model) :
        model(model),
        n_expert((int32_t) model.hparams.n_expert),
        n_slots(model.moe_slot_bank_size()),
        layers(model.layers.size()),
        predicted(model.layers.size()),
        deferred(model.layers.size()),
        miss_per_layer(model.layers.size(), 0),
        userdata(model.layers.size()) {
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto & state = layers[il];
            const auto & layer = model.layers[il];

            bind(state, layer.ffn_gate_up_exps, n_slots);
            bind(state, layer.ffn_gate_exps,    n_slots);
            bind(state, layer.ffn_up_exps,      n_slots);
            bind(state, layer.ffn_down_exps,    n_slots);

            if (state.bindings.empty()) {
                continue;
            }
            if (!layer.ffn_down_exps ||
                (!layer.ffn_gate_up_exps && (!layer.ffn_gate_exps || !layer.ffn_up_exps))) {
                throw std::runtime_error(format("incomplete MoE slot bindings for layer %zu", il));
            }

            state.enabled = true;
            finalize_pack(state);
            state.slot_to_expert.assign(n_slots, kSlotFree);
            state.expert_to_slot.assign(n_expert, kSlotFree);
            state.slot_age.assign(n_slots, 0);
            // window_start is not computed here — decode_window is not known until resolve_decode_window(),
            // so it is still 0 at this point. Assign it there in one place; otherwise all layers share [0, W),
            // causing them to overwrite one another (a real bug encountered previously).
            userdata[il] = { nullptr, (int) il };
            n_enabled++;
            last_moe_layer = (int) il;
            if (state.pack_ok) {
                n_pack_ok++;
            }
            for (const auto & b : state.bindings) {
                if (b.tensor && b.tensor->view_src) {
                    shared_bank = true;
                    break;
                }
            }
        }
        size_t stride0 = 0;
        if (last_moe_layer >= 0) {
            stride0 = layers[(size_t) last_moe_layer].pack_bytes;
            for (const auto & layer : layers) {
                if (layer.enabled && layer.pack_bytes > 0) {
                    stride0 = layer.pack_bytes;
                    break;
                }
            }
        }
        // Experts loading preadv's the packed expert span straight into its slot, which requires both a
        // packed layout and CPU-addressable slot tensors. Do not guess either, and do not silently fall
        // back to an upload path: private Metal buffers (non-unified memory) expose tensor->data as a
        // virtual address that is not writable from the CPU.
        for (const auto & layer : layers) {
            if (layer.enabled && !layer.pack_ok) {
                throw std::runtime_error(
                    "MoE experts layer is not packed; preadv loading requires gate/up/down contiguous per expert");
            }
            for (const auto & b : layer.bindings) {
                if (!b.tensor->buffer || !ggml_backend_buffer_is_host_addressable(b.tensor->buffer)) {
                    throw std::runtime_error(format(
                        "MoE slot tensor '%s' is not CPU-addressable; preadv loading requires host-visible slots",
                        ggml_get_name(b.tensor)));
                }
            }
        }

        moe_log("init pack=%s pack_layers=%d/%d bank=%d shared=%s experts=%d io_threads=%d stride0=%zu preadv=on park_window=%s",
            n_pack_ok == n_enabled && n_enabled > 0 ? "on" : (n_pack_ok > 0 ? "partial" : "off"),
            n_pack_ok, n_enabled, (int) n_slots, shared_bank ? "on" : "off",
            (int) n_expert, moe_ssd_io_threads(), stride0,
            park_window_enabled() ? "on" : "off");

        lookahead.init((int32_t) n_slots, n_expert, last_moe_layer,
            [this](int layer, int expert, int slot) {
                if (slot < 0) {
                    install_prefetch_into_window(layer, expert);
                } else {
                    read_expert_directv(layers[(size_t) layer], expert, slot);
                }
            });
        resolve_decode_window();
        {
            const char * e = std::getenv("LLAMA_MOE_MATMUL_SPLIT");
            matmul_split = e ? std::atoi(e) : 0;
            if (matmul_split > 0) {
                moe_log("matmul-split: defer reads for missing experts at the last %d positions (overlap reads with compute)",
                        (int) matmul_split);
            }
        }

        // Binding the decode-window tensors must wait until resolve_decode_window(); it depends on shared_bank
        // and last_moe_layer, both of which are determined only in the layer loop above.
        if (decode_window > 0) {
            // The per-layer tensor width on the model side is c_il + K (cache region + prefetch staging region) and **varies by layer**.
            // See the partition description in qwen35moe.cpp and install_prefetch_into_window.
            int n_dec = 0;
            std::string fail;
            for (size_t il = 0; il < model.layers.size() && fail.empty(); ++il) {
                auto & state = layers[il];
                if (!state.enabled) {
                    continue;
                }
                const auto & layer = model.layers[il];
                const std::pair<ggml_tensor *, const char *> fam[4] = {
                    { layer.ffn_gate_up_exps_dec, "gate_up" },
                    { layer.ffn_gate_exps_dec,    "gate"    },
                    { layer.ffn_up_exps_dec,      "up"      },
                    { layer.ffn_down_exps_dec,    "down"    },
                };
                for (int k = 0; k < 4 && fail.empty(); ++k) {
                    if (!fam[k].first) {
                        continue;
                    }
                    try {
                        const int32_t expect = layer_width[il] > 0 ? layer_width[il]
                                                                   : decode_window + prefetch_slots;
                        bind(state, fam[k].first, expect, &state.dec_bindings);
                    } catch (const std::exception & e) {
                        ggml_tensor * t = fam[k].first;
                        fail = format("layer %zu family %s: name=%s ne2=%lld nbytes=%zu "
                                      "win=%d :: %s",
                                      il, fam[k].second, ggml_get_name(t),
                                      (long long) t->ne[2], ggml_nbytes(t),
                                      (int) decode_window, e.what());
                    }
                }
                if (fail.empty() && state.dec_bindings.empty()) {
                    fail = format("layer %zu has no decode tensors", il);
                }
                if (fail.empty()) {
                    finalize_pack(state.dec_bindings, state.dec_pack_order, state.dec_pack_ok,
                                  state.dec_pack_base, state.dec_pack_stride, state.dec_pack_path);
                    n_dec++;
                }
            }
            if (!fail.empty()) {
                // If binding fails, fall back to the shared bank (the correct path), but log it **loudly** —
                // a silent fallback would suggest the window is active and make benchmark results misleading.
                moe_log("decode-cache: binding failed; falling back to shared bank (window disabled). Reason: %s",
                        fail.c_str());
                for (auto & st : layers) {
                    st.dec_bindings.clear();
                    st.dec_pack_order.clear();
                    st.dec_pack_ok = false;
                }
                decode_window = 0;
            } else {
                moe_log("decode-cache: bound decode tensors for %d layers (%d slots/layer = %d cache + %d prefetch, pack %s)",
                        n_dec, (int) (decode_window + prefetch_slots), (int) decode_window,
                        (int) prefetch_slots, layers[0].dec_pack_ok ? "ok" : "off");
            }
        }
        sweep_init();
        prewarm_init();
    }

    /// Prefetch only: read an expert into the **target layer's own window**, choosing a slot with the window LRU.
    ///
    /// Shares the same slot-selection rules as process() installation, so the prefetch result is
    /// an ordinary cache hit for that layer — no ping-pong region and no later copy needed.
    ///
    /// The call sequence guarantees safety: prefetch is armed during layer L-1's process(), and layer L's process()
    /// waits for it at take() near the start, so layer L is not executing while the worker modifies its window.
    /// Clear a layer's cache mapping table.
    ///
    /// Preserve `kSlotPending`: the prefetch worker has claimed that slot and is writing to it. Normally this clear
    /// occurs only at the prefill-to-decode boundary (when no worker is active), but preserving the check is safer —
    /// otherwise the worker's slot could be marked free and allocated by the main path, causing a collision.
    static void clear_layer_cache(layer_state & st) {
        for (auto & e : st.slot_to_expert) {
            if (e != kSlotPending) {
                e = kSlotFree;
            }
        }
        // Do not clear expert_to_slot wholesale: pending experts have not been registered yet, so clearing all of it is safe.
        std::fill(st.expert_to_slot.begin(), st.expert_to_slot.end(), kSlotFree);
    }

    /// Prefetch: read a predicted expert into the **target layer's own window**, sharing its slot pool with the cache.
    ///
    /// Key design: **claim the slot under the lock, read from disk without the lock, then publish under the lock**.
    ///
    /// Why this is necessary: previously, this function held `st.mutex` for the whole disk read (about 1 ms per read),
    /// while the main thread's process() needed the same lock, so disk I/O by the worker blocked the main thread.
    /// Measurements showed 43 ms of wasted time per token. **That was the real source of cache/prefetch interference**,
    /// not an inherent conflict between caching and prefetch.
    ///
    /// While claimed, the slot is marked `kSlotPending`; main-path allocation and eviction both skip it,
    /// so the main thread neither overwrites it nor uses data that has not been fully written.
    /// Claim a slot and mark it `kSlotPending` (contents not written yet). **Caller must already hold st.mutex.**
    ///
    /// `cache_only` determines which region to choose from:
    ///   true  → choose only from the **cache region** [window_start, window_start + c_il) (for prewarming;
    ///           prewarmed experts are expected to be reused across tokens and therefore belong in the cache region)
    ///   false → choose from the **prefetch region** (for prefetch; the entire layer window in unified mode)
    ///
    /// Returns -1 if the expert is already in the window or there is no free slot.
    int32_t claim_slot_in_window(layer_state & st, int32_t expert, bool cache_only) {
        if (decode_window <= 0 || expert < 0 || expert >= n_expert) {
            return -1;
        }
        if (st.expert_to_slot[(size_t) expert] >= 0) {
            return -1;   // Already present (in the cache or prefetch region); skip the redundant read
        }
        const int32_t w_il = st.window_width > 0 ? st.window_width : (decode_window + prefetch_slots);
        const int32_t c_il = std::max<int32_t>(1, w_il - prefetch_slots);
        const int32_t lo = st.window_start;
        const int32_t hi = cache_only ? st.window_start + c_il
                                      : st.window_start + w_il;
        const int32_t from = cache_only ? lo
                                        : (prefetch_unified ? lo : st.window_start + c_il);

        int32_t slot = -1;
        for (int32_t cand = from; cand < hi; ++cand) {
            if (st.slot_to_expert[(size_t) cand] == kSlotFree) {
                slot = cand;
                break;
            }
        }
        if (slot < 0) {
            uint64_t oldest = UINT64_MAX;
            for (int32_t cand = from; cand < hi; ++cand) {
                if (st.slot_to_expert[(size_t) cand] == kSlotPending) {
                    continue;
                }
                if (st.slot_age[(size_t) cand] < oldest) {
                    oldest = st.slot_age[(size_t) cand];
                    slot = cand;
                }
            }
        }
        if (slot < 0) {
            return -1;
        }
        const int32_t evicted = st.slot_to_expert[(size_t) slot];
        if (evicted >= 0) {
            st.expert_to_slot[(size_t) evicted] = kSlotFree;
        }
        st.slot_to_expert[(size_t) slot] = kSlotPending;   // Claimed; contents have not been written yet
        return slot;
    }

    /// Register a completed read as a valid cache entry. **Caller must already hold st.mutex.**
    /// Returns whether it was registered (if the main path loaded the expert during the read, discard this copy).
    bool publish_slot_in_window(layer_state & st, int32_t expert, int32_t slot) {
        // The main path may have loaded this expert during the read; if so, discard the duplicate.
        if (st.expert_to_slot[(size_t) expert] >= 0) {
            st.slot_to_expert[(size_t) slot] = kSlotFree;
            return false;
        }
        st.slot_to_expert[(size_t) slot] = expert;
        st.expert_to_slot[(size_t) expert] = slot;
        st.slot_age[(size_t) slot] = ++st.age;
        st.n_prefetch_published++;
        return true;
    }

    void install_prefetch_into_window(int layer, int32_t expert) {
        if (layer < 0 || layer >= (int) layers.size() || decode_window <= 0 || prefetch_slots <= 0) {
            return;
        }
        auto & st = layers[(size_t) layer];
        if (!st.enabled || expert < 0 || expert >= n_expert) {
            return;
        }

        // Allocate only within the **prefetch staging region** [window_start + c, window_start + c + K).
        // This region is refilled every step and is not part of cross-token LRU eviction, so prefetch **does not evict** cache
        // entries. This is essential to the union coverage from combining caching and prefetch:
        //   If the predicted expert is already in the cache region, skip the redundant read (checked below).
        //   Otherwise, read it into staging so that use by this layer does not land on the critical path.
        // In unified mode, prefetch may use the entire layer window (see the prefetch_unified comment).
        int32_t slot = -1;
        {
            std::lock_guard<std::mutex> lock(st.mutex);
            slot = claim_slot_in_window(st, expert, /*cache_only=*/false);
        }
        if (slot < 0) {
            return;
        }

        // Read from disk outside the lock; this no longer blocks the main thread.
        read_expert_directv(st, expert, slot, true);

        std::lock_guard<std::mutex> lock(st.mutex);
        publish_slot_in_window(st, expert, slot);
    }

    /// ── Cache prewarming (LLAMA_MOE_PREWARM) ───────────────────────────────────
    ///
    /// Each layer window is empty at the start of a generation round (cleared when entering window mode and again by park).
    /// Prewarming reads a **static table** of expert IDs for each layer into its decode window, so the first step is no longer
    /// an "all-miss" step.
    ///
    /// Key point: the decode window is **physically separate** from the prefill region (see the base layout in qwen35moe.cpp),
    /// so prefill cannot overwrite data read before prefill. At the phase switch, only the mappings need to be
    /// **republished** (`publish_prewarm_placed`, integer operations only); the experts need not be read again.
    /// This is why the app calls `prewarm_experts()` after unloading mmproj and before prefill:
    /// those reads are completely removed from the first token's critical path.
    ///
    /// Three phases (structured like install_prefetch_into_window, but batched):
    ///   1. Claim: hold each layer lock and find a free slot / evict the oldest entry in the **cache region** for each expert;
    ///   2. Read: hold no layer locks and read concurrently using io_threads threads;
    ///   3. Publish: lock each layer, publish mappings, and record placements (layer, expert, slot) for reattachment at phase switch.
    ///
    /// Pass the layer number in `cur_layer` if the caller (process) already holds that layer's lock; pass -1 when called
    /// from the public API (no layer lock is held). `std::mutex` is not recursive, so locking it again would deadlock.
    void prewarm_windows(int cur_layer) {
        if (!prewarm_enabled || decode_window <= 0) {
            return;
        }
        const int64_t t0 = now_ns();
        struct flat { int layer; int32_t expert; int32_t slot; };

        std::vector<std::vector<std::pair<int32_t, int32_t>>> tasks(layers.size());
        size_t n_req = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            if (il >= prewarm_table.size() || prewarm_table[il].empty()) {
                continue;
            }
            auto & st = layers[il];
            if (!st.enabled) {
                continue;
            }
            n_req += prewarm_table[il].size();
            auto claim_all = [&]() {
                for (int32_t e : prewarm_table[il]) {
                    const int32_t s = claim_slot_in_window(st, e, /*cache_only=*/true);
                    if (s >= 0) {
                        tasks[il].push_back({ e, s });
                    }
                }
            };
            if ((int) il == cur_layer) {
                claim_all();
            } else {
                std::lock_guard<std::mutex> lk(st.mutex);
                claim_all();
            }
        }

        std::vector<flat> q;
        for (size_t il = 0; il < tasks.size(); ++il) {
            for (const auto & t : tasks[il]) {
                q.push_back({ (int) il, t.first, t.second });
            }
        }
        const int n = (int) q.size();
        const int n_th = std::max(1, std::min(moe_ssd_io_threads(), n));
        std::atomic<int> next{0};
        std::atomic<int> n_read{0};
        std::atomic<int64_t> n_bytes{0};
        auto read_one = [&]() {
            for (;;) {
                const int i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= n) {
                    break;
                }
                auto & st = layers[(size_t) q[i].layer];
                try {
                    read_expert_directv(st, q[i].expert, q[i].slot, true);
                    n_read.fetch_add(1, std::memory_order_relaxed);
                    n_bytes.fetch_add((int64_t) st.pack_bytes, std::memory_order_relaxed);
                } catch (...) {
                    q[i].slot = -1;
                }
            }
        };
        if (n > 0) {
            if (n_th == 1) {
                read_one();
            } else {
                std::vector<std::thread> th;
                th.reserve((size_t) n_th);
                for (int t = 0; t < n_th; ++t) {
                    th.emplace_back(read_one);
                }
                for (auto & h : th) {
                    h.join();
                }
            }
        }

        size_t n_pub = 0;
        std::vector<prewarm_place> placed;
        placed.reserve((size_t) n);
        for (size_t il = 0; il < tasks.size(); ++il) {
            if (tasks[il].empty()) {
                continue;
            }
            auto & st = layers[il];
            auto pub_all = [&]() {
                for (const auto & t : tasks[il]) {
                    if (t.second >= 0 && publish_slot_in_window(st, t.first, t.second)) {
                        placed.push_back({ (int32_t) il, t.first, t.second });
                        n_pub++;
                    }
                }
            };
            if ((int) il == cur_layer) {
                pub_all();
            } else {
                std::lock_guard<std::mutex> lk(st.mutex);
                pub_all();
            }
        }
        // Record placements so mappings can be republished at the prefill-to-decode switch without rereading the data.
        prewarm_placed = std::move(placed);
        prewarm_ready = !prewarm_placed.empty();

        moe_log("prewarm: table %zu experts -> claimed %d / read %d / published %zu, %.1f MiB, wall=%.1f ms "
                "(%d threads, cache region)",
                n_req, n, (int) n_read.load(), n_pub,
                (double) n_bytes.load() / (1024.0 * 1024.0), ms_of(now_ns() - t0), n_th);
    }

    /// Republish mappings from the prewarm placement records when switching from prefill to decode.
    ///
    /// The data is still present (decode and prefill regions are physically separate); only the mapping table was cleared —
    /// so this performs integer operations only, without disk I/O. The caller already holds `cur_layer`'s lock (as in prewarm_windows).
    void publish_prewarm_placed(int cur_layer) {
        for (const auto & p : prewarm_placed) {
            if (p.layer < 0 || p.layer >= (int) layers.size()) {
                continue;
            }
            auto & st = layers[(size_t) p.layer];
            if (!st.enabled) {
                continue;
            }
            if (p.expert < 0 || p.expert >= n_expert ||
                p.slot < 0 || p.slot >= (int32_t) st.slot_to_expert.size()) {
                continue;
            }
            auto do_pub = [&]() {
                // The mapping was just cleared (all entries are kSlotFree), so register directly.
                // If the slot or expert is already occupied (not expected), skip it to avoid a conflict.
                if (st.slot_to_expert[(size_t) p.slot] == kSlotFree &&
                    st.expert_to_slot[(size_t) p.expert] < 0) {
                    st.slot_to_expert[(size_t) p.slot] = p.expert;
                    st.expert_to_slot[(size_t) p.expert] = p.slot;
                    st.slot_age[(size_t) p.slot] = ++st.age;
                }
            };
            if ((int) p.layer == cur_layer) {
                do_pub();
            } else {
                std::lock_guard<std::mutex> lk(st.mutex);
                do_pub();
            }
        }
    }

    /// Public entry point (`llama_moe_prewarm_experts`): read the static table into each layer's decode window and publish mappings.
    /// Call while idle and before prefill (app: after unloading mmproj). No decode operation may be in flight.
    void prewarm_experts() {
        if (!prewarm_enabled) {
            return;
        }
        prewarm_windows(-1);   // -1 = no layer lock held; acquire locks as needed
    }

    /// Determine the decode-cache window width and log the result.
    ///
    /// Why prefill cannot use the window: measurements show 105–215 experts per layer during prefill (`unique/l`),
    /// far more than the bank/40 slots available. Prefill must therefore keep the full bank and clear it per layer;
    /// decode can use windows instead. They are naturally separate phases, switched in process() based on `n_tokens == 1`.
    void resolve_decode_window() {
        const char * s = std::getenv("LLAMA_MOE_DECODE_WINDOW");
        if (!s || !*s) {
            moe_log("decode-cache: off (set LLAMA_MOE_DECODE_WINDOW to enable; leave empty to use the bank maximum)");
            return;
        }
        if (!shared_bank) {
            moe_log("decode-cache: off (per-layer windows require shared-bank mode)");
            return;
        }
        // The **model side** (`moe_decode_window()`) is the sole source of window width.
        // Tensor layout was built from it during model loading; calculating a different width at runtime would mismatch —
        // and the old limit (bank / layer count) applied to shared tensors and is no longer valid:
        // each layer has its own tensor, so c is limited only by RAM and ne[2] <= 1024.
        const int32_t want = model.moe_decode_window();
        if (want < 1) {
            moe_log("decode-cache: off (window is not enabled on the model side)");
            return;
        }
        decode_window = want;
        // Prefetch staging size equals prefetch depth (K). If prefetch is disabled, it is 0, leaving only the cache region.
        prefetch_slots = lookahead.enabled ? std::max(0, (int32_t) lookahead.depth) : 0;
        {
            // Unified LRU by default: prefetch and the main path share the entire window (c+K).
            // Set `LLAMA_MOE_PREFETCH_UNIFIED=0` to revert to partitioned mode (for A/B testing only).
            const char * u = std::getenv("LLAMA_MOE_PREFETCH_UNIFIED");
            prefetch_unified = !(u && *u && *u == '0');
            moe_log("decode-cache: %s (window %d slots, K=%d)",
                    prefetch_unified ? "prefetch and cache share one LRU window" : "partitioned prefetch/cache",
                    (int) (decode_window + prefetch_slots), (int) prefetch_slots);
        }

        // Window starts are **no longer il*c** — each layer now has its own decode tensor
        // (a view into base), with slot space [0, width), so the start is always 0.
        // This simplifies runtime handling: layers cannot conflict with one another.
        //
        // **Width varies by layer**: the model side has already laid out tensors according to LLAMA_MOE_LAYER_WINDOWS (base capacity is
        // the sum of all layer widths), so this must use **the exact same** data or the tensor layout will mismatch.
        {
            const std::vector<int32_t> per_layer_c = model.moe_layer_windows();
            layer_width.assign(layers.size(), 0);
            for (size_t il = 0; il < layers.size(); ++il) {
                const int32_t c_il = il < per_layer_c.size() ? per_layer_c[il] : decode_window;
                const int32_t w = c_il + prefetch_slots;
                layers[il].window_start = 0;
                layers[il].window_width = w;
                layer_width[il] = w;
            }
            // Print the distribution to verify that per-layer sizing is active.
            std::string dist;
            for (size_t il = 0; il < layer_width.size() && il < 6; ++il) {
                dist += format("%zu:%d ", il, (int) layer_width[il]);
            }
            moe_log("decode-cache: per-layer widths %s (first few layers: %s)",
                    (getenv("LLAMA_MOE_LAYER_WINDOWS") ? "on" : "off (uniform)"), dist.c_str());
        }

        // ── How large a batch can the window cover? ──────────────────────────────
        //
        // In addition to single-token decode, speculative verification (DSpark) submits small batches with width n_max+1.
        // Letting them use the window enables **cross-round** expert reuse (consecutive decode steps have highly overlapping routing).
        // The limit is capacity-based: the worst-case union of experts for the batch must fit in the cache region.
        //   Worst-case union = min(rows × top-k, n_expert); each row uses at most top-k, and there are n_expert total experts.
        //
        // Clamp using the **narrowest layer**, since per-layer widths can vary (LLAMA_MOE_LAYER_WINDOWS);
        // the clamp must fit even the narrowest layer or process() will throw for that layer.
        {
            const char * rows_env = std::getenv("LLAMA_MOE_WINDOW_MAX_ROWS");
            const int32_t rows_want = std::max(1, rows_env && *rows_env ? atoi(rows_env) : 1);

            int32_t cache_min = 0;
            for (size_t il = 0; il < layer_width.size() && il < layers.size(); ++il) {
                if (!layers[il].enabled || layer_width[il] <= 0) {
                    continue;
                }
                const int32_t usable = layer_width[il] -
                    (prefetch_unified ? 0 : prefetch_slots);
                if (usable > 0 && (cache_min == 0 || usable < cache_min)) {
                    cache_min = usable;
                }
            }
            if (cache_min <= 0) {
                cache_min = decode_window;
            }

            const int32_t n_used = (int32_t) model.hparams.n_expert_used;
            int32_t rows = rows_want;
            while (rows > 1 && cache_min < std::min(rows * n_used, n_expert)) {
                rows--;
            }
            window_max_rows = rows;

            if (rows != rows_want) {
                moe_log("decode-cache: only %d cache slots; worst-case union for %d rows is %d, "
                        "so the supported window batch width is clamped from %d to %d",
                        (int) cache_min, (int) rows_want,
                        (int) std::min(rows_want * n_used, n_expert), (int) rows_want, (int) rows);
            }
            moe_log("decode-cache: window covers batch widths 1..%d%s", (int) window_max_rows,
                    window_max_rows > 1 ? " (speculative verification also uses the window)" : " (single-token decode only)");
        }

        // **Keep prefetch**, but have it write into the target layer's own window (see install_prefetch_into_window).
        //
        // Why not disable it? Prefetch reads bytes concurrently while the previous layer is computing, so if
        // `prefetch bytes / bandwidth <= one layer's compute time`, the reads are fully hidden and effectively free.
        // Only bytes on the **critical path** (those that delay the next layer) add to elapsed time.
        // Caching also helps turn prefetch from "over budget and partly exposed" into "fully hidden":
        //   No cache, depth 8: 14.2 MiB/layer -> 8.2 ms > 6 ms budget  (over budget)
        //   Cache 24 + K=8: 6.6 MiB/layer -> 3.8 ms < 6 ms budget (within budget)
        // The two mechanisms complement each other; they should not be treated as alternatives.
        if (lookahead.enabled && !lookahead.window_mode) {
            lookahead.window_mode = true;
            moe_log("decode-cache: keeping prefetch; redirecting it to the target layer window (depth=%d)",
                    (int) lookahead.depth);
        }

        const int32_t n_moe_layers = model.moe_experts_layer_count();
        const size_t total = (size_t) decode_window * (size_t) n_moe_layers;
        moe_log("decode-cache: on, cache=%d + prefetch=%d slots/layer × %d layers = %zu slots "
                "(%.0f MiB, sharing one base allocation with the %d-slot prefill bank)",
                (int) decode_window, (int) prefetch_slots, (int) n_moe_layers, total,
                (double) total * (double) layers[0].pack_bytes / 1024.0 / 1024.0,
                (int) n_slots);
    }

    ~impl() {
        if (cur.n_layers > 0) {
            flush_ubatch(cur.n_tokens <= 1 ? "decode" : "prefill",
                cur.n_tokens <= 1 ? n_decode_step : n_prefill_ubatch);
        }
        lookahead.report();
        if (miss_pos_steps > 0) {
            std::string s;
            for (int i = 0; i < 16; ++i) {
                if (miss_pos[i] == 0 && i >= 8) break;
                s += format("%d:%.2f ", i, (double) miss_pos[i] / (double) miss_pos_steps);
            }
            moe_log("miss position (score order, avg over %llu steps): %s",
                    (unsigned long long) miss_pos_steps, s.c_str());
        }
        if (miss_per_layer_steps > 0 && decode_window > 0) {
            // Per-layer miss distribution: determine whether every layer has a few misses or only a few layers have many.
            std::string s;
            for (size_t il = 0; il < miss_per_layer.size(); ++il) {
                if (miss_per_layer[il] == 0) continue;
                s += format("%zu:%.2f ", il, (double) miss_per_layer[il] / (double) miss_per_layer_steps);
            }
            moe_log("miss/layer (avg over %llu steps): %s",
                    (unsigned long long) miss_per_layer_steps, s.c_str());
        }
        lookahead.stop();
        log_io("final");
        report_merge();
        if (sweep_fp) {
            fclose(sweep_fp);
            sweep_fp = nullptr;
        }
        for (const auto & item : fds) {
#ifdef _WIN32
            _close(item.second);
#else
            close(item.second);
#endif
        }
    }

    void bind(layer_state & state, ggml_tensor * tensor, int32_t expect_slots,
              std::vector<binding> * out = nullptr) {
        if (!tensor) {
            return;
        }
        const auto * entry = model.moe_experts_entry(ggml_get_name(tensor));
        if (!entry) {
            throw std::runtime_error(format("missing experts entry for tensor '%s'", ggml_get_name(tensor)));
        }
        if (tensor->ne[2] != expect_slots ||
            ggml_nbytes(tensor) != entry->bytes_per_expert * (size_t) expect_slots) {
            throw std::runtime_error(format("invalid virtual slot tensor '%s'", ggml_get_name(tensor)));
        }
        (out ? *out : state.bindings).push_back({ tensor, entry });
    }

    using pack_out = layer_state;

    static void finalize_pack(std::vector<binding> & bindings, std::vector<size_t> & order,
                              bool & ok, size_t & base, size_t & stride, std::string & path) {
        ok = false;
        if (bindings.empty()) {
            return;
        }
        const std::string & p = bindings[0].entry->path;
        const size_t st = bindings[0].entry->expert_stride;
        if (st == 0) {
            return;
        }
        size_t min_off = (size_t) -1;
        size_t max_end = 0;
        for (const auto & b : bindings) {
            if (b.entry->path != p || b.entry->expert_stride != st) {
                return;
            }
            min_off = std::min(min_off, b.entry->offset);
            max_end = std::max(max_end, b.entry->offset + b.entry->bytes_per_expert);
        }
        if (min_off == (size_t) -1 || max_end <= min_off || (max_end - min_off) > st) {
            return;
        }
        for (const auto & b : bindings) {
            if (b.entry->offset < min_off) {
                return;
            }
            if ((b.entry->offset - min_off) + b.entry->bytes_per_expert > st) {
                return;
            }
        }
        ok = true;
        base = min_off;
        stride = st;
        path = p;

        order.resize(bindings.size());
        for (size_t i = 0; i < bindings.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return bindings[a].entry->offset < bindings[b].entry->offset;
        });
    }

    static void finalize_pack(layer_state & state) {
        finalize_pack(state.bindings, state.pack_order, state.pack_ok,
                      state.pack_base, state.pack_stride, state.pack_path);
        state.pack_bytes = state.pack_stride;
    }

    /// LLAMA_MOE_F_NOCACHE: whether expert reads bypass the page cache.
    ///   unset / "0" → never bypass (default)
    ///   "auto"     → phase-dependent (bypass during prefill, not during decode)
    ///   "1"        → always bypass
    static int nocache_mode() {
        static const int m = [] {
            const char * e = std::getenv("LLAMA_MOE_F_NOCACHE");
            if (!e || !*e || std::strcmp(e, "0") == 0) {
                return 0;
            }
            return std::strcmp(e, "auto") == 0 ? -1 : 1;
        }();
        return m;
    }

    bool experts_nocache_desired() const {
        const int m = nocache_mode();
        return m < 0 ? want_nocache.load(std::memory_order_relaxed) : m != 0;
    }

    /// Toggle F_NOCACHE on already-open expert file descriptors. Changes cache behavior only, not data.
    void apply_experts_nocache(bool on) {
#ifdef __APPLE__
        std::lock_guard<std::mutex> lock(fd_mutex);
        for (const auto & kv : fds) {
            if (fcntl(kv.second, F_NOCACHE, on ? 1 : 0) != 0) {
                moe_log("F_NOCACHE switch failed: %s", std::strerror(errno));
            }
        }
#endif
        // Log each transition direction only once to avoid flooding long sessions.
        static bool logged[2] = { false, false };
        if (!logged[on ? 0 : 1]) {
            logged[on ? 0 : 1] = true;
            moe_log("expert reads %s the page cache (%s phase)", on ? "bypass" : "use",
                    on ? "prefill" : "decode");
        }
    }

    int fd_for(const std::string & path) {
        std::lock_guard<std::mutex> lock(fd_mutex);
        const auto found = fds.find(path);
        if (found != fds.end()) {
            return found->second;
        }
#if defined(_WIN32)
        const int fd = _open(path.c_str(), _O_RDONLY | _O_BINARY);
#elif defined(__linux__)
        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | (experts_nocache_desired() ? O_DIRECT : 0));
#else
        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (experts_nocache_desired() && fcntl(fd, F_NOCACHE, 1) != 0) {
            moe_log("F_NOCACHE configuration failed: %s", std::strerror(errno));
        }
#endif
        if (fd < 0) {
            throw std::runtime_error(format("failed to open MoE experts file '%s': %s", path.c_str(), std::strerror(errno)));
        }
        fds.emplace(path, fd);
        return fd;
    }

    // read one contiguous file span into several disjoint buffers, adjusting the iovecs on partial reads
    void readv_span(int fd, size_t offset, struct iovec * iov, int niov) {
        {
            size_t n = 0;
            for (int i = 0; i < niov; ++i) {
                n += iov[i].iov_len;
            }
            // The only instrumentation point that captures **all I/O**: both the main path and prefetch worker pass through here.
            // Looking only at bytes= on decode log lines misses prefetch reads (which were measured to be the bulk of the I/O).
            io_bytes_all.fetch_add(n, std::memory_order_relaxed);
            io_calls_all.fetch_add(1, std::memory_order_relaxed);
        }
        size_t total = 0;
        for (int i = 0; i < niov; ++i) {
            total += iov[i].iov_len;
        }
        size_t done = 0;
        while (done < total) {
            struct iovec cur[8];
            int cn = 0;
            size_t skip = done;
            for (int i = 0; i < niov; ++i) {
                if (skip >= iov[i].iov_len) {
                    skip -= iov[i].iov_len;
                    continue;
                }
                cur[cn].iov_base = (uint8_t *) iov[i].iov_base + skip;
                cur[cn].iov_len  = iov[i].iov_len - skip;
                skip = 0;
                cn++;
            }
#if defined(_WIN32)
            std::lock_guard<std::mutex> lock(io_mutex);
            if (_lseeki64(fd, (int64_t) (offset + done), SEEK_SET) < 0) {
                throw std::runtime_error("failed to seek MoE experts");
            }
            const ssize_t result = _read(fd, cur[0].iov_base, (unsigned int) cur[0].iov_len);
#elif defined(__linux__)
            const ssize_t result = preadv(fd, cur, cn, (off_t) (offset + done));
            if (result < 0 && errno == EINVAL && experts_nocache_desired()) {
                throw std::runtime_error(format(
                    "failed to read MoE experts at offset %zu (bad alignment for LLAMA_MOE_F_NOCACHE)", offset + done));
            }
#else
            const ssize_t result = preadv(fd, cur, cn, (off_t) (offset + done));
#endif
            if (result < 0 && errno == EINTR) {
                continue;
            }
            if (result <= 0) {
                throw std::runtime_error(format(
                    "failed to read MoE experts at offset %zu", offset + done));
            }
            done += (size_t) result;
        }
    }

    // Reads the packed gate|up|down span (one contiguous run in the experts) with a single preadv,
    // writing each family straight into its own slot. One syscall, one write per byte: no staging
    // buffer and no second copy into the device buffer.
    int64_t read_expert_directv(const layer_state & state, int32_t expert, int32_t slot,
                                bool window_mode = false) {
        // Diagnostic: LLAMA_MOE_SKIP_READ=1 performs all accounting but does not actually read from disk.
        //
        // This separates the cost of the CPU section plus per-layer round trips from the cost of real disk reads:
        //   step (real) - step (empty read) = I/O exposed on the critical path
        //   step (empty read) - resident baseline = structural overhead (40 GPU/CPU round trips + scheduling)
        // The output will be incorrect (expert contents are garbage); use this only to measure timing.
        // Three diagnostic modes separate reading from writing into the GPU buffer:
        //   0 (default) normal: preadv writes directly into the slot tensor (GPU-visible)
        //   1 dummy: preadv reads into a scratch buffer — same I/O, but does not touch the GPU buffer
        //   2 skip: do not read at all
        // Interpretation: if 0 >> 1 ≈ 2, the cost is in writing the GPU buffer (cache-line invalidation);
        //               if 0 ≈ 1 >> 2, the cost is the I/O itself.
        static const int mode = [] {
            const char * d = std::getenv("LLAMA_MOE_DUMMY_READ");
            if (d && *d && *d != '0') return 1;
            const char * s = std::getenv("LLAMA_MOE_SKIP_READ");
            if (s && *s && *s != '0') return 2;
            return 0;
        }();

        // During decode, read into this layer's own window tensor (a view into base); during prefill, read into the shared region.
        // The tensors, file offsets, and strides differ, so switch the entire set together.
        const std::vector<binding> & bindings = window_mode ? state.dec_bindings   : state.bindings;
        const std::vector<size_t>  & order    = window_mode ? state.dec_pack_order : state.pack_order;
        const std::string          & path     = window_mode ? state.dec_pack_path : state.pack_path;
        const size_t base   = window_mode ? state.dec_pack_base   : state.pack_base;
        const size_t stride = window_mode ? state.dec_pack_stride : state.pack_stride;

        struct iovec iov[8];
        const int n = (int) order.size();
        if (n <= 0 || n > 8) {
            throw std::runtime_error("invalid MoE packed family count for preadv");
        }
        size_t nbytes = 0;
        size_t max_exp = 0;
        for (int i = 0; i < n; ++i) {
            max_exp = std::max(max_exp, bindings[order[(size_t) i]].entry->bytes_per_expert);
        }
        if (mode == 1) {
            // dummy: read into a persistent scratch buffer (not the GPU buffer). This still performs real disk I/O,
            // so I/O volume matches normal mode; only the destination differs.
            static std::vector<uint8_t> scratch;
            if (scratch.size() < max_exp) {
                scratch.resize(max_exp);
            }
            for (int i = 0; i < n; ++i) {
                const binding & b = bindings[order[(size_t) i]];
                iov[i].iov_base = scratch.data();
                iov[i].iov_len  = b.entry->bytes_per_expert;
                nbytes += b.entry->bytes_per_expert;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                const binding & b = bindings[order[(size_t) i]];
                iov[i].iov_base = (uint8_t *) b.tensor->data + (size_t) slot * b.entry->bytes_per_expert;
                iov[i].iov_len  = b.entry->bytes_per_expert;
                nbytes += b.entry->bytes_per_expert;
            }
        }
        const int fd = fd_for(path);
        const size_t offset = base + (size_t) expert * stride;

        if (mode == 2) {
            // Keep accounting as if the read succeeded, but do not access the disk — used to isolate I/O from structural overhead.
            stats.n_miss.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }

        // LLAMA_MOE_READ_DELAY_US: add a fixed delay to every read.
        // Used to test whether read time lies 1:1 on the critical path: if total time increases by
        // delay × read count, reads lie linearly on the critical path; a larger increase indicates amplification.
        static const int delay_us = [] {
            const char * d = std::getenv("LLAMA_MOE_READ_DELAY_US");
            return d ? std::atoi(d) : 0;
        }();
        if (delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        }

        const int64_t t0 = now_ns();
        readv_span(fd, offset, iov, n);
        const int64_t dt = now_ns() - t0;
        stats.pread_ns.fetch_add(dt, std::memory_order_relaxed);
        stats.n_bytes.fetch_add(nbytes, std::memory_order_relaxed);
        stats.n_miss.fetch_add(1, std::memory_order_relaxed);
        return dt;
    }

    static const uint8_t * host_data(const ggml_tensor * tensor) {
        if (!tensor || !tensor->data || (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer))) {
            return nullptr;
        }
        return (const uint8_t *) tensor->data;
    }

    static uint8_t * host_data(ggml_tensor * tensor) {
        return const_cast<uint8_t *>(host_data((const ggml_tensor *) tensor));
    }

    void read_i32_ids(const ggml_tensor * tensor, std::vector<int32_t> & ids) const {
        const int64_t n_ids = tensor->ne[0];
        const int64_t n_tokens = tensor->ne[1];
        const size_t row_bytes = (size_t) n_ids * sizeof(int32_t);
        ids.resize((size_t) (n_ids * n_tokens));
        if (const uint8_t * src = host_data(tensor)) {
            for (int64_t token = 0; token < n_tokens; ++token) {
                memcpy(ids.data() + token * n_ids, src + (size_t) token * tensor->nb[1], row_bytes);
            }
            return;
        }
        for (int64_t token = 0; token < n_tokens; ++token) {
            ggml_backend_tensor_get(
                tensor,
                ids.data() + token * n_ids,
                (size_t) token * tensor->nb[1],
                row_bytes);
        }
    }

    void write_i32_ids(ggml_tensor * tensor, const std::vector<int32_t> & ids) const {
        const int64_t n_ids = tensor->ne[0];
        const int64_t n_tokens = tensor->ne[1];
        const size_t row_bytes = (size_t) n_ids * sizeof(int32_t);
        if (ids.size() != (size_t) (n_ids * n_tokens)) {
            throw std::runtime_error("MoE slot id count does not match tensor shape");
        }
        if (uint8_t * dst = host_data(tensor)) {
            for (int64_t token = 0; token < n_tokens; ++token) {
                memcpy(dst + (size_t) token * tensor->nb[1], ids.data() + token * n_ids, row_bytes);
            }
            return;
        }
        for (int64_t token = 0; token < n_tokens; ++token) {
            ggml_backend_tensor_set(
                tensor,
                ids.data() + token * n_ids,
                (size_t) token * tensor->nb[1],
                row_bytes);
        }
    }

    // Count contiguous runs among pending entries to determine whether reads of adjacent experts can be merged.
    //
    // Background: experts are stored expert-major, so expert k's bytes are located at
    // base + k * expert_stride in the file; therefore consecutive expert IDs mean consecutive file addresses.
    // pending is built in ascending expert-ID order, so it naturally forms contiguous runs that can be merged.
    //
    // Two mergeable cases:
    //   1. Slots are also consecutive -> one iovec can cover the whole run, with zero copies and one syscall per run
    //   2. Slots are not consecutive -> read into scratch first, then scatter to each slot, requiring one extra memory copy
    //
    // In shared-bank prefill, LRU is cleared per layer and slots are allocated sequentially from 0; pending is also
    // appended in ascending expert-ID order, so case 1 should dominate. These statistics verify that.
    static run_stats accumulate_runs(const std::vector<pending_install> & pending) {
        run_stats st;
        const size_t n = pending.size();
        if (n == 0) {
            return st;
        }
        size_t i = 0;
        while (i < n) {
            size_t j = i + 1;   // Extend only across consecutive expert IDs; check slots separately
            while (j < n && pending[j].expert == pending[j - 1].expert + 1) {
                ++j;
            }
            const int run_len = (int) (j - i);

            bool slots_consecutive = true;
            for (size_t k = i + 1; k < j; ++k) {
                if (pending[k].slot != pending[k - 1].slot + 1) {
                    slots_consecutive = false;
                    break;
                }
            }

            st.n++;
            st.experts += run_len;
            st.max = std::max(st.max, run_len);
            if (slots_consecutive) {
                st.ok_n++;
                st.ok_ex += run_len;
            } else {
                st.scatter_n++;
                st.scatter_ex += run_len;
            }
            i = j;
        }
        return st;
    }

    static void add_runs(run_stats & dst, const run_stats & src) {
        dst.n          += src.n;
        dst.experts    += src.experts;
        dst.max         = std::max(dst.max, src.max);
        dst.ok_n       += src.ok_n;
        dst.ok_ex      += src.ok_ex;
        dst.scatter_n  += src.scatter_n;
        dst.scatter_ex += src.scatter_ex;
    }

    install_timing install_pending(layer_state & state, const std::vector<pending_install> & pending,
                                   bool window_mode) {
        install_timing total;
        const int n = (int) pending.size();
        if (n <= 0) {
            return total;
        }
        for (int offset = 0; offset < n; offset += kMoeSsdIoChunk) {
            const int n_wave = std::min(kMoeSsdIoChunk, n - offset);
            std::vector<pending_install> wave(
                pending.begin() + offset, pending.begin() + offset + n_wave);
            const install_timing t = install_packed_wave(state, wave, window_mode);
            total.n += t.n;
            total.n_th = std::max(total.n_th, t.n_th);
            total.pack = true;
            total.wall_ns += t.wall_ns;
            total.pread_sum_ns += t.pread_sum_ns;
            total.pread_max_ns = std::max(total.pread_max_ns, t.pread_max_ns);
            total.bytes += t.bytes;
        }
        return total;
    }

    install_timing install_packed_wave(layer_state & state, const std::vector<pending_install> & pending,
                                       bool window_mode) {
        install_timing t;
        const int n = (int) pending.size();
        if (n <= 0) {
            return t;
        }

        t.n = n;
        t.pack = true;
        t.bytes = (size_t) n * state.pack_bytes;
        std::vector<std::string> errs((size_t) n);
        std::vector<int64_t> dts((size_t) n, 0);
        auto read_one = [&](int i) {
            try {
                dts[(size_t) i] = read_expert_directv(
                    state, pending[(size_t) i].expert, pending[(size_t) i].slot, window_mode);
            } catch (const std::exception & e) {
                errs[(size_t) i] = e.what();
            }
        };

        t.n_th = std::max(1, std::min(moe_ssd_io_threads(), n));
        const int64_t wall0 = now_ns();
        // Note: in single-thread mode, calling only read_one(0) would load only the first expert,
        // silently skipping the remaining N-1 and leaving stale weights from the previous layer/token in their slots;
        // the model output would be wrong without an error. This path is entered only when io_threads == 1,
        // so it remained hidden for a long time because the default is 4; making it configurable exposed the bug.
        // Single-thread mode has another effect: pending is built in ascending expert-ID order, while experts are
        // expert-major (offset = base + expert * stride), so single-thread mode performs strictly sequential
        // file reads and may approach sequential-read throughput (measured faster than interleaved random reads with 4 threads).
        if (t.n_th == 1) {
            for (int i = 0; i < n; ++i) {
                read_one(i);
            }
        } else {
            std::atomic<int> next{0};
            std::vector<std::thread> threads;
            threads.reserve((size_t) t.n_th);
            for (int th = 0; th < t.n_th; ++th) {
                threads.emplace_back([&]() {
                    for (;;) {
                        const int i = next.fetch_add(1, std::memory_order_relaxed);
                        if (i >= n) {
                            break;
                        }
                        read_one(i);
                    }
                });
            }
            for (auto & th : threads) {
                th.join();
            }
        }
        t.wall_ns = now_ns() - wall0;
        for (int i = 0; i < n; ++i) {
            if (!errs[(size_t) i].empty()) {
                throw std::runtime_error(errs[(size_t) i]);
            }
            t.pread_sum_ns += dts[(size_t) i];
            t.pread_max_ns = std::max(t.pread_max_ns, dts[(size_t) i]);
        }
        return t;
    }

    static double ms_of(int64_t ns) {
        return ns / 1e6;
    }

    static double mib_of(size_t bytes) {
        return bytes / (1024.0 * 1024.0);
    }

    static double par_of(int64_t sum_ns, int64_t wall_ns) {
        if (wall_ns <= 0) {
            return 0.0;
        }
        return (double) sum_ns / (double) wall_ns;
    }

    bool should_log_ubatch(int n_tokens, uint64_t idx) const {
        // LLAMA_MOE_LOG_EVERY_STEP=1: log every step for performance attribution.
        // With sparse logging, it is easy to misread the counters' accumulation window and span (observed during testing);
        // per-step logging keeps span fixed at 1, so consumers need no normalization.
        static const bool every = [] {
            const char * s = std::getenv("LLAMA_MOE_LOG_EVERY_STEP");
            return s && *s && *s != '0';
        }();
        if (every) {
            return true;
        }
        if (idx <= 8) {
            return true;
        }
        if (n_tokens <= 1) {
            return idx <= 4 || (idx % 32) == 0;
        }
        return (idx % 8) == 0;
    }

    void flush_ubatch(const char * kind, uint64_t idx) {
        if (cur.n_layers <= 0) {
            return;
        }
        // Complete I/O for this step: main path + prefetch. Use global-counter deltas to avoid mixing accounting scopes.
        const uint64_t all_now = io_bytes_all.load(std::memory_order_relaxed);
        const uint64_t calls_now = io_calls_all.load(std::memory_order_relaxed);
        const int64_t proc_now = process_ns.load(std::memory_order_relaxed);
        const uint64_t arm_skip_now = lookahead.n_arm_skipped.load(std::memory_order_relaxed);
        const uint64_t wait_now     = lookahead.n_waits.load(std::memory_order_relaxed);
        const int64_t  wait_ns_now  = lookahead.n_wait_ns.load(std::memory_order_relaxed);
        const uint64_t pred_read_now = lookahead.n_pred_read.load(std::memory_order_relaxed);
        const uint64_t pf_hits_now = lookahead.n_hits.load(std::memory_order_relaxed);
        const uint64_t pf_pub_now  = lookahead.n_pred_published.load(std::memory_order_relaxed);
        const uint64_t all_delta = all_now - last_io_bytes;
        const uint64_t calls_delta = calls_now - last_io_calls;
        const int64_t proc_delta = proc_now - last_process_ns;
        const uint64_t arm_skip_delta = arm_skip_now - last_arm_skipped;
        const uint64_t wait_delta = wait_now - last_wait_count;
        const int64_t wait_ns_delta = wait_ns_now - last_wait_ns;
        const uint64_t pred_read_delta = pred_read_now - last_pred_read;
        const uint64_t pf_hits_delta = pf_hits_now - last_pf_hits;
        const uint64_t pf_pub_delta  = pf_pub_now - last_pf_pub;
        last_io_bytes = all_now;
        last_io_calls = calls_now;
        last_process_ns = proc_now;
        last_arm_skipped = arm_skip_now;
        last_wait_count = wait_now;
        last_wait_ns = wait_ns_now;
        last_pred_read = pred_read_now;
        last_pf_hits = pf_hits_now;
        last_pf_pub  = pf_pub_now;

        // These counters advance only on flush, and flushes are throttled by should_log_ubatch.
        // Therefore, log how many steps this flush covers; otherwise consumers may interpret
        // a multi-step total as a per-step value (it was overstated by 24x in testing). That is what span= indicates.
        const uint64_t span = idx > last_flush_idx ? idx - last_flush_idx : 1;
        last_flush_idx = idx;
        const double wall_s = cur.wall_ns / 1e9;
        const double mibs = wall_s > 0.0 ? (cur.bytes / (1024.0 * 1024.0)) / wall_s : 0.0;
        const double uniq = cur.n_layers > 0 ? (double) cur.unique_sum / (double) cur.n_layers : 0.0;
        const double avg_run = cur.runs.n > 0 ? (double) cur.runs.experts / (double) cur.runs.n : 0.0;
        moe_log("%s #%llu tok=%d used=%d layers=%d unique/l=%.0f miss=%d hit=%d pack=%d/%d nth=%d "
                "pread_wall=%.0fms pread_sum=%.0fms bytes=%.0fMiB par=%.2f io=%.0fMiB/s "
                "readall=%.0fMiB calls=%llu realio=%.0fMiB/s span=%llu pmgmt=%.1fms "
                "pfskip=%llu pfwait=%llu/%.1fms pfread=%llu pfpub=%llu pfhit=%llu pfpredict=%llu pevsave=%llu pevforced=%llu pfcached=%llu usedpf=%llu "
                "runs=%d avg_run=%.1f max_run=%d zerocopy=%.0f%%",
            kind,
            (unsigned long long) idx,
            cur.n_tokens,
            cur.n_used,
            cur.n_layers,
            uniq,
            cur.miss_sum,
            cur.unique_sum - cur.miss_sum,
            cur.pack_layers,
            cur.n_layers,
            cur.n_th,
            ms_of(cur.wall_ns),
            ms_of(cur.pread_sum_ns),
            mib_of(cur.bytes),
            par_of(cur.pread_sum_ns, cur.wall_ns),
            mibs,
            mib_of(all_delta),
            (unsigned long long) calls_delta,
            // Actual bandwidth: total bytes / wall-clock time for this step (includes compute, so this is a lower bound).
            wall_s > 0.0 ? (double) all_delta / (1024.0 * 1024.0) / wall_s : 0.0,
            (unsigned long long) span,
            ms_of(proc_delta),
            (unsigned long long) arm_skip_delta,
            (unsigned long long) wait_delta,
            ms_of(wait_ns_delta),
            (unsigned long long) pred_read_delta,
            (unsigned long long) pf_pub_delta,
            (unsigned long long) pf_hits_delta,
            (unsigned long long) lookahead.n_predict_only.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_evict_saved.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_evict_forced.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_pred_cached.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_used_from_pf.load(std::memory_order_relaxed),
            cur.runs.n,
            avg_run,
            cur.runs.max,
            cur.runs.experts > 0 ? 100.0 * (double) cur.runs.ok_ex / (double) cur.runs.experts : 0.0);
        cur = ubatch_acc{};
    }

    /// Process-wide summary of whether merging reads is feasible. It answers one question:
    /// How many syscalls can be saved, and how large will transfers become, if reads of adjacent experts are merged into larger preadv calls?
    void report_merge() {
        if (g_runs.experts <= 0) {
            return;
        }
        const double avg_run = (double) g_runs.experts / (double) g_runs.n;
        const double zc_pct  = 100.0 * (double) g_runs.ok_ex / (double) g_runs.experts;
        // Assuming full merging: reduce syscall count from one per expert to one per run.
        const double syscall_ratio = (double) g_runs.n / (double) g_runs.experts;
        moe_log("merge-scan: experts=%d runs=%d avg_run=%.2f max_run=%d "
                "zerocopy=%.1f%% scatter_experts=%d "
                "syscalls=%.1f%% of current (avg read = %.2f experts)",
            g_runs.experts, g_runs.n, avg_run, g_runs.max,
            zc_pct, g_runs.scatter_ex,
            100.0 * syscall_ratio, avg_run);
    }

    void log_io(const char * tag) {        const uint64_t miss = stats.n_miss.load(std::memory_order_relaxed);
        if (miss == 0) {
            return;
        }
        const int64_t pread_ns = stats.pread_ns.load(std::memory_order_relaxed);
        const uint64_t bytes = stats.n_bytes.load(std::memory_order_relaxed);
        moe_log("total[%s]: miss=%llu bytes=%.1f MiB pread=%.0fms",
            tag,
            (unsigned long long) miss,
            mib_of((size_t) bytes),
            pread_ns / 1e6);
    }

    const llama_model & model;
    int32_t n_expert;
    int32_t n_slots;
    int n_enabled = 0;
    int n_pack_ok = 0;
    bool shared_bank = false;
    // Per-layer decode cache: each layer gets a private slot window [layer*W, (layer+1)*W),
    // with LRU preserved across tokens within the window. Prefill does not use the window (100+ experts per layer exceed W),
    // and continues to use the full bank, cleared once per layer.
    //
    // W = 0 disables this (restores old behavior: all layers share all slots, cleared per layer).
    int32_t decode_window = 0;
    /// Maximum batch width (rows) supported by the window. 1 serves single-token decode only (legacy behavior); larger values let
    /// small batches such as speculative verification (DSpark, batch width = n_max+1) use the window too, saving disk reads through cross-round expert reuse.
    ///
    /// The requested value comes from LLAMA_MOE_WINDOW_MAX_ROWS, then resolve_decode_window() clamps it to cache capacity.
    /// Clamping is necessary: if the batch does not fit, process() throws at runtime rather than falling back.
    /// The upper bound is also constrained by batch width < 32; larger batches use mul_mat_id, which cannot address
    /// each layer's `.dec` view.
    int32_t window_max_rows = 1;
    /// Number of slots in each layer's **prefetch staging region** (= prefetch depth K).
    ///
    /// Why keep this separate: prefetched entries may be needed next step, while cross-token reuse has a different value.
    /// With a shared LRU, prefetch evicts cross-token cache entries when the window fills,
    /// meaning prefetch destroys the cache. Partitioning prevents interference; coverage becomes the union of:
    ///   Cache region [window_start, window_start + c)          — cross-token LRU
    ///   Prefetch region [window_start + c, window_start + c + K) — refilled every step, not subject to LRU eviction
    int32_t prefetch_slots = 0;
    /// Total width of each layer's window (c_il + K), indexed by layer. See layer_state::window_width.
    std::vector<int32_t> layer_width;
    /// Unified LRU: prefetch and main path share the **entire window** (no partition). **Enabled by default.**
    ///
    /// Why partitioning is slower (measured over 24 steps):
    ///   Partitioned: misses 53 | prefetch writes 3022 | used experts from prefetch region 4568 | pmgmt 174 ms
    ///   Unified:     misses 29 | prefetch writes 1932 | used experts from prefetch region 888  | pmgmt 101 ms
    /// With partitioning, the prefetch region retains entries for only ~3 steps (it is refilled every step), and 78% of used experts land there —
    /// so hot experts are flushed immediately after use and must be reread. Unified LRU lets them naturally enter the cross-token LRU.
    ///
    /// Partitioning was originally intended to prevent prefetch from evicting many cache entries at once, but filtering during arm() removes that premise:
    /// prefetch adds only about 2.4 new experts per layer per step (not 8), so it does not cause mass eviction.
    bool prefetch_unified = true;
    bool was_window = false;
    /// Toggled only when LLAMA_MOE_F_NOCACHE=auto (phase-dependent): bypass during prefill, not during decode.
    /// Must be atomic: the main thread writes it in process() without holding fd_mutex, while fd_for() is called
    /// and reads it from a prefetch worker thread, causing unsynchronized concurrent access. relaxed is sufficient:
    /// only atomicity matters; no ordering guarantee is needed.
    std::atomic<bool> want_nocache{false};
    int last_moe_layer = -1;
    uint64_t n_prefill_ubatch = 0;
    uint64_t n_decode_step = 0;
    ubatch_acc cur;
    run_stats g_runs;   // Process-wide cumulative stats; process() runs only on ith == 0, single-threaded, so atomics are unnecessary
    std::vector<layer_state> layers;
    std::vector<map_userdata> userdata;
    std::unordered_map<std::string, int> fds;
    std::mutex fd_mutex;
    std::mutex io_mutex;
    io_stats stats;
    lookahead_expert_prefetch lookahead;
    ggml_tensor * lookahead_pending = nullptr;

    std::atomic<bool> parked{false};
    std::mutex park_mutex;

    // ── Hit-rate sweep (diagnostics, disabled by default) ──────────────────────
    // LLAMA_MOE_SWEEP=<path> enables this and writes one raw record per decode step and layer:
    //
    //   L<layer> E:<experts actually used by this layer> P:<experts predicted for this layer by the previous layer, sorted by score descending>
    //
    // Why dump raw data instead of computing tables in C++: hit rate depends only on routing, not hardware,
    // so the file can be generated once on a Mac and analyzed anywhere. After dumping, sweep cap/K as desired
    // without recompiling each combination. See analyze_hits.py in the project documentation.
    FILE * sweep_fp = nullptr;
    std::vector<std::vector<int32_t>> sweep_pred;   // sweep_pred[L] = prediction for layer L (from layer L-1)

    // ── Cache prewarm table (LLAMA_MOE_PREWARM=<path>) ─────────────────────────
    // File format: one layer per line; line number = model layer index (same numbering as L<layer> in sweep dumps),
    // expert IDs on each line are comma- or whitespace-separated (# starts a comment; blank/comment lines do not count as layers).
    // First line = layer 0, second line = layer 1, and so on.
    std::vector<std::vector<int32_t>> prewarm_table;
    bool prewarm_enabled = false;

    /// Placement records after prewarming (layer, expert, slot index within the window).
    /// Used to republish mappings at phase switches — prefill does not touch the data (physically separate); only mappings are cleared.
    struct prewarm_place { int32_t layer; int32_t expert; int32_t slot; };
    std::vector<prewarm_place> prewarm_placed;
    /// Whether the data has been loaded and not yet consumed by a phase switch. Set to false when consumed (one-shot),
    /// and also cleared by park() at the end of each round — prewarm_experts() must be called again next round.
    bool prewarm_ready = false;

    void prewarm_init() {
        const char * path = std::getenv("LLAMA_MOE_PREWARM");
        if (!path || !*path) {
            return;
        }
        FILE * fp = std::fopen(path, "r");
        if (!fp) {
            moe_log("prewarm: cannot open table file %s", path);
            return;
        }
        prewarm_table.assign(layers.size(), {});
        char buf[16384];
        int li = 0;
        int total = 0;
        while (li < (int) layers.size() && std::fgets(buf, sizeof(buf), fp)) {
            const char * p = buf;
            while (*p == ' ' || *p == '\t') {
                ++p;
            }
            if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') {
                continue;   // Comments and blank lines do not count as layers
            }
            std::vector<int32_t> ids;
            while (*p) {
                while (*p == ' ' || *p == '\t' || *p == ',') {
                    ++p;
                }
                if (!*p || *p == '\n' || *p == '\r' || *p == '#') {
                    break;
                }
                char * end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (end == p) {
                    ++p;
                    continue;
                }
                if (v >= 0 && v < n_expert) {
                    ids.push_back((int32_t) v);
                }
                p = end;
            }
            prewarm_table[(size_t) li] = std::move(ids);
            total += (int) prewarm_table[(size_t) li].size();
            li++;
        }
        std::fclose(fp);
        prewarm_enabled = total > 0 && decode_window > 0;
        moe_log("prewarm: table %s -> %d layers / %d experts, decode_window=%d, enabled=%s",
                path, li, total, (int) decode_window, prewarm_enabled ? "on" : "off");
        if (decode_window <= 0) {
            moe_log("prewarm: window is disabled (LLAMA_MOE_DECODE_WINDOW=0); prewarming will have no effect");
        }
    }

    void sweep_init() {
        const char * path = std::getenv("LLAMA_MOE_SWEEP");
        if (!path || !*path) {
            return;
        }
        sweep_fp = fopen(path, "w");
        if (!sweep_fp) {
            moe_log("sweep: cannot open %s", path);
            return;
        }
        sweep_pred.assign(layers.size(), {});
        fprintf(sweep_fp, "# layer real_experts | predicted_experts(desc)\n");
        moe_log("sweep: recording to %s (one line per layer per decode step)", path);
    }

    void sweep_write(int layer, const std::vector<int32_t> & used, const std::vector<int32_t> & pred) {
        if (!sweep_fp) {
            return;
        }
        fprintf(sweep_fp, "L%d E:", layer);
        for (size_t i = 0; i < used.size(); ++i) {
            fprintf(sweep_fp, "%s%d", i ? "," : "", used[i]);
        }
        fprintf(sweep_fp, " P:");
        for (size_t i = 0; i < pred.size(); ++i) {
            fprintf(sweep_fp, "%s%d", i ? "," : "", pred[i]);
        }
        fprintf(sweep_fp, "\n");
    }

    /// During prefill, record experts used by the **final token** (`P<layer> E:...`).
    ///
    /// Purpose: verify whether prefill can leave an initial cache for decode.
    /// Prefill uses 100+ experts per layer, too many for the per-layer window (c=24), so it must use
    /// the full bank and clear it per layer. At the end, only the last layer's data remains in the bank. But the 8 experts used by the
    /// final token are the most likely neighbors of decode's first step, so they can be retained as seeds.
    void sweep_write_prefill(int layer, const int32_t * ids, int n_ids, int n_tokens) {
        if (!sweep_fp || n_tokens <= 0) {
            return;
        }
        const int32_t * last = ids + (size_t) (n_tokens - 1) * n_ids;
        fprintf(sweep_fp, "P%d E:", layer);
        for (int i = 0; i < n_ids; ++i) {
            fprintf(sweep_fp, "%s%d", i ? "," : "", last[i]);
        }
        fprintf(sweep_fp, "\n");
    }

    std::vector<ggml_backend_buffer_t> expert_buffers() const {
        std::set<ggml_backend_buffer_t> uniq;
        auto add = [&](const std::vector<binding> & bs) {
            for (const auto & b : bs) {
                ggml_tensor * t = b.tensor;
                if (!t) {
                    continue;
                }
                ggml_backend_buffer_t buf = t->buffer;
                if (!buf && t->view_src) {
                    buf = t->view_src->buffer;
                }
                if (buf) {
                    uniq.insert(buf);
                }
            }
        };
        for (const auto & st : layers) {
            add(st.bindings);
            add(st.dec_bindings);
        }
        return { uniq.begin(), uniq.end() };
    }

    void park_expert_memory() {
        if (!park_window_enabled()) {
            return;
        }
        lookahead.drain();
        std::lock_guard<std::mutex> lk(park_mutex);
        if (parked.load(std::memory_order_relaxed)) {
            return;
        }
        // End of a round: window contents have churned during decode, so the placement records are invalid.
        prewarm_ready = false;
        prewarm_placed.clear();
        int n_ok = 0;
        for (ggml_backend_buffer_t buf : expert_buffers()) {
            metal_buf_fn fn = lookup_metal_buf_fn(buf, "ggml_backend_metal_buffer_park");
            if (fn && fn(buf)) {
                n_ok++;
            }
        }
        for (auto & st : layers) {
            if (st.enabled) {
                clear_layer_cache(st);
            }
        }
        was_window = false;
        parked.store(true, std::memory_order_release);
        moe_log("park expert memory: %d metal buffer(s)", n_ok);
    }

    void ensure_unparked() {
        if (!parked.load(std::memory_order_acquire)) {
            return;
        }
        std::lock_guard<std::mutex> lk(park_mutex);
        if (!parked.load(std::memory_order_relaxed)) {
            return;
        }
        for (ggml_backend_buffer_t buf : expert_buffers()) {
            metal_buf_fn fn = lookup_metal_buf_fn(buf, "ggml_backend_metal_buffer_unpark");
            if (fn && !fn(buf)) {
                throw std::runtime_error("failed to unpark MoE expert Metal buffer");
            }
        }
        parked.store(false, std::memory_order_release);
        moe_log("unparked expert memory");
    }
};

void llama_moe_slot_runtime::map_custom(
        ggml_tensor * dst,
        const ggml_tensor * src,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * data = static_cast<llama_moe_slot_runtime::impl::map_userdata *>(userdata);
    // This runs during graph execution (CPU custom op), so its runtime is directly on the critical path.
    // It runs once per layer, 40 times per token; measuring it reveals whether it is a bottleneck.
    const int64_t t0 = now_ns();
    data->runtime->process(data->layer, src, dst);
    data->runtime->add_process_ns(now_ns() - t0);
}

void llama_moe_slot_runtime::map_custom_lookahead(
        ggml_tensor * dst,
        const ggml_tensor * a,
        const ggml_tensor * b,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * data = static_cast<llama_moe_slot_runtime::impl::map_userdata *>(userdata);
    const int64_t t0 = now_ns();
    data->runtime->process(data->layer, a, dst, b);
    data->runtime->add_process_ns(now_ns() - t0);
}

llama_moe_slot_runtime::llama_moe_slot_runtime(const llama_model & model) :
    pimpl(std::make_unique<impl>(model)) {
    for (auto & item : pimpl->userdata) {
        item.runtime = this;
    }
}

llama_moe_slot_runtime::~llama_moe_slot_runtime() = default;

bool llama_moe_slot_runtime::uses_layer(int layer) const {
    return layer >= 0 && layer < (int) pimpl->layers.size() && pimpl->layers[layer].enabled;
}

int llama_moe_slot_runtime::matmul_split() const {
    return pimpl->matmul_split;
}

/// A2 phase B node: copy slot_ids to an output (so this node naturally depends on phase A),
/// and also depend on the preceding experts' matmul output `b`, causing the scheduler to place it after the preceding section finishes
/// and before the following matmul.
void llama_moe_slot_runtime::map_custom_deferred(
        ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b,
        int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(b);
    if (ith != 0) {
        return;
    }
    auto * data = static_cast<llama_moe_slot_runtime::impl::map_userdata *>(userdata);
    memcpy(dst->data, a->data, ggml_nbytes(a));
    data->runtime->process_deferred(data->layer);
}

ggml_tensor * llama_moe_slot_runtime::build_deferred_trigger(
        ggml_context * ctx, ggml_tensor * slot_ids, ggml_tensor * dep, int layer) {
    return ggml_map_custom2(ctx, slot_ids, dep, map_custom_deferred, 1, &pimpl->userdata[(size_t) layer]);
}

void llama_moe_slot_runtime::add_process_ns(int64_t ns) {
    pimpl->process_ns.fetch_add(ns, std::memory_order_relaxed);
}

bool llama_moe_slot_runtime::lookahead_prefetch_enabled() const {
    return pimpl->lookahead.enabled;
}

int llama_moe_slot_runtime::lookahead_prefetch_depth() const {
    return pimpl->lookahead.enabled ? (int) pimpl->lookahead.depth : 0;
}

int llama_moe_slot_runtime::decode_window() const {
    return (int) pimpl->decode_window;
}

int llama_moe_slot_runtime::window_max_rows() const {
    return (int) pimpl->window_max_rows;
}

void llama_moe_slot_runtime::park_expert_memory() {
    pimpl->park_expert_memory();
}

void llama_moe_slot_runtime::ensure_unparked() {
    pimpl->ensure_unparked();
}

void llama_moe_slot_runtime::prewarm_experts() {
    pimpl->prewarm_experts();
}

llama_moe_slot_runtime::counters llama_moe_slot_runtime::counters_snapshot() const {
    counters c;
    c.process  = pimpl->process_ns.load(std::memory_order_relaxed);
    c.readids  = pimpl->readids_ns.load(std::memory_order_relaxed);
    c.install  = pimpl->install_ns.load(std::memory_order_relaxed);
    c.writeids = pimpl->writeids_ns.load(std::memory_order_relaxed);
    c.miss     = (int64_t) pimpl->stats.n_miss.load(std::memory_order_relaxed);
    c.bytes    = (int64_t) pimpl->stats.n_bytes.load(std::memory_order_relaxed);
    c.unique   = pimpl->unique_sum.load(std::memory_order_relaxed);
    return c;
}

void llama_moe_slot_runtime::set_lookahead_prefetch_depth(int depth) {
    pimpl->lookahead.set_depth(depth);
}

void llama_moe_slot_runtime::set_lookahead_pred(ggml_tensor * pred_ids) {
    pimpl->lookahead_pending = pred_ids;
}

ggml_tensor * llama_moe_slot_runtime::build_slot_ids_tensor(
        ggml_context * ctx,
        ggml_tensor * expert_ids,
        int layer) {
    if (!uses_layer(layer)) {
        return expert_ids;
    }
    ggml_tensor * pred = pimpl->lookahead_pending;
    pimpl->lookahead_pending = nullptr;
    if (pimpl->lookahead.enabled && pred) {
        return ggml_map_custom2(ctx, expert_ids, pred, map_custom_lookahead, 1, &pimpl->userdata[(size_t) layer]);
    }
    return ggml_map_custom1(ctx, expert_ids, map_custom, 1, &pimpl->userdata[layer]);
}

void llama_moe_slot_runtime::process(
        int layer,
        const ggml_tensor * expert_ids,
        ggml_tensor * slot_ids,
        const ggml_tensor * lookahead_pred) {
    pimpl->ensure_unparked();
    if (!uses_layer(layer) ||
        expert_ids->type != GGML_TYPE_I32 ||
        slot_ids->type != GGML_TYPE_I32 ||
        expert_ids->ne[0] != slot_ids->ne[0] ||
        expert_ids->ne[1] != slot_ids->ne[1]) {
        throw std::runtime_error(format(
            "invalid MoE slot map tensors at layer %d type=%s/%s shape=%lldx%lld vs %lldx%lld",
            layer,
            ggml_type_name(expert_ids ? expert_ids->type : GGML_TYPE_COUNT),
            ggml_type_name(slot_ids ? slot_ids->type : GGML_TYPE_COUNT),
            expert_ids ? (long long) expert_ids->ne[0] : -1,
            expert_ids ? (long long) expert_ids->ne[1] : -1,
            slot_ids ? (long long) slot_ids->ne[0] : -1,
            slot_ids ? (long long) slot_ids->ne[1] : -1));
    }

    auto & state = pimpl->layers[layer];

    const int32_t n_rows = (int32_t) expert_ids->ne[1];

    const bool decode = n_rows == 1;
    // The window also covers small batches such as speculative verification (batch width = n_max+1): they use
    // the same mul_mv_id path and per-layer `.dec` tensors as single-token decode, reusing the same experts across rounds.
    // Large batches (prefill) do not use it — they use the shared prefill view and can only use [0, prefill_cap).
    //
    // The upper bound is clamped by window_max_rows according to cache capacity (see resolve_decode_window),
    // eliminating the risk that the union will not fit at runtime.
    const bool window_mode = pimpl->decode_window > 0 && n_rows <= pimpl->window_max_rows;
    const bool lookahead_step = pimpl->lookahead.enabled && decode;

    // In auto mode, choose the expert-cache policy by phase: bypass during prefill, do not bypass during decode.
    //
    // Use batch width as the criterion rather than window_mode — the latter also requires the decode window to be enabled. When the window is off,
    // decode still exists, so using window_mode would bypass the cache during decode too. This criterion shares its source with llama-context's
    // moe_decode_phase. This code has no effect unless auto mode is enabled (the default is not auto).
    if (llama_moe_slot_runtime::impl::nocache_mode() < 0) {
        const bool want = n_rows > pimpl->window_max_rows;
        if (pimpl->want_nocache.load(std::memory_order_relaxed) != want) {
            pimpl->want_nocache.store(want, std::memory_order_relaxed);
            pimpl->apply_experts_nocache(want);
        }
    }

    // Only window mode needs to wait for prefetch to finish before acquiring the lock.
    //
    // In window mode, the prefetch worker locks the target layer mutex to write into its window. If this function already holds
    // state.mutex while waiting for the worker, lock order is reversed and a deadlock occurs (this has happened in practice).
    // In non-window mode, the worker writes only to the trailing ping-pong buffer and touches no layer lock, so take() remains after the lock as before.
    if (lookahead_step && window_mode) {
        std::vector<lookahead_expert_prefetch::hit> ignored;   // Window mode does not return hits
        pimpl->lookahead.take(layer, ignored);
    }

    std::lock_guard<std::mutex> lock(state.mutex);

    // On phase switch, prefill uses the full bank / shared prefill view, which may physically overlap per-layer windows,
    // so all per-layer mappings must be cleared when entering window mode.
    //
    // Clearing only the W slots inside the window is insufficient: prefill leaves many stale expert_to_slot mappings
    // pointing outside the window. process() then wrongly treats those experts as resident and skips loading them,
    // so matmul reads another expert's weights — incorrect output without an error. (This bug occurred in practice.)
    if (window_mode && !pimpl->was_window) {
        for (auto & st : pimpl->layers) {
            if (!st.enabled) {
                continue;
            }
            // The mutex for the current layer (layer 0) is already held by the lock_guard above;
            // locking it again would self-deadlock (std::mutex is not recursive). The mapping is cleared directly below.
            if (&st == &state) {
                continue;
            }
            std::lock_guard<std::mutex> lk(st.mutex);
            llama_moe_slot_runtime::impl::clear_layer_cache(st);
        }
        // Clear the current layer while its lock is held (do not lock it again).
        llama_moe_slot_runtime::impl::clear_layer_cache(state);

        // Cache prewarming: the window has just opened and the cache has just been cleared (once per generation round).
        //
        // If prewarm_experts() was already called before prefill this round (the app's normal path),
        // those experts have already been read into the window, and prefill cannot touch them (decode and prefill regions are physically separate).
        // Only republish the mappings here (integer operations only, zero I/O); otherwise fall back to the old path that reads them now.
        // The current layer 0 lock is already held, so neither function locks the current layer again.
        if (pimpl->prewarm_ready) {
            const size_t n_before = pimpl->prewarm_placed.size();
            pimpl->publish_prewarm_placed(layer);
            pimpl->prewarm_ready = false;   // One-shot: consumed for this round
            moe_log("prewarm: republished %zu prewarmed experts (no I/O; data survived prefill)", n_before);
        } else {
            pimpl->prewarm_windows(layer);
        }
    }
    // The phase criterion must be window_mode, not decode: speculative verification (batch width = n_max+1) and
    // single-token decode are the **same phase** and must not clear the cache between them — otherwise the window
    // is cleared every step and cross-round reuse becomes impossible (measured hit rate dropped to zero).
    pimpl->was_window = window_mode;

    if (pimpl->shared_bank && !window_mode) {
        llama_moe_slot_runtime::impl::clear_layer_cache(state);
    }
    std::vector<int32_t> experts;
    {
        const int64_t t0 = now_ns();
        pimpl->read_i32_ids(expert_ids, experts);
        pimpl->readids_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
    const size_t count = experts.size();
    std::vector<int32_t> slots(count);
    std::vector<uint8_t> requested(pimpl->n_expert, 0);

    // Diagnostic: identify which positions in this token will miss (the expert is not cached yet).
    // Slot allocation has not happened yet, so use expert_to_slot only to test residency.
    if (window_mode && expert_ids->ne[1] == 1) {
        pimpl->miss_pos_steps++;
        for (size_t i = 0; i < count && i < 16; ++i) {
            if (experts[i] >= 0 && experts[i] < pimpl->n_expert &&
                state.expert_to_slot[experts[i]] < 0) {
                pimpl->miss_pos[i]++;
            }
        }
    }

    for (size_t i = 0; i < count; ++i) {
        if (experts[i] < 0 || experts[i] >= pimpl->n_expert) {
            throw std::runtime_error("MoE expert id is out of range");
        }
        requested[experts[i]] = 1;
    }

    if (pimpl->lookahead.enabled && !decode) {
        pimpl->lookahead.reset();
    }
    // Non-window mode: take prefetch results here (hits are in the ping-pong region and must be moved to target slots).
    // In window mode, the worker has already written results directly into the target window, so nothing is needed here.
    if (lookahead_step && !window_mode) {
        std::vector<lookahead_expert_prefetch::hit> hits;
        pimpl->lookahead.take(layer, hits);
        for (const auto & h : hits) {
            if (h.expert < 0 || h.expert >= pimpl->n_expert) {
                continue;
            }
            if (!requested[(size_t) h.expert] || state.expert_to_slot[h.expert] >= 0) {
                continue;
            }
            state.slot_to_expert[h.slot] = h.expert;
            state.expert_to_slot[h.expert] = h.slot;
            state.slot_age[h.slot] = ++state.age;
            pimpl->lookahead.n_hits.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Allocation range. In window mode, find free slots/evictions only within this layer's window — this is the key idea:
    // each layer has its own slot range, so LRU is independent across layers and cached entries survive across tokens.
    // Non-window mode still uses the whole bank (reserving the trailing ping-pong region when prefetch is enabled).
    int32_t alloc_lo = 0;
    int32_t alloc_hi = pimpl->n_slots;
    if (window_mode) {
        alloc_lo = state.window_start;
        // Partitioned mode: the main path allocates only from the cache region, leaving staging slots for prefetch (no competition).
        // Unified mode: the main path may use the whole window, so used experts naturally enter the cross-token LRU
        // and are not flushed when the prefetch region is refilled every step.
        // Use this layer's width for boundaries (widths vary by layer; see layer_state::window_width).
        const int32_t w_il = state.window_width > 0
            ? state.window_width : (pimpl->decode_window + pimpl->prefetch_slots);
        alloc_hi = state.window_start + w_il -
                   (pimpl->prefetch_unified ? 0 : pimpl->prefetch_slots);
    } else if (lookahead_step) {
        alloc_hi = pimpl->lookahead.slot_floor();
    }

    // The point where cache and prediction combine (zero I/O cost):
    // Some experts predicted by the previous layer are not used in this layer now, but will be used next step. Eviction must
    // avoid them — otherwise the oldest entry might be one just predicted to be used and would be evicted.
    //
    // This is not prefetch (no bytes are read); it uses existing prediction signals to improve eviction decisions.
    std::vector<uint8_t> keep;
    if (window_mode && pimpl->lookahead.enabled) {
        std::vector<int32_t> pred;
        {
            std::lock_guard<std::mutex> lk(pimpl->pred_mtx);
            auto & slot = pimpl->predicted[(size_t) layer];
            if (!slot.empty()) {
                pred.swap(slot);
            }
        }
        if (!pred.empty()) {
            keep.assign((size_t) pimpl->n_expert, 0);
            for (int32_t e : pred) {
                if (e >= 0 && e < pimpl->n_expert) {
                    keep[(size_t) e] = 1;
                }
            }
        }
    }

    std::vector<llama_moe_slot_runtime::impl::pending_install> pending;
    pending.reserve((size_t) std::min((int32_t) pimpl->n_slots, pimpl->n_expert));

    // A2: use **rank** to decide whether to read now or defer. Missing experts among the last matmul_split ranks
    // are deferred to phase B — measurements show misses are concentrated near the end (rank 7 has 7 times
    // as many misses as rank 0), where an actual read is most likely to be needed.
    //
    // Split by **fixed rank** rather than sorting by residency, to preserve summation order —
    // this allows byte-for-byte output verification.
    //
    // The criterion must use rank (pos % n_used), not the linear index: the flattened order is
    // pos = rank + n_used * row (ne[0] changes fastest; see read_i32_ids), so
    // "the last K linear indices" across multiple rows would cover all ranks only in the last K/n_used rows, whereas the graph split
    // (slice_ids in build_moe_ffn) splits ranks [split_at, n_used) across all rows.
    // If the two do not match, phase A may read a slot before phase B has written it (silently using incorrect expert weights).
    // For a single token, pos == rank, so both approaches are equivalent.
    auto & defer_list = pimpl->deferred[(size_t) layer];
    defer_list.clear();
    const int32_t n_used_row = (int32_t) expert_ids->ne[0];
    const int32_t split_at   = n_used_row - pimpl->matmul_split;
    const bool defer_by_rank = window_mode && pimpl->matmul_split > 0 && split_at > 0;

    for (size_t pos = 0; pos < count; ++pos) {
        const int32_t expert = experts[pos];
        if (state.expert_to_slot[expert] >= 0) {
            slots[pos] = state.expert_to_slot[expert];   // Already resident; no read needed
            continue;
        }

        int32_t slot = -1;
        for (int32_t candidate = alloc_lo; candidate < alloc_hi; ++candidate) {
            if (state.slot_to_expert[candidate] == kSlotFree) {
                slot = candidate;
                break;
            }
        }
        if (slot < 0) {
            // Skip kSlotPending during eviction: the prefetch worker is currently writing that slot;
            // evicting it would make the worker write into a slot that already belongs to someone else.
            //
            // Two-pass selection: pass 0 considers only entries not predicted to be used next step (keep[] == 0);
            // pass 1 relaxes this constraint. This lets predicted entries survive one more step at zero I/O cost.
            for (int pass = 0; pass < 2 && slot < 0; ++pass) {
                uint64_t oldest = UINT64_MAX;
                for (int32_t candidate = alloc_lo; candidate < alloc_hi; ++candidate) {
                    const int32_t resident = state.slot_to_expert[candidate];
                    if (resident == kSlotPending) {
                        continue;
                    }
                    if (pass == 0 && !keep.empty() && keep[(size_t) resident]) {
                        continue;   // Predicted to be used next step; leave it alone for now
                    }
                    if (!requested[resident] && state.slot_age[candidate] < oldest) {
                        oldest = state.slot_age[candidate];
                        slot = candidate;
                    }
                }
                if (pass == 0) {
                    if (slot >= 0) {
                        pimpl->lookahead.n_evict_saved.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        // No candidates were found in pass 0 — all candidates are in the prediction set,
                        // meaning the "protect predicted entries" policy is genuinely constraining eviction.
                        pimpl->lookahead.n_evict_forced.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
        if (slot < 0) {
            throw std::runtime_error(format(
                "layer %d requires more than %d unique experts in one graph; reduce ubatch or increase --moe-slot-bank",
                layer, alloc_hi - alloc_lo));
        }

        const int32_t evicted = state.slot_to_expert[slot];
        if (evicted >= 0) {
            state.expert_to_slot[evicted] = kSlotFree;
        }
        state.slot_age[slot] = ++state.age;
        slots[pos] = slot;
        const bool defer_this = defer_by_rank &&
            (int32_t) (pos % (size_t) n_used_row) >= split_at;
        if (defer_this) {
            // Defer: mark the slot kSlotPending to reserve it (so nothing else touches it); phase B reads and publishes it later.
            state.slot_to_expert[slot] = kSlotPending;
            defer_list.push_back({ expert, slot });
        } else {
            state.slot_to_expert[slot] = expert;
            state.expert_to_slot[expert] = slot;
            pending.push_back({ expert, slot });
        }
    }

    int n_unique = 0;
    for (int32_t expert = 0; expert < pimpl->n_expert; ++expert) {
        if (requested[expert]) {
            n_unique++;
        }
    }
    pimpl->unique_sum.fetch_add(n_unique, std::memory_order_relaxed);

    const llama_moe_slot_runtime::impl::install_timing timing =
        pimpl->install_pending(state, pending, window_mode);
    pimpl->install_ns.fetch_add(timing.wall_ns, std::memory_order_relaxed);

    if (!pending.empty()) {
        // Statistics only; does not affect the read path. Use per-layer deltas to avoid double-counting cumulative cur values.
        const auto st = llama_moe_slot_runtime::impl::accumulate_runs(pending);
        llama_moe_slot_runtime::impl::add_runs(pimpl->cur.runs, st);
        llama_moe_slot_runtime::impl::add_runs(pimpl->g_runs, st);
    }

    if (lookahead_step && lookahead_pred) {
        std::vector<int32_t> pred;
        pimpl->read_i32_ids(lookahead_pred, pred);
        // Cache + prediction integration point (zero I/O): retain predictions for the target layer,
        // so its next eviction can avoid those entries. This is not prefetch — no bytes are read.
        {
            std::lock_guard<std::mutex> lk(pimpl->pred_mtx);
            pimpl->predicted[(size_t) layer + 1] = pred;
        }
        // Before arm(), remove predicted experts that are already in the target layer cache.
        //
        // They do not need to be read (the worker also skips them after checking the map), but they make the arm batch appear larger
        // and keep the worker busy longer — and arm() is discarded immediately (pfskip) while the worker is busy.
        // Measurements showed about 9 arm calls per token were discarded this way (22% of 40), wasting predictions.
        // Filtering first sends only experts that really need reading to the worker, making batches smaller, faster, and less likely to be dropped.
        std::vector<int32_t> need_read;
        need_read.reserve(pred.size());
        {
            auto & tgt = pimpl->layers[(size_t) layer + 1];
            std::lock_guard<std::mutex> lk(tgt.mutex);
            for (int32_t e : pred) {
                if (e >= 0 && e < pimpl->n_expert && tgt.expert_to_slot[(size_t) e] < 0) {
                    need_read.push_back(e);
                }
            }
        }
        pimpl->lookahead.n_pred_cached.fetch_add(pred.size() - need_read.size(),
                                                 std::memory_order_relaxed);
        if (!need_read.empty()) {
            pimpl->lookahead.arm(layer + 1, need_read);
        }
        // Record predictions for layer+1 for use in that layer's sweep record.
        // Predictions are computed from this layer's hidden state (see qwen35moe.cpp), so
        // layer 0 never receives predictions — the structural upper bound for prefetch coverage is 39/40.
        if (pimpl->sweep_fp && layer + 1 < (int) pimpl->sweep_pred.size()) {
            pimpl->sweep_pred[layer + 1] = pred;
        }
    }

    // Hit-rate sweep: during decode, record all E experts actually used; during prefill, record only E for the final token
    // (to test whether prefill can leave an initial cache for decode).
    if (pimpl->sweep_fp) {
        if (decode) {
            std::vector<int32_t> used;
            used.reserve(32);
            for (int32_t e = 0; e < pimpl->n_expert; ++e) {
                if (requested[e]) {
                    used.push_back(e);
                }
            }
            pimpl->sweep_write(layer, used, pimpl->sweep_pred[layer]);
        } else {
            pimpl->sweep_write_prefill(layer, experts.data(),
                                       (int) expert_ids->ne[0], (int) expert_ids->ne[1]);
        }
    }

    const int n_tokens = (int) expert_ids->ne[1];
    const int n_used = (int) expert_ids->ne[0];
    const int n_miss = (int) pending.size();
    const int n_hit = n_unique - n_miss;

    if (layer == 0) {
        if (pimpl->cur.n_layers > 0) {
            pimpl->flush_ubatch(pimpl->cur.n_tokens <= 1 ? "decode" : "prefill",
                pimpl->cur.n_tokens <= 1 ? pimpl->n_decode_step : pimpl->n_prefill_ubatch);
        }
        if (n_tokens <= 1) {
            pimpl->n_decode_step++;
            pimpl->miss_per_layer_steps++;   // Once per decode step (not once per layer)
        } else {
            pimpl->n_prefill_ubatch++;
        }
        pimpl->cur.n_tokens = n_tokens;
        pimpl->cur.n_used = n_used;
    }

    pimpl->cur.n_layers++;
    pimpl->cur.unique_sum += n_unique;
    pimpl->cur.miss_sum += n_miss;
    pimpl->cur.n_th = std::max(pimpl->cur.n_th, timing.n_th);
    pimpl->cur.pack_layers += state.pack_ok ? 1 : 0;
    pimpl->cur.wall_ns += timing.wall_ns;
    pimpl->cur.pread_sum_ns += timing.pread_sum_ns;
    pimpl->cur.bytes += timing.bytes;
    if (window_mode && n_tokens <= 1 && layer < (int) pimpl->miss_per_layer.size()) {
        pimpl->miss_per_layer[(size_t) layer] += (uint64_t) n_miss;
    }

    const uint64_t ubatch_idx = n_tokens <= 1 ? pimpl->n_decode_step : pimpl->n_prefill_ubatch;
    const bool sample_layer = (layer == 0 || layer == 1 ||
        layer == pimpl->last_moe_layer / 2 || layer == pimpl->last_moe_layer);
    if (ubatch_idx <= 2 && sample_layer && (n_miss > 0 || layer == 0)) {
        moe_log("layer l=%d tok=%d used=%d unique=%d miss=%d hit=%d pack=%d nth=%d "
                "pread_wall=%.1fms pread_sum=%.1fms pread_max=%.1fms bytes=%.1fMiB par=%.2f",
            layer, n_tokens, n_used, n_unique, n_miss, n_hit,
            timing.pack ? 1 : 0, timing.n_th,
            llama_moe_slot_runtime::impl::ms_of(timing.wall_ns),
            llama_moe_slot_runtime::impl::ms_of(timing.pread_sum_ns),
            llama_moe_slot_runtime::impl::ms_of(timing.pread_max_ns),
            llama_moe_slot_runtime::impl::mib_of(timing.bytes),
            llama_moe_slot_runtime::impl::par_of(timing.pread_sum_ns, timing.wall_ns));
    }

    if (layer == pimpl->last_moe_layer) {
        const char * kind = n_tokens <= 1 ? "decode" : "prefill";
        const uint64_t idx = n_tokens <= 1 ? pimpl->n_decode_step : pimpl->n_prefill_ubatch;
        if (pimpl->should_log_ubatch(n_tokens, idx)) {
            pimpl->flush_ubatch(kind, idx);
        } else {
            pimpl->cur = llama_moe_slot_runtime::impl::ubatch_acc{};
        }
        if (idx == 8 || (idx > 8 && (idx % 8) == 0)) {
            pimpl->log_io(kind);
        }
    }

    // slots[] has already been filled by position in the allocation loop (deferred entries also have valid slot addresses). Here, only update
    // LRU timestamps: later positions are newer, as before.
    for (size_t i = 0; i < count; ++i) {
        if (slots[i] < 0) {
            throw std::runtime_error("MoE expert was not installed");
        }
        state.slot_age[(size_t) slots[i]] = ++state.age;
        // Diagnostic: did this expert come from the prefetch region? That region is refilled every step, so the entry disappears after use —
        // if needed again, it must be reread. Ideally, it should be promoted to the cache region.
        if (window_mode && pimpl->prefetch_slots > 0 &&
            slots[i] >= state.window_start + (state.window_width > 0
                        ? state.window_width - pimpl->prefetch_slots : pimpl->decode_window)) {
            pimpl->lookahead.n_used_from_pf.fetch_add(1, std::memory_order_relaxed);
        }
    }
    {
        const int64_t t0 = now_ns();
        pimpl->write_i32_ids(slot_ids, slots);
        pimpl->writeids_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
}

/// A2 phase B: read missing experts deferred during phase A, then publish them into the window.
///
/// Called after the earlier experts (positions < n - matmul_split) have finished computing.
/// The GPU is then computing those resident experts while these reads happen concurrently — this is the source of A2's benefit.
///
/// Locking: disk reads do not hold a lock (the slots were marked kSlotPending in phase A and cannot be touched elsewhere); only publishing
/// (writing expert_to_slot / slot_to_expert) takes the layer lock, matching the prefetch worker's approach.
void llama_moe_slot_runtime::process_deferred(int layer) {
    if (layer < 0 || layer >= (int) pimpl->deferred.size()) {
        return;
    }
    auto & list = pimpl->deferred[(size_t) layer];
    if (list.empty()) {
        return;
    }
    auto & st = pimpl->layers[(size_t) layer];
    if (!st.enabled) {
        list.clear();
        return;
    }

    for (const auto & d : list) {
        pimpl->read_expert_directv(st, d.expert, d.slot, true);
    }
    {
        std::lock_guard<std::mutex> lk(st.mutex);
        for (const auto & d : list) {
            st.slot_to_expert[(size_t) d.slot] = d.expert;
            st.expert_to_slot[(size_t) d.expert] = d.slot;
        }
        st.n_prefetch_published += list.size();
    }
    list.clear();
}
