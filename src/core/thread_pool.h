#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "kernel/kernel_define.h"
#include "utils.h"

// clang-format off
#ifndef INFER_PAUSE
# if defined __GNUC__ && (defined __i386__ || defined __x86_64__)
#   if !defined(__SSE2__)
      static inline void non_sse_mm_pause() { __asm__ __volatile__ ("rep; nop"); }
#     define _mm_pause non_sse_mm_pause
#   else
#       include <immintrin.h>
#   endif
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { _mm_pause(); } } while (0)
# elif defined __GNUC__ && defined __aarch64__
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("yield" ::: "memory"); } } while (0)
# elif defined __GNUC__ && defined __arm__
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("" ::: "memory"); } } while (0)
# elif defined __GNUC__ && defined __riscv
// PAUSE HINT is not part of RISC-V ISA yet, but is under discussion now. For details see:
// https://github.com/riscv/riscv-isa-manual/pull/398
// https://github.com/riscv/riscv-isa-manual/issues/43
// #   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("pause"); } } while (0)
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("nop"); } } while (0)
# else
#   warning "Can't detect 'pause' (CPU-yield) instruction on the target platform. Specify INFER_PAUSE() definition via compiler flags."
#   define INFER_PAUSE(...) do { /* no-op: works, but not effective */ } while (0)
# endif
#endif // MTDA_PAUSE
// clang-format on


namespace lightInfer {

struct Worker {

public:
    Worker(std::function<void()>&& run) : thread{run} {}
    ~Worker() {thread.join();}
    std::thread thread;
    std::atomic<bool> work_flag{false};

};

/**
 * \brief ThreadPool execute the task in multi-threads(nr_threads>1) mode , it
 * will fallback to single-thread mode if nr_thread is 1.
 */
class ThreadPool {

public:
    ThreadPool(uint32_t nr_threads);
    void add_task(const MultiThreadingTask& task, uint32_t nr_task);
    inline void sync();
    inline void active();
    void deactive();
    ~ThreadPool();

    uint32_t nr_threads() const {return m_nr_threads;}

    static constexpr int MAIN_THREAD_ACTIVE_WAIT = 10000;
    static constexpr int WORKER_ACTIVE_WAIT = 2000;
    static constexpr int ACTIVE_WAIT_PAUSE_LIMIT = 16;

private:
    uint32_t m_nr_threads = 1;
    uint32_t m_nr_task = 0;
    uint32_t m_task_per_thread = 0;
    std::atomic_bool m_stop {false};
    std::atomic_bool m_active {false};
    MultiThreadingTask m_task;
    std::vector<Worker*> m_workers;
    std::condition_variable m_cv;
    std::mutex m_mutex;

};

}