#include <coroutine>
#include <vector>
#include <utility>
#include <iostream>
#include <thread>
#include <semaphore>

template <typename... Args>
struct Task
{
    struct promise;
private:
    using handle = std::coroutine_handle<promise>;
    handle handle_;
public:
    Task(void (*coroutine)(Args...), Args... args)
    {
        coroutine(args...);
        handle_ = handle::from_address(std::exchange(promise::current_address, nullptr));
    }

    ~Task() { if(handle_) handle_.destroy(); }

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    Task(Task&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr))
    {
    }

    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy(); 
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    void resume() const { handle_.resume(); } 
    bool done() const { return handle_.done(); } 

    struct promise
    {
        static auto initial_suspend() { return std::suspend_always(); }
        static auto final_suspend() noexcept { return std::suspend_always(); }
        static auto yield_value(int) noexcept { return std::suspend_always(); }
        static void return_void() {}
        static void unhandled_exception() { throw; }
        thread_local static void* current_address;
        void get_return_object()
        { 
            current_address = handle::from_promise(*this).address();
        }
    };
};

template <typename... Args>
thread_local void* Task<Args ...>::promise::current_address = nullptr;

template <typename... Args>
struct std::coroutine_traits<void, Args...> {
    using promise_type = Task<Args...>::promise;
};

#define __global__ static
#define __shared__ thread_local
#define __syncthreads() co_yield 0

__global__ void kernel_function(int* x, double y)
{
    __shared__ int sdata;
    //std::cout << "start" << std::endl;
    __syncthreads();
    //std::cout << "finish" << std::endl;
    for(int i=0; i<10; ++i) {
        __syncthreads();
        sdata += 1;
        //std::cout << "doing: " << i << " sdata: " << sdata << std::endl;
    }
}

constexpr unsigned char max_cores = std::numeric_limits<unsigned char>::max();
std::counting_semaphore<max_cores> core_semaphore(std::thread::hardware_concurrency());


struct CoreManager
{
    static void acquire_core() {
        core_semaphore.acquire();
    }

    static void release_core() {
        core_semaphore.release();
    }

};

struct CoreReleaser {
    ~CoreReleaser() { CoreManager::release_core(); }
};

void run_block()
{
    const int threadDim = 256;

    CoreReleaser releaser;

    int x;
    std::vector<Task<int*, double>> tasks;
    for(int i=0; i<threadDim; ++i)
        tasks.emplace_back(&kernel_function, &x, 0.1);

    bool all_done = false;
    while(!all_done)
    {
        all_done = true;
        for(const auto& task : tasks)
        {
            if (!task.done()) {
                task.resume();
                all_done = false;
            }
        }
    }
}

int main()
{
    const auto blockDim = 1024;

    std::vector<std::thread> threads;
    for(int i=0; i<blockDim; ++i)
    {
        CoreManager::acquire_core();
        threads.emplace_back(run_block);
    }

    for(auto& thread: threads)
    {
        thread.join();
    }

    return 0; 
}