

#ifndef THREAD_POOL
#define THREAD_POOL
#include <atomic>
#include <thread>
#include <vector>
#include "async_queue.h"

using namespace std;
class thread_pool {
  const unsigned _count{thread::hardware_concurrency()};
  vector<thread> _threads;
  vector<async_queue> _q{_count};
  atomic<unsigned> _index{0};
  const unsigned K = 4;

 public:
  thread_pool() {
    for (unsigned n = 0; n != _count; ++n) {
      _threads.emplace_back([&, n] { run(n); });
    }
  }
  ~thread_pool() {
    for (auto& q : _q)
      q.done();
    for (auto& t : _threads)
      t.join();
  }
  void run(unsigned i) {
    while (true) {
      function<void()> f;
      for (unsigned n; n < _count; ++n)
        if (_q[(i + n) % _count].try_pop(f))
          break;
      if (!f && !_q[i].pop(f))
        break;
      f();
    }
  }

  template <typename F>
  void _async(F&& f) {
    auto index = _index++;
    for (unsigned n; n < _count * K; ++n)
      if (_q[(index + n) % _count].try_push(forward<F>(f)))
        return;
    _q.at(index % _count).push(forward<F>(f));
  }
};

#endif