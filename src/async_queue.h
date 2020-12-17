#ifndef ASYNC_QUEUE
#define ASYNC_QUEUE

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>

using namespace std;

using lock_t = unique_lock<mutex>;

class async_queue {
  deque<function<void()>> _q;
  mutex _mutex;
  condition_variable _ready;
  bool _done{false};

 public:
  void done() {
    lock_t lock{_mutex};
    _done = true;
    _ready.notify_all();
  }
  bool pop(function<void()>& x) {
    lock_t lock{_mutex};
    while (_q.empty() && !_done)
      _ready.wait(lock);
    if (_q.empty())
      return false;
    x = move(_q.front());
    _q.pop_front();
    return true;
  }

  bool try_pop(function<void()>& x) {
    lock_t lock{_mutex, try_to_lock};
    if (!lock || _q.empty())
      return false;
    x = move(_q.front());
    _q.pop_front();
    return true;
  }

  template <typename T>
  void push(T&& f) {
    lock_t lock{_mutex};
    _q.emplace_back(forward<T>(f));
    _ready.notify_one();
  }

  template <typename T>
  bool try_push(T&& f) {
    lock_t lock{_mutex, try_to_lock};
    if (!lock)
      return false;
    _q.emplace_back(forward<T>(f));
    _ready.notify_one();
    return true;
  }
};

#endif