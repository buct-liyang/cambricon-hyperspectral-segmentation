#ifndef MODULE_RWLOCK_H_
#define MODULE_RWLOCK_H_

#include <pthread.h>  // for pthread_rwlock_t
#include <memory>
#include <utility>

// FIXME
class RwLock {
 public:
  RwLock() { pthread_rwlock_init(&rwlock, NULL); }
  ~RwLock() { pthread_rwlock_destroy(&rwlock); }
  void wrlock() { pthread_rwlock_wrlock(&rwlock); }
  void rdlock() { pthread_rwlock_rdlock(&rwlock); }
  void unlock() { pthread_rwlock_unlock(&rwlock); }

 private:
  pthread_rwlock_t rwlock;
};

class RwLockWriteGuard {
 public:
  explicit RwLockWriteGuard(RwLock& lock) : lock_(lock) { lock_.wrlock(); }
  ~RwLockWriteGuard() { lock_.unlock(); }

 private:
  RwLock& lock_;
};

class RwLockReadGuard {
 public:
  explicit RwLockReadGuard(RwLock& lock) : lock_(lock) { lock_.rdlock(); }
  ~RwLockReadGuard() { lock_.unlock(); }

 private:
  RwLock& lock_;
};


#endif
