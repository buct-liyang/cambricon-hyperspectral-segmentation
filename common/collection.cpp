#include "collection.h"

#include <memory>
#include <string>
#include <utility>

void Collection::Add(const std::string& tag, std::unique_ptr<any>&& value) {
  RwLockWriteGuard lk(rw_lock_);
  if (data_.end() != data_.find(tag)) {
#if !defined(_LIBCPP_NO_RTTI)
    LOGF(COLLECTION) << "Data tagged by [" << tag << "] had been added, and value type is ["
                     << data_[tag]->type().name() << "]. Current type is [" << value->type().name() << "].";
#else
    LOGF(COLLECTION) << "Data tagged by [" << tag << "] had been added.";
#endif
  }
  data_[tag] = std::forward<std::unique_ptr<any>>(value);
}

bool Collection::AddIfNotExists(const std::string& tag, std::unique_ptr<any>&& value) {
  RwLockWriteGuard lk(rw_lock_);
  if (data_.end() != data_.find(tag)) {
    VLOG2(COLLECTION) << "Data tagged by [" << tag << "] had been added. Current data will not be added.";
    return false;
  }
  data_[tag] = std::forward<std::unique_ptr<any>>(value);
  return true;
}

bool Collection::HasValue(const std::string& tag) {
  RwLockReadGuard lk(rw_lock_);
  return data_.end() != data_.find(tag);
}

#if !defined(_LIBCPP_NO_RTTI)
const std::type_info& Collection::Type(const std::string& tag) {
  RwLockReadGuard lk(rw_lock_);
  auto iter = data_.find(tag);
  if (data_.end() == iter) {
    LOGF(COLLECTION) << "No data tagged by [" << tag << "] was been added.";
  }
  return iter->second->type();
}
#endif
