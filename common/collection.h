#ifndef MODULE_COLLECTION_HPP_
#define MODULE_COLLECTION_HPP_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "logging.h"
#include "any.h"
#include "rwlock.h"

// Used by FrameInfo::Collection, the tags of data used by modules
static constexpr char kDataFrameTag[] = "DataFrame"; /*!< value type in FrameInfo::Collection : DataFramePtr. */
static constexpr char kInferObjsTag[] = "InferObjs"; /*!< value type in FrameInfo::Collection : InferObjsPtr. */
static constexpr char kImageIDTag[] = "ImageID";
static constexpr char kImageNameTag[] = "ImageName";

class NonCopyable {
 protected:
  /*!
   * @brief Constructs an instance with empty value.
   *
   * @param None.
   *
   * @return  None.
   */
  NonCopyable() = default;
  /*!
   * @brief Destructs an instance.
   *
   * @param None.
   *
   * @return  None.
   */
  ~NonCopyable() = default;

 private:
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable(NonCopyable &&) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
  NonCopyable &operator=(NonCopyable &&) = delete;
};

/**
 * @class Collection
 *
 * @brief Collection is a class storing structured data of variable types.
 *
 * @note This class is thread safe.
 */
class Collection : public NonCopyable {
 public:
  /*!
   * @brief Constructs an instance with empty value.
   *
   * @return  No return value.
   */
  Collection() = default;
  /*!
   * @brief Destructs an instance.
   *
   * @return  No return value.
   */
  ~Collection() = default;
  /**
   * @brief Gets the reference to the object of typename ValueT if it exists, otherwise crashes.
   *
   * @param[in] tag The unique identifier of the data.
   *
   * @return Returns the reference to the object of typename ValueT which is tagged by `tag`.
   */
  template <typename ValueT>
  ValueT& Get(const std::string& tag);
  /**
   * @brief Adds data tagged by `tag`. Crashes when there is already a piece of data tagged by `tag`.
   *
   * @param[in] tag The unique identifier of the data.
   * @param[in] value Value to be add.
   *
   * @return Returns the reference to the object of typename ValueT which is tagged by `tag`.
   */
  template <typename ValueT>
  ValueT& Add(const std::string& tag, const ValueT& value);
  /**
   * @brief Adds data tagged by `tag` using move semantics. Crashes when there is already a piece of data tagged by
   * `tag`.
   *
   * @param[in] tag The unique identifier of the data.
   * @param[in] value Value to be add.
   *
   * @return Returns the reference to the object of typename ValueT which is tagged by `tag`.
   */
  template <typename ValueT>
  ValueT& Add(const std::string& tag, ValueT&& value);

  /**
   * @brief Adds data tagged by `tag`, only if there is no piece of data tagged by `tag`.
   *
   * @param[in] tag The unique identifier of the data.
   * @param[in] value Value to be add.
   *
   * @return Returns true if the data is added successfully, otherwise returns false.
   */
  template <typename ValueT>
  bool AddIfNotExists(const std::string& tag, const ValueT& value);
  /**
   * @brief Adds data tagged by `tag` using move semantics, only if there is no piece of data tagged by
   * `tag`.
   *
   * @param[in] tag The unique identifier of the data.
   * @param[in] value Value to be add.
   *
   * @return Returns true if the data is added successfully, otherwise returns false.
   */
  template <typename ValueT>
  bool AddIfNotExists(const std::string& tag, ValueT&& value);

  /**
   * @brief Checks whether there is the data tagged by `tag`.
   *
   * @param[in] tag The unique identifier of the data.
   *
   * @return Returns true if there is already a piece of data tagged by `tag`, otherwise returns false.
   */
  bool HasValue(const std::string& tag);

#if !defined(_LIBCPP_NO_RTTI)
  /**
   * @brief Gets type information for data tagged by `tag`.
   *
   * @param[in] tag The unique identifier of the data.
   *
   * @return Returns type information of the data tagged by `tag`.
   */
  const std::type_info& Type(const std::string& tag);

  /**
   * @brief Checks if the type of data tagged by `tag` is `ValueT` or not.
   *
   * @param tag The unique identifier of the data.
   *
   * @return Returns true if the type of data tagged by `tag` is ``ValueT``, otherwise returns false.
   */
  template <typename ValueT>
  bool TaggedIsOfType(const std::string& tag);
#endif

 private:
  void Add(const std::string& tag, std::unique_ptr<any>&& value);
  bool AddIfNotExists(const std::string& tag, std::unique_ptr<any>&& value);

 private:
  std::map<std::string, std::unique_ptr<any>> data_;
  RwLock rw_lock_;
};  // class Collection

template <typename ValueT>
ValueT& Collection::Get(const std::string& tag) {
  RwLockReadGuard lk(rw_lock_);
  auto iter = data_.find(tag);
  if (data_.end() == iter) {
    LOGF(COLLECTION) << "No data tagged by [" << tag << "] has been added.";
  }
  try {
    return any_cast<ValueT&>(*iter->second);
  } catch (bad_any_cast& e) {
#if !defined(_LIBCPP_NO_RTTI)
    LOGF(COLLECTION) << "The type of data tagged by [" << tag << "]  is [" << iter->second->type().name()
                     << "]. Expect type is [" << typeid(ValueT).name() << "].";
#else
    LOGF(COLLECTION) << "The type of data tagged by [" << tag << "] is not the expected data type."
#endif
  }

  // never be here.
  return any_cast<ValueT&>(*iter->second);
}

template <typename ValueT>
inline ValueT& Collection::Add(const std::string& tag, const ValueT& value) {
  Add(tag, std::unique_ptr<any>(new any(value)));
  return Get<ValueT>(tag);
}

template <typename ValueT>
inline ValueT& Collection::Add(const std::string& tag, ValueT&& value) {
  Add(tag, std::unique_ptr<any>(new any(std::forward<ValueT>(value))));
  return Get<ValueT>(tag);
}

template <typename ValueT>
inline bool Collection::AddIfNotExists(const std::string& tag, const ValueT& value) {
  return AddIfNotExists(tag, std::unique_ptr<any>(new any(value)));
}

template <typename ValueT>
inline bool Collection::AddIfNotExists(const std::string& tag, ValueT&& value) {
  return AddIfNotExists(tag, std::unique_ptr<any>(new any(std::forward<ValueT>(value))));
}

#if !defined(_LIBCPP_NO_RTTI)
template <typename ValueT>
inline bool Collection::TaggedIsOfType(const std::string& tag) {
  if (!HasValue(tag)) return false;
  return typeid(ValueT) == Type(tag);
}
#endif

using CollectionPtr = std::shared_ptr<Collection>;

#endif  // COLLECTION_HPP_
