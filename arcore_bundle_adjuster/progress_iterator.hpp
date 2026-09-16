#pragma once

#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/range.hpp>
#include <boost/timer/progress_display.hpp>
#include <memory>

namespace arcore_bundle_adjuster
{
template <typename Iterator>
class progress_iterator
  : public boost::iterator_adaptor<progress_iterator<Iterator>, Iterator>
{
   private:
    friend class boost::iterator_core_access;
    std::shared_ptr<boost::timer::progress_display> progress;

    void increment()
    {
        ++this->base_reference();
        ++*progress;
    }

   public:
    progress_iterator(
      Iterator it,
      size_t size,
      const std::shared_ptr<boost::timer::progress_display>& progress)
      : progress_iterator::iterator_adaptor_(it), progress(progress)
    {
    }
};

template <typename Range>
auto progress(Range&& r, const std::string& message = "")
{
    using iterator_type = decltype(std::begin(r));
    const auto size     = std::size(r);
    auto progress       = std::make_shared<boost::timer::progress_display>(
      size, std::cout, message + ": \n");
    auto begin_it =
      progress_iterator<iterator_type>(std::begin(r), size, progress);
    auto end_it = progress_iterator<iterator_type>(std::end(r), size, progress);
    return boost::make_iterator_range(begin_it, end_it);
}

} // namespace arcore_bundle_adjuster