#include "util.hpp"

#include <boost/format.hpp>
#include <boost/log/trivial.hpp>

using namespace arba;

scoped_timed_log::scoped_timed_log(const std::string &message)
  : message(message), start(boost::chrono::high_resolution_clock::now())
{
}

scoped_timed_log::~scoped_timed_log()
{
    const auto end = boost::chrono::high_resolution_clock::now();
    const auto duration =
      boost::chrono::duration_cast<boost::chrono::milliseconds>(end - start);
    BOOST_LOG_TRIVIAL(info)
      << boost::format("%1% (%2% ms)") % message % duration.count();
}