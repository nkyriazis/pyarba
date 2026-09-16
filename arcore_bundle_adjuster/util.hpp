#pragma once

#include <boost/chrono.hpp>
#include <string>

namespace arba
{
struct scoped_timed_log
{
    scoped_timed_log(const std::string &message);
    ~scoped_timed_log();

   private:
    const std::string message;
    boost::chrono::high_resolution_clock::time_point start;
};
} // namespace arba