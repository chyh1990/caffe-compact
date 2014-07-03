#ifndef GLOG_COMPACT_HPP
#define GLOG_COMPACT_HPP

#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>

class LogMessage{
	std::string level;
	std::ostream &ofs;
	public:
		LogMessage(const std::string &l)
			:level(l), ofs(std::cerr){
			stream() << "[" << level << "]\t";
		}
		LogMessage(std::ostream &o)
			:level("ERROR"), ofs(o){
			stream() << "[" << level << "]\t";
		}
		inline std::ostream &stream(){ 
			return ofs;
		}
		~LogMessage() { 
			stream() << std::endl; 
		}
};

#define 	LOG(type)   LogMessage(#type).stream()
#define 	DLOG(type)   LogMessage(#type).stream()
#define 	VLOG(level)   if ((level) <= FLAGS_v) LOG(INFO)

#define CHECK(x) if(x) {} else LOG(ERROR)
#define DCHECK(x) CHECK(x)

#define 	CHECK_EQ(x, y)   CHECK((x) == (y))
#define 	CHECK_LT(x, y)   CHECK((x) < (y))
#define 	CHECK_GT(x, y)   CHECK((x) > (y))
#define 	CHECK_LE(x, y)   CHECK((x) <= (y))
#define 	CHECK_GE(x, y)   CHECK((x) >= (y))
#define 	CHECK_NE(x, y)   CHECK((x) != (y))

#endif
