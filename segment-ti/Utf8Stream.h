#ifndef UTF8_STREAM_H
#define UTF8_STREAM_H

//MAIN
//#include "LogStream.h"

//STL
#include <iostream>
#include <string>
using namespace std;

class Utf8Stream
{
public:
	Utf8Stream(const string & utf8_src);
	~Utf8Stream(void);

	bool ReadOneCharacter(string & utf8_char_str);

private:

	string m_utf8_src;

	size_t m_byte_idx;

	size_t get_utf8_char_len(char & byte);

};



#endif //UTF8_STREAM_H

