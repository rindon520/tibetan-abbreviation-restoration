#include "Utf8Stream.h"


Utf8Stream::Utf8Stream(const string & utf8_src): m_utf8_src(utf8_src),
												 m_byte_idx(0)
{
}


Utf8Stream::~Utf8Stream(void)
{
}


bool Utf8Stream::ReadOneCharacter(string & utf8_char_str)
{
	// return true 表示读取成功，
	// return false 表示已经读取到流末尾
	//utf8_char_str = "";

	if(m_utf8_src.size() == m_byte_idx)
		return false;

	size_t utf8_char_len = get_utf8_char_len(m_utf8_src[m_byte_idx]);

	if( 0 == utf8_char_len )
	{
		cerr << "Get a illegal utf-8 character." << endl;
		utf8_char_str = m_utf8_src[m_byte_idx++];
		return true;
	}

	size_t next_idx = m_byte_idx + utf8_char_len;
	if( m_utf8_src.size() < next_idx )
	{
		cerr << "Get utf8 first byte out of input src string." << endl;
		next_idx = m_utf8_src.size();
	}

	//输出UTF-8的一个字符
	utf8_char_str = m_utf8_src.substr(m_byte_idx, next_idx - m_byte_idx);

	//重置偏移量
	m_byte_idx = next_idx;

	return true;
}


size_t Utf8Stream::get_utf8_char_len(char & byte)
{
	// return 0 表示错误
	// return 1-6 表示正确值
	// 不会 return 其他值 

	//UTF8 编码格式：
	//     U-00000000 - U-0000007F: 0xxxxxxx  
    //     U-00000080 - U-000007FF: 110xxxxx 10xxxxxx  
    //     U-00000800 - U-0000FFFF: 1110xxxx 10xxxxxx 10xxxxxx  
    //     U-00010000 - U-001FFFFF: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx  
    //     U-00200000 - U-03FFFFFF: 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx  
    //     U-04000000 - U-7FFFFFFF: 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx  

	size_t len = 0;
	unsigned char mask = 0x80;

	while( byte & mask )
	{
		len++;
		if( len > 6 )
		{
			cerr << "The mask get len is over 6." << endl;
			return 0;
		}

		mask >>= 1;
	}

	if( 0 == len )
	{
		return 1;
	}

	return len;

}

