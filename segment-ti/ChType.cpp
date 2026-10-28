#include <fstream>
#include <iostream>
#include "ChType.h"

CChType::CChType()
{
}

CChType::~CChType()
{
}

//初始化
bool CChType::Initialize(const string &punc, const string &alph, const string &date, const string &num)
{
	if (!Load(punc, m_sPunctuation))
	{
		cout << "Load punctuation fail" << endl;
		return false;
	}

	if (!Load(alph, m_sAlphabet))
	{
		cout << "Load alphabet fail" << endl;
		return false;
	}

	if (!Load(date, m_sDate))
	{
		cout << "Load date fail" << endl;
		return false;
	}

	if (!Load(num, m_sNumber))
	{
		cout << "Load number fail" << endl;
		return false;
	}

	return true;
}

//加载符号集合
bool CChType::Load(const string &file, set<string> &keys)
{
	ifstream fin(file.c_str());
	if (!fin)
	{
		cout << "Can not open file: " << file << endl;
		return false;
	}
	
	string line;

	while (getline(fin, line))
	{
		set<string>::iterator it = keys.find(line);

		if (it == keys.end())
		{
			keys.insert(line);
		}
	}

	fin.close();
	return true;
}

//字符是否在集合中
bool CChType::InSet(const string &ch, const set<string> &keys)
{
	set<string>::const_iterator it = keys.find(ch);

	return it != keys.end();
}

//获得字符类型
int CChType::GetType(const string &ch)
{
	if (InSet(ch, m_sPunctuation))
	{
		return 0;
	}

	if (InSet(ch, m_sAlphabet))
	{
		return 1;
	}

	if (InSet(ch, m_sDate))
	{
		return 2;
	}

	if (InSet(ch, m_sNumber))
	{
		return 3;
	}

	return 4;
}
