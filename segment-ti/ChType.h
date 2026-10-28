#ifndef CHTYPE_H
#define CHTYPE_H

#include <string>
#include <set>

using namespace std;

class CChType
{
public:
	CChType();
	~CChType();
	bool Initialize(const string &punc, const string &alph, const string &date, const string &num);
	int GetType(const string &ch);

private:
	bool Load(const string &file, set<string> &keys);
	bool InSet(const string &ch, const set<string> &keys);

	set<string> m_sAlphabet;
	set<string> m_sPunctuation;
	set<string> m_sDate;
	set<string> m_sNumber;

};

#endif
