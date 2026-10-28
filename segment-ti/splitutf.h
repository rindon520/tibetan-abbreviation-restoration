#pragma once


#include <string>
#include <set>
#include <vector>
#include <fstream>
#include <sstream>
#include "Utf8Stream.h"
using namespace std;

class splitutf
{
public:
	splitutf(void);
	~splitutf(void);

	set<string> tiseg;//must segment to word
	set<string> chseg;//must segment to word
	set<string> numbzd;//tibeton num
	set<string> semiseg;//only splite the yinjie
	set<string> semisegsg;//only splite the yinjie
	set<string> dict;

	string dot;
	string separator1;
	string separator2;
	string suffix;
	set<string> en_set;

	set<string> numfirst;
	set<string> numiner;

	void readFile(const string & file,set<string> &s);
	void preSplit( string &s,vector<pair<string,bool> > &block);
	void processSym(string &s,vector<pair<string,bool> > &block,bool isEnd);
};
