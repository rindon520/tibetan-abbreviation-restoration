#include "splitutf.h"
#include <stdlib.h>

splitutf::splitutf(void)
{
	readFile("utf/tisegutf",this->tiseg);
	readFile("utf/chsegutf",this->chseg);
	readFile("utf/numutf",this->numbzd);
	readFile("utf/semisegutf",this->semiseg);
	readFile("utf/semisegsgutf",this->semisegsg);
	readFile("utf/numinerutf",this->numiner);
	readFile("utf/numfirst",this->numfirst);
	//readFile("bzd/dict",this->dict);
	char dotchar[4]={static_cast<char>(0xe0),static_cast<char>(0xbc),static_cast<char>(0x8b),0};
	dot = string(dotchar);
	char spchar1[4]={static_cast<char>(0xe0),static_cast<char>(0xbc),static_cast<char>(0x8d),0};
	separator1 = string(spchar1);
	char spchar2[4]={static_cast<char>(0xe0),static_cast<char>(0xbc),static_cast<char>(0x8e),0};
	separator2 = string(spchar2);
	char sfchar[4]={static_cast<char>(0xe0),static_cast<char>(0xbd),static_cast<char>(0xa0),0};
	suffix = string(sfchar);
	//en_set.
	en_set.insert("{");
	en_set.insert("}");
	en_set.insert("[");
	en_set.insert("]");
	en_set.insert("(");
	en_set.insert(")");

	
}

splitutf::~splitutf(void)
{
}

void splitutf::readFile(const string & file,set<string> &s)
{
	string line;
	ifstream fin(file.c_str());
    if(!fin)
    {
        cerr<<"can not open file "<<file<<endl;
        exit(1);
    }
	while(getline(fin,line))
		s.insert(line);
	fin.close();
}

void splitutf::processSym(string &s,vector<pair<string,bool> > &block,bool isEnd)//isEnd=false,Ӧ����s����һ�����ڵ㣻
{
	if(this->tiseg.find(s)!=this->tiseg.end())
	{
		if(isEnd)
			block.push_back(make_pair(s,true));
		else
			block.push_back(make_pair(s+dot,true));
	}
	else
	{
		int len = s.size();
		if(len>6)//�ǲ��ǳ������������һ�����з�???????
		{
			string tmp = s.substr(len-6);
			if(this->semiseg.find(tmp)!=this->semiseg.end())
			{
				string rest = s.substr(0,len-6);
				//if(dict.find(rest)!=dict.end())
				block.push_back(make_pair(rest,false));
				if(!isEnd)
					tmp +=dot;
				block.push_back(make_pair(tmp,false));

				return;

			}
			if(this->semisegsg.find(tmp)!=this->semisegsg.end())
			{
				string rest = s.substr(0,len-6);
				block.push_back(make_pair(rest,false));
				
				if(!isEnd)
					tmp +=dot;
				block.push_back(make_pair(tmp,false));

				return;
			}
			

		}
		
		if(len >3)
		{
			string tmp = s.substr(len-3);
			if(this->semisegsg.find(tmp)!=this->semisegsg.end())//???�ǲ��ǳ������������һ�����з�
			{
				string rest = s.substr(0,len-3);
				//bool isseg=false;

				/*if(dict.find(rest+suffix)!=dict.end())
				{
					isseg = true;
					block.push_back(make_pair(rest+suffix,isseg));
				}
				else*/

				block.push_back(make_pair(rest,false));
				
				if(!isEnd)
					tmp +=dot;
				block.push_back(make_pair(tmp,false));
			}
			else
			{

				if(isEnd)
					block.push_back(make_pair(s,false));
				else
					block.push_back(make_pair(s+dot,false));


			}
		}
		else
		{

			if(isEnd)
				block.push_back(make_pair(s,false));
			else
				block.push_back(make_pair(s+dot,false));
		}
	}//else
}

void splitutf::preSplit( string &s,vector<pair<string,bool> > &block)
{
	int i;
	block.clear();

	stringstream ss;
	ss<<s;
	string linechar;
	while(ss >> linechar)
	{
		Utf8Stream utfStream(linechar);
		string u8char;
		vector<string> line;//store utf8 characters
		while(utfStream.ReadOneCharacter(u8char))
		{
			line.push_back(u8char);
		}
		int len = line.size();
		string sym;
		bool isLastArabNum=false;
		for(i=0;i<len;)
		{
			//cout<<i<<endl;
			if(line[i].size()==1 && en_set.find(line[i])==en_set.end())
			{
				string bk;

				bool isArabNum = false;
				if(numfirst.find(line[i])!=numfirst.end())
					isArabNum = true;

				if(isArabNum)
				{
					while(i<len &&line[i].size()==1 && en_set.find(line[i])==en_set.end())//english characters and without {}()
					{
						if(numfirst.find(line[i])==numfirst.end())
							isArabNum = false;
						bk += line[i++];
					}//while
				}
				else
				{
					while(i<len &&line[i].size()==1 && en_set.find(line[i])==en_set.end())//english characters and without {}()
					{
						bk += line[i++];
					}//while
				}

				if(sym.size()!=0)
				{
					processSym(sym,block,true);
					sym = "";
				}
				string tmp = bk;

				if(isLastArabNum)
				{
					if(isArabNum)
					{
						//isLastArabNum = true;
						if(block.empty())
							block.push_back(make_pair(tmp,true));
						else
						{
							vector<pair<string,bool> >::reference rf = block.back();
							rf.first +=tmp;
						}
					}
					else
					{
						isLastArabNum = false;
						block.push_back(make_pair(tmp,true));
					}
				}
				else
				{
					if(isArabNum)
					{
						isLastArabNum = true;
						
						block.push_back(make_pair(tmp,true));
						
					}
					else
					{
						
						block.push_back(make_pair(tmp,true));
					}
				}
				
				
			}
			else if(line[i].size() ==1 && en_set.find(line[i])!=en_set.end())//appear {}()
			{
				isLastArabNum = false;
				if(sym.size()!=0)
				{
					processSym(sym,block,true);
					sym = "";
				}
				
				block.push_back(make_pair(line[i],true));
				++i;
			}
			else
			{
				//cout<<i<<endl;
				bool lastIsNum=false;
				while(i<len && line[i].size()>1)
				{
					//cout<<i<<endl;
					string tmp = line[i];
					++i;
					sym +=tmp;

					if(this->numbzd.find(tmp)!=this->numbzd.end())//������ֲ��ĵ����֣�����һ���顣�������ֵĺ�����û�е�ţ�
					{
						isLastArabNum = false;
						int symLen = sym.size();
						if(symLen>3)
						{
							sym = sym.substr(0,symLen-3);
							processSym(sym,block,true);
						}
						if(lastIsNum)
						{
							if(!block.empty())
							{
								vector<pair<string,bool> >::reference rf = block.back();
								rf.first +=tmp;
							}
							else
							{
								block.push_back(make_pair(tmp,true));
							}
						}
						else
							block.push_back(make_pair(tmp,true));
						lastIsNum = true;
						sym = "";
					}
					else if(this->numiner.find(tmp)!=this->numiner.end())//������ֳ������ֿ�1.2% ��������
					{
						if(isLastArabNum && !block.empty())
						{
							vector<pair<string,bool> >::reference rf = block.back();
							rf.first +=tmp;
						}
						else
						{
							isLastArabNum = true;
							if(sym.size()==3)
								block.push_back(make_pair(sym,true));
							else
							{
								sym = sym.substr(0,sym.size()-3);
								processSym(sym,block,true);
								block.push_back(make_pair(tmp,true));
							}
						}

						sym = "";
					}
					else if(this->chseg.find(tmp)!=this->chseg.end())//����к��ֵı����ţ����ߺ��ֵ����ֱ��
					{
						isLastArabNum = false;
						lastIsNum = false;
						int symLen = sym.size();
						if(symLen==3)
							block.push_back(make_pair(sym,true));
						else
						{
							sym = sym.substr(0,symLen-3);
							processSym(sym,block,true);
							block.push_back(make_pair(tmp,true));
						}
						
						sym = "";
					}
					else if(tmp==dot)//�������ڵ�
					{
						//
						isLastArabNum = false;
						if(lastIsNum)
						{
							vector<pair<string,bool> >::reference rf = block.back();
							rf.first +=dot;
						}
						else
						{
							
							int symLen = sym.size();
							if(symLen==3)
							{
								block.push_back(make_pair(sym,true));
							}
							else
							{
								sym = sym.substr(0,symLen-3);
								processSym(sym,block,false);
							}
						}

						lastIsNum = false;
						sym = "";
					}
					else if(tmp == separator1 || tmp ==separator2)
					{
						//
						isLastArabNum = false;
						lastIsNum = false;
						int symLen = sym.size();
						
						if(symLen>3)
						{
							sym = sym.substr(0,symLen-3);
							processSym(sym,block,true);
						}
						if(!block.empty())
						{
							vector<pair<string,bool> >::reference rf = block.back();
							if(rf.first!=separator1 && rf.first!=separator2)
								block.push_back(make_pair(tmp,true));
							else
								rf.first +=tmp;
						}
						else
							block.push_back(make_pair(tmp,true));

						sym = "";
					}
					else if(i>=len)//��������ڵ�������ܽ���
					{
						//
						isLastArabNum = false;
						lastIsNum = false;
						processSym(sym,block,true);

					}
					else
					{
						isLastArabNum = false;
						lastIsNum = false;
					}

				}//while
			}//else
			/*if(line[i]<0 && i+1==len)//������һ���ַ���С��0�ģ�ֻ��������
			{
				string tmp2;
				tmp2[0] = line[i];
				block.push_back(make_pair(tmp2,true));
				break;
			}*/

		}//for

	}//while

	
}
