#include <sstream>
#include <string>
#include <strstream>
#include <fstream>
#include <iostream>
#include <strstream>
#include <queue>
#include "Segmentor.h"
#include <stdlib.h>
#include "splitutf.h"
CSegmentor::CSegmentor()
{
}

CSegmentor::~CSegmentor()
{
}
// string &trim(string &s)
// {
//         size_t beg = s.find_first_not_of(" \r\t\n");
//         if(beg == string::npos)
//         s = "";
//         else
//         {
//              size_t end = s.find_last_not_of(" \r\t\n");
//              s = s.substr(beg,end+1-beg);
//         }

//         return s;
// }

string CSegmentor::MySeg(const char* input){
	splitutf sp;
	string line = string(input);
	line = trim(line); 
	string seg;  
	vector<pair<string,bool> > blocklist;
	stringstream ss;
	ss<<line;
	string part;
	while(ss >> part)
	{
		size_t cma=0;
		if(part[0]=='<' && part[part.size()-1]=='>' && (cma = part.find(","))!=string::npos )
		{
			blocklist.push_back(make_pair(part.substr(cma +1 ,part.size()-2-cma ),true));
		}
		else
		{
			vector<pair<string,bool> > tmpBlocklist;
			sp.preSplit(part,tmpBlocklist);
			for(int i=0;i<tmpBlocklist.size();++i)
				blocklist.push_back(tmpBlocklist[i]);
		}
	}
	for(int i=0;i<blocklist.size();)
	{
		if(blocklist[i].second or i == blocklist.size() - 1)
		{
			seg += " " + trim(blocklist[i].first);
			++i;
		}
		else
		{
			string block,temp;
			vector<string> vword;
			while(i<blocklist.size() && !blocklist[i].second)
			{
				vword.push_back(blocklist[i].first);
				++i;
			}
			cout<<"DEBUG: The size of vword in PerSeg is :["<<vword.size()<<"]"<<endl;
			cout << "The vword is as follows:"<<endl;
			for(int i=0; i < vword.size(); i++){
				cout << vword[i];
			}
			cout << endl;
			cout<<"DEBUG: The input of Segment:[vowrd] is : ";
			for(int i = 0 ; i < vword.size(); i++){
				cout<<vword[i];
			}
			cout<<endl;
			cout<<"DEBUG: The input of Segment:[temp] is :"<<temp<<endl;
			Segment(vword, temp);
			cout << "The temp after segment is " << temp << endl;
			//for(int i=0;i<vword.size();i++)
			//	temp = temp + vword[i];
			seg += " " + trim(temp);
			//seg += " " +temp;
		}

	
	}
	string rst;
	rst= trim(seg);
	cout<<"DEBUG: The rst is : "<<rst<<endl;
	return rst;
}

//��ʼ��
bool CSegmentor::Initialize()
{
	/*if (!m_cChType.Initialize("data/punc", "data/alph", "data/date", "data/num"))
	{
		return false;
	}*/

	if (!m_cPerceptron.Initialize("data/model"))
	{
		return false;
	}

	ifstream fin("data/data.txt");

	string line;
	while(getline(fin,line))
	{
		dict.insert(trim(line));
	}
	fin.close();
	
	ifstream fin_("data_bpe.txt");
	while(getline(fin_,line))
	{
		dict_bpe.insert(trim(line));
	}
	fin_.close();
	
	//�����ܴʳ�
	m_nScanLen = 10;//10;//34

    indegree = 5;


	return true;
}
string & CSegmentor::trim(string &s)
{
	size_t beg = s.find_first_not_of(" \t\n\r");
	if(beg == string::npos)
	{
		s = "";
		return s;
	}

	size_t end = s.find_last_not_of(" \t\n\r");
	s = s.substr(beg,end-beg+1);
	return s;
}
//�з�
string CSegmentor::Segment(const vector<string> &src, string &tgt)
{
	tgt.clear();
	vector<CTagElem> elems;
	Split(src, elems);
	cout<<"DEBUG: The elems in Segment after Split() is :";
	for(int i = 0; i < elems.size(); i++){
		cout<<elems[i].ch;
	}
	cout<<"DEBUG: After Split Force Cast  "<<src[0].c_str()<<endl;
	cout<<endl;
	//��������
	vector<vector<string> > featSets;
	vector<string> nullFeats;
	featSets.push_back(nullFeats);
	featSets.push_back(nullFeats);
	featSets.push_back(nullFeats);//feat
	featSets.push_back(nullFeats);//feat

	int len = (int)elems.size();
	int i;
	int from = 4;//feat
	int to = len - 5;//feat

	for (i = from; i <= to; ++i)
	{
		vector<string> feats;
		GenFeats(elems, i, feats);
		featSets.push_back(feats);
	}

	featSets.push_back(nullFeats);
	featSets.push_back(nullFeats);
	featSets.push_back(nullFeats);//feat
	featSets.push_back(nullFeats);//feat

	//��ȡ����÷�
	vector<map<string, double> > scores;
	GetScores(featSets, scores);
	cout<<"DEBUG: The tgt before BestPath is :["<<tgt<<"]"<<endl;
	//�滮���·��
	BestPath(elems, scores,tgt);
	cout<<"DEBUG: The tgt after BestPath is :["<<tgt<<"]"<<endl;
	//�����зֽ��
	//TraceSeg(elems, tgt);
	return tgt;
}
// string CSegmentor::Segment(char* vsrc, int size) {



// 	// vector<string> src(vsrc[0], vsrc+size-1);
// 	std::vector<string> src;
// 	for(int i = 0; i < size; i++){
// 		src.push_back(string(1,vsrc[i]));
// 	}
// 	cout<<"DEBUG: src content test : ";
// 	for(int i=0; i < src.size(); i++){
// 		cout<<src[i];
// 	}
// 	cout<<endl;
// 	// for(int i = 0; i<size; i++) {
// 	// 	src.emplace_back(string(vsrc[i]));
// 	// }
// 	// src.emplace_back(string(vsrc));
// 	// src.emplace_back(string(vsrc));
// 	cout<<"DEBUG: The size of src is ["<<src.size()<<"]"<<endl;
// 	// src.emplace_back(string(vsrc));
// 	cout<<"DEBUG: The input is : [" << vsrc <<"]"<<endl;
// 	// cout<<vsrc<<endl;
// 	// cout<<"The Ti is : " << "སྨྱུག་གུ་འདི་ཁྱོདཀྱིས་གང་ནས་ཉོས་པ་རེད།"<<endl;

//     string tgt;
//     vector<CTagElem> elems;

//     Split(src, elems);        
	
// 	cout<<"DEBUG: The elems in Segment after Split() is :";
//  	for(int i = 0; i < elems.size(); i++){
//  		cout<<elems[i].ch;
//  	}

//     //��������    
//     vector<vector<string> > featSets;    
//     vector<string> nullFeats;    
//     featSets.push_back(nullFeats);    
//     featSets.push_back(nullFeats);    
//     featSets.push_back(nullFeats);//feat    
//     featSets.push_back(nullFeats);//feat

//     int len = (int)elems.size();    

//     int i; 
//     int from = 4;//feat    
//     int to = len - 5;//feat
//     for (i = from; i <= to; ++i)    {
//     	vector<string> feats;        
//     	GenFeats(elems, i, feats);        
//     	featSets.push_back(feats);    
//     }
//     featSets.push_back(nullFeats);    
//     featSets.push_back(nullFeats);    
//     featSets.push_back(nullFeats);//feat    
//     featSets.push_back(nullFeats);//feat
//     //��ȡ����÷�    
//     vector<map<string, double> > scores;    
//     GetScores(featSets, scores);
//     //�滮���·��    
// 	cout<<"DEBUG: The tgt before best path is : ["<< tgt << "]"<<endl;
//     BestPath(elems, scores,tgt);
//     //�����зֽ��    
//     // TraceSeg(elems, tgt);
// 	cout<<"DEBUG: The tgt after best path is : ["<< tgt<< "]"<<endl;
//     return tgt;
// }

void CSegmentor::test(char* size){
	cout<<size<<endl;
	return;
}
//�з�Ϊ��������
void CSegmentor::Split(const vector<string> &src, vector<CTagElem> &elems)
{
	elems.clear();

	int i;
	for (i = 0; i < 4; ++i)//feat
	{
		CTagElem tip("<s>", -1);
		elems.push_back(tip);
	}

	for(i =0;i<src.size();++i)
	{
		

		int type = 0;
		CTagElem elem(src[i], type);

		elems.push_back(elem);
		
	}

	for (i = 0; i < 4; ++i)//feat
	{
		CTagElem tip("</s>", -1);
		elems.push_back(tip);
	}
}

//���ɸû����µ�����
void CSegmentor::GenFeats(const vector<CTagElem> &elems, const int &cur, vector<string> &feats)
{
	/*feats.clear();

	char feat[32];
	int n;

	//Cn
	for (n = -2; n <= 2; ++n)
	{
		sprintf(feat, "C%d==%s", n, elems[cur + n].ch.c_str());
		feats.push_back(feat);
	}

	//CnCn+1
	for (n = -2; n <= 1; ++n)
	{
		sprintf(feat, "C%dC%d==%s%s", n, n + 1, elems[cur + n].ch.c_str(), elems[cur + n + 1].ch.c_str());
		feats.push_back(feat);
	}

	//C-1C1
	sprintf(feat, "C-1C1==%s%s", elems[cur - 1].ch.c_str(), elems[cur + 1].ch.c_str());
	feats.push_back(feat);

	//Pu(C0)
	sprintf(feat, "PuC0==%d", elems[cur].type == 0 ? 1 : 0);
	feats.push_back(feat);

	//T(C-2)T(C-1)T(C0)T(C1)T(C2)
	sprintf(feat, "TC-22==%d%d%d%d%d", elems[cur - 2].type, elems[cur - 1].type, elems[cur].type, elems[cur + 1].type, elems[cur + 2].type);
	feats.push_back(feat);*/

		feats.clear();

	char feat[5000];
	int n;

	//Cn
	for (n = -4; n <= 4; ++n)//feat
	{
		sprintf(feat, "C%d==%s", n, elems[cur + n].ch.c_str());
		feats.push_back(feat);
	}

	//CnCn+1
	for (n = -2; n <= 1; ++n)
	{
		sprintf(feat, "C%dC%d==%s%s", n, n + 1, elems[cur + n].ch.c_str(), elems[cur + n + 1].ch.c_str());
		feats.push_back(feat);
	}
	
	//C-1C1
	sprintf(feat, "C-1C1==%s%s", elems[cur - 1].ch.c_str(), elems[cur + 1].ch.c_str());
	feats.push_back(feat);

	//Pu(C0)
	//sprintf(feat, "PuC0==%d", elems[cur].type == 0 ? 1 : 0);
	//feats.push_back(feat);

	////T(C-2)T(C-1)T(C0)T(C1)T(C2)
	//sprintf(feat, "TC-22==%d%d%d%d%d", elems[cur - 2].type, elems[cur - 1].type, elems[cur].type, elems[cur + 1].type, elems[cur + 2].type);

	//feat
	//CnCn+1Cn+2Cn+3
	for(n= -4;n<=1;++n)
	{
		sprintf(feat, "C%dC%dC%dC%d==%s%s%s%s", n, n + 1,n+2,n+3, elems[cur + n].ch.c_str(), elems[cur + n + 1].ch.c_str(),elems[cur + n + 2].ch.c_str(),elems[cur + n + 3].ch.c_str());
		feats.push_back(feat);
	}

	sprintf(feat, "C-2C-1C1C2==%s%s%s%s", elems[cur - 2].ch.c_str(),elems[cur - 1].ch.c_str(), elems[cur + 1].ch.c_str(), elems[cur + 2].ch.c_str());
	/////////////////////////////Co

	//CoCn
	for (n = -4; n <= 4; ++n)//feat
	{
		sprintf(feat, "C0C%d==%s%s", n,elems[cur].ch.c_str(), elems[cur + n].ch.c_str());
		feats.push_back(feat);
	}

	//CoCnCn+1
	for (n = -2; n <= 1; ++n)
	{
		sprintf(feat, "C0C%dC%d==%s%s%s", n, n + 1,elems[cur].ch.c_str(), elems[cur + n].ch.c_str(), elems[cur + n + 1].ch.c_str());
		feats.push_back(feat);
	}
	
	//CoC-1C1
	sprintf(feat, "C0C-1C1==%s%s%s",elems[cur].ch.c_str() ,elems[cur - 1].ch.c_str(), elems[cur + 1].ch.c_str());
	feats.push_back(feat);

	//CoCnCn+1Cn+2Cn+3
	for(n= -4;n<=1;++n)
	{
		sprintf(feat, "C0C%dC%dC%dC%d==%s%s%s%s%s", n, n + 1,n+2,n+3,elems[cur].ch.c_str() ,elems[cur + n].ch.c_str(), elems[cur + n + 1].ch.c_str(),elems[cur + n + 2].ch.c_str(),elems[cur + n + 3].ch.c_str());
		feats.push_back(feat);
	}

	sprintf(feat, "C0C-2C-1C1C2==%s%s%s%s%s", elems[cur].ch.c_str(),elems[cur - 2].ch.c_str(),elems[cur - 1].ch.c_str(), elems[cur + 1].ch.c_str(), elems[cur + 2].ch.c_str());


	//feat
	feats.push_back(feat);
}

//���������ӳ�䵽tag�ĵ÷�
double CSegmentor::FeatsPred(const vector<string> &feats, const string &tag)
{
	double score = 0.0;
	int cnt = (int)feats.size();
	int k;

	for (k = 0; k < cnt; ++k)
	{
		string pred = feats[k] + "=>" + tag;
		score += m_cPerceptron.PredWeight(pred);
	}

	return score;
}

//�����Span�ķ���
void CSegmentor::GetScores(const vector<vector<string> > &featSets, vector<map<string, double> > &scores)
{
	scores.clear();

	int len = (int)featSets.size();
	int lb = 4;//feat
	int rb = len - 5;//feat
	int i;

	scores.resize(len);

	for (i = lb; i <= rb; ++i)
	{
		scores[i].insert(map<string, double>::value_type("b", FeatsPred(featSets[i], "b")));
		scores[i].insert(map<string, double>::value_type("m", FeatsPred(featSets[i], "m")));
		scores[i].insert(map<string, double>::value_type("e", FeatsPred(featSets[i], "e")));
		scores[i].insert(map<string, double>::value_type("s", FeatsPred(featSets[i], "s")));
	}
}

//��ע��to��β����len�Ĵ�
void CSegmentor::TagWord(vector<CTagElem> &elems, const int &to, const int &len)
{
	int from = to - len + 1;
	int p;

	//��ע�ôʶ�
	for (p = from; p <= to; ++p)
	{
		if (p == from)
		{
			if (len == 1)
			{
				elems[p].tag = "s";
			}
			else
			{
				elems[p].tag = "b";
			}
		}
		else
		{
			if (p == to)
			{
				elems[p].tag = "e";
			}
			else
			{
				elems[p].tag = "m";
			}
		}
	}
}

//��̬�滮���·��
void CSegmentor::BestPath(vector<CTagElem> &elems, vector<map<string, double> > &scores,string &tgt)
{
	//׼���滮
	/*elems[3].best = 0.0;//feat

	//�滮����β����
	int lb = 4;//feat
	int rb = (int)elems.size() - 5;//feat
	int i;

	for (i = lb; i <= rb; ++i)
	{
		bool first = true;
		int dis;

		//[i - dis, i]�Ĵ�
		for (dis = 0; dis < m_nScanLen && i - dis >= lb; ++dis)
		{

			//��ע�ôʶ�
			TagWord(elems, i, dis + 1);

			//�÷����÷�
			double score = elems[i - dis - 1].best;
			int p;

			for (p = i - dis; p <= i; ++p)
			{
				map<string, double>::const_iterator it  = scores[p].find(elems[p].tag);
				score += it->second;
			}

			//�������ֵ
			if (first || score > elems[i].best)
			{
				elems[i].best = score;
				elems[i].length = dis + 1;
			}

			first = false;
		}
	}*/


	elems[3].best = 0.0;//feat

	//�滮����β����
	int lb = 4;//feat
	int rb = (int)elems.size() - 5;//feat
	int i;

	int Len = rb - lb+1;
	vector<vector<cand > > lattice;
	lattice.resize(Len);

	for (i = lb; i <= rb; ++i)
	{
		bool first = true;
		int dis;
		
		priority_queue<cand,vector<cand>, cmp> cands;


		//[i - dis, i]�Ĵ�
		for (dis = 0; dis < m_nScanLen && i - dis >= lb; ++dis)
		{

			
			//��ע�ôʶ�
			TagWord(elems, i, dis + 1);

			//�÷����÷�
			double score = elems[i - dis - 1].best;
			int p;

			for (p = i - dis; p <= i; ++p)
			{
				map<string, double>::const_iterator it  = scores[p].find(elems[p].tag);
				score += it->second;
			}

			cand c1;
			c1.from = i-dis-4;
			c1.to = i-4;
			c1.score = score;
			c1.spanScore = score - elems[i - dis - 1].best;

			//c1.path =  - c1.spanScore; // add

			string dicterm;

			for(int qq=i-dis;qq<=i;++qq)
			{
				dicterm +=elems[qq].ch;
			}

            char dotchar[4]={static_cast<char>(0xe0),static_cast<char>(0xbc),static_cast<char>(0x8b),0};
            string dot = string(dotchar);          
            if(dicterm.size()>=3 && dicterm.substr(dicterm.size()-3,3)==dot)               
                dicterm = dicterm.substr(0,dicterm.size()-3);
			if(dict.find(dicterm)!=dict.end())
			{
				//cout<<"find"<<endl;
				c1.path = 1;
				//c1.score = 7;
			}/*
			else if(dict_bpe.find(dicterm)!=dict_bpe.end())
			{
				c1.score = 3;//3;	
			}*/
			else 
			{
				//cout<<"no"<<endl;
				c1.path = 7;
				//c1.score = 1;
			}

			if(cands.size()>=indegree)
			{
				if(c1.score>cands.top().score)
				{
					cands.pop();
					cands.push(c1);
				}
			}
			else
			{
				cands.push(c1);
			}

			

			//�������ֵ
			if (first || score > elems[i].best)
			{
				elems[i].best = score;
				elems[i].length = dis + 1;
			}

			first = false;
		}//for (dis = 0; dis < m_nScanLen && i - dis >= lb; ++dis)
		int num = cands.size();
		lattice[i-4].resize(num);
		int j = num-1;
		if(j<0)
		{
			cerr<<"no nbest"<<endl;
			exit(1);
		}
		for(;j>=0;--j)
		{
			lattice[i-4][j] = cands.top();
			cands.pop();
		}
		



	}

	vector<pair<int,int> > path;
	path.resize(Len);

	for(int i=0;i<Len;++i)
	{
		path[i].first = 100000;
		for(int j=0;j<lattice[i].size();++j)
		{
			int pre = lattice[i][j].from-1;
			int score;
			if(pre >=0)
				score = lattice[i][j].path+path[pre].first;
			else
				score = lattice[i][j].path;
			if(score <path[i].first)
			{
				path[i].first = score;
				path[i].second = lattice[i][j].to - lattice[i][j].from +1;
			}
		}
	}

	int pre = Len - path[Len-1].second -1;
	tgt = "";
	int cur = Len-1;
	while(pre>=0)
	{
		tgt = " "+tgt;
		for(int i=cur;i>=pre+1;--i)
			tgt = elems[i+4].ch+tgt;

		cur = pre;
		pre = cur - path[cur].second;
	}

	tgt = " "+tgt;
	for(int i=cur;i>=0;--i)
		tgt = elems[i+4].ch+tgt;





}

//�������·������tag����
void CSegmentor::TraceSeg(vector<CTagElem> &elems, string &seg)
{
	seg.clear();

	int from = 4;//feat
	int p = static_cast<int>(elems.size() - 5);//feat

	while (p >= from)
	{
		int len = elems[p].length;
		int k;

		for (k = 0; k < len; ++k)
		{
			seg = elems[p].ch + seg;
			--p;
		}
	
		if (p >= from)
		{
			seg = "  " + seg;
		}
	}
}
