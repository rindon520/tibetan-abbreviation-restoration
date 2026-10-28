#ifndef DATASTRUCT_H
#define DATASTRUCT_H
#include <string>
using namespace std;
//��עԪ��
class CTagElem
{
public:
	string ch;
	int type;
	string tag;

	double best;
	int length;

	CTagElem(const string &chVal, const int &typeVal)
	{
		ch = chVal;
		type = typeVal;
	}
};


class cand{
	
public:
	int from;
	int to;
	double spanScore;
	double score;
	int path;

	cand()
	{
		from =0;
		to =0;
		score=0.0;
		spanScore=0.0;
		path = 1;
	}

	cand(int from,int to,double score,double spanScore,int path)
	{
		this->from = from;
		this->to = to;
		this->score = score;
		this->spanScore = spanScore;
		this->path = path;
	}
};

class cmp{
	
public:
 bool operator()(const cand &a,const cand &b)
 {
  //return a>b;//����
	 return a.score>b.score;//��С��
 }


};

#endif
