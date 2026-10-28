#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>
#include <algorithm>
#include "Segmentor.h"
#include "splitutf.h"

using namespace std;

string &trim(string &s)
{
        size_t beg = s.find_first_not_of(" \r\t\n");
        if(beg == string::npos)
        s = "";
        else
        {
             size_t end = s.find_last_not_of(" \r\t\n");
             s = s.substr(beg,end+1-beg);
        }

        return s;
}

int main(int argc, char *argv[])
{
	vector<string> cmds;
	int i;

	for (i = 1; i < argc; ++i)
	{
		cmds.push_back(argv[i]);
	}

	vector<string>::iterator it;
	it = find(cmds.begin(), cmds.end(), "-s");

	if (it == cmds.end() || it + 1 == cmds.end())
	{
		cout << "Para list should be: -s src-file -t tgt-file" << endl;
		exit(0);
	}

	string srcFile = *(it + 1);

	it = find(cmds.begin(), cmds.end(), "-t");
	it = find(cmds.begin(), cmds.end(), "-t");

	if (it == cmds.end() || it + 1 == cmds.end())
	{
		cout << "Para list should be: -s src-file -t tgt-file" << endl;
		exit(0);
	}

	string tgtFile = *(it + 1);
    cout << srcFile << tgtFile << endl;
	ifstream fin(srcFile.c_str());
	if (!fin)
	{
		cout << "Can not open src-file: " << srcFile << endl;
		exit(0);
	}

	cout << "Reading " << srcFile << " ... ";
	ifstream fin_cnt(srcFile.c_str());
    string str;
    int num = 0;
    while (getline(fin_cnt, str)) {
		num++;
		//str = trim(str);
        //if (str.length() > 0) {
        //    num++;
        //}
    }
	cout << num << " lines" << endl;
	int point_every = ceil(num/100.);
	int number_every = ceil(num/10.);
	//cout << num/100. << " - " << num/10. << " - " << point_every << " - " << number_every << endl;

	ofstream fout(tgtFile.c_str());
	if (!fout)
	{
		cout << "Can not open tgt-file: " << tgtFile << endl;
		exit(0);
	}

	CSegmentor segmentor;

	cout << "Initialize segmentor ... ";
	flush(cout);

	if (!segmentor.Initialize())
	{
		cout << "Segmentor initialize fail" << endl;
		exit(0);
	}

	cout << "succeed" << endl;

	//��ʱ��ʼ
	long begin = clock();

	string line;
	int cnt = 0;
    splitutf sp;
	cout << "Starting to segment ..." << endl;
	while (getline(fin, line))
	{
        cout<<"DEBUG: The line in PerSeg is ["<<line<<"]"<<endl;
		line = trim(line);  // 1
        if(line.empty())
        {
            fout<<endl;
            ++cnt;
            cout << "-NULL-";
            continue;
        }
		//cout << cnt << endl;
		string seg;  
		vector<pair<string,bool> > blocklist; // 2
		stringstream ss;
		ss<<line;
		string part;
		while(ss >> part)
		{
			size_t cma=0;
			if(part[0]=='<' && part[part.size()-1]=='>' && (cma = part.find(","))!=string::npos )
			{
				//cout << part.substr(cma +1 ,part.size()-2-cma ) << endl;
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
        //cout <<"split ok"<<endl;
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
				segmentor.Segment(vword, temp);
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
        /*for(vector<pair<string,bool> >::iterator it = blocklist.begin();it!=blocklist.end();++it)
            fout<<it->first<<":"<<it->second<<" ";
        fout<<endl;*/

		fout << rst << endl;
		
		++cnt;
		//cout << cnt % point_every << endl;
		//cout << cnt % number_every << endl;
		if ( cnt % point_every == 0 )
		{
			cerr << '.';
		}
		if ( cnt % number_every == 0 )
		{
			cerr << cnt;
		}
		/*
		if (cnt % 1 == 0)
		{
			cout << cnt << endl;
		}*/
	}
	cout << endl;
	
	//��ʱ����
	long end = clock();

	//�зֺ�ʱ
	double sec = (double)(end - begin) / (double)CLOCKS_PER_SEC;
	cout << "\nSucceed, time cost: " << sec << " secs" << endl;

	fin.close();
	fout.close();
	return 0;
}
