#include <fstream>
#include <iostream>
#include "Perceptron.h"
#include <stdlib.h>

CPerceptron::CPerceptron()
{
}

CPerceptron::~CPerceptron()
{
}

//初始化
bool CPerceptron::Initialize(const string &model)
{
	ifstream fin(model.c_str());
	if (!fin)
	{
		cout << "Can not open model file: " << model << endl;
		return false;
	}

	string line;

	while (getline(fin, line))
	{
		size_t split = line.find('\t');
		string pred = line.substr(0, split);
		string weight = line.substr(split + 1, line.length() - split - 1);
		double weightVal = atof(weight.c_str());

		m_mPredWeight.insert(map<string, double>::value_type(pred, weightVal));
	}

	fin.close();
	return true;
}

//获取该推断的分值
double CPerceptron::PredWeight(const string &pred)
{
	map<string, double>::iterator it = m_mPredWeight.find(pred);

	if (it != m_mPredWeight.end())
	{
		return it->second;
	}
	else
	{
		return 0.0;
	}
}
