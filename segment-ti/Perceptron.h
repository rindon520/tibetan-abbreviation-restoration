#ifndef PERCEPTRON
#define PERCEPTRON

#include <string>
#include <vector>
#include <map>

using namespace std;

class CPerceptron
{
public:
	CPerceptron();
	~CPerceptron();
	bool Initialize(const string &model);
	double PredWeight(const string &pred);

private:
	map<string, double> m_mPredWeight;

};

#endif
