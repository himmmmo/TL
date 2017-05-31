#ifndef TRANLEARN_H_
#define TRANLEARN_H_

#include "SVDTrainer.h"

class TranLearn : public SVDTrainer {
protected:
	float alphas;
	vector<vector<Node> > mD2DMatrix;
private:
	void init();
	void loadD2DFile(string fileName, string separator);
public:
	TranLearn(int dim = 8, bool isTr = false);
	void train(float alpha, float lambda, int nIter);
	void loadFile(string mTrainFileName, string mTestFileName, string separator, 
		string mHisFileName = "",string mD2DFileName="");
	void predict(string mOutputFileName, string separator);
	virtual ~TranLearn();
};

#endif
