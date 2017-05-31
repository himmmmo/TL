#include "TranLearn.h"

TranLearn::TranLearn(int dim, bool isTr) :
	SVDTrainer(dim, isTr) {
	alphas = 0;
}
void TranLearn::loadD2DFile(string fileName, string separator) {
	fstream file;
	int userId, itemId, mLineNum = 0;
	float rate;
	string mLine;
	map<int, int>::iterator iter;
	file.open(fileName.c_str(), ios::in);
	while (getline(file, mLine)) {
		userId = atoi(strtok((char *)mLine.c_str(), separator.c_str()));
		itemId = atoi(strtok(NULL, separator.c_str()));
		if (isTranspose) {
			int temp = userId;
			userId = itemId;
			itemId = temp;
		}
		rate = atof(strtok(NULL, separator.c_str()));
		mLineNum++;
		if (mLineNum % 1000 == 0)
			cout << mLineNum << " lines read" << endl;
		mD2DMatrix[mUserId2Map[userId]].push_back(Node(mItemId2Map[itemId], rate));
	}
	file.close();
}
void TranLearn::loadFile(string mTrainFileName, string mTestFileName,
	string separator, string mHisFileName, string mD2DFileName) {
	SVDTrainer::loadFile(mTrainFileName, mTestFileName, separator);
	if (mD2DFileName.length() == 0) {
		mD2DMatrix = mRateMatrix;
	}
	else {
		mD2DMatrix = vector<vector<Node> >(mUserNum + 1);
		loadD2DFile(mD2DFileName, separator);
	}
	cout << "------Initing------" << endl;
	init();
	cout << "------Init complete------" << endl;
}
void TranLearn::init() {
	float beta = 2;
	int fh=1;
	vector<Node> mf;
	for (int i = 1; i <= mItemNum; i++) {
		mf.push_back(Node(i,0));
	}
	for (int i = 1; i <= mUserNum; i++) {
		for (unsigned int j = 0; j < mD2DMatrix[i].size(); j++) {
			if (mD2DMatrix[i][j].getRate())
				mf[j] = Node(mf[j].getId(), mf[j].getRate() + 1);
		}
	}
	for (int i = 0; i < mItemNum; i++) {
		if (mf[i].getRate())
			fh++;
	}
	alphas = pow(beta, fh)*tgammaf(beta) / tgammaf(beta + mItemNum);
	for (int i = 0; i < mItemNum; i++)
		if (mf[i].getRate())
			alphas = alphas*tgammaf(mf[i].getRate());
	alphas = 1;
}
void TranLearn::train(float alpha, float lambda, int nIter) {
	cout << "------start training------" << endl;
	long double Rmse = 0, mLastRmse = 100000;
	int nRateNum = 0;
	float rui = 0;
	for (int n = 1; n <= nIter; n++) {
		Rmse = 0;
		nRateNum = 0;
		for (int i = 1; i <= mUserNum; i++)
			for (unsigned int j = 0; j < mRateMatrix[i].size(); j++) {
				rui = mean + bu[i] + bi[mRateMatrix[i][j].getId()]
					+ mt->getInnerProduct(p[i],q[mRateMatrix[i][j].getId()], dim);
				if (rui > mMaxRate)
					rui = mMaxRate;
				else if (rui < mMinRate)
					rui = mMinRate;
				float e = mRateMatrix[i][j].getRate() - rui;
				//更新bu,bi,p,q
				bu[i] += alpha * (e - lambda * bu[i]);
				bi[mRateMatrix[i][j].getId()] += alpha* (e - lambda * bi[mRateMatrix[i][j].getId()]);
				for (int k = 0; k < dim; k++) {
					p[i][k] += alpha* (e * q[mRateMatrix[i][j].getId()][k]- lambda * p[i][k]);
					q[mRateMatrix[i][j].getId()][k] += alpha* (e * p[i][k]- lambda * q[mRateMatrix[i][j].getId()][k]);
				}
				Rmse += e * e;
				nRateNum++;
			}
		for (int i = 1; i <= mUserNum; i++)
			for (unsigned int j = 0; j < mD2DMatrix[i].size(); j++) {
				rui = mean + bu[i] + bi[mD2DMatrix[i][j].getId()]
					+ mt->getInnerProduct(p[i], q[mD2DMatrix[i][j].getId()], dim);
				if (rui > mMaxRate)
					rui = mMaxRate;
				else if (rui < mMinRate)
					rui = mMinRate;
				float e = mD2DMatrix[i][j].getRate() - rui;
				//更新bu,bi,p,q
				bu[i] += alpha * alphas*(e - lambda * bu[i]);
				bi[mD2DMatrix[i][j].getId()] += alpha* alphas*(e - lambda * bi[mD2DMatrix[i][j].getId()]);
				for (int k = 0; k < dim; k++) {
					p[i][k] += alpha* alphas*(e * q[mD2DMatrix[i][j].getId()][k]- lambda * p[i][k]);
					q[mD2DMatrix[i][j].getId()][k] += alpha* alphas*(e * p[i][k]- lambda * q[mD2DMatrix[i][j].getId()][k]);
				}
				Rmse += e * e;
				nRateNum++;
			}
		Rmse = sqrt(Rmse / nRateNum);
		cout << "n = " << n << " Rmse = " << Rmse << endl;
		if (Rmse > mLastRmse)
			break;
		mLastRmse = Rmse;
		alpha *= 0.9;
	}
	cout << "------training complete!------" << endl;
	char *filename = "e:\\P2.txt";
	ofstream fout(filename);
	for (int i = 1; i <= mUserNum; i++) {
		for (unsigned int j = 1; j <=mItemNum; j++) {
			float rui = mean + bu[i] + bi[j]
				+ mt->getInnerProduct(p[i], q[j], dim);
			if (rui > mMaxRate)
				rui = mMaxRate;
			else if (rui < mMinRate)
				rui = mMinRate;
			cout << rui << " ";
			fout << rui << " ";
		}
		cout << endl;
		fout << endl;
	}
}
void TranLearn::predict(string mOutputFileName, string separator) {
	cout << "------predicting------" << endl;
	fstream file;
	fstream out;
	int userId, itemId;
	float rate;
	string mLine;
	long double Rmse = 0;
	int nNum = 0;
	if (mOutputFileName.length() != 0)
		out.open(mOutputFileName.c_str(), ios::out);
	file.open(mTestFileName.c_str(), ios::in);
	while (getline(file, mLine)) {
		userId = atoi(strtok((char *)mLine.c_str(), mSeparator.c_str()));
		itemId = atoi(strtok(NULL, mSeparator.c_str()));
		if (isTranspose) {
			int temp = userId;
			userId = itemId;
			itemId = temp;
		}
		rate = atof(strtok(NULL, mSeparator.c_str()));
		float rui = mean + bu[mUserId2Map[userId]] + bi[mItemId2Map[itemId]]
			+ mt->getInnerProduct(p[mUserId2Map[userId]],
				q[mItemId2Map[itemId]], dim);
		if (rui > mMaxRate)
			rui = mMaxRate;
		else if (rui < mMinRate)
			rui = mMinRate;
		Rmse += (rate - rui) * (rate - rui);
		nNum++;
		if (mOutputFileName.length() != 0) {
			out << userId << separator << itemId << separator << rui << endl;
		}
	}
	cout << "test file Rmse = " << sqrt(Rmse / nNum) << endl;
	file.close();
	out.close();
}
TranLearn::~TranLearn() {
}