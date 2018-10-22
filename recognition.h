///****************************************************************************************************************************///
///********************************* Developed By Fazle Rabby Sourav && Ahnaf Rownak ******************************************///
///****************************************************************************************************************************///


#ifndef RECOGNITION_H
#define RECOGNITION_H

#include <opencv2/opencv.hpp>

#include <fstream>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <stack>
#include <queue>
#include <algorithm>
#include <iterator>
#include <utility>


#define REP(i,n)for (i=0;i<n;i++)
#define FOR(i,p,k) for (i=p; i<k;i++)
#define FORE(i, p, k) for(i=p; i<=k; i++)
#define FOREACH(it,x) for(__typeof((x).begin()) it=(x.begin()); it!=(x).end(); ++it)
#define FORD(i,n)    for(i=n;i>=0;i--)

#define READ(f) freopen(f, "r", stdin)
#define WRITE(f) freopen(f, "w", stdout)
#define REV(s, e) reverse(s, e)

#define CLR(p) memset(p, 0, sizeof(p))
#define mset(p, v) memset(p, v, sizeof(p))
#define ALL(c) c.begin(), c.end()
#define SZ(c) (int)c.size()
#define pb(x) push_back(x)

///#define type
#define ll long long int
#define vs vector<string>
#define vi vector<int>
#define vii vector< pair<int, int> >
#define pii pair< int, int >
#define psi pair< string, int >
#define vmat vector <Mat>


#define fs first
#define sc second
#define MP(x, y) make_pair(x, y)
#define pq priority_queue

#define LOG(x,BASE) (log10(x)/log10(BASE))
#define EQ(a,b)     (fabs(a-b)<ERR)


#define csprint printf("Case %d: ", ++t);
#define PI acos(-1)
#define ERR 10E-5
#define INF (1<<31)

#define pointAt(inpImg, k, j) inpImg.at<uchar>(Point(k,j))

using namespace std;
using namespace cv;

#define HEIGHT 33
#define WIDTH 33


struct distanceStruct{
    int index;
    int distance;
};

Mat HistogramValue(const Mat &src_img, vector<int> &mhist,
                     bool isHor = true) {
    int sz = (isHor) ? src_img.rows : src_img.cols;
    int i, j, v, width, height, maximum = 0;
    Mat data, histo;

    mhist.clear();
    for (j = 0; j < sz; ++j) {
        data = (isHor) ? src_img.row(j) : src_img.col(j);
        v = countNonZero(data);
        mhist.push_back(v);
        if (v > maximum) maximum = v;
    }
    return histo;
}


void borderMarginHor(const Mat src, Mat &dvp){


    vector<int> leftRightDist ;
    int numOfRows = src.rows, numOfNonZeros,cntBit;
    unsigned max=0, numOfNonZerosPerRows[numOfRows];
    dvp = Mat::zeros(Size(src.cols, src.rows), CV_8U);
    for(int i=0; i<src.rows; i++){

        cntBit=0;
        for(int j=0; j<src.cols; j++){
            Scalar ints = src.at<uchar>(Point(j,i));
            if(ints.val[0]<100)
            {
                cntBit++;
            }
            else{
               // dvp.at<uchar>(i,j) = 250;
                break;
            }
        }
        leftRightDist.push_back(cntBit);
        line(dvp, Point(0, i), Point(cntBit,i), 255);

    }
    for(int i=0; i<src.rows; i++){

        cntBit=0;
        for(int j=src.cols; j>=0; j--){
            Scalar ints = src.at<uchar>(Point(j,i));
            if(ints.val[0]<100)
            {
                cntBit++;
            }
            else{
               // dvp.at<uchar>(Point(j,i)) = 255;
                break;
            }
        }
        leftRightDist.push_back(cntBit);
        line(dvp, Point(src.cols, i), Point(src.cols-cntBit,i), 255);

    }
    imshow("border" , dvp);
    waitKey();

}

void getLine(const Mat &inpImg, vector<Mat> trainCharVec){

    Scalar inpImgScl, trainImgScl;
    Mat dvp = inpImg.clone();
    threshold(dvp, dvp, 180, 255, THRESH_BINARY_INV);
    int numOfRows = dvp.rows, numOfNonZeros;
    unsigned max=0, numOfNonZerosPerRows[numOfRows];
    for(int i=0; i< HEIGHT ; i++){
        for(int j=0; j< WIDTH ; j++){

            if(inpImg.at<uchar>(Point(j,i))<255){
                numOfNonZeros = j;
                numOfNonZerosPerRows[i] = numOfNonZeros;
                break;
            }
        }
    }
    for (int  i = 0; i < HEIGHT; ++i){
            line(dvp, Point(0, i), Point(numOfNonZerosPerRows[i], i), 255);
        }
        imshow("DVP", dvp);
        waitKey();
}

int compareByShifting(const Mat &inpImg, vector<Mat> trainCharVec){

    int minMatch = 9999999 , matchedIndx=-1;
    vector<pii> score;
    score.clear();

    for(int i=0; i< trainCharVec.size(); i++){
        Scalar inpImgScl, trainImgScl;
        int misMatchCount=0, minMismatch=99999999;

        for(int j=0; j< HEIGHT ; j++ ){

            for(int k=0; k< WIDTH ; k++){

                trainImgScl = trainCharVec.at(i).at<uchar>(Point(k,j));
                inpImgScl = pointAt(inpImg, k, j);

                if(inpImgScl.val[0] != trainImgScl.val[0]){
                    misMatchCount++;
                }
                //cout << inpImgScl.val[0] << " ";
            }
            //cout <<"\n";

        }
        minMismatch= min(minMismatch, misMatchCount);


        misMatchCount=0;
        for(int j=0; j< HEIGHT ; j++ ){

            for(int k=1; k< WIDTH-1; k++){

                trainImgScl = trainCharVec.at(i).at<uchar>(Point(k,j));
                inpImgScl = pointAt(inpImg, k-1, j);

                if(inpImgScl.val[0] != trainImgScl.val[0]){
                    misMatchCount++;
                }
                //cout << inpImgScl.val[0] << " ";
            }
            //cout <<"\n";

        }
        minMismatch= min(minMismatch, misMatchCount);


        misMatchCount=0;
        for(int j=0; j< HEIGHT ; j++ ){

            for(int k=2; k< WIDTH ; k++){

                trainImgScl = trainCharVec.at(i).at<uchar>(Point(k,j));
                inpImgScl = pointAt(inpImg, k-2, j);

                if(inpImgScl.val[0] != trainImgScl.val[0]){
                    misMatchCount++;
                }
                //cout << inpImgScl.val[0] << " ";
            }
            //cout <<"\n";

        }
        minMismatch= min(minMismatch, misMatchCount);

//        misMatchCount=0;
//        for(int j=0; j< HEIGHT ; j++ ){
//
//            for(int k=3; k< WIDTH ; k++){
//
//                trainImgScl = trainCharVec.at(i).at<uchar>(Point(k,j));
//                inpImgScl = pointAt(inpImg, k-3, j);
//
//                if(inpImgScl.val[0] != trainImgScl.val[0]){
//                    misMatchCount++;
//                }
//                //cout << inpImgScl.val[0] << " ";
//            }
//            //cout <<"\n";
//
//        }
//        minMismatch= min(minMismatch, misMatchCount);

        if(minMismatch<minMatch)
        {
            minMatch= minMismatch;
            matchedIndx= i;
        }

        //score.pb(MP(minMismatch, i));
    }

   // sort(ALL(score));
    Mat candidateImage= trainCharVec.at(matchedIndx);
//
//    for(int i=1; i<10; i++)
//    {
//        candidateImage.pb(trainCharVec.at(score[i].sc));
//    }
//    cout<<score[0].sc<<endl;
//    imshow("Input Image", inpImg);
//    imshow("New" ,candidateImage);
//    waitKey();
    return matchedIndx;
}


int compareWithHisto(Mat &inpImg, vector<Mat> trainCharVec)
{
    int minDist = (int)(1<<30) , matchedIndx=-1;

    vector<int> histoVecInp, histoVecInpHor, histoVecTrain, histoVecTrainHor;
    vector<pii> score;

    histoVecInp.clear(); histoVecInpHor.clear();
    histoVecTrain.clear(); histoVecTrainHor.clear();
    score.clear();

    HistogramValue(inpImg, histoVecInp, true);
    HistogramValue(inpImg, histoVecInpHor, false);
//    REP(i, SZ(histoVecInp))
//    {
////        cout<<i<<" "<<histoVecInp[i]<<endl;
//    }
    for(int i=0; i< trainCharVec.size(); i++){
        Mat currTrainMat = trainCharVec.at(i);
        histoVecTrain.clear(); histoVecTrainHor.clear();

        HistogramValue(currTrainMat, histoVecTrain, true);
        HistogramValue(currTrainMat, histoVecTrainHor, false);

        ///Compare both vector
        int distV=0, distH=0, k=0;
        REP(k, SZ(histoVecInp))
        {
            distV+= abs(histoVecInp[k]-histoVecTrain[k]);
            distH+= abs(histoVecInpHor[k]-histoVecTrainHor[k]);
        }
//        cout<<"i: "<<i<<" "<<"DistV: "<< distV << "DistH "<< distH << " "<<(distV+distH) << endl;
//        imshow("TarinImage" ,currTrainMat);
        score.pb(MP((distV+distH), i));

        if( (distV+distH)<minDist)
        {
            minDist= (distV+distH);
            matchedIndx= i;
        }
    }

    sort(ALL(score));
    Mat candidateImage= trainCharVec.at(score[0].sc);

    for(int i=1; i<10; i++)
    {
        candidateImage.pb(trainCharVec.at(score[i].sc));
    }

    imshow("Histo" ,candidateImage);
//    cout<<"Matched Indx"<<matchedIndx<<endl;
//    waitKey();
}


vector<int> leftRightborderDistance(const Mat src){


    vector<int> distanceFromBorder ;
    distanceFromBorder.clear();
    Mat dvp ;
    int numOfRows = src.rows, numOfNonZeros,cntBit;
    unsigned max=0, numOfNonZerosPerRows[numOfRows];
    dvp = Mat::zeros(Size(src.cols, src.rows), CV_8U);
    //left to right
    for(int i=0; i<src.rows; i+=3){

        cntBit=0;
        for(int j=0; j<src.cols; j++){
            Scalar ints = src.at<uchar>(Point(j,i));
            if(ints.val[0]<100)
            {
                cntBit++;
            }
            else{
               // dvp.at<uchar>(i,j) = 250;
                break;
            }
        }
        distanceFromBorder.push_back(cntBit);
        line(dvp, Point(0, i), Point(cntBit,i), 255);


    }
    // right to left
    for(int i=0; i<src.rows; i+=3){

        cntBit=0;
        for(int j=src.cols; j>=0; j--){
            Scalar ints = src.at<uchar>(Point(j,i));
            if(ints.val[0]<100)
            {
                cntBit++;
            }
            else{
               // dvp.at<uchar>(Point(j,i)) = 255;
                break;
            }
        }

        distanceFromBorder.push_back(cntBit);
        line(dvp, Point(src.cols, i), Point(src.cols-cntBit,i), 255);

    }

    //up down
    for(int i=0; i<src.cols; i+=3){

        cntBit=0;
        for( int j=0; j<src.rows; j++){
            Scalar ints = src.at<uchar>(Point(i,j));
            if(ints.val[0]<100)
            {
                cntBit++;
            }
            else{
               // dvp.at<uchar>(i,j) = 250;
                break;
            }
        }
        distanceFromBorder.push_back(cntBit);
        line(dvp, Point(i, 0), Point(i,cntBit), 255);


    }
    //down to up

    for(int i=0; i<src.cols; i+=3){

        cntBit=0;
        for(int j=src.rows; j>=0; j--){
            Scalar ints = src.at<uchar>(Point(i,j));
            if(ints.val[0]<100)
            {
                cntBit++;
            }
            else{
                //dvp.at<uchar>(Point(i,j)) = 255;
                break;
            }
        }
        distanceFromBorder.push_back(cntBit);
        line(dvp, Point(i,src.rows), Point(i,src.rows-cntBit), 255);

    }
    //imshow("border" , dvp);
    //waitKey();
    return distanceFromBorder;
}
//hamming Distance between two vector
int hammingDistance( vector<int> inpImgDist , vector<int> trainImgDist ){

    int distance = 0;
    for(int i=0; i<inpImgDist.size(); i++){
        distance += abs(inpImgDist.at(i) - trainImgDist.at(i));
    }
    inpImgDist.clear();
    trainImgDist.clear();
    return distance;
}
//quick sort
bool compareByLength(const distanceStruct &a, const distanceStruct &b)
{
    return a.distance < b.distance;
}
//comparing image with trainset using border Distance feature
int compareWithBorderDist(const Mat &inpImg, vector<Mat> trainCharVec){
    vector<distanceStruct> distAndindxVec;
    vector<int> inpImgDist = leftRightborderDistance(inpImg);
    vector<int> trainImgDist;
    int minDist = 99999, resltIndx=-1;
    for(int i=0; i< trainCharVec.size(); i++){
        trainImgDist = leftRightborderDistance(trainCharVec.at(i));
        int distance = hammingDistance(inpImgDist, trainImgDist);
       // cout << "train Image Num: " << i+1 << " distance : " << distance << "\n" ;
        distAndindxVec.push_back(distanceStruct());
        distAndindxVec[i].index = i;
        distAndindxVec[i].distance = distance;
        if( distance < minDist ){
            minDist = distance;

        }
    }
    std::sort(distAndindxVec.begin(), distAndindxVec.end(), compareByLength);


    Mat outputShow = trainCharVec.at(distAndindxVec.at(0).index) ;
    cout << distAndindxVec.at(0).index+1 << " = " << distAndindxVec.at(0).distance << endl;
    for(int i=1; i<10; i++){
        outputShow.push_back(trainCharVec.at(distAndindxVec.at(i).index));
        cout << distAndindxVec.at(i).index+1 << " = " << distAndindxVec.at(i).distance << endl;
    }
    imshow("Distance Line" , outputShow);
    distAndindxVec.clear();

}


int compareWithBox(Mat &inpImg, vector<Mat> trainCharVec, int nBoxSize)
{
    int cntBitInBoxInp[12][12], cntBitInBoxInpShift1[12][12], cntBitInBoxInpShift1_[12][12], cntBitInBoxInpShift2_[12][12], cntBitInBoxInpShift2[12][12], cntBitInBoxInpShift3[12][12], cntBitInBoxInpShift3_[12][12], cntBitInBoxTrain[12][12];

    int HEIGHT_NEW, WEIGHT_NEW;

    memset(cntBitInBoxInp, 0, sizeof(cntBitInBoxInp) );
    memset(cntBitInBoxInpShift1, 0 , sizeof(cntBitInBoxInpShift1));
    memset(cntBitInBoxInpShift1_, 0 , sizeof(cntBitInBoxInpShift1_));
    memset(cntBitInBoxInpShift2, 0 , sizeof(cntBitInBoxInpShift2));
    memset(cntBitInBoxInpShift2_, 0 , sizeof(cntBitInBoxInpShift2_));
    memset(cntBitInBoxInpShift3, 0 , sizeof(cntBitInBoxInpShift3));
    memset(cntBitInBoxInpShift3_, 0 , sizeof(cntBitInBoxInpShift3_));
    memset(cntBitInBoxTrain, 0, sizeof(cntBitInBoxTrain) );

    vector <pii> score;     score.clear();
    Scalar inpImgScl, trainImgScl;


//    cout<<"Loop started in camparewiTHbOX"<<endl;
    /// for inpImage
    for(int i=0; i<HEIGHT; i++)
    {
        for(int j=0; j<WIDTH; j++)
        {
            inpImgScl = pointAt(inpImg, j, i);
            if(inpImgScl.val[0]!=0)
                cntBitInBoxInp[i/nBoxSize][j/nBoxSize]++;
        }
    }

    /// for inpImage shifted to Shift 1 Bit
    for(int i=0; i<HEIGHT; i++)
    {
        for(int j=0; j<WIDTH-1; j++)
        {
            int k=j+1;
            inpImgScl = pointAt(inpImg, j, i);
            if(inpImgScl.val[0]!=0)
                cntBitInBoxInpShift1[i/nBoxSize][k/nBoxSize]++;  ///K for relative position for boxcount
        }
    }

    /// for inpImage shifted to Shift -1 Bit
    for(int i=0; i<HEIGHT; i++)
    {
        for(int j=1; j<WIDTH; j++)
        {
            int k=j-1;
            inpImgScl = pointAt(inpImg, j, i);
            if(inpImgScl.val[0]!=0)
                cntBitInBoxInpShift1_[i/nBoxSize][k/nBoxSize]++;  ///K for relative position for boxcount
        }
    }

    /// for inpImage shifted to Shift 2 Bit
    for(int i=0; i<HEIGHT; i++)
    {
        for(int j=0; j<WIDTH-2; j++)
        {
            int k=j+2;
            inpImgScl = pointAt(inpImg, j, i);
            if(inpImgScl.val[0]!=0)
                cntBitInBoxInpShift2[i/nBoxSize][k/nBoxSize]++;  ///K for relative position for boxcount
        }
    }

     /// for inpImage shifted to Shift -2 Bit
    for(int i=0; i<HEIGHT; i++)
    {
        for(int j=2; j<WIDTH; j++)
        {
            int k=j-2;
            inpImgScl = pointAt(inpImg, j, i);
            if(inpImgScl.val[0]!=0)
                cntBitInBoxInpShift2_[i/nBoxSize][k/nBoxSize]++;  ///K for relative position for boxcount
        }
    }

    /// for inpImage shifted to Shift 3 [3 is Optimal] Bit
    for(int i=0; i<HEIGHT; i++)
    {
        for(int j=0; j<WIDTH-3; j++)
        {
            int k=j+3;
            inpImgScl = pointAt(inpImg, j, i);
            if(inpImgScl.val[0]!=0)
                cntBitInBoxInpShift3[i/nBoxSize][k/nBoxSize]++;  ///K for relative position for boxcount
        }
    }

    /// for inpImage shifted to Shift -3 [3 is Optimal] Bit
    for(int i=0; i<HEIGHT; i++)
    {
        for(int j=3; j<WIDTH; j++)
        {
            int k=j-3;
            inpImgScl = pointAt(inpImg, j, i);
            if(inpImgScl.val[0]!=0)
                cntBitInBoxInpShift3_[i/nBoxSize][k/nBoxSize]++;  ///K for relative position for boxcount
        }
    }


    ///for every train Image character
    for(int t=0; t<SZ(trainCharVec); t++)
    {
        memset(cntBitInBoxTrain, 0, sizeof(cntBitInBoxTrain) );
        for(int i=0; i<HEIGHT; i++)
        {
            for(int j=0; j<WIDTH; j++)
            {
                trainImgScl = trainCharVec.at(t).at<uchar>(Point(j,i));

                if(trainImgScl.val[0]!=0)
                    cntBitInBoxTrain[i/nBoxSize][j/nBoxSize]++;
            }
        }

        /// compare Both BoxCount
        int cntDiff=0, cntDiffShifted1=0, cntDiffShifted1_=0 , cntDiffShifted2=0, cntDiffShifted2_=0, cntDiffShifted3=0, cntDiffShifted3_=0;
        int cntDiffOfBox=0, cntDiffOfBoxShifted1=0, cntDiffOfBoxShifted1_=0, cntDiffOfBoxShifted2=0, cntDiffOfBoxShifted2_=0, cntDiffOfBoxShifted3=0,  cntDiffOfBoxShifted3_=0;
        int BitInBox=8;


        for(int i=0; i<(HEIGHT/nBoxSize); i++)
        {
            for(int j=0; j<(WIDTH/nBoxSize); j++)
            {
                cntDiff+= abs(cntBitInBoxInp[i][j] - cntBitInBoxTrain[i][j]);
                cntDiffShifted1+= abs(cntBitInBoxInpShift1[i][j]- cntBitInBoxTrain[i][j]);
                cntDiffShifted1_+= abs(cntBitInBoxInpShift1_[i][j]- cntBitInBoxTrain[i][j]);
                cntDiffShifted2+= abs(cntBitInBoxInpShift2[i][j]- cntBitInBoxTrain[i][j]);
                cntDiffShifted2_+= abs(cntBitInBoxInpShift2_[i][j]- cntBitInBoxTrain[i][j]);
                cntDiffShifted3+= abs(cntBitInBoxInpShift3[i][j]- cntBitInBoxTrain[i][j]);
                cntDiffShifted3_+= abs(cntBitInBoxInpShift3_[i][j]- cntBitInBoxTrain[i][j]);

                if(abs(cntBitInBoxInp[i][j] - cntBitInBoxTrain[i][j]) >=BitInBox )
                {
                    cntDiffOfBox++;
                }
                if(abs(cntBitInBoxInpShift1[i][j] - cntBitInBoxTrain[i][j]) >=BitInBox )
                {
                    cntDiffOfBoxShifted1++;
                }
                if(abs(cntBitInBoxInpShift1_[i][j] - cntBitInBoxTrain[i][j]) >=BitInBox )
                {
                    cntDiffOfBoxShifted1_++;
                }
                if(abs(cntBitInBoxInpShift2[i][j] - cntBitInBoxTrain[i][j]) >=BitInBox )
                {
                    cntDiffOfBoxShifted2++;
                }
                if(abs(cntBitInBoxInpShift2_[i][j] - cntBitInBoxTrain[i][j]) >=BitInBox )
                {
                    cntDiffOfBoxShifted2_++;
                }
                if(abs(cntBitInBoxInpShift3[i][j] - cntBitInBoxTrain[i][j]) >=BitInBox )
                {
                    cntDiffOfBoxShifted3++;
                }
                if(abs(cntBitInBoxInpShift3_[i][j] - cntBitInBoxTrain[i][j]) >=BitInBox )
                {
                    cntDiffOfBoxShifted3_++;
                }
            }
        }

        int boxW=9;
        cntDiffOfBox*=boxW; /// per square size is 9
        cntDiffOfBoxShifted1*=boxW;
        cntDiffOfBoxShifted1_*=boxW;
        cntDiffOfBoxShifted2*=boxW;
        cntDiffOfBoxShifted2_*=boxW;
        cntDiffOfBoxShifted3*=boxW;
        cntDiffOfBoxShifted3_*=boxW;

//        resize(TrainSingleChar, TrainSingleChar, Size(HEIGHT_NEW, WEIGHT_NEW));
//        resize(inpImg, inpImg, Size(HEIGHT_NEW, WEIGHT_NEW));

        int minWeight=(cntDiff+cntDiffOfBox) ;

        minWeight= min(minWeight, (cntDiffShifted1+cntDiffOfBoxShifted1));
        minWeight= min(minWeight, (cntDiffShifted1_+cntDiffOfBoxShifted1_));
        minWeight= min(minWeight, (cntDiffShifted2+cntDiffOfBoxShifted2));
        minWeight= min(minWeight, (cntDiffShifted2_+cntDiffOfBoxShifted2_));
//        minWeight= min(minWeight, (cntDiffShifted3+cntDiffOfBoxShifted3));
//        minWeight= min(minWeight, (cntDiffShifted3_+cntDiffOfBoxShifted3_));

        score.pb(MP(  minWeight , t));
//        score.pb(MP(cntDiff+cntDiffOfBox, t));
    }

    sort(ALL(score));

    Mat candidateImage= trainCharVec.at(score[0].sc);
    for(int i=1; i<10; i++)
    {
        candidateImage.pb(trainCharVec.at(score[i].sc));
//        cout<<score[i].fs<<endl;
    }
//    cout<<endl;
//    imshow("Inp", inpImg);
//    imshow("Box" ,candidateImage);
//    waitKey();
    return score[0].sc;

}

#endif
