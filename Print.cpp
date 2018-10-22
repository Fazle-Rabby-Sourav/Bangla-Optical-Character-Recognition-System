///************************************** BANGLA OCR Segmentation & Recognition***********************************///
///********************************* Fazle Rabby Sourav && Ahnaf Rownak ******************************************///
///***************************************************************************************************************///

#include<opencv2/opencv.hpp>
#include "cca_.h"

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



//const string SegmentedFileName  = "train/crickOut.bmp";
const string TrainFileName      = "train/ben.train";
const string TrainImageFileName = "train/ben.tif";
const string TrainBoxFileName   = "train/ben.box";
const string TrainPhoneFileName = "train/ben.phon";

const char* sourceFileName      = "image/kore.jpg";
const char* dstFileName         = "image/koreOutFinal.bmp";

const string inpTextFile        = "train/ben.txt";
const string outTextFile        = "image/1OutTextFinal.txt";

#define TotalNumOfChar 256

int DEBUG= 0;
int RECOGDEBUG= 0;



template< class T > T _abs(T n) { return (n < 0 ? -n : n); }
template< class T > T sq(T x) { return x * x; }
template< class T > T gcd(T a, T b) { return (b != 0 ? gcd<T>(b, a%b) : a); }
template< class T > T lcm(T a, T b) { return (a / gcd<T>(a, b) * b); }
template< class T > T _max(T a, T b) { return (!(a < b) ? a : b); }
template< class T > T _min(T a, T b) { return (a < b ? a : b); }

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

#define fs first
#define sc second
#define MP(x, y) make_pair(x, y)
#define pq priority_queue

#define LOG(x,BASE) (log10(x)/log10(BASE))
#define EQ(a,b)     (fabs(a-b)<ERR)


#define csprint printf("Case %d: ", ++t);
#define PI acos(-1)
#define ERR 10E-5


// #define TRAIN_FILE_AVAILABLE
#define KNN


using namespace cv;
using namespace std;


const int HH = 24, WW = 24;

typedef vector<int> vecInt;

const int LineHeightThreshold = 10;
const int WordWidthThreshold = 3;

int threshForBitMap = 170;  ///* It is needed to be 160-170 specifically !!!
int threshForSegment = 3;
float resizeFact = 1.0;

const char *ConsoleChar = "#";
const char *ConsoleSpace = " ";
enum { HORIZONTAL = 1, VERTICAL = 0 };


void print_roi_image(const Mat &roi) {
    int r = roi.rows, c = roi.cols;
    const uchar *data;

    for (int j = 0; j < r; j++) {
        data = roi.ptr<uchar>(j);
        for (int i = 0; i < c; i++)
            printf(data[i] ? "@" : ".");

        printf("\n");
    }

    printf("\n");
}

void getActualWordHeightData(const Mat word, int &wordStartTopPos, int &presentWordHeight) {
    int wordHeight = word.rows, wordEndBottomPos;
    for (int k = 0; k < wordHeight; k++) {
        if (countNonZero(word.row(k))) {
            wordStartTopPos = k;
            break;
        }
    }

    for (int k = wordHeight - 1; k >= 0; k--) {
        if (countNonZero(word.row(k))) {
            wordEndBottomPos = k;
            break;
        }
    }

    presentWordHeight = wordEndBottomPos - wordStartTopPos;
}

int getMatraPos(const vector<int> &hHist, int &maxHHIdx,
                 int &matraStart, int &matraEnd) {

    int lastIdx;
    maxHHIdx = 0;
    int r_2 = hHist.size() * 0.4, maxHH = hHist[0];
    for (int i = 1; i <= r_2; i++) {
        int temp = hHist[i];
        if (temp > maxHH) maxHH = temp, maxHHIdx = i;
    }

    double matraTH = maxHH * 0.75;
    matraStart = matraEnd = maxHHIdx;

    for (int i = maxHHIdx - 1; i >= 0; i--)
        if (hHist[i] >= matraTH) matraStart = i;
        else break;

    for (int i = maxHHIdx + 1; i <= r_2; i++)
        if (hHist[i] >= matraTH) matraEnd = i;
        else break;


    for(int i=(int)(hHist.size()); i>=0; i-- )
    {
        if(hHist[i]>=3)
        {
            lastIdx=i; break;
        }
    }

    matraEnd++;
    return lastIdx;
}

void resizeImage(Mat src, Mat &dst){

    int new_rowNum = (int) src.rows * resizeFact , new_colNum = (int) src.cols * resizeFact ;

    resize(src,dst,Size(new_colNum,new_rowNum));
}

int countBlackBits(Mat &src, vecInt &numOfWhiteBitsList, int allignment ){

    int length,  numOfWhiteBits, max=0;
    Mat line;

    length = (allignment == HORIZONTAL)? src.rows : src.cols ;
    numOfWhiteBitsList.clear();
    for(int i=0; i<length; i++)
    {
        line = (allignment == HORIZONTAL )? src.row(i) : src.col(i) ;
        numOfWhiteBits = countNonZero(line);

        numOfWhiteBitsList.push_back(numOfWhiteBits);

        if(max<numOfWhiteBits) max = numOfWhiteBits;
    }
    return max;
}

int countBlackBitsCurve(Mat &src, vecInt &numOfWhiteBitsListUpper, vecInt &numOfWhiteBitsListMiddle,  vecInt &numOfWhiteBitsListLower){

    int k, length, height, numOfWhiteBits, MAX=0;
    Mat line, lineUpper, lineLower, srcUpper, srcLower;

    length = src.cols ;

    numOfWhiteBitsListUpper.clear();
    numOfWhiteBitsListLower.clear();
    numOfWhiteBitsListMiddle.clear();

    for(int i=0; i<length; i++)
    {
        line = src.col(i);

        height= line.rows;

        int cntUpper=0, cntLower=0 , cntMiddle=0;
        for(k=2; k< (height*0.20); k++)
        {
            if(line.at<uchar>(k)!=0)
            {
                cntUpper++;
            }
        }
        for( ; k< (height*0.60); k++)
        {
            if(line.at<uchar>(k)!=0)
            {
                cntMiddle++;
            }
        }
        for( ; k<height; k++)
        {
            if(line.at<uchar>(k)!=0)
            {
                cntLower++;
            }
        }
        if(DEBUG)cout<<cntUpper<<" "<<cntMiddle<<" "<< cntLower<<endl;
        numOfWhiteBitsListUpper.push_back(cntUpper);
        numOfWhiteBitsListMiddle.push_back(cntMiddle);
        numOfWhiteBitsListLower.push_back(cntLower);
    }
}

void horVertHistogram(Mat &src, vecInt &numOfWhiteBitsList, Mat &dst, int allignment){

    int maximumBits, length, width, height;

    maximumBits = countBlackBits(src, numOfWhiteBitsList, allignment);
    length = (allignment == HORIZONTAL) ? src.rows : src.cols;

    if(allignment == HORIZONTAL){
        width = maximumBits;
        height = numOfWhiteBitsList.size();
        dst = Mat::zeros(Size(width,height),CV_8U);

        for(int i=0; i< height; i++){
            line(dst, Point(0,i),Point(numOfWhiteBitsList.at(i),i), 255 );
        }
    }else{
        width =  numOfWhiteBitsList.size();
        height = maximumBits ;
        dst = Mat::zeros(Size(width,height),CV_8U);

        for(int i=0; i< width; i++){
            line(dst, Point(i,height),Point(i,height-numOfWhiteBitsList.at(i)), 255 );
        }
    }


}

void getSpacePosition(const vecInt &numOfWhiteBitsList,
            vecInt &start_pos, vecInt &stop_pos) {
    int  numOfWhiteBits;
    bool isStart = false, isStop = true;

    start_pos.clear();
    stop_pos.clear();
    for (int i = 0; i < numOfWhiteBitsList.size(); i++) {
        numOfWhiteBits = numOfWhiteBitsList.at(i);
        if (numOfWhiteBits) {
            isStop = false;
            if (isStart) continue;
            else {
                start_pos.push_back(i);
                isStart = true;
            }
        }
        else {
            isStart = false;
            if (isStop) continue;
            else {
                stop_pos.push_back(i);
                isStop = true;
            }
        }
    }
}

void print_workingRegion_image(const Mat &workingRegion) {
    int i, j, r = workingRegion.rows, c = workingRegion.cols;
    const uchar *data;

    for (j = 0; j < r; j++) {
        data = workingRegion.ptr<const uchar>(j);
        for (i = 0; i < c; i++)
            printf(data[i] ? ConsoleChar : ConsoleSpace);
        printf("\n");
    }

    printf("\n");
}


vs TrainVec;
ofstream outputFile;

void BuildTrainVec()
{
    ifstream inpFile(inpTextFile);

    TrainVec.clear();

    int ind=1;
    string tmp;


    TrainVec.pb(" ");

    while(ind<=TotalNumOfChar)
    {
        getline(inpFile, tmp);
        TrainVec.pb(tmp);
       // cout<<tmp;
        ind++;
    }

    cout<<"Done!!!"<<endl;

}

bool NeedToSwap(int ind1, int ind2)
{
    if((ind1==34 || ind1==39 || ind1==40) && (ind2!=34 || ind2!=39 || ind2!=40) )
        return true;
    else
        return false;
}

int main(){

    double exec_time = (double)getTickCount();

    BuildTrainVec();
    outputFile.open(outTextFile);



#ifndef TRAIN_FILE_AVAILABLE
    Mat roi, samples_train, responses_train;

    Mat img_train = imread(TrainImageFileName, 0);
    if (img_train.empty()) {
        printf("Unable to open train image");
        exit(-1);
    }



    threshold(img_train, img_train, 0, 1, 1);

    ifstream boxFile(TrainBoxFileName);
    if (!boxFile.is_open()) {
        printf("Unable to open box file");
        exit(-1);
    }



    string line;
    int l, b, r, t;
    int w = img_train.cols, h = img_train.rows;
//     cout<<w<<" "<< h<<endl;
    for (int i = 0; getline(boxFile, line); i++) {

//        cout<<i<<endl;
        sscanf(line.c_str(), "%d %d %d %d", &l, &b, &r, &t);

        roi = img_train(Rect(l, h - t, r - l, t - b));


        resize(roi, roi, Size(HH, WW));
        roi.reshape(1, 1).convertTo(roi, CV_32FC1);

       // imshow("Train", roi);
//        waitKey();

        samples_train.push_back(roi);
        responses_train.push_back(i);
     //   cout<<l<<" "<<b<<" "<<i<<endl;
    }


    responses_train.convertTo(responses_train, CV_32FC1);
    FileStorage trainFile(TrainFileName, FileStorage::WRITE);
   // trainFile << "responsesData" << responses_train;
    trainFile << "samplesData" << samples_train;
    trainFile.release();
    boxFile.close();
#else

#endif

//    ifstream benPhon(TrainPhoneFileName);
//    if (!benPhon.is_open()) {
//        cout << "Unable to open phone file\n";
//        exit(-1);
//    }

//    string phon;
//    vector<string> phonList;
//    while (getline(benPhon, phon)) phonList.push_back(phon);

#ifndef KNN

#else
    cout << "Training and Recognizing using KNN ...\n\n";
    int K=32;
    Mat sampleIdx;
    KNearest model = KNearest();
    responses_train.convertTo(responses_train, CV_32FC1);
    model.train(samples_train, responses_train.reshape(1, 1), sampleIdx, false, K);
#endif

    exec_time = ((double)getTickCount() - exec_time) * 1000. / getTickFrequency();
    cout << "\ntime needed to train: " << exec_time << " milisec\n\n";


    ///*********************************************************************************************///
    /// ************************Segmentation Code starts Here***************************************///
    ///*********************************************************************************************///

    int maximumBits, imgWidth , imgHeight;
    int matraStart, matraEnd, maxMatraLineIdx, matraEndWord, matraStartWord;

    Mat sourceImg, threshedImg, lineBox, wordBox, histogramImg;
    vecInt numOfWhiteBitsList, lineStartPosList, lineStopPosList, wordStartPosList, wordStopPosList;


    sourceImg = imread(sourceFileName , 0);
    print_roi_image(sourceImg);

    //time Start
    exec_time = (double)getTickCount();
        //resize
    resizeImage(sourceImg,threshedImg);
    //convert the image into binary image
    threshold(threshedImg, threshedImg, threshForBitMap, 255, THRESH_BINARY_INV);

    imgWidth = threshedImg.cols;
    imgHeight = threshedImg.rows;
    return 0;
    //Horizontal histogram
    maximumBits = countBlackBits(threshedImg, numOfWhiteBitsList,HORIZONTAL);
    //find out no bits zone vertically, means line position
    getSpacePosition(numOfWhiteBitsList, lineStartPosList, lineStopPosList );

    int lineStartPosList_length = lineStartPosList.size();
    int lineStartPos, lineStopPos, lineHeight, maxlineHeight=0, lastIdxOfLine;



    ///**********************for calculting MaxmaxlineHeight before process*******************////
    for(int i=0; i< lineStartPosList_length; i++)
    {

        try {
            lineStartPos = lineStartPosList.at(i);
            lineStopPos = lineStopPosList.at(i);
            lineHeight = lineStopPos - lineStartPos;
            maxlineHeight= max(maxlineHeight, lineHeight);
            if(lineHeight < LineHeightThreshold) continue ;
        }
        catch(...){
            break;
        }
    }



    //loop through all the lines
    for(int i=0; i< lineStartPosList_length; i++)
    {

        try {
            lineStartPos = lineStartPosList.at(i);
            lineStopPos = lineStopPosList.at(i);
            lineHeight = lineStopPos - lineStartPos;
            if(lineHeight < LineHeightThreshold) continue ;
        }
        catch(...){
            break;
        }
        //capture individual lines
        lineBox = threshedImg(Rect(0, lineStartPos, imgWidth, maxlineHeight));
        countBlackBits(lineBox, numOfWhiteBitsList , HORIZONTAL);

        if(DEBUG)imshow("Line:", lineBox);
        if(DEBUG)waitKey();

        ///for word matra I have changed to line matra
        lastIdxOfLine = getMatraPos(numOfWhiteBitsList, maxMatraLineIdx, matraStart, matraEnd);

        //Vertical histogram
        horVertHistogram(lineBox, numOfWhiteBitsList, histogramImg, VERTICAL );

        //find no bits zone horizontally, means words position
        getSpacePosition(numOfWhiteBitsList, wordStartPosList, wordStopPosList);
        int wordStartPosList_length = wordStartPosList.size();

        //find out ErrorWordWidthThreshold
        int totalGap = 0, noOfGap = 0;
        for (int j = 0; j < wordStartPosList_length; j++) {
            try {
                totalGap += wordStartPosList.at(j + 1) - wordStopPosList.at(j);
                noOfGap++;
            } catch (...) {
                break;
            }
        }
        int  ErrorWordWidthThreshold = noOfGap ? std::ceil(0.5 * totalGap / noOfGap) : 2;
        bool errorWordOccured=false;
        int presentWordStartPos , presentWordStopPos, presentWordWidth, nextWordStartPos, nextWordStopPos, errorWordLength;

        // loop through all the words
        for(int j=0; j<wordStartPosList_length; j++)
        {

            //find if the word is too short
            try {
                presentWordStartPos = wordStartPosList.at(j);
                presentWordStopPos = wordStopPosList.at(j);
                presentWordWidth = presentWordStopPos - presentWordStartPos;
                if (presentWordWidth <= WordWidthThreshold) continue;
            } catch (...) {
                break;
            }
            //find if any character is separated from word for some gap
            while (1) {
                try {
                     errorWordLength = wordStartPosList.at(j + 1) - wordStopPosList.at(j);
                    if (errorWordLength > ErrorWordWidthThreshold) break;
                    else presentWordStopPos = wordStopPosList.at(1 + j++);
                } catch (...) {
                    break;
                }
            }

            presentWordWidth = presentWordStopPos - presentWordStartPos;


            wordBox = lineBox(Rect(presentWordStartPos, 0, presentWordWidth, maxlineHeight));
            int wordStartTopPos, presentWordHeight=maxlineHeight;


            ///**there is no need to check if (wordheight*.75<wordWidth)*//
            /// there is no need to predict MatraIndex
            /// Predicting Working Region height Was Wrong !!!
            ///************************************************///
            ///Line Matra and Word matra calculated seperately///
            ///***********************************************///
            int maxMatraLineIdx ;
            countBlackBits(wordBox, numOfWhiteBitsList , HORIZONTAL);
            getMatraPos(numOfWhiteBitsList, maxMatraLineIdx, matraStartWord, matraEndWord);

            ///if the deviation is not so much then we can take the wordMatra as Working Matra
            if( abs(matraEnd-matraEndWord) <=3 )
            {
                int workingRegionHeight = (maxlineHeight-matraEndWord-1)* (.65);
                lastIdxOfLine = matraEnd + ( (maxlineHeight-matraEndWord-1)* (.66) );

                Mat workingRegion = wordBox(Rect(0, matraEndWord+1, presentWordWidth, workingRegionHeight));

//                if(j==0) imshow("workingRegion", workingRegion);
//                if(j==0)waitKey();

                vector< int > numOfWhiteBitsListUpper, numOfWhiteBitsListMiddle, numOfWhiteBitsListLower;

                countBlackBits(workingRegion, numOfWhiteBitsList, VERTICAL);
                countBlackBitsCurve(workingRegion, numOfWhiteBitsListUpper, numOfWhiteBitsListMiddle , numOfWhiteBitsListLower);

                for (int k = 0; k < presentWordWidth; k++)
                {

                    Mat Col, Matra;
                    Matra = wordBox(Rect(0, matraStartWord, presentWordWidth, (matraEndWord- matraStartWord)  ));

                    if(DEBUG)imshow("Matra", Matra);

                    Col = wordBox.col(k);



                    ///* three different segmentation
                    if( k>=2 && k<presentWordWidth-3 )
                    {
                        if( (numOfWhiteBitsListUpper[k]<=0  && numOfWhiteBitsListMiddle[k-2]<=0 && numOfWhiteBitsListMiddle[k-1]<=1 && numOfWhiteBitsListLower[k+1]<=2 ) //curve segmentation
                            or (numOfWhiteBitsList[k]<=1 && numOfWhiteBitsListMiddle[k-4]<=1 && numOfWhiteBitsListLower[k]==0 /*for go */)
                            or (numOfWhiteBitsList[k]<=0 && numOfWhiteBitsListMiddle[k-1]<=0 )    /*straigtLineSegment*/
                            //or (numOfWhiteBitsListMiddle[k]==3 && numOfWhiteBitsListLower[k]<=0 && numOfWhiteBitsListUpper[k]<=0) //no e to
                            /*or (numOfWhiteBitsListUpper[k]<1 && numOfWhiteBitsListMiddle[k]<1 && numOfWhiteBitsListLower[k]<=1 */

                          )
                        {
                            if(DEBUG)cout<<numOfWhiteBitsListUpper[k]<<" "<<numOfWhiteBitsListLower[k+1]<<endl;
                            for (int l = matraStartWord - 2; l <= matraEndWord+1; l++)      //here changed
                            {
                                Col.at<uchar>(l) = 0;
                            }

                        }

                        ///****Upper Segmentation******///
//                        Col.at<uchar>(matraStartWord-1) = 0;
//                        Col.at<uchar>(matraStartWord-2) = 0;
//                        Col.at<uchar>(matraStartWord-3) = 0;


                     //   Col.at<uchar>(lastIdxOfLine-1) = 0;
                     //   Col.at<uchar>(lastIdxOfLine) = 0;
                     //   Col.at<uchar>(lastIdxOfLine+1) = 0;
                    }
                }
            }



            ///*********************************************************************************************************///
            ///***********************************Recognition starts here***********************************************///
            ///*********************************** Each Word Recognition************************************************///
            ///*********************************************************************************************************///
            if(RECOGDEBUG)cout<<"New Word Recognition result:"<<endl<<endl<<endl;

//            cout<<TrainVec.size()<<"vec size\n";


            {
                size_t noOfLabel;
                vi vecInd;  vecInd.clear();



                vector<CCA> CC = LabelingImpl(wordBox, 8, noOfLabel);
                sort(CC.begin(), CC.end(), [](const CCA & a, const CCA & b) -> bool {
                    Rect ra = a.getBoundingBox();
                    Rect rb = b.getBoundingBox();
                    int xa = ra.x, xb = rb.x, ya = ra.y, yb = rb.y;
                    return (std::abs(yb - ya) <= 36) ? (xb > xa) : (yb > ya);
                });

                int py, px, result;
                for (CCA cc : CC) {
                    if (cc.getPixelCount() > 50) {
                        Rect r = cc.getBoundingBox();
                        roi = Mat::zeros(r.height + 1, r.width + 1, CV_8U);
                        for (Point2i p : cc.getPixels()) {
                            py = r.height - (r.y - p.y);
                            px = p.x - r.x;
                            roi.at<uchar>(py, px) = 1;
                        }

                        resize(roi, roi, Size(HH, WW));

                        roi.reshape(1, 1).convertTo(roi, CV_32FC1);

                        result = model.find_nearest(roi, 1);
                        vecInd.pb(result+1);
                     //   cout<<result<<endl;
                     //   outputFile<<TrainVec[result+1];
//                        if(RECOGDEBUG)cout << phonList[result] <<endl;
                      //  waitKey();
                    }
                }

                /// swapping
                int ii;
                REP(ii, SZ(vecInd))
                {
                    if(ii+1<SZ(vecInd) and NeedToSwap(vecInd[ii], vecInd[ii+1] ))
                    {
                        swap(vecInd[ii], vecInd[ii+1]);
                        ii++;
                    }
                }

              //  cout<<SZ(vecInd)<<endl;
                /// Printing in Text File
                REP(ii, SZ(vecInd))
                {
                    outputFile<<TrainVec[ vecInd[ii] ];
                }
                outputFile<<" ";
                vecInd.clear();

            }






            ///*********************************************************************************************************///
            ///***********************************Recognition Ends here*************************************************///
            ///*********************************************************************************************************///

            if(RECOGDEBUG)imshow("word", wordBox);
            if(RECOGDEBUG)waitKey();






        }
    }

//    int i;
//    cout<<TrainVec.size()<<" vec size at last....\n";
//    REP(i, SZ(TrainVec))
//    {
//        cout<<i<<endl;
//        cout<<TrainVec[i];
//        outputFile<<TrainVec[i];
//    }

    //time end
    exec_time = ((double)getTickCount() - exec_time) * 1000. / getTickFrequency();
    cout << "time needed to run [Segmentation and recognition]: " << exec_time << " milisec\n";
    threshold(threshedImg, threshedImg, 180, 255, THRESH_BINARY_INV);
    imwrite(dstFileName, threshedImg);
    outputFile.close();
    waitKey();
    return 0;
}
