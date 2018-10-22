///****************************************************************************************************************************///
///********************************* Developed By Fazle Rabby Sourav && Ahnaf Rownak ******************************************///
///****************************************************************************************************************************///


#ifndef TRAIN_H
#define TRAIN_H

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



///****************************************////
#define trainFileNum 3







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


void BuildTrainedTextVector(vs& TrainedTextVector)
{
    int ind=1;
    TrainedTextVector.clear();
    TrainedTextVector.pb(" ");

    for(int i=1; i<=trainFileNum; i++)
    {
        string fileName;
        stringstream sts;

        sts<<i;
        sts>>fileName;
        fileName="train/"+fileName+".txt";

        string tmp;
        ifstream inpFile(fileName);

        for (int k = 0; getline(inpFile, tmp); k++)
        {
            TrainedTextVector.pb(tmp);
        }

        cout<<SZ(TrainedTextVector)<<endl;
        cout<<"Done Building TrainedTextVector"<<endl;
    }
}

/// New Training Method
vector<Mat> trainFromFile(){

    Mat roi;
    vector<Mat> trainChar;
    string line;
    int l, b, r, t;

    cout<<"start Tarin From File"<<endl;

    for(int i=1; i<=trainFileNum; i++)
    {
        string BoxFileName, imgFileName;
        stringstream sts;

        sts<<i;
        sts>>BoxFileName;
        BoxFileName="train/"+BoxFileName+".box";

        sts.clear();

        sts<<i;
        sts>>imgFileName;
        imgFileName="train/"+imgFileName+".bmp";

        Mat img_train = imread(imgFileName, 0);
        threshold(img_train, img_train, 180, 255, THRESH_BINARY_INV);

        cout<<i<<" :Load Image file"<<endl;

        ifstream boxFile(BoxFileName);
        if (!boxFile.is_open()) {
            printf("Unable to open box file");
            exit(-1);
        }

        int w = img_train.cols, h = img_train.rows;
//        cout<<w<<" "<< h<<endl;
        for (int k = 0; getline(boxFile, line); k++)
        {
//            cout<<k<<endl;
            sscanf(line.c_str(), "%d %d %d %d", &l, &b, &r, &t);

//            cout<<l<<" "<<b<<" "<<r<<" "<<t<<endl;

            roi = img_train(Rect(l, b, abs(l-r), abs(b-t)+1));

//            cout<<"Done!"<<endl;
            resize(roi, roi, Size(HEIGHT, WIDTH));
            threshold(roi, roi, 180, 255, THRESH_BINARY);
            trainChar.push_back(roi);
        }
//        cout<<"done"<<endl;
    }
    printf("[[%d]]", SZ(trainChar));
    return trainChar;
}

#endif
