///****************************************************************************************************************************///
///********************************* Developed By Fazle Rabby Sourav && Ahnaf Rownak ******************************************///
///****************************************************************************************************************************///


#ifndef CCMP_H
#define CCMP_H

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

#define HEIGHT 35
#define WIDTH 35


int xx[8]= {1, 1,0,-1, -1, -1, 0, 1};
int yy[8]= {0, 1, 1, 1, 0, -1, -1, -1};



///Connected Component
vmat connected_comp(Mat wordBox, int matraStart, int matraEnd)
{
    int rowNum= wordBox.rows, colNum= wordBox.cols , CntComponent=0;

    Mat WorkingBoard;
    WorkingBoard= Mat::zeros(rowNum, colNum, CV_8U);

    vmat retMat;
    int taken[rowNum+5][colNum+5], cntInd=0;
    mset(taken, 0);

//    imshow("work", wordBox);
//    imshow("xxx", WorkingBoard);

    for(int c=0; c<colNum; c++)
    {
        for(int r=0 ; r<rowNum; r++)
        {
            if(wordBox.at<uchar>(r,c) != 0 && taken[r][c]==0)
            {
                ///Need to uncomment latter
                if(r>= matraStart-5 && r<=matraEnd+5)
                    continue;

                CntComponent++;
                {
                    int i, j, k, from_x,from_y, to_x, to_y, cntBit=0;
                    int UpperX= 999999, UpperY=999999, LowerX=0, LowerY=0;

                    queue <int> Q;
                    Q.push(r);
                    Q.push(c);
                    taken[r][c]=CntComponent;

                    while(!Q.empty())
                    {
                        from_x= Q.front();
                        Q.pop();
                        from_y= Q.front();
                        Q.pop();
                        cntBit++;

                        for(i=0; i<8; i++)
                        {
                            to_x= from_x+xx[i];
                            to_y= from_y+yy[i];

                            if(to_x>=0 && to_x<rowNum && to_y>=0 && to_y <colNum && taken[to_x][to_y]==0 && wordBox.at<uchar>(to_x, to_y) != 0)
                            {
                                cntBit++;
                                Q.push(to_x);
                                Q.push(to_y);
                                taken[to_x][to_y]=CntComponent;
                                WorkingBoard.at<uchar>(to_x, to_y)= wordBox.at<uchar>(to_x, to_y);
                            }
                        }

                        UpperX  = min(UpperX, from_y);
                        LowerX  = max(LowerX, from_y);

                        UpperY  = min(UpperY, from_x);
                        LowerY  = max(LowerY, from_x);
                    }
//                    cout<<" cntBit "<<cntBit<<endl;
                    if(cntBit>10)
                    {
                        ///******************This Line are important for image ratio
//                        wordBox = lineBox(Rect(presentWordStartPos, 0, presentWordWidth, maxlineHeight));
//                        printf("Matrastart=%d MatraEnd=%d upperY=%d LowerY=%d UX=%d Lx=%d\n", matraStart, matraEnd, UpperY, LowerY, UpperX, LowerX);
                        UpperY= min(matraEnd, UpperY);
//                        LowerY= max(rowNum, LowerY );
//                        printf("Matrastart=%d MatraEnd=%d upperY=%d LowerY=%d UX=%d Lx=%d\n\n", matraStart, matraEnd, UpperY, LowerY, UpperX, LowerX);

                        Mat CompoSingle = WorkingBoard(Rect(UpperX, UpperY,(LowerX-UpperX)+1 , (LowerY-UpperY)+1 ) );
                        resize(CompoSingle, CompoSingle, Size(HEIGHT, WIDTH));
                        threshold(CompoSingle, CompoSingle, 180, 255, THRESH_BINARY);

//                        imshow("singleComp", CompoSingle);
//                        waitKey();

                        retMat.pb(CompoSingle);

                        WorkingBoard= Mat::zeros(rowNum, colNum, CV_8U);
                        cntInd++;
                    }
                }
            }
        }
    }
//    for(int k=0; k<SZ(retMat); k++)
//    {
//        imshow("---After Pushback in ret vector", retMat.at(k));
//        waitKey();
//    }
//    cout<<CntComponent<<"ComponentNum"<<endl;
//    waitKey();

    return retMat;
}



#endif
