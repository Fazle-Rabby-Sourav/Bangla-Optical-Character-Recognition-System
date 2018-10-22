///****************************************************************************************************************************///
///********************************* Developed By Fazle Rabby Sourav && Ahnaf Rownak ******************************************///
///****************************************************************************************************************************///


#ifndef WORDPOSTPROCESS_H
#define WORDPOSTPROCESS_H

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

///************ Dari**********************///
#define dariIndex1 11

///************ A-kar index****************//
#define aKarIndex1 33
#define aKarIndex2 42
#define aKarIndex3 85

#define soreOindex 22

#define fotaIndex 12

#define fotaOkkhorStart 259
#define fotaOkkhorEnd 288

#define refIndex 41


bool IsEkar_Ikar_Ukar(int ind1)
{
    if((ind1==34 || ind1==38 || ind1==39) )
        return true;
    else
        return false;
}




void doWordPostProcess(vector<int> &wordIndexVector)
{

    /// Convert akar to Dari at the end of sentece
//    if( SZ(wordIndexVector)>0 and (wordIndexVector[0]==aKarIndex1 or wordIndexVector[0]==aKarIndex2 or wordIndexVector[0]==aKarIndex3   ) )
//    {
//        wordIndexVector[0]= dariIndex1;
//    }
//    cout<<"WordPostProcess"<< SZ(wordIndexVector) <<endl;

    for(int i=0; i<SZ(wordIndexVector); i++)
    {
        /// when Dari is at the middle of any word, convert it to aKar
//        if(i>0 && wordIndexVector[i]==dariIndex1)
//        {
//            wordIndexVector[i]= aKarIndex1;
//        }

        /// Ref
        if(i+1<SZ(wordIndexVector) and wordIndexVector[i+1]==refIndex)
        {
            swap(wordIndexVector[i], wordIndexVector[i+1]);
            i++;
        }

        /// Sore_o + aKar = soreA
        if(i+1<SZ(wordIndexVector) and
           wordIndexVector[i]==soreOindex and (wordIndexVector[i+1]==aKarIndex1 or wordIndexVector[i+1]==aKarIndex2 or wordIndexVector[i+1]==aKarIndex3) )
        {
            wordIndexVector[i]=0;
            wordIndexVector[i+1]= soreOindex+1;
        }

        /// Ekar , rossso ikar, oikar swaping
        if(i+2<SZ(wordIndexVector)
           and IsEkar_Ikar_Ukar(wordIndexVector[i])
           and ( (wordIndexVector[i+1]>= fotaOkkhorStart and wordIndexVector[i+1]<=fotaOkkhorEnd) or (wordIndexVector[i+1]==531) )
           and (wordIndexVector[i+2]==fotaIndex or wordIndexVector[i+2]==fotaIndex+1 ) )
        {
            swap(wordIndexVector[i], wordIndexVector[i+1]);
            swap(wordIndexVector[i+1], wordIndexVector[i+2]);
            i+=2;
        }
        else if(i+1<SZ(wordIndexVector) and IsEkar_Ikar_Ukar(wordIndexVector[i]) )
        {
            swap(wordIndexVector[i], wordIndexVector[i+1]);
            i++;
        }
    }
//    cout<<"return"<<endl;
    return;
}

#endif
