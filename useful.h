///****************************************************************************************************************************///
///********************************* Developed By Fazle Rabby Sourav && Ahnaf Rownak ******************************************///
///****************************************************************************************************************************///

#ifndef USEFUL_H
#define USEFUL_H

#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

void print_roi_image(const Mat &roi) {
    int r = roi.rows, c = roi.cols;
    const uchar *data;

    for (int j = 0; j < r; j++) {
        data = roi.ptr<uchar>(j);
        for (int i = 0; i < c; i++)
            printf(data[i] ? "@" : " ");

        printf("\n");
    }

    printf("\n");
}

void getMatraPos(const vector<int> &hHist, int &maxHHIdx,
                 int &matraStart, int &matraEnd) {
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

    matraEnd++;
}

void GetPos(const vector<int> &mHist,
            vector<int> &start_pos, vector<int> &stop_pos) {
    int i, hist, len_mHist = mHist.size();
    bool isStart = false, isStop = true;

    for (i = 0; i < len_mHist; i++) {
        hist = mHist.at(i);
        if (hist) {
            isStop = false;
            if (isStart) continue;
            else {
                start_pos.push_back(i);
                isStart = true;
            }
        } else {
            isStart = false;
            if (isStop) continue;
            else {
                stop_pos.push_back(i);
                isStop = true;
            }
        }
    }
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

Mat HorVertHistogram(const Mat &src_img, vector<int> &mhist,
                     bool isHor = true, bool isHistOn = false) {
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

    if (isHistOn) {
        if (isHor) {
            width = maximum, height = sz;
            histo = Mat::zeros(Size(width, height), CV_8U);
            for (i = 0; i < height; ++i)
                line(histo, Point(0, i), Point(mhist.at(i), i), 255);
        } else {
            width = sz, height = maximum;
            histo = Mat::zeros(Size(width, height), CV_8U);
            for (i = 0; i < width; ++i)
                line(histo, Point(i, maximum), Point(i, maximum - mhist.at(i)), 255);
        }
    }

    return histo;
}

#endif // USEFUL_H
