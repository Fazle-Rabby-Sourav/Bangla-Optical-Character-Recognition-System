///****************************************************************************************************************************///
///********************************* Developed By Fazle Rabby Sourav && Ahnaf Rownak ******************************************///
///****************************************************************************************************************************///

#ifndef CCA_H
#define CCA_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef ushort dtype;
typedef const int Cint;
typedef const bool Cbool;
typedef const uchar Cuchar;
typedef const size_t Csize_t;
typedef const dtype Cdtype;
typedef vector<Point_<dtype>> vecPoint;

class CCA {
    Rect m_bb;
    int m_pixel_count;
    dtype minX, minY, maxX, maxY;
    vecPoint m_pixels;

public:
    CCA() {
        m_bb = Rect(0, 0, 0, 0);
        minX = minY = 2147483647;
        maxX = maxY = m_pixel_count = 0;
        m_pixels = vecPoint();
    }

    inline void addPixel(dtype x, dtype y) {
        m_pixel_count++;
        if (x > maxX) maxX = x;
        if (x < minX) minX = x;
        if (y > maxY) maxY = y;
        if (y < minY) minY = y;
        m_bb = Rect(minX, maxY, maxX - minX, maxY - minY);
        m_pixels.push_back(Point(x, y));
    }

    inline int getBoundingBoxArea() const { return (m_bb.width * m_bb.height); }
    inline Rect getBoundingBox() const { return m_bb; }
    inline int getPixelCount() const { return m_pixel_count; }
    inline vecPoint getPixels() const { return m_pixels; }
};

class DisjointSet {
    dtype *P, count;

public:
    DisjointSet(int length) {
        P = new dtype[length];
        P[0] = 0;
        count = 1;
    }

    ~DisjointSet() { delete[] P; }
    inline dtype get(dtype i) { return P[i]; }
    inline dtype add() {
        P[count] = count++;
        return count - 1;
    }

    inline dtype findRoot(dtype i) {
        dtype root = i;
        while (P[root] < root) root = P[root];
        return root;
    }

    inline void setRoot(dtype i, dtype root) {
        dtype j;
        while (P[i] < i) j = P[i], P[i] = root, i = j;
        P[i] = root;
    }

    inline dtype find(dtype i) {
        dtype root = findRoot(i);
        setRoot(i, root);
        return root;
    }

    inline dtype set_union(dtype i, dtype j) {
        dtype root = findRoot(i), rootj;
        if (i != j) {
            rootj = findRoot(j);
            if (root > rootj) root = rootj;
            setRoot(j, root);
        }

        setRoot(i, root);
        return root;
    }

    inline size_t flattenL() {
        dtype t;
        size_t k = 1;
        for (size_t i = 1; i < count; ++i)
            t = P[i], P[i] = t < i ? P[t] : k++;

        return k;
    }
};

inline static vector<CCA>
LabelingImpl(const Mat &I, uchar connectivity, size_t &nLabels) {
    CV_Assert(connectivity == 8 || connectivity == 4);

    Mat_<dtype> L(I.size());
    Cint rows = L.rows, cols = L.cols;
    Csize_t Plength = 4 * ((size_t(rows + 2) / 3) * (size_t(cols + 2) / 3));
    DisjointSet ds(Plength);


    if (connectivity == 8) {
        for (int r_i = 0; r_i < rows; r_i++) {
            dtype *Lr = L.ptr<dtype>(r_i);
            dtype *Lr_p = L.ptr<dtype>(r_i - 1);
            dtype *Lrow = Lr, *Lrow_prev = Lr_p;
            Cuchar *Irow = I.ptr<uchar>(r_i);
            Cuchar *Irow_prev = I.ptr<uchar>(r_i - 1);
            Cuchar *Ir = Irow, *Ir_p = Irow_prev;
            Cbool T_r = (r_i - 1) >= 0;

            for (int c_i = 0; c_i < cols; Irow++, c_i++) {
                if (!*Irow) { Lr[c_i] = 0; continue; }

                Irow_prev = Ir_p + c_i;
                Lrow = Lr + c_i;
                Lrow_prev = Lr_p + c_i;
                Cbool T_a = T_r && (c_i - 1) >= 0 && *(Irow_prev - 1);
                Cbool T_b = T_r && *Irow_prev;
                Cbool T_c = T_r && (c_i + 1) < cols && *(Irow_prev + 1);
                Cbool T_d = (c_i - 1) >= 0 && *(Irow - 1);

                *Lrow = T_b ? *Lrow_prev :
                        T_c ? T_a ? ds.set_union(*(Lrow_prev + 1), *(Lrow_prev - 1)) :
                        T_d ? ds.set_union(*(Lrow_prev + 1), *(Lrow - 1)) : *(Lrow_prev + 1) :
                        T_a ? *(Lrow_prev - 1) : T_d ? *(Lrow - 1) : ds.add();
            }
        }
    } else {
        for (int r_i = 0; r_i < rows; r_i++) {
            dtype *Lr = L.ptr<dtype>(r_i);
            dtype *Lr_p = L.ptr<dtype>(r_i - 1);
            dtype *Lrow = Lr, *Lrow_prev = Lr_p;
            Cuchar *Irow = I.ptr<uchar>(r_i);
            Cuchar *Irow_prev = I.ptr<uchar>(r_i - 1);
            Cuchar *Ir = Irow, *Ir_p = Irow_prev;
            Cbool T_r = (r_i - 1) >= 0;

            for (int c_i = 0; c_i < cols; Irow++, c_i++) {
                if (!*Irow) { Lr[c_i] = 0; continue; }

                Irow_prev = Ir_p + c_i;
                Lrow = Lr + c_i;
                Lrow_prev = Lr_p + c_i;
                Cbool T_b = T_r && *Irow_prev;
                Cbool T_d = (c_i - 1) >= 0 && *(Irow - 1);

                *Lrow = T_b ? T_d ? ds.set_union(*(Lrow - 1), *Lrow_prev) : *Lrow_prev :
                        T_d ? *(Lrow - 1) : ds.add();
            }
        }
    }

    nLabels = ds.flattenL();
    vector<CCA> CC(nLabels);
    for (int r_i = 0; r_i < rows; r_i++) {
        dtype *Lrow = L.ptr<dtype>(r_i);
        for (int c_i = 0; c_i < cols; c_i++) {
            Cdtype l = ds.get(*Lrow);
            if (l) CC[l].addPixel(c_i, r_i);
            *Lrow++ = l;
        }
    }

    return CC;
}

#endif // CCA_H
