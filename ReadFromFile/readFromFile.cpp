// basic file operations
#include <iostream>
#include <fstream>
#include <iomanip>      // For std::hex
#include <iostream>     // For console output
#include <string>
#include <vector>     // For STL strings
#define TOTAL_NUM_OF_CHAR 286
using namespace std;
void readFileChar(FILE* file1){

    ifstream file("ben.txt");
    string temp;
    ofstream myfile;
    myfile.open ("output.txt");
    int index=0;
    vector<string> vec;
    temp = " ";
    vec.push_back(temp);
    int lineNumb =0;
        while(!EOF){
                //Do with temp
                getline(file, temp);
                vec.push_back(temp);
                cout << temp << endl ;
                lineNumb ++;
            }
    while(scanf("%d", &index)==1){

            myfile << vec[index];
    }
    myfile.close();
}

int main ()
{
    FILE *fp;
    printf("press the number of character and enter  (you can get the number from ben.txt)\n");
    printf("To stop press ctr+z\n");
    fp = fopen("text.txt", "r");
    readFileChar(fp);

}

