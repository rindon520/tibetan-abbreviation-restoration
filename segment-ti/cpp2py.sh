#!/bash/bin

python cpp2python.py
g++ tiseg.cc -o tiseg.so -shared -fPIC -I /home/wh/anaconda3/envs/Ti/include/python2.7/ -L /usr/lib64 -l python2.7 -lboost_python27
