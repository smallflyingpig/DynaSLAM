echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building DynaSLAM ..."

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DPYTHON_INCLUDE_DIR=/home/lailo/anaconda3/envs/py27/include -DPYTHON_LIBRARY=/home/lailo/anaconda3/envs/py27/lib ..  #Release
make -j4
