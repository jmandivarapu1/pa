set -x

echo "Compiling ..."
nvcc pyramid_image.cu `pkg-config --cflags --libs opencv` -o blur-effect.out

echo "Executing ..."
./blur-effect.out ./jpeg-home.jpg  9 256 652 10
