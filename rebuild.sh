cd /data2/chethan/mtech/PhysiWorldUAV
docker stop isaac-sim && docker rm isaac-sim
docker build -t isaac-sim-novnc:4.1.0 .
docker run ...   # same run command as Step 6