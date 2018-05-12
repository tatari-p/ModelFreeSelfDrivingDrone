# ModelFreeSelfDrivingDrone

Tensorflow implementation of self driving drone using Deep Deterministic Policy Gradient and Galaga method.

Microsoft AirSim is needed.

For random target, it achieves about 82% success rate using 18 layer ResNET and 2 layer fully connected only.

It uses depth map for image processing, and I think deeper network, like 34 or 50 layer ResNET, will be needed to use raw RGB input.

* Training Environment
- CPU: AMD Ryzen 7 1700
- GPU: nvidia GTX1080Ti GDDR5X 11GB (Factory Overclocked)
- Training Time: under 8 hours
