## Eigenface
> A Python class that implements the Eigenfaces algorithm for face recognition, using eigen decomposition and principle component analysis.
> We use the AT&T data set and my own face, in order to reduce the number of computations.
> Additionally, we use a small set of celebrity images to find the best dataset matches to them.
> All images should have the same size, namely (92 width, 112 height).
### 项目文件结构

本次项目文件夹中有如下文件（夹），其作用分别对应如下表所示。

|  **文件/文件夹名**   |                          **功能**                          |
| :------------------: | :--------------------------------------------------------: |
|     **Dataset**      |       存放AT&T数据集的40份人脸样本及1份我的人脸样本        |
|   **changePCs.py**   | 通过更改能量调节PCs数目，记录不同PCs参与时识别准确率并作图 |
|  **Eigenfaces.py**   |                   Eigenface类的定义文件                    |
| **myreconstruct.py** |     人脸重建文件，用于实现不同PCs参与下的人脸重建效果      |
|    **mytest.py**     |        利用给定的一张图片进行测试，测试模型识别效果        |
|    **mytrain.py**    |                  对模型进行训练并输出模型                  |

### Algorithm Reference

[Link](http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html#algorithmic-description) to the description of the algorithm in the OpenCV documentation.
