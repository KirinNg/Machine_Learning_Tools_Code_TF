import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.style.use('ggplot')
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# 画出数据点和边界
def border_of_classifier(sklearn_cl, x, y):
        """
        param sklearn_cl : skearn 的分类器
        param x: np.array
        param y: np.array
        """
        ## 1 生成网格数据
        x_min, y_min = x.min(axis=0) - 9
        x_max, y_max = x.max(axis=0) + 9
        # 利用一组网格数据求出方程的值，然后把边界画出来。
        x_values, y_values = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        # 计算出分类器对所有数据点的分类结果 生成网格采样
        mesh_output = sklearn_cl.predict(np.c_[x_values.ravel(), y_values.ravel()])
        # 数组维度变形
        mesh_output = mesh_output.reshape(x_values.shape)
        fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
        ## 会根据 mesh_output结果自动从 cmap 中选择颜色
        plt.pcolormesh(x_values, y_values, mesh_output, cmap='rainbow')
        plt.scatter(x[:, 0], x[:, 1], c=y, s=100, edgecolors='steelblue', linewidth=1, cmap=plt.cm.Spectral)
        plt.xlim(x_values.min(), x_values.max())
        plt.ylim(y_values.min(), y_values.max())
        # 设置x轴和y轴
        plt.xticks((np.arange(np.ceil(min(x[:, 0]) - 1), np.ceil(max(x[:, 0]) + 1), 1.0)))
        plt.yticks((np.arange(np.ceil(min(x[:, 1]) - 1), np.ceil(max(x[:, 1]) + 1), 1.0)))

        plt.plot([0, 0], [-10, 10])
        plt.show()

fr = open('/home/kirin/Python_Code/Sawtooth/super_toy/super2_toy_data.txt', 'rb')
# fr = open('/home/kirin/Python_Code/Sawtooth/super_toy/super4_toy_data.txt', 'rb')

data_1000 = pickle.load(fr)

x, y = data_1000[0], data_1000[1]

# dtree1 = DecisionTreeClassifier(max_depth=2)
# dtree2 = DecisionTreeClassifier(max_depth=50)
# dtree3 = DecisionTreeClassifier(max_depth=200)
# dtree1.fit(x, y)
# dtree2.fit(x, y)
# dtree3.fit(x, y)
# border_of_classifier(dtree1, x, y)
# border_of_classifier(dtree2, x, y)
# border_of_classifier(dtree3, x, y)


svc_line1 = SVC(C=0.001, kernel='rbf')
svc_line2 = SVC(C=50.0, kernel='rbf')
svc_line3 = SVC(C=5000.0, kernel='rbf')
svc_line1.fit(x, y)
svc_line2.fit(x, y)
svc_line3.fit(x, y)
border_of_classifier(svc_line1, x, y)
border_of_classifier(svc_line2, x, y)
border_of_classifier(svc_line3, x, y)


# nn1 = MLPClassifier(hidden_layer_sizes=(8, 16, 8, 8, 8), max_iter=2000)
# nn2 = MLPClassifier(hidden_layer_sizes=(8, 16, 16, 8, 8), max_iter=2000)
# nn3 = MLPClassifier(hidden_layer_sizes=(8, 16, 16, 16, 8), max_iter=2000)
# nn1.fit(x, y)
# nn2.fit(x, y)
# nn3.fit(x, y)
# border_of_classifier(nn1, x, y)
# border_of_classifier(nn2, x, y)
# border_of_classifier(nn3, x, y)
