# 机器学习与交通运输
## 入门与基础
## 项目: 泰坦尼克号乘客幸存率
### 安装要求

这个项目需要安装 **Python** 以及安装下列python库

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

你还需要安装和运行 Jupyter Notebook。

如果你还没有安装python，这里强烈推荐你安装Anaconda，一个包含了项目需要的所有库和软件的python发行版本。这里介绍了如何安装Anaconda。
如果你使用macOS系统并且对命令行比较熟悉，可以安装homebrew，以及brew版python。

```bash
$ brew install python
```
再用下列命令安装所需要的python库

```bash
$ pip install numpy pandas matplotlib scikit-learn scipy jupyter
```

### 代码

核心代码在 `titanic survival exploration.ipynb` 文件中，辅助代码在 `titanic_visualizations.py` 文件中。尽管已经提供了一些代码帮助你上手，你还是需要补充些代码使得项目要求的功能能够成功实现。
### 运行

在命令行中，确保当前目录为 titanic_survival_exploration/ 文件夹的最顶层（目录包含本 README 文件），运行下列命令：
```bash
jupyter notebook titanic_survival_exploration.ipynb
```
or
```bash
ipython notebook titanic_survival_exploration.ipynb
```

这会启动 Jupyter Notebook 把项目文件打开在你的浏览器中。

### 数据

​ 这个项目的数据包含在 `titanic_data.csv`。 文件包含下列特征：

**Survived**：是否存活（0代表否，1代表是）
- **Pclass**：社会阶级（1代表上层阶级，2代表中层阶级，3代表底层阶级）
- **Name**：船上乘客的名字
- **Sex**：船上乘客的性别
- **Age**：船上乘客的年龄（可能存在 `NaN`）
- **SibSp**：乘客在船上的兄弟姐妹和配偶的数量
- **Parch**：乘客在船上的父母以及小孩的数量
- **Ticket**：乘客船票的编号
- **Fare**：乘客为船票支付的费用
- **Cabin**：乘客所在船舱的编号（可能存在 `NaN`）
- **Embarked**：乘客上船的港口（C 代表从 Cherbourg 登船，Q 代表从 Queenstown 登船，S 代表从 Southampton 登船）

对jupyter不熟悉的同学可以看一下下面链接。
[Jupyter使用视频教程](http://cn-static.udacity.com/mlnd/how_to_use_jupyter.mp4)
- [为什么使用jupyter？](https://www.zhihu.com/question/37490497)