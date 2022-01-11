
import random
import pickle
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time

def comp_similarity(indv, target):
    """计算相似度，计算两个array的差值平方和
    :param indv: 图片像素
    :param target: 目标图片像素
    """
    score = np.sum(np.square((indv - target)))
    return score

# pixel array-like; out name.malplotlib支持保存为png，若想转换为jpeg，使用PIL
def save_img(pixel, out):
    fig = plt.figure(figsize=(5.3,5.3))
    plt.axis("off")
    plt.tight_layout()
    plt.imsave(out, arr=pixel)
    plt.close()

#随机生成个体染色体
def init_genes(id, x, y, z):
    """初始化个体基因
    :param id: 一个数字，与其他初始化相区别
    :param x: 图片高
    :param y: 图片长
    :param z: 4表示rgba, 3表示rgb;png是rgba, jpeg是rgb
    """
    np.random.seed(id)
    pixel = np.random.random((x, y, z)) #3维数组
    return pixel

#生成个体染色体，并计算和原始目标的相似度
def init_indv(id, x, y, z, target):
    """初始化个体数据，初始话像素基因，与目标相似度分数
    :param id: 一个数字，与其他初始化相区别
    :param x: 图片高
    :param y: 图片长
    :param z: 4表示rgba, 3表示rgb
    :param target: 目标像素
    """
    pixel = init_genes(id, x, y, z)
    score = comp_similarity(pixel, target)
    indv = {}
    indv['score'] = score
    indv["gene"] = pixel
    return indv

#种群染色体
def init_pop(target, p=20, jobs=5): #默认20，不过有传实参则变
    """初始化群体数据
    :param target: 目标像素
    :param p: 群体大小
    :param jobs: 进程数 多进程创建；如果种群数量很大有用
    """
    x, y, z = target.shape
    f_init = partial(init_indv, x=x, y=y, z=z,target=target) #partial是一种调用函数的方法；比如调用f_init，就相当于调用了init_indv，不过一些参数已经提前设置好了
    with Pool(jobs) as pl:
        for i,v in enumerate(pl.map(f_init, list(range(p)))): #ENUMRATE ->INDEX;list(range(p))->[0,1,2,...p-1],迭代作为init_indv的id实参
            data_pool[i] = v


def breed(p1, p2, mutation, width=5):
    """初始化群体数据
    :param p1: 个体1
    :param p2: 个体2
    :param mutation: 突变率
    :param width: 变异宽度
    """
    x, y, z = p1.shape
    new_p = p1.copy()
    x1_idx = random.sample(range(x), int(x/2))
    y1_idx = random.sample(range(y), int(y/2))
    x2_idx = list(set(range(x)) - set(x1_idx))
    y2_idx = list(set(range(y)) - set(y1_idx))

    #交叉 new_p是交叉得到的个体
    '''
    可以理解为矩阵替换。  先取个体2的x1，即左半部分，然后左半的上半部分替换为个体1的左上；同理右下部分替换为x1的。
    不过注意x1，y1是随机选的，所以是随机交叉，不是很规整的上半下半这样
        x1   x2
    y1  x1y1 x2y1
    
    y2 x1y2  x2y2
    '''
    temp1 = p2[x1_idx]  #注意是三维数组，但是由于没有在z维上再随机交叉，可以理解为二维数组的交叉。
    temp1[:,y1_idx] = p1[x1_idx][:, y1_idx] #其中一半替换为p1
    new_p[x1_idx] = temp1   #temp1.shape  250,500,3

    temp2 = p2[x2_idx]
    temp2[:,y2_idx] = p1[x2_idx][:, y2_idx]
    new_p[x2_idx] = temp2

    #变异 x*mutation 中的 y*mutation 个像素变异；仍可以用二维矩阵来理解
    m_x = random.sample(range(x), int(x*mutation))
    m_y = random.sample(range(y), int(y*mutation))


    indv_m = new_p.copy()

    for i,j in zip(m_x, m_y):
        channel = random.randint(0,z-1) #每个像素有三个channel，随机选一个突变
        center_p = new_p[i,j][channel] #取像素点的值,INT
        sx = list(range(max(0, i-width), min(i+width, x-1))) #width为变异宽度。从i左右2*width的范围内所有像素都突变为 i的正态分布
        sy = list(range(max(0, j-width), min(j+width, y-1)))
        mtemp = indv_m[sx]   #mtemp[:,sy].shape[:2]  (10, 10);  shape[0:2]第0维和第1维的长度
        normal_rgba = np.random.normal(center_p,.01, size=mtemp[:,sy].shape[:2]) #正态分布，均值为像素点的值

        # 大于1小于0的超范围了 normal_rgba[normal_rgba>1]这种访问方式可以访问所有大于1的元素
        normal_rgba[normal_rgba>1] = 1
        normal_rgba[normal_rgba<0] = 0

        mtemp[:,sy, channel] = normal_rgba # x*mutation 中的 y*mutation 个像素变异
        indv_m[sx] = mtemp
    return new_p, indv_m   #new_p normal; m mutation

'''
正态分布，均值为像素点的值， 比如像素值为0.3905757079134994，得到的就是
0.3905757079134994 [[0.38747752 0.38721341 0.39317362 0.39978442 0.39823821 0.39357596
  0.39235361 0.39924616 0.37682846 0.38807273]
 [0.3832875  0.39683169 0.38461452 0.39011044 0.38575063 0.39745371
  0.36966595 0.40141263 0.3856352  0.39080591]



'''


def crossover(g, target, pair, mutation):
    pi = data_pool.keys()
    males = random.sample(pi, int(len(data_pool)/2)) #取一半染色体作为父本
    females = set(pi) - set(males) #母本
    mm = random.sample(males, pair) #从父本中再取pair个.此处即是交叉概率，pair*2/种群大小
    fm = random.sample(females, pair)
    f_mate = lambda pair: breed(data_pool[pair[0]]["gene"], data_pool[pair[1]]["gene"], mutation) #此处pair是参数，不是数量，而是zip(mm,fm)得到的pair
    for idx, sps in enumerate(map(f_mate, zip(mm,fm))):
        n_indv = f"g{g}_{idx}_n"
        m_indv = f"g{g}_{idx}_m"
        data_pool[n_indv] = {}
        data_pool[m_indv] = {}
        data_pool[n_indv]["gene"] = sps[0]
        data_pool[n_indv]["score"] = comp_similarity(sps[0], target)
        data_pool[m_indv]["gene"] = sps[1]
        data_pool[m_indv]["score"] = comp_similarity(sps[1], target)

def evol(godie):
    scores = [(k, v['score']) for k, v in data_pool.items()]
    scores = sorted(scores, key=lambda x: x[1], reverse=True) #升序
    for i in range(godie):
        del data_pool[scores[i][0]]
    best_id, best_score = scores[-1][0], scores[-1][1] #最后一位
    best_indv = data_pool[best_id]
    return best_id, best_indv

# jiaran.jpeg 500*500 pixel
def main(args):
    pkl = args.evol_info # 50000 + 20000
    img = args.img
    p = args.population #20
    pair = args.pair #8
    generations = args.generation #50000
    mutation = args.mutation #0.05
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    global data_pool
    data_pool = {}
    einfo = {"data": data_pool, "g": 0} #g表示当前到了第几代
    target = mpimg.imread(img) # 读取jiaran.png，是RGB，https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html;  RGB和RGBA和png/jpeg间没有必然联系
    # 遇到问题是读取的是0-255的整数值，而上面的变异和初始种群生成，都是按照0-1的值处理的，  如75/255
    #查了下文档，只有png按照0-1处理，其他的都是int，需要额外处理/255；这里就不整了，直接png
    if args.c: #若之前有数据
        h =  open(pkl, "rb")
        einfo = pickle.load(h)
        data_pool = einfo["data"]
        h.close()
    else:
        init_pop(target, p=p, jobs=5) #初始种群大小20
    fcross = partial(crossover, target=target, pair=pair, mutation=mutation)
    for i in range(0+einfo["g"], einfo["g"]+500001):  #从保存的/初始的代数，进化50000代
        if i % 5000 == 0: #每5000代保存一次进化信息，即当前datapool，当前染色体
            einfo["data"] = data_pool
            einfo["g"] = i
            with open(pkl, "wb") as h:
                pickle.dump(einfo, h)
        fcross(i)
        best_id, best_indv = evol(pair*2)
        print(i, best_id, best_indv['score'])
        if i % 500 == 0: #每500代，保存一次当前最好的图片
            pixel = best_indv['gene']
            out = f"{outdir}/{best_id}_{best_indv['score']:.1f}.png" # 保存的图片命名 id（种群内id）_score   id=g31998_6_m 代数_种群内_
            save_img(pixel, out)

def command_parser():
    parser = argparse.ArgumentParser(description='toy program using genetic algorithm')
    parser.add_argument('img', help='目标图片')
    parser.add_argument('-pop', '--population', type=int, default=20, help='种群大小, default=20')
    parser.add_argument('-pair', type=int, default=8, help='几对夫妻繁衍下一代, default=8')
    parser.add_argument('-g', '--generation', type=int, default=50000, help='进化代数, default=50000')
    parser.add_argument('-m', '--mutation', type=float, default=.05, help='突变率,default=0.05')
    parser.add_argument('-ei', '--evol_info', default="evol_info.ga", help='存储进化信息文件，方便修改参数后再次运行,default=evol_info.ga')
    parser.add_argument('-c', action='store_true', help='使用evol_info继续进化')
    parser.add_argument('-o', '--outdir', required=True, help='输出目录')
    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    command_parser()
    #python MyGene.py jiaran.png -o generated

'''
https://www.jianshu.com/p/27c3684dfdf9
理论上先初始化n个体得到初始种群
计算每个染色体适应性
轮盘选择得到父本
按交叉概率父本两两交叉得到n个个体
n个个体按照变异概率变异
达到迭代次数则停止返回最优解，否则重复第二步
'''