import random
import pickle
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from itertools import accumulate
from bisect import bisect_right

import time

# calculate similarity score
# param chromo: 图片像素
# param target: 目标图片像素
def cal_similarity(chromo,target):
    score = np.sum(np.square((chromo - target)))
    return score

# init one chromosome
# param id: chromosome id
# param x,y,z: img size
# param target: 目标图片像素
# img RGB/RGBA
def init_chromosome(id, x, y, z,target):
    np.random.seed(id)
    img = np.random.random((x, y, z))
    score = cal_similarity(img, target)
    chromo = {}
    chromo['score'] = score
    chromo["gene"] = img
    return chromo

#init population
def init_population(target,population):
    x,y,z = target.shape
    func_init = partial(init_chromosome, x = x, y = y, z = z,target = target)
    with Pool(5) as pl:
        for i,v in enumerate(pl.map(func_init, list(range(population)))):
            data_pool[i] = v

def imp_mutation(orig_indv,mutation_rate, width=5):
    #mutation
    x, y, z = orig_indv.shape
    mutation_index_x = random.sample(range(x), int(x * mutation_rate))
    mutation_index_y = random.sample(range(y), int(y * mutation_rate))

    mutation_indv = orig_indv.copy()

    for i, j in zip(mutation_index_x, mutation_index_y):
        channel = random.randint(0, z - 1)  # 每个像素有三个channel，随机选一个突变
        center_p = orig_indv[i, j][channel]  # 取像素点的值,INT
        sx = list(range(max(0, i - width), min(i + width, x - 1)))  # width为变异宽度。从i左右2*width的范围内所有像素都突变为 i的正态分布
        sy = list(range(max(0, j - width), min(j + width, y - 1)))
        mtemp = mutation_indv[sx]  # mtemp[:,sy].shape[:2]  (10, 10);  shape[0:2]第0维和第1维的长度
        normal_rgba = np.random.normal(center_p, .01, size=mtemp[:, sy].shape[:2])  # 正态分布，均值为像素点的值
        # 大于1小于0的超范围了 normal_rgba[normal_rgba>1]这种访问方式可以访问所有大于1的元素
        normal_rgba[normal_rgba > 1] = 1
        normal_rgba[normal_rgba < 0] = 0
        mtemp[:, sy, channel] = normal_rgba  # x*mutation 中的 y*mutation 个像素变异
        mutation_indv[sx] = mtemp

    return mutation_indv


def imp_cross(father,mother):
    #cross randomly
    x, y, z = father.shape
    cross_indv = father.copy()
    index_x_1 = random.sample(range(x), int(x / 2))
    index_y_1 = random.sample(range(y), int(y / 2))

    index_x_2 = list(set(range(x)) - set(index_x_1))
    index_y_2 = list(set(range(y)) - set(index_y_1))

    tmp_gene1 = mother[index_x_1]  # 注意是三维数组，但是由于没有在z维上再随机交叉，可以理解为二维数组的交叉。
    tmp_gene1[:, index_y_1] = father[index_x_1][:, index_y_1]  # 其中一半替换为p1
    cross_indv[index_x_1] = tmp_gene1  # temp1.shape  250,500,3

    tmp_gene_2 = mother[index_x_2]
    tmp_gene_2[:, index_y_2] = father[index_x_2][:, index_y_2]
    cross_indv[index_x_2] = tmp_gene_2

    return cross_indv

# 交叉、变异的实现
# param p1: 个体1
# param p2: 个体2
# param mutation_rate: 突变率
# param width: 变异宽度width*2+1
# 变异概率100%，即2i个个体交叉生成i个个体，i个个体都变异得到i个个体，共生成2i个新个体
def imp_cross_mutation(father, mother, mutation_rate, width=5):
    cross_indv = imp_cross(father,mother)
    mutaion_indv = imp_mutation(cross_indv,mutation_rate, width)
    #return
    return cross_indv, mutaion_indv  # new_p normal; m mutation

#roulette wheel -> a pair of father, mother
# 未采用
def imp_select_wheel():
    fit = [indv["score"] for indv in data_pool]
    min_fit = min(fit)
    fit = [(i - min_fit) for i in fit]
    # Create roulette wheel.
    sum_fit = sum(fit)
    wheel = list(accumulate([i / sum_fit for i in fit])) #accumulate，累加；从而得到1/5，2/5，3/5...1这样的区间
    # Select a father and a mother.
    father_idx = bisect_right(wheel, random()) # random() 生成[0-1)之间的随机数，落在哪个区间，则取哪个gene作为父本
    father = data_pool[father_idx]
    mother_idx = (father_idx + 1) % len(wheel)
    mother = data_pool[mother_idx]
    return father, mother

#Delete individuals with low similarity; select individuals with high similarity;population-(population-pair*2)=pair*2
#population num = 32 = 4*pair
#4 pair - 2 pair = 2 pair
#2 pair + pair(cross) + pair(mutation) = 4 pair
def imp_select(delete_num):
    scores = [(key, value['score']) for key, value in data_pool.items()]
    scores = sorted(scores, key=lambda x: x[1], reverse=True) #降序
    for i in range(delete_num):
        del data_pool[scores[i][0]]

def select_cross_mutation(generation, target, pair, mutation_rate,population):
    imp_select(population-pair*2) #Delete individuals with low similarity; select individuals with high similarity;population-(population-pair*2)=pair*2
    pi = data_pool.keys()
    males = random.sample(pi, int(len(data_pool)/2))
    females = set(pi) - set(males)
    func_cross_mutation = lambda pair: imp_cross_mutation(data_pool[pair[0]]["gene"], data_pool[pair[1]]["gene"], mutation_rate) #此处pair是参数，不是数量，而是zip(mm,fm)得到的pair
    for id, gene in enumerate(map(func_cross_mutation, zip(males,females))):
        cross_indv = f"generation{generation}_id{id}_cross"
        mutation_indv = f"generation{generation}_id{id}_mutation"
        data_pool[cross_indv] = {}
        data_pool[mutation_indv] = {}
        data_pool[cross_indv]["gene"] = gene[0]
        data_pool[cross_indv]["score"] = cal_similarity(gene[0], target)
        data_pool[mutation_indv]["gene"] = gene[1]
        data_pool[mutation_indv]["score"] = cal_similarity(gene[1], target)



def best(delete_num):
    scores = [(k, v['score']) for k, v in data_pool.items()]
    scores = sorted(scores, key=lambda x: x[1], reverse=True) #升序
    best_id, best_score = scores[-1][0], scores[-1][1] #最后一位
    best_indv = data_pool[best_id]
    return best_id, best_indv

def save_img(pixel, out):
    fig = plt.figure(figsize=(5.3,5.3))
    plt.axis("off")
    plt.tight_layout()
    plt.imsave(out, arr=pixel)
    plt.close('all')

def frame(args):
    img = args.img
    target = mpimg.imread(img) #target img
    population = args.population #32
    pair = args.pair #8
    global data_pool
    data_pool = {}  # generated genes
    evolution_info = {"gene": data_pool, "iteration": 0}  # g表示当前到了第几代
    iterations = args.generation #100000
    mutation_rate = args.mutation #0.05
    output_dir = Path(args.outdir)
    output_dir.mkdir(exist_ok=True)
    t_start = time.time()
    pre_evolution_info = args.evol_info
    #use pre evolution info
    if args.continu:
        file_handler =  open(pre_evolution_info, "rb")
        pre_evoinfo = pickle.load(file_handler)
        data_pool = pre_evoinfo["gene"]
        file_handler.close()
    else:
        init_population(target, population=population) #初始种群大小32

    func_select_cross_mutation = partial(select_cross_mutation, target=target, pair=pair, mutation_rate=mutation_rate,population=population)
    for i in range(0+evolution_info["iteration"], evolution_info["iteration"]+iterations+1):  
        if i % 5000 == 0:
            evolution_info["gene"] = data_pool
            evolution_info["iteration"] = i
            with open(pre_evolution_info, "wb") as handler:
                pickle.dump(evolution_info, handler)
        func_select_cross_mutation(i)
        
        best_id, best_generatedimg = best(pair*2)
        print(i, best_id, best_generatedimg['score'])

        if i % 500 == 0: #save the best generated img
            img = best_generatedimg['gene']
            name = f"{output_dir}/gene-id_{best_id}_similarity-score_{best_generatedimg['score']:.1f}.png" # 保存的图片命名 id（种群内id）_score   id=g31998_6_m 代数_种群内_
            save_img(img, name)

            print(len(data_pool))
    t_end = time.time()
    print('time cost', t_end - t_start)

            

def command_parser():
    parser = argparse.ArgumentParser(description='toy program using genetic algorithm')
    parser.add_argument('img', help='目标图片')
    parser.add_argument('-pop', '--population', type=int, default=32, help='种群大小, default=32')
    parser.add_argument('-pair', type=int, default=8, help='几对夫妻繁衍下一代, default=8')
    parser.add_argument('-g', '--generation', type=int, default=100000, help='进化代数, default=100000')
    parser.add_argument('-m', '--mutation', type=float, default=.05, help='突变率,default=0.05')
    parser.add_argument('-ei', '--evol_info', default="evol_info.ga", help='存储进化信息文件，方便修改参数后再次运行,default=evol_info.ga')
    parser.add_argument('-continu', action='store_true', help='使用evol_info继续进化')
    parser.add_argument('-o', '--outdir', required=True, help='输出目录')
    parser.set_defaults(func=frame)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    command_parser()


'''
   usage：    #python Gene_comment.py target.png -o outputdir

（1）初始化n个体（染色体）得到初始种群，n = 4 mate 
（2）依据适应度计算函数计算每个染色体适应性
（3）依据选择算法得到mate对父本、母本
（4）按交叉概率父本母本两两交叉得到mate个交叉个体
（5）mate个个体按照变异概率变异，得到mate个变异个体
（6）mate对父本、母本和mate个交叉个体、mate个变异个体构成新的种群
（7）达到迭代次数则停止迭代，返回当前种群最优解，否则重复（2）-（6）步

'''