# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

#Importing required modules
import math
import random
import matplotlib.pyplot as plt

A = [1]*240+[2]*60+[1]*300+[2]*60+[1]*60 # 定义站点每分钟的人数增长
total_people = sum(A) # 一个站点在一天产生的总人数是固定的
total_time = len(A) # 一天的运行时间是固定的
D = [0.9]*3+[0.85]+[0.9]*3+[0.85]+[0.9]*4 # 定义每个站点的下车率
cap = 50 # 定义车的总容量
n = 50 # 定义班次数
L = [[0] * 12] * n # 定义L(i,j) := i班车离开j站点瞬间的人数  
W = [[0] * 12] * n # 定义W(i,j) := j站点在i班车离开瞬间的等待人数
# S = [i for i in range(n)] # 定义每班车的发车时间（迭代单元，决策变量）

def I(i,j):
    return sum(A[k] for k in range(i,j+1))
def d(j):
    return D[j]
def w(i,j):
    if i < 0 or j < 0:
        return 0
    return W[i][j]
def l(i,j):
    if i < 0 or j < 0:
        return 0
    return L[i][j]

def computeLandM(S):
    def s(i):
        return S[i]
    # 处理0号车到达前的人数积累
    for j in range(0,12):
        L[0][j] = min(cap, math.floor(d(j) * l(0, j-1)) + I(0, s(0)+j))
        W[0][j] = max(0, math.floor(d(j) * l(0, j-1)) + I(0, s(0)+j) - cap)

    # 迭代
    for i in range(1,n):
        for j in range(0,12):
            L[i][j] = min(cap, math.floor(d(j) * l(i, j-1)) + w(i-1, j) + I(s(i-1)+j+1, s(i)+j-1))
            W[i][j] = max(0, math.floor(d(j) * l(i, j-1)) + w(i-1, j) + I(s(i-1)+j+1, s(i)+j-1) - cap)


#First function to optimize
def function1(S):
    computeLandM(S)
    def s(i):
        return S[i]
    obj1 = 0
    for j in range(0,12):
        final = w(n-1, j) + I(s(n-1)+j+1, total_time-1)
        obj1 += total_people - final
    return -obj1

#Second function to optimize
def function2(S):
    computeLandM(S)
    obj2 = 0
    for j in range(0,12):
        waiting = 0
        for t in range(total_time):
            if t in S:
                waiting = w(S.index(t), j)
            else:
                waiting += I(t, t)
            obj2 += waiting
    return obj2

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1) + 1e5)
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2) + 1e5)
    return distance

#Function to carry out the crossover
def crossover(a,b):
    new = []
    combined  = list(set(a+b))
    return sorted(random.sample(combined, n))

#Function to carry out the mutation operator
def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob <1:
        mutation_point_1 = random.randint(0,len(solution)-1)
        mutation_point_2 = random.randint(0,len(solution)-1)
        t = min(mutation_point_1,mutation_point_2)
        mutation_point_2 = max(mutation_point_1,mutation_point_2)
        mutation_point_1 = t

        max_start_value = -1
        while max_start_value < 0:
            # 生成 n-1 个差值
            differences = [random.randint(1, 10) for _ in range(mutation_point_2 - mutation_point_1)]

            # 确定起始值的最大可能值
            max_start_value = solution[mutation_point_2] - sum(differences)
        
        # 随机选择起始值
        start_value = random.randint(0, max_start_value)

        # 构建整个列表
        random_list = [start_value]
        for diff in differences:
            random_list.append(random_list[-1] + diff)

        solution[mutation_point_1+1:mutation_point_2] = random_list
    return solution

#Main program starts here
pop_size = 30
max_gen = 100

#Initialization
min_x=-55
max_x=55
solution = []
for i in range(pop_size):
    max_start_value = -1
    while max_start_value < 0:
        # 生成 n-1 个差值
        differences = [random.randint(1, 10) for _ in range(n - 1)]

        # 确定起始值的最大可能值
        max_start_value = 705 - sum(differences)

    # 随机选择起始值
    start_value = random.randint(0, max_start_value)

    # 构建整个列表
    random_list = [start_value]
    for diff in differences:
        random_list.append(random_list[-1] + diff)

    solution.append(random_list)

gen_no=0
while(gen_no<max_gen):
    function1_values = [function1(solution[i])for i in range(0,pop_size)]
    function2_values = [function2(solution[i])for i in range(0,pop_size)]
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    print("The best front for Generation number ",gen_no, " is")
    for valuez in non_dominated_sorted_solution[0]:
        print(f"[{solution[valuez][0]}...{solution[valuez][-1]}]",end=" ")
        print(solution[valuez])
    print("\n")
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    solution2 = solution[:]
    #Generating offsprings
    while(len(solution2)!=2*pop_size):
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        solution2.append(crossover(solution[a1],solution[b1]))
    function1_values2 = [function1(solution2[i])for i in range(0,2*pop_size)]
    function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1

#Lets plot the final front now
function1 = [i for i in function1_values]
function2 = [j for j in function2_values]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()


