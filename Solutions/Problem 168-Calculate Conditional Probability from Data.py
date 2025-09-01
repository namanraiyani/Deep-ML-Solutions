# https://www.deep-ml.com/problems/168

def conditional_probability(data, x, y):
    x_num=y_num=inter=0
    for d in data:
        if d[0]==x:
            x_num+=1
        if d[0]==x and d[1]==y:
            inter+=1
    if x_num==0:
        return 0.0
    return round(inter / x_num, 4)
