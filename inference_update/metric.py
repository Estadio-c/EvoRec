import json
import math
import argparse
import os

def score(rank, truth, k):
    len_t = len(truth)
    hr = 0
    ndcg = 0
    for i in range(len(rank)):
        for j in range(min(k, len(rank[i]))):
            if truth[i] == rank[i][j]:
                hr += 1
                ndcg += 1 / math.log2(j+2)
    hr /= len_t
    ndcg /= len_t
    return hr, ndcg

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--edit_user', type=int)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--score_file', type=str)
    parser.add_argument('--generate_file', type=str)
    args = parser.parse_args()
    
    
    truth = []
    with open(args.test_file, 'r') as f:
        for line in f:
            truth.append(json.loads(line))
    res = []
    with open(args.generate_file, 'r') as f:
        for line in f:
            res.append(json.loads(line))
            
            
    t = []
    for i in truth:
        t.append(i['messages'][2]['content'].split('||')[0])
    r = []
    for i in res:
        r.append(i.split('||'))


    #计算edit域的性能
    print('Ua域上的性能为')
    hr_1, ndcg_1 = score(r, t, 1)
    print('hr@1:', hr_1)
    hr_3, ndcg_3 = score(r, t, 3)
    print('hr@3:', hr_3, 'ndcg@3:', ndcg_3)
    print("*"*30)
    
    score_data = {
        "checkpoint":args.checkpoint,
        "hr@1": hr_1,
        "hr@3": hr_3,
        "ndcg@3": ndcg_3
    }
    print(args.score_file)
    with open(args.score_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(score_data, ensure_ascii=False) + '\n')