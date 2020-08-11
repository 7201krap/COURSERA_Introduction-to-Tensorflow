def solution(clothes):
    hash_set = {}

    for name, category in clothes:
        hash_set[category] = 0

    for name, category in clothes:
        print(name, category)
        hash_set[category] = hash_set[category] + 1


    # clothes = {y:x for x,y in clothes.iteritems()}


    print(hash_set)

    answer = 0
    return answer

# solution([['yellow_hat', 'headgear'], ['blue_sunglasses', 'eyewear'], ['green_turban', 'headgear']])
solution([['crow_mask', 'face'], ['blue_sunglasses', 'face'], ['smoky_makeup', 'face']])
