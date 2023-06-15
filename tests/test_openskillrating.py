from scipy.stats import norm

from pufferlib.rating import OpenSkillRating


def test_openskillrating():
    rating_obj = OpenSkillRating(mu=1000, anchor_mu=1000, sigma=100 / 3)
    for i in range(1, 11):
        rating_obj.add_policy(i)
    rating_obj.set_anchor('anchor')

    # run 100 updates on the policies
    for j in range(100):
        policy_ids = []
        scores = []
        for i in range(1, 11):
            # score each policy by sampling from a Normal Distribution centered around a mean of their id number
            # in the limit, this should order the policies by their id number
            policy_ids.append(i)
            scores.append(norm.rvs(loc=i, scale=1))
        rating_obj.update(policy_ids, scores=scores)

    # print the mu of each policy
    del rating_obj.ratings['anchor']
    # print(rating_obj.stats)
    # print(sorted(rating_obj.stats.items(), key=lambda x: x[1]))
    sorted_rating = sorted(rating_obj.stats.items(), key=lambda x: x[1])
    passed = False
    for i in range(1, 11):
        if sorted_rating[i - 1][0] == i:
            passed = True
        else:
            passed = False
            break
    assert passed, 'OpenSkillRating failed'


if __name__ == '__main__':
    test_openskillrating()