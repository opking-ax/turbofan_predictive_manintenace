import math

def phm08_score(actual_rul_list, predicted_rul_list):
    total_score = 0

    for i in range(len(predicted_rul_list)):
        error = predicted_rul_list[i] - actual_rul_list[i]

        if error < 0:
            penalty = math.exp(-error / 13) - 1
        else:
            penalty = math.exp(error / 10) - 1

        total_score += penalty

    return total_score
