def curriculum_horizon(epoch):
    if epoch < 10:
        return 1
    if epoch < 25:
        return 5
    if epoch < 50:
        return 10
    if epoch < 75:
        return 20
    return 50