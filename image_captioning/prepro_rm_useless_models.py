import os

# 110: 20
# 111: 32
# 112: 23
# 113: 34
# 114: 34
# 115: 27
# 116: 33
# 117: 25


if __name__ == '__main__':
    seed = 110
    save_epoch = 18

    model_path = 'models/encoder_decoder_inception_v4_seed_' + str(seed)

    for i in range(save_epoch + 11):
        if i != save_epoch:
            current_remove_model_path = os.path.join(model_path, 'model_epoch-' + str(i) + '.pth')
            current_remove_optimizer_path = os.path.join(model_path, 'optimizer_epoch-' + str(i) + '.pth')

            print('removing {}'.format(current_remove_model_path))
            os.system('rm ' + current_remove_model_path)

            print('removing {}'.format(current_remove_optimizer_path))
            os.system('rm ' + current_remove_optimizer_path)
