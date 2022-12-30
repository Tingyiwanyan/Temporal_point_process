from lib.train import temporal_point_process

if __name__ == '__main__':
    train_net = temporal_point_process()
    train_net.create_memory_bank()
    #train_net.train_temporal_progression()
