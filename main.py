import random
import datetime
from utils import *
from model import *
from ops import *


client = MultirotorClient()
env = ModelFreeAirSimEnv(client)

model_name = "DeepDeterministicPolicyGradient-0"
save_path = "./Models/"+model_name+".ckpt"
restore_from_saved_model = True

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

buffer_size = 2000
train_size = 30
num_train_per_epoch = 50
step = 10
save_period = 20
write_meta = not restore_from_saved_model

with tf.Session(config=config) as sess:
    main = DeepDeterministicPolicyGradient("main")
    target = DeepDeterministicPolicyGradient("target")

    main.build((144, 256))
    target.build((144, 256))

    print("*** Building model is completed.")

    if restore_from_saved_model:
        tf.train.Saver().restore(sess, save_path)
        print("*** Model is restored successfully.")

    else:
        sess.run(tf.global_variables_initializer())

    print("*** Train of model", model_name, "began at", datetime.datetime.now())
    saver = tf.train.Saver()
    max_epochs = 3000

    env.set_to_drive()
    print("*** Get ready to train...")

    train_buffer = []
    env.head_to_target()

    for epoch in range(max_epochs):
        l_rate_Q = 0.0005*0.5**epoch//200
        l_rate_p = 0.00005*0.5**epoch//200

        print("*** Epoch %d started..." % epoch)
        print("*** Target point was set to (%.1f, %.1f, %.1f)" % (env.target[0], env.target[1], env.target[2]))

        state, reward, done = env.get_state()

        episode_step = 0

        noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(3))

        while True:
            action = np.clip(main.predict_p(np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0), np.expand_dims(state[2], axis=0))[0]+noise(), -1., 1.)
            env.drive(action)
            next_state, reward, done = env.get_state()

            if episode_step % step == 0:
                p = env.get_position()
                print("*** Current position is (%.1f, %.1f, %.1f)" % (p[0], p[1], p[2]))

            episode_step += 1

            if episode_step > 100:
                reward = -1.
                done = True
                print("*** Drone is straying. Environment is reset.")
                env.reset()

            train_buffer.append((state, action, reward, done, next_state))

            if len(train_buffer) > buffer_size:
                train_buffer.pop(0)

            if done:
                break

            state = next_state

        loss_buffer = []

        if len(train_buffer) >= buffer_size/5 and epoch % 5 == 0:

            for _ in range(num_train_per_epoch):
                batch = random.sample(train_buffer, train_size)
                Q_loss = replay_DDPG(main, target, batch, l_rate_Q, l_rate_p)
                loss_buffer.append(Q_loss)

            target.update("main")
            print("*** Neural network has been updated.")
            exp = np.mean(loss_buffer)
            print("*** Q_loss: {}".format(exp))

            if epoch % save_period == 0:
                saver.save(sess, save_path, write_meta_graph=write_meta)

                if write_meta:
                    write_meta = False

                print("*** Model was saved successfully.")