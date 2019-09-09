from game import Interceptor_V2
from game.Interceptor_V2 import Init, Draw, Game_step, World
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

gamma = 0.99


class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def Draw2(alternative_matrix, score):
    dim_x, dim_y = alternative_matrix.shape
    plt.cla()
    plt.rcParams['axes.facecolor'] = 'black'
    for x in range(0, dim_x):
        for y in range(0, dim_y):
            val = alternative_matrix[x, y]
            if val == rocket_symb:
                plt.plot(x, y, '.y')
            if val == inter_symb:
                plt.plot(x, y, 'or')
                C1 = plt.Circle((x, y), radius=1, linestyle='--', color='gray', fill=False)
                ax = plt.gca()
                ax.add_artist(C1)
            if val == city_symb:
                plt.plot(x, y, 'Xy')

    # for e in explosion_list:
    #     P1 = plt.Polygon(e.verts1, True, color='yellow')
    #     P2 = plt.Polygon(e.verts2, True, color='red')
    #     ax = plt.gca()
    #     ax.add_artist(P1)
    #     ax.add_artist(P2)
    # plt.plot(turret.x, turret.y,'oc', markersize=12)
    # plt.plot([turret.x, turret.x + 100*np.sin(np.deg2rad(turret.ang))],
    #          [turret.y, turret.y + 100*np.cos(np.deg2rad(turret.ang))],'c', linewidth = 3)
    # plt.plot(turret.x_hostile, turret.y_hostile,'or', markersize=12)
    # plt.axes().set_aspect('equal')
    # plt.axis([-world.width / 2, world.width / 2, 0, world.height])
    plt.title('Score: ' + str(score))
    plt.draw()
    plt.pause(0.001)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # try to load a saved model
    model_loader = tf.train.Saver()
    current_model_folder = "./trained_models/backup/model-1"
    if (os.path.exists(current_model_folder)):
        print("Loading pre calculaated model")
        model_loader.restore(sess, current_model_folder + "/model.ckpt")
    else:
        print("Creating the folder for the model to be stored in")
        os.makedirs(current_model_folder)



    Init()
    should_save_model = True
    #Alternative
    x_downsample = 50
    y_downsample = 20
    rocket_symb = 1
    inter_symb = 2
    city_symb = 3

    #World
    width = World.width
    half_w = int(width / 2)
    height = World.height
    half_h = int(height / 2)
    print("World width: " + str(World.width))
    for game in range(100):
        #Save the model
        if should_save_model:
            print("Saving most recent model")
            save_path = model_loader.save(sess, current_model_folder + "/model.ckpt")
            print("Model saved in path: %s" % save_path)

        for stp in range(1000):
            action_button = 3#np.random.randint(0, 4)
            r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
            alternative_matrix = np.zeros((201, 201))
            for r in r_locs:
                if r[0]+ half_w < 0 or r[0]+ half_w > width or r[1] < 0 or r[1] > height:
                    continue
                al_rx = int((r[0] + half_w) / x_downsample)
                al_ry = int((r[1]) / y_downsample)
                alternative_matrix[al_rx, al_ry] = rocket_symb
            for i in i_locs:
                if i[0]+ half_w < 0 or i[0]+ half_w > width or i[1] < 0 or i[1] > height:
                    continue
                al_ix = int((i[0] + half_w) / x_downsample)
                al_iy = int((i[1]) / y_downsample)
                alternative_matrix[al_ix, al_iy] = inter_symb
            for c in c_locs:
                al_cx = int((c[0] + half_w) / x_downsample)
                al_cy = 0
                alternative_matrix[al_cx, al_cy] = city_symb
            # plt.matshow(alternative_matrix)
            Draw2(alternative_matrix, score)
            print("Step: " + str(stp))
            print("r_locs: " + str(r_locs))
            print("i_locs: " + str(i_locs))
            print("c_locs: " + str(c_locs))
            print("and: " + str(ang))
            print("score: " + str(score))
            # time.sleep(1)
            # Draw()

