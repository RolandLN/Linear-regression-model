import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2017)

x_train = np.array([[3.3], [4.4],
                    [5.5], [6.71],
                    [6.93], [4.168],
                    [9.779], [6.182],
                    [7.59], [2.167],
                    [7.042], [10.791],
                    [5.313], [7.997], [3.1]],
                   dtype=np.float32)

y_train = np.array([[1.7], [2.76],
                    [2.09], [3.19],
                    [1.694], [1.573],
                    [3.366], [2.596],
                    [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]],
                   dtype=np.float32)

# %matplotlibinline
plt.plot(x_train, y_train, 'bo')
plt.show()

x = tf.constant(x_train, name='x')
y = tf.constant(y_train, name='y')

w = tf.Variable(initial_value=tf.random_normal(shape=(), seed=2017),
                dtype=tf.float32, name='weight')
b = tf.Variable(initial_value=0, dtype=tf.float32, name='biase')

with tf.variable_scope('Linear_Model'):
    y_pred = w*x+b

print(w.name)
print(y_pred.name)
# 开启交互式会话
sess = tf.InteractiveSession()
# 一定要有初始化这一步!!!
sess.run(tf.global_variables_initializer())
# %matplotlibinline
# 要先将`tensor`的内容`fetch`出来
y_pred_numpy = y_pred.eval(session=sess)
plt.plot(x_train, y_train, 'bo', label='real')
plt.plot(x_train, y_pred_numpy, 'ro', label='estimated')
plt.legend()
# plt.show()


loss = tf.reduce_mean(tf.square(y-y_pred))
# 看看在当前模型下的误差有多少
print(loss.eval(session=sess))

w_grad, b_grad = tf.gradients(loss, [w, b])

lr = 0.01
w_update = w.assign_sub(lr*w_grad)
b_update = b.assign_sub(lr*b_grad)
sess.run([w_update, b_update])
print('w_grad: %.4f' % w_grad.eval(session=sess))
print('b_grad: %.4f' % b_grad.eval(session=sess))

y_pred_numpy = y_pred.eval(session=sess)
# plt.plot(x_train, y_train, 'bo', label='real')
plt.plot(x_train, y_pred_numpy, 'go', label='estimated1')
plt.legend()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
# plt.ion()
# fig.show()
# fig.canvas.draw()
# sess.run(tf.global_variables_initializer())

for e in range(10):
    sess.run([w_update, b_update])
    y_pred_numpy = y_pred.eval(session=sess)
    loss_numpy = loss.eval(session=sess)
    ax.clear()
    ax.plot(x_train, y_train, 'bo', label='real')
    ax.plot(x_train, y_pred_numpy, 'ro', label='estimated')
    ax.legend()
    fig.canvas.draw()
    plt.pause(0.5)
    print('epoch: {}, loss: {}'.format(e, loss_numpy))

print('w_grad: %.4f' % w_grad.eval(session=sess))
print('b_grad: %.4f' % b_grad.eval(session=sess))

plt.plot(x_train, y_train, 'bo', label='real')
plt.plot(x_train, y_pred_numpy, 'ro', label='estimated')
plt.legend()

plt.show()
sess.close()
