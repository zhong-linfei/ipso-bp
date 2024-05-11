import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer

class PSOOptimizer(Optimizer):
    def __init__(self, num_particles, c1, c2, w_min, w_max, name=None, **kwargs):
        super(PSOOptimizer, self).__init__(name, **kwargs)
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.w_min = w_min
        self.w_max = w_max
        
    def get_config(self):
        config = super(PSOOptimizer, self).get_config()
        config.update({
            'num_particles': self.num_particles,
            'c1': self.c1,
            'c2': self.c2,
            'w_min': self.w_min,
            'w_max': self.w_max,
        })
        return config
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'position')
            self.add_slot(var, 'velocity')
    
    def _prepare(self):
        self._learning_rate = self._get_hyper('learning_rate')
    
    def _compute_current_learning_rate(self, var_dtype):
        return self._learning_rate
    
    def _resource_apply_dense(self, grad, var):
        position = self.get_slot(var, 'position')
        velocity = self.get_slot(var, 'velocity')
        
        if position is None:
            position = tf.Variable(var.initialized_value(), trainable=False)
            self.add_slot(var, 'position', position)
        
        if velocity is None:
            velocity = tf.Variable(tf.zeros_like(var), trainable=False)
            self.add_slot(var, 'velocity', velocity)
        
        # 粒子更新规则
        w = self.w_max - (self.w_max - self.w_min) * self.iterations / self.max_iterations
        r1 = tf.random.uniform(var.shape)
        r2 = tf.random.uniform(var.shape)
        velocity.assign(w * velocity + self.c1 * r1 * (position - var) + self.c2 * r2 * (self.best_position - var))
        position.assign(position + velocity)
        
        # 更新参数
        var.assign(position)
        
        # 记录最佳位置
        if self.best_position is None:
            self.best_position = tf.Variable(tf.zeros_like(var), trainable=False)
            self.best_loss = tf.Variable(np.inf, trainable=False)
        loss = self._loss([var])
        if loss < self.best_loss:
            self.best_loss.assign(loss)
            self.best_position.assign(position)
    
    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")
