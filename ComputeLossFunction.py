import tensorflow as tf


def compute_loss(L, t_f, E, K, model, batch_vars, random_vars, batch_size, real_solution):

  # Termos fonte para input na DeepONet branch
  U = tf.convert_to_tensor(batch_vars[0])
  P = tf.convert_to_tensor(batch_vars[1])

  # Condições iniciais para input na DeepONet branch
  u0 = tf.convert_to_tensor(batch_vars[2])
  p0 = tf.convert_to_tensor(batch_vars[3])

  # Treinamento em x e t aleatórios
  x = tf.convert_to_tensor(random_vars[0])
  t = tf.convert_to_tensor(random_vars[1])

  with tf.GradientTape(persistent=True) as g:
      g.watch([x, t])
      with tf.GradientTape(persistent=True) as gg:
          gg.watch([x, t])
          trunk_input = tf.concat([x, t], axis=-1)
          u, p = model({'branch_input_u0': u0, 'branch_input_p0': p0, 'branch_input_U': U, 'branch_input_P': P, 'trunk_input': trunk_input})
      u_x = gg.gradient(u, x)
      p_x = gg.gradient(p, x)
  u_xx = g.gradient(u_x, x)
  u_xt = g.gradient(u_x, t)
  p_xx = g.gradient(p_x, x)

  loss_eq_1 = tf.reduce_mean(tf.square(- E * u_xx + p_x - random_vars[2]))
  loss_eq_2 = tf.reduce_mean(tf.square(u_xt - K * p_xx - random_vars[3]))
  loss_eq = loss_eq_1 / E ** 2 + loss_eq_2

  loss_u_real = tf.reduce_mean(tf.abs((u - real_solution[0]) / real_solution[0]))
  loss_p_real = tf.reduce_mean(tf.abs((p - real_solution[1]) / real_solution[1]))

  # Treinamento no contorno esquerdo (x=0)

  x = tf.constant(0, shape=(batch_size, 1), dtype=tf.float64)
  t = tf.convert_to_tensor(random_vars[1])

  with tf.GradientTape(persistent=True) as g:
      g.watch([x, t])
      trunk_input = tf.concat([x, t], axis=-1)
      u, p = model({'branch_input_u0': u0, 'branch_input_p0': p0, 'branch_input_U': U, 'branch_input_P': P, 'trunk_input': trunk_input})
  u_x = g.gradient(u, x)

  loss_bc_u_l = tf.reduce_mean(tf.square(u_x))
  loss_bc_p_l = tf.cast(tf.reduce_mean(tf.square(p)), tf.float64)
  loss_bc_l = loss_bc_u_l + loss_bc_p_l

  # Treinamento no contorno direito (x=L)

  x = tf.constant(L, shape=(batch_size, 1), dtype=tf.float64)
  t = tf.convert_to_tensor(random_vars[1])

  with tf.GradientTape(persistent=True) as g:
      g.watch([x, t])
      trunk_input = tf.concat([x, t], axis=-1)
      u, p = model({'branch_input_u0': u0, 'branch_input_p0': p0, 'branch_input_U': U, 'branch_input_P': P, 'trunk_input': trunk_input})
  p_x = g.gradient(p, x)

  loss_bc_u_r = tf.cast(tf.reduce_mean(tf.square(u)), tf.float64)
  loss_bc_p_r = tf.reduce_mean(tf.square(p_x))
  loss_bc_r = loss_bc_u_r + loss_bc_p_r

  # Treinamento na condição inicial (t=0)

  x = tf.convert_to_tensor(random_vars[0])
  t = tf.constant(0, shape=(batch_size, 1), dtype=tf.float64)
  trunk_input = tf.concat([x, t], axis=-1)
  u, p = model({'branch_input_u0': u0, 'branch_input_p0': p0, 'branch_input_U': U, 'branch_input_P': P, 'trunk_input': trunk_input})

  loss_ic_u = tf.reduce_mean(tf.square(u - random_vars[4]))
  loss_ic_p = tf.reduce_mean(tf.square(p - random_vars[5]))
  loss_ic = tf.cast(loss_ic_u + loss_ic_p, tf.float64)

  # Loss total
  loss_total = loss_eq + loss_bc_l + loss_bc_r + loss_ic

  return loss_total, [loss_eq_1 / E ** 2, loss_eq_2, loss_bc_l, loss_bc_r, loss_ic], [loss_u_real, loss_p_real]
  
