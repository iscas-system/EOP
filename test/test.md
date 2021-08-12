[Var(data, ty=TensorType([1, 1, 224, 224], float32)), Var(conv1_weight, ty=TensorType([64, 1, 7, 7], float32))]
[Var(data, ty=TensorType([1, 1, 224, 224], float32)), Var(conv1_weight, ty=TensorType([64, 1, 7, 7], float32))]

[CallNode(Op(nn.conv2d), [Var(data, ty=TensorType([1, 1, 224, 224], float32)), Var(conv1_weight, ty=TensorType([64, 1, 7, 7], float32))], relay.attrs.Conv2DAttrs(0x561d81bab158), [TensorType([1, 1, 224, 224], float32), TensorType([64, 1, 7, 7], float32)]), Var(batch1_gamma, ty=TensorType([64], float32)), Var(batch1_beta, ty=TensorType([64], float32)), Var(batch1_moving_mean, ty=TensorType([64], float32)), Var(batch1_moving_var, ty=TensorType([64], float32))]


[Var(0, ty=TensorType([1, 64, 112, 112], float32)), Var(batch1_gamma, ty=TensorType([64], float32)), Var(batch1_beta, ty=TensorType([64], float32)), Var(batch1_moving_mean, ty=TensorType([64], float32)), Var(batch1_moving_var, ty=TensorType([64], float32))]