def @main(%data: Tensor[(1, 32, 112, 112), float32], %graph_bn_gamma: Tensor[(32), float32], %graph_bn_beta: Tensor[(32), float32], %graph_bn_moving_mean: Tensor[(32), float32], %graph_bn_moving_var: Tensor[(32), float32]) 

-> Tensor[(1, 32, 112, 112), float32] 
{
  %0 = nn.batch_norm(%data, %graph_bn_gamma, %graph_bn_beta, %graph_bn_moving_mean, %graph_bn_moving_var) 
  %0.0
}